"""
IMC Prosperity Round 3 Trader — v3

Root cause fixes from v1/v2 analysis:
  v1 bug: IVs were ~0.27 (underestimate) → sold options too cheaply → short options lost
  v2 bug: IVs were ~0.30-0.33 (overestimate) → bot thought ask < fair → bought ask aggressively
          every tick, hit position limit of 300 immediately, paid spread on every fill = huge loss

v3 approach: NEVER use hardcoded IVs to decide aggression direction.
  Instead:
  - Treat the market mid as the best estimate of fair value (calibrated EWM)
  - For options: post passive bids at bid_price+1 and asks at ask_price-1 (inside spread)
  - Only take aggressive fills when price is STRICTLY inside the current spread (true mispricings)
  - Inventory skew: widen/flip quotes as position grows
  - HYDROGEL + VEV: clean mean-reversion market making
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


# ─────────────────────────────────────────────────────────────────
# Position limits
# ─────────────────────────────────────────────────────────────────

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300,
    "VEV_4500": 300,
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 300,
    "VEV_5400": 300,
    "VEV_5500": 300,
    "VEV_6000": 300,
    "VEV_6500": 300,
}

VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_PRODUCTS = {K: f"VEV_{K}" for K in VOUCHER_STRIKES}


# ─────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────

class State:
    def __init__(self):
        # EWM fair value per product (smoothed mid price)
        self.fair: Dict[str, float] = {}

    def update(self, product: str, new_mid: float, alpha: float = 0.3):
        prev = self.fair.get(product, new_mid)
        self.fair[product] = alpha * new_mid + (1 - alpha) * prev

    def get(self, product: str, default: float) -> float:
        return self.fair.get(product, default)

    def to_json(self) -> str:
        return json.dumps({"fair": self.fair})

    @classmethod
    def from_json(cls, s: str) -> "State":
        obj = cls()
        if not s:
            return obj
        try:
            d = json.loads(s)
            obj.fair = d.get("fair", {})
        except Exception:
            pass
        return obj


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def best_bid_ask(od: OrderDepth):
    bid = max(od.buy_orders) if od.buy_orders else None
    ask = min(od.sell_orders) if od.sell_orders else None
    return bid, ask


def mid_price(od: OrderDepth) -> Optional[float]:
    bid, ask = best_bid_ask(od)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None


def buy_cap(product: str, pos: int) -> int:
    return max(0, LIMITS[product] - pos)


def sell_cap(product: str, pos: int) -> int:
    return max(0, LIMITS[product] + pos)


# ─────────────────────────────────────────────────────────────────
# Core market-making logic
# ─────────────────────────────────────────────────────────────────

def make_orders(
    product: str,
    od: OrderDepth,
    pos: int,
    fair: float,
    passive_spread: int,   # half-width of our passive quotes around fair
    passive_lot: int,      # size of each passive quote
    aggr_threshold: int,   # how far inside spread to hit aggressively (> 0)
    aggr_lot: int,         # max aggressive fill size
    max_pos_frac: float = 0.7,  # start skewing when |pos|/limit > this
) -> List[Order]:
    """
    Two-sided market making:
    - Post passive bid at (fair - passive_spread), ask at (fair + passive_spread)
    - Skew quotes toward flattening inventory
    - Only hit aggressive if price crosses fair by aggr_threshold
    """
    orders: List[Order] = []
    limit = LIMITS[product]

    bid, ask = best_bid_ask(od)

    # ── Aggressive: only when price is clearly wrong vs our fair ──────────
    # Buy only if someone sells BELOW (fair - aggr_threshold)
    if ask is not None and ask < fair - aggr_threshold:
        qty = min(-od.sell_orders[ask], buy_cap(product, pos), aggr_lot)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            pos += qty

    # Sell only if someone buys ABOVE (fair + aggr_threshold)
    if bid is not None and bid > fair + aggr_threshold:
        qty = min(od.buy_orders[bid], sell_cap(product, pos), aggr_lot)
        if qty > 0:
            orders.append(Order(product, bid, -qty))
            pos -= qty

    # ── Inventory skew: fade quotes as position grows ─────────────────────
    skew_ratio = pos / limit  # range [-1, 1]
    # When long (pos > 0), lower both bid and ask to encourage selling
    # When short (pos < 0), raise both to encourage buying
    skew = -round(skew_ratio * passive_spread)

    bid_price = int(math.floor(fair - passive_spread + skew))
    ask_price = int(math.ceil(fair + passive_spread + skew))

    # Ensure bid < ask
    if bid_price >= ask_price:
        bid_price = ask_price - 1

    # Scale down lot size as position approaches limit
    pos_frac = abs(pos) / limit
    scale = max(0.2, 1.0 - max(0.0, pos_frac - max_pos_frac) / (1.0 - max_pos_frac))
    lot = max(1, int(passive_lot * scale))

    bq = min(lot, buy_cap(product, pos))
    sq = min(lot, sell_cap(product, pos))

    if bq > 0:
        orders.append(Order(product, bid_price, bq))
    if sq > 0:
        orders.append(Order(product, ask_price, -sq))

    return orders


# ─────────────────────────────────────────────────────────────────
# Product-specific configs
# ─────────────────────────────────────────────────────────────────

# (passive_spread, passive_lot, aggr_threshold, aggr_lot)
CONFIGS = {
    "VELVETFRUIT_EXTRACT": (2,  25, 3,  30),
    "HYDROGEL_PACK":       (5,  25, 8,  30),
    # Options: spread is 1-6 ticks, we post 1 tick inside market, aggress only on clear cross
    "VEV_4000":  (3, 10, 8, 15),
    "VEV_4500":  (3, 10, 8, 15),
    "VEV_5000":  (2, 15, 5, 20),
    "VEV_5100":  (2, 15, 4, 20),
    "VEV_5200":  (1, 15, 3, 20),
    "VEV_5300":  (1, 15, 3, 20),
    "VEV_5400":  (1, 10, 2, 15),
    "VEV_5500":  (1, 10, 2, 15),
    # Far OTM: skip (no edge, prices stuck at 0-1)
    "VEV_6000":  None,
    "VEV_6500":  None,
}


# ─────────────────────────────────────────────────────────────────
# Main Trader
# ─────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        st = State.from_json(state.traderData or "")
        result: Dict[str, List[Order]] = {}

        all_products = list(state.order_depths.keys())

        for product in all_products:
            cfg = CONFIGS.get(product)
            if cfg is None:
                continue  # skip VEV_6000, VEV_6500

            od = state.order_depths[product]
            pos = state.position.get(product, 0)

            # Update fair value from current mid
            m = mid_price(od)
            if m is not None:
                st.update(product, m, alpha=0.3)

            # Get smoothed fair (fall back to raw mid or a sensible default)
            defaults = {
                "VELVETFRUIT_EXTRACT": 5262.0,
                "HYDROGEL_PACK": 9990.0,
                "VEV_4000": 1262.0, "VEV_4500": 762.0,
                "VEV_5000": 267.0,  "VEV_5100": 176.0,
                "VEV_5200": 102.0,  "VEV_5300": 50.0,
                "VEV_5400": 16.0,   "VEV_5500": 6.5,
            }
            fair = st.get(product, defaults.get(product, m or 0))

            if fair <= 0:
                continue

            passive_spread, passive_lot, aggr_thresh, aggr_lot = cfg
            orders = make_orders(
                product, od, pos, fair,
                passive_spread, passive_lot,
                aggr_thresh, aggr_lot,
            )

            if orders:
                result[product] = orders

        trader_data = st.to_json()
        return result, 0, trader_data