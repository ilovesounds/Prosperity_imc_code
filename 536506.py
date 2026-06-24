"""
IMC Prosperity Round 3 Trader — v4 (Final)

ROOT CAUSE ANALYSIS from v3 log (tradeHistory + activitiesLog):

v3 actual earned PnL: ~874 total
  HYDROGEL_PACK:       +537  (29 trades, avg ~6-8 tick adverse fill)
  VELVETFRUIT_EXTRACT: + 68
  VEV_4000:            + 94
  VEV_4500:            + 95
  VEV_5000:            -  4
  VEV_5100/5300/5400/5500: 0  (ZERO fills despite active markets!)
  VEV_5200:            + 84

KEY FINDINGS:
1. HYDROGEL_PACK: Mark 38 is the ONLY market maker (sole counterparty in 29/32 trades).
   His spread is consistently 16 ticks wide. In v3 we were filling at the MIDDLE of his
   spread paying 5-8 adverse ticks per trade instead of capturing 7-8 ticks.
   FIX: Post bid = market_bid + 1 and ask = market_ask - 1 (inside his spread).
   Never aggress (aggr_threshold=20 > spread of 16).

2. VELVETFRUIT_EXTRACT: Mark 55 is primary, avg spread ~5 ticks.
   She buys at 5264 avg, sells at 5256 avg — directional noise trader.
   FIX: Post inside, lot=40 for more volume capture.

3. VEV_5300/5400/5500: Mark 01 buys from Mark 22 at BID price — Mark 22 is the
   passive seller at the bid level. We need to undercut Mark 22 (post ask-1 to be
   at a better price than his ask so when Mark 01 lifts, he hits us first).
   FIX: quote_inside_market with lot=20.

4. VEV_5100: 0 trades in entire round despite liquid book. v3 aggr_threshold was 4,
   missing all opportunities. FIX: lower to 2-3, post more competitively.

5. VEV_6000/6500: Only trade at 0.0 between Mark 01 and Mark 22. Skip.

EXPECTED IMPROVEMENT per product (vs v3):
  HYDROGEL: +1200-1500 (better fill prices, same frequency)
  VEV_5300/5400/5500: +200-400 (now capturing Mark 01 flow)
  VEV_5100: +80-100 (now competitive)
  Other products: +50-100 from tighter spreads
  Total projected: ~2500-3000
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


# ─────────────────────────────────────────────────────────────────
# Position limits
# ─────────────────────────────────────────────────────────────────

LIMITS = {
    "HYDROGEL_PACK":       200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000":  300,
    "VEV_4500":  300,
    "VEV_5000":  300,
    "VEV_5100":  300,
    "VEV_5200":  300,
    "VEV_5300":  300,
    "VEV_5400":  300,
    "VEV_5500":  300,
    "VEV_6000":  300,
    "VEV_6500":  300,
}


# ─────────────────────────────────────────────────────────────────
# Persistent state (EWM fair value)
# ─────────────────────────────────────────────────────────────────

class State:
    def __init__(self):
        self.fair: Dict[str, float] = {}

    def update(self, product: str, new_mid: float, alpha: float = 0.15):
        """Slow EWM (alpha=0.15): smooth fair value estimate, resilient to noise."""
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
# Core: inside-spread quoting with inventory skew
# ─────────────────────────────────────────────────────────────────

def make_orders(
    product: str,
    od: OrderDepth,
    pos: int,
    fair: float,
    passive_lot: int,
    aggr_threshold: int,
    aggr_lot: int,
    max_pos_frac: float = 0.5,
    skew_ticks: int = 1,
) -> List[Order]:
    """
    Inside-spread market making.

    Passive quotes:
      bid_price = market_bid + 1 + skew   (best bid in book → gets first fills)
      ask_price = market_ask - 1 + skew   (best ask in book → gets first fills)

    Inventory skew: when long, shift both prices down to encourage sells.
    When |pos|/limit > max_pos_frac, start scaling lot down.
    When |pos|/limit > 0.8, suppress the side that adds to position.

    Aggressive: only when price is aggr_threshold ticks inside our fair value.
    """
    orders: List[Order] = []
    limit = LIMITS[product]
    bid, ask = best_bid_ask(od)

    # ── Aggressive orders ─────────────────────────────────────────────────
    if ask is not None and ask < fair - aggr_threshold:
        qty = min(-od.sell_orders[ask], buy_cap(product, pos), aggr_lot)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            pos += qty

    if bid is not None and bid > fair + aggr_threshold:
        qty = min(od.buy_orders[bid], sell_cap(product, pos), aggr_lot)
        if qty > 0:
            orders.append(Order(product, bid, -qty))
            pos -= qty

    # ── Inventory skew ────────────────────────────────────────────────────
    pos_frac = pos / limit
    skew = max(-skew_ticks, min(skew_ticks, round(-pos_frac * skew_ticks)))

    # ── Quote inside the spread ───────────────────────────────────────────
    if bid is not None and ask is not None:
        spread = ask - bid
        if spread >= 3:
            bid_price = bid + 1 + skew
            ask_price = ask - 1 + skew
        elif spread == 2:
            bid_price = bid + 1 + skew
            ask_price = ask - 1 + skew  # may equal bid_price, handled below
        else:
            # 1-tick spread: join best price on each side
            bid_price = bid + skew
            ask_price = ask + skew
    elif bid is not None:
        bid_price = bid + 1 + skew
        ask_price = int(math.ceil(fair + 2 + skew))
    elif ask is not None:
        bid_price = int(math.floor(fair - 2 + skew))
        ask_price = ask - 1 + skew
    else:
        return orders

    # Guard: bid must be strictly below ask
    if bid_price >= ask_price:
        ask_price = bid_price + 1

    # ── Scale lot size near position limit ───────────────────────────────
    abs_frac = abs(pos) / limit
    if abs_frac < max_pos_frac:
        scale = 1.0
    else:
        scale = max(0.1, 1.0 - (abs_frac - max_pos_frac) / (1.0 - max_pos_frac) * 0.9)
    lot = max(1, int(passive_lot * scale))

    # Suppress side that increases a large position
    long_heavy  = pos_frac >  0.8
    short_heavy = pos_frac < -0.8

    bq = 0 if long_heavy  else min(lot, buy_cap(product, pos))
    sq = 0 if short_heavy else min(lot, sell_cap(product, pos))

    if bq > 0:
        orders.append(Order(product, int(bid_price), bq))
    if sq > 0:
        orders.append(Order(product, int(ask_price), -sq))

    return orders


# ─────────────────────────────────────────────────────────────────
# Product configs: (lot, aggr_threshold, aggr_lot, max_pos_frac, skew_ticks)
# ─────────────────────────────────────────────────────────────────
#
# HYDROGEL_PACK: Mark 38 sole MM, 16-tick spread, ~12 units/side, limit=200
#   lot=30 fills comfortably vs his volume. aggr_threshold=20 (never aggress).
#
# VELVETFRUIT_EXTRACT: 5-tick spread, Mark 55 as noise trader. lot=40.
#
# VEV_4000/4500: Mark 38 counterparty, 20-21 tick spread. lot=20.
#
# VEV_5000-5200: 3-7 tick spread, multiple counterparties. lot=25.
#
# VEV_5300/5400/5500: Mark 01 buys from Mark 22 at bid. We undercut Mark 22.
#   1-2 tick spread. lot=20. Lower aggr_threshold to catch Mark 01 crossing.
#
# VEV_6000/6500: Skip (all @ 0.0, no edge).

CONFIGS = {
    "HYDROGEL_PACK":       (30, 20,  5, 0.5, 1),
    "VELVETFRUIT_EXTRACT": (40,  6, 15, 0.5, 1),
    "VEV_4000":  (20, 25,  5, 0.5, 1),
    "VEV_4500":  (20, 20,  5, 0.5, 1),
    "VEV_5000":  (25,  8, 10, 0.5, 1),
    "VEV_5100":  (25,  3, 10, 0.5, 1),
    "VEV_5200":  (25,  3, 15, 0.5, 1),
    "VEV_5300":  (20,  3, 15, 0.5, 1),
    "VEV_5400":  (20,  2, 15, 0.5, 1),
    "VEV_5500":  (20,  2, 15, 0.5, 1),
    "VEV_6000":  None,  # skip
    "VEV_6500":  None,  # skip
}

# Calibrated defaults from round 3 historical data
DEFAULTS = {
    "VELVETFRUIT_EXTRACT": 5270.0,
    "HYDROGEL_PACK":       10030.0,
    "VEV_4000":  1265.0,
    "VEV_4500":   765.0,
    "VEV_5000":   267.0,
    "VEV_5100":   174.0,
    "VEV_5200":    97.0,
    "VEV_5300":    44.5,
    "VEV_5400":    14.0,
    "VEV_5500":     4.5,
}


# ─────────────────────────────────────────────────────────────────
# Main Trader
# ─────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        st = State.from_json(state.traderData or "")
        result: Dict[str, List[Order]] = {}

        for product, od in state.order_depths.items():
            cfg = CONFIGS.get(product)
            if cfg is None:
                continue  # skip VEV_6000, VEV_6500

            pos = state.position.get(product, 0)

            # Update smoothed fair value (alpha=0.15)
            m = mid_price(od)
            if m is not None:
                st.update(product, m, alpha=0.15)

            fair = st.get(product, DEFAULTS.get(product, m or 0))
            if fair <= 0:
                continue

            lot, aggr_thresh, aggr_lot, max_pos_frac, skew_ticks = cfg

            orders = make_orders(
                product, od, pos, fair,
                lot, aggr_thresh, aggr_lot,
                max_pos_frac=max_pos_frac,
                skew_ticks=skew_ticks,
            )

            if orders:
                result[product] = orders

        return result, 0, st.to_json()