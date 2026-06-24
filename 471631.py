"""
IMC Prosperity Round 3 Trader
Products: HYDROGEL_PACK, VELVETFRUIT_EXTRACT, VEV_4000..VEV_6500
Strategy:
  - HYDROGEL_PACK / VELVETFRUIT_EXTRACT: mean-reversion market making
  - VEV options: Black-Scholes fair value pricing with IV smile, market make around theoretical price
  - Deep ITM vouchers (VEV_4000, VEV_4500): buy aggressively since price ≈ intrinsic value
  - Near OTM vouchers (VEV_5200, VEV_5300): active market making
  - Far OTM (VEV_6000, VEV_6500): avoid / small sell positions
"""

from datamodel import (
    OrderDepth, TradingState, Order,
    ConversionObservation, Listing, Observation,
    ProsperityEncoder, Symbol, Product, Position, UserId, ObservationValue
)
from typing import Dict, List, Any
import json
import math


# ─────────────────────────────────────────────
# Black-Scholes helpers
# ─────────────────────────────────────────────

def norm_cdf(x: float) -> float:
    """Standard normal CDF using math.erfc for no-import approach."""
    return 0.5 * math.erfc(-x / math.sqrt(2))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Delta of a European call."""
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

POSITION_LIMITS = {
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

# Round 3 → TTE = 5 days at start; within the day timestamps run 0..999900
# We model T as days/365 for B-S (annualised vol)
# Historical IV smile (from day-2 calibration): roughly flat ~0.27 for ATM
# We use a simple vol smile table keyed by strike
IV_SMILE = {
    4000: 0.22,  # deep ITM — use intrinsic + small time value
    4500: 0.22,
    5000: 0.30,
    5100: 0.28,
    5200: 0.27,
    5300: 0.27,
    5400: 0.25,
    5500: 0.27,
    6000: 0.45,
    6500: 0.68,
}

# Strikes list
VOUCHER_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
VOUCHER_PRODUCTS = {K: f"VEV_{K}" for K in VOUCHER_STRIKES}

# Round 3 starts with TTE = 5 days.
# Each Solvenarian day has 1_000_000 timestamps.
# TTE decreases linearly from 5 days to ~0 over the round.
TTE_START_DAYS = 5          # days remaining at t=0 of this round
ROUND_DURATION_TS = 1_000_000
DAYS_PER_YEAR = 365.0


# ─────────────────────────────────────────────
# Trader State (persisted via traderData JSON)
# ─────────────────────────────────────────────

class TraderState:
    def __init__(self):
        self.vev_mid_history: List[float] = []          # last N VEV mid prices
        self.hydrogel_mid_history: List[float] = []
        self.vev_fair_value: float = 5250.0              # EWM fair value estimate

    def to_json(self) -> str:
        return json.dumps({
            "vev_mid_history": self.vev_mid_history[-50:],
            "hydrogel_mid_history": self.hydrogel_mid_history[-50:],
            "vev_fair_value": self.vev_fair_value,
        })

    @classmethod
    def from_json(cls, s: str) -> "TraderState":
        obj = cls()
        if not s:
            return obj
        try:
            d = json.loads(s)
            obj.vev_mid_history = d.get("vev_mid_history", [])
            obj.hydrogel_mid_history = d.get("hydrogel_mid_history", [])
            obj.vev_fair_value = d.get("vev_fair_value", 5250.0)
        except Exception:
            pass
        return obj


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────

def get_mid(od: OrderDepth) -> float | None:
    if od.buy_orders and od.sell_orders:
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        return (best_bid + best_ask) / 2.0
    if od.buy_orders:
        return float(max(od.buy_orders))
    if od.sell_orders:
        return float(min(od.sell_orders))
    return None


def clamp_qty(qty: int, product: str, position: int, side: str) -> int:
    limit = POSITION_LIMITS.get(product, 300)
    if side == "buy":
        return min(qty, limit - position)
    else:
        return min(qty, limit + position)


def ewm(history: List[float], alpha: float = 0.15) -> float:
    if not history:
        return 0.0
    val = history[0]
    for x in history[1:]:
        val = alpha * x + (1 - alpha) * val
    return val


# ─────────────────────────────────────────────
# Sub-strategies
# ─────────────────────────────────────────────

def trade_mean_reversion(
    product: str,
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    half_spread: float = 2,
    order_size: int = 10,
    aggressive_edge: float = 1.0,
) -> List[Order]:
    """
    Place passive limit orders around fair_value, and hit aggressive orders
    when price deviates significantly.
    """
    orders: List[Order] = []
    limit = POSITION_LIMITS[product]

    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

    # --- Aggressive: take mispricings ---
    if best_ask is not None and best_ask < fair_value - aggressive_edge:
        qty = min(-order_depth.sell_orders[best_ask], limit - position)
        if qty > 0:
            orders.append(Order(product, best_ask, qty))

    if best_bid is not None and best_bid > fair_value + aggressive_edge:
        qty = min(order_depth.buy_orders[best_bid], limit + position)
        if qty > 0:
            orders.append(Order(product, best_bid, -qty))

    # --- Passive market making ---
    bid_price = math.floor(fair_value - half_spread)
    ask_price = math.ceil(fair_value + half_spread)

    buy_qty = clamp_qty(order_size, product, position, "buy")
    sell_qty = clamp_qty(order_size, product, position, "sell")

    if buy_qty > 0:
        orders.append(Order(product, bid_price, buy_qty))
    if sell_qty > 0:
        orders.append(Order(product, ask_price, -sell_qty))

    return orders


def compute_voucher_fair_value(
    strike: int,
    S: float,
    timestamp: int,
) -> float:
    """
    Compute theoretical call price using Black-Scholes.
    T is time to expiry in years.
    At timestamp=0, TTE = TTE_START_DAYS.
    TTE decreases linearly to 0 at timestamp = ROUND_DURATION_TS.
    """
    tte_days = TTE_START_DAYS * (1.0 - timestamp / ROUND_DURATION_TS)
    tte_days = max(tte_days, 0.0001)
    T = tte_days / DAYS_PER_YEAR
    sigma = IV_SMILE.get(strike, 0.27)
    return bs_call_price(S, float(strike), T, sigma)


def trade_voucher(
    product: str,
    strike: int,
    order_depth: OrderDepth,
    position: int,
    S: float,
    timestamp: int,
) -> List[Order]:
    """
    Market make each voucher around its BS theoretical price.
    For deep ITM (4000, 4500): aggressively buy if price < intrinsic.
    For OTM (5400+): small position / avoid if far OTM.
    For near ATM (5000-5300): active market making.
    """
    orders: List[Order] = []
    limit = POSITION_LIMITS[product]
    fair = compute_voucher_fair_value(strike, S, timestamp)

    best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

    # Determine strategy by moneyness
    intrinsic = max(S - strike, 0.0)
    moneyness = S / strike  # >1 = ITM

    if strike in (4000, 4500):
        # Deep ITM: price ≈ intrinsic + small time value
        # Buy aggressively if price is below fair value
        if best_ask is not None and best_ask < fair - 1:
            qty = min(-order_depth.sell_orders[best_ask], limit - position)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        # Passive bid just below fair
        bid_price = int(fair - 2)
        buy_qty = clamp_qty(15, product, position, "buy")
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        # Passive ask just above fair
        ask_price = int(fair + 2)
        sell_qty = clamp_qty(15, product, position, "sell")
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

    elif strike in (5000, 5100, 5200, 5300):
        # Near ATM / slightly OTM: active market making
        edge = 2
        if best_ask is not None and best_ask < fair - edge:
            qty = min(-order_depth.sell_orders[best_ask], limit - position, 20)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        if best_bid is not None and best_bid > fair + edge:
            qty = min(order_depth.buy_orders[best_bid], limit + position, 20)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        bid_price = int(math.floor(fair - edge))
        ask_price = int(math.ceil(fair + edge))
        buy_qty = clamp_qty(10, product, position, "buy")
        sell_qty = clamp_qty(10, product, position, "sell")
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

    elif strike in (5400, 5500):
        # Slightly OTM: smaller sizes, wider spreads
        edge = 3
        if best_ask is not None and best_ask < fair - edge:
            qty = min(-order_depth.sell_orders[best_ask], limit - position, 10)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
        if best_bid is not None and best_bid > fair + edge:
            qty = min(order_depth.buy_orders[best_bid], limit + position, 10)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

        bid_price = max(1, int(math.floor(fair - edge)))
        ask_price = int(math.ceil(fair + edge))
        buy_qty = clamp_qty(5, product, position, "buy")
        sell_qty = clamp_qty(5, product, position, "sell")
        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

    else:
        # Far OTM (6000, 6500): sell small if overpriced, otherwise ignore
        if best_bid is not None and best_bid > fair + 1:
            qty = min(order_depth.buy_orders[best_bid], limit + position, 5)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))

    return orders


# ─────────────────────────────────────────────
# Main Trader class
# ─────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):
        # ── Load persistent state ──────────────────────────────────────────
        ts = TraderState.from_json(state.traderData or "")

        result: Dict[str, List[Order]] = {}
        timestamp = state.timestamp

        # ── Estimate VEV (underlying) fair value ───────────────────────────
        vev_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vev_mid = get_mid(vev_od) if vev_od else None

        if vev_mid is not None:
            ts.vev_mid_history.append(vev_mid)
        vev_fair = ewm(ts.vev_mid_history, alpha=0.2) if ts.vev_mid_history else 5250.0
        ts.vev_fair_value = vev_fair
        S = vev_fair  # underlying price for options

        # ── Estimate HYDROGEL fair value ───────────────────────────────────
        hyd_od = state.order_depths.get("HYDROGEL_PACK")
        hyd_mid = get_mid(hyd_od) if hyd_od else None
        if hyd_mid is not None:
            ts.hydrogel_mid_history.append(hyd_mid)
        hyd_fair = ewm(ts.hydrogel_mid_history, alpha=0.2) if ts.hydrogel_mid_history else 9990.0

        # ── Trade VELVETFRUIT_EXTRACT ──────────────────────────────────────
        if vev_od:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            orders = trade_mean_reversion(
                product="VELVETFRUIT_EXTRACT",
                order_depth=vev_od,
                position=pos,
                fair_value=vev_fair,
                half_spread=2,
                order_size=15,
                aggressive_edge=1,
            )
            if orders:
                result["VELVETFRUIT_EXTRACT"] = orders

        # ── Trade HYDROGEL_PACK ────────────────────────────────────────────
        if hyd_od:
            pos = state.position.get("HYDROGEL_PACK", 0)
            orders = trade_mean_reversion(
                product="HYDROGEL_PACK",
                order_depth=hyd_od,
                position=pos,
                fair_value=hyd_fair,
                half_spread=5,
                order_size=20,
                aggressive_edge=3,
            )
            if orders:
                result["HYDROGEL_PACK"] = orders

        # ── Trade VEV Vouchers (options) ───────────────────────────────────
        for strike in VOUCHER_STRIKES:
            product = VOUCHER_PRODUCTS[strike]
            od = state.order_depths.get(product)
            if od is None:
                continue
            pos = state.position.get(product, 0)
            orders = trade_voucher(
                product=product,
                strike=strike,
                order_depth=od,
                position=pos,
                S=S,
                timestamp=timestamp,
            )
            if orders:
                result[product] = orders

        # ── Persist state ──────────────────────────────────────────────────
        trader_data = ts.to_json()

        conversions = 0
        return result, conversions, trader_data