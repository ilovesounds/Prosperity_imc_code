from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional
import json
import math


HYDROGEL = "HYDROGEL_PACK"
VELVET = "VELVETFRUIT_EXTRACT"
VOUCHER = "VELVETFRUIT_EXTRACT_VOUCHER"

HYDRO_MAKER = "Mark 38"
VELVET_FLOW = "Mark 55"

LIMITS = {
    HYDROGEL: 200,
    VELVET: 200,
    VOUCHER: 300,
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

DEFAULT_FAIRS = {
    HYDROGEL: 10030.0,
    VELVET: 5270.0,
    "VEV_4000": 1265.0,
    "VEV_4500": 765.0,
    "VEV_5000": 267.0,
    "VEV_5100": 174.0,
    "VEV_5200": 97.0,
    "VEV_5300": 44.5,
    "VEV_5400": 14.0,
    "VEV_5500": 4.5,
}

OPTION_CONFIGS = {
    "VEV_4000": (20, 25, 5, 2),
    "VEV_4500": (20, 20, 5, 2),
    "VEV_5000": (25, 8, 10, 2),
    "VEV_5100": (25, 3, 10, 1),
    "VEV_5200": (25, 3, 15, 1),
    "VEV_5300": (20, 3, 15, 1),
    "VEV_5400": (20, 2, 15, 1),
    "VEV_5500": (20, 2, 15, 1),
}


class State:
    def __init__(self):
        self.fair: Dict[str, float] = {}
        self.flow: Dict[str, float] = {}
        self.voucher_basis: float = 0.0

    def update_fair(self, product: str, mid: float, alpha: float) -> None:
        prev = self.fair.get(product, mid)
        self.fair[product] = alpha * mid + (1.0 - alpha) * prev

    def decay(self) -> None:
        for product, value in list(self.flow.items()):
            new_value = value * 0.65
            if abs(new_value) < 0.05:
                self.flow.pop(product, None)
            else:
                self.flow[product] = new_value

    def bump_flow(self, product: str, signed_qty: float) -> None:
        self.flow[product] = self.flow.get(product, 0.0) + signed_qty

    def get_fair(self, product: str, fallback: float) -> float:
        return self.fair.get(product, fallback)

    def to_json(self) -> str:
        return json.dumps(
            {
                "fair": self.fair,
                "flow": self.flow,
                "voucher_basis": self.voucher_basis,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "State":
        obj = cls()
        if not raw:
            return obj
        try:
            payload = json.loads(raw)
            obj.fair = payload.get("fair", {})
            obj.flow = payload.get("flow", {})
            obj.voucher_basis = payload.get("voucher_basis", 0.0)
        except Exception:
            pass
        return obj


def best_bid_ask(depth: OrderDepth) -> tuple[Optional[int], Optional[int]]:
    bid = max(depth.buy_orders) if depth.buy_orders else None
    ask = min(depth.sell_orders) if depth.sell_orders else None
    return bid, ask


def mid_price(depth: OrderDepth) -> Optional[float]:
    bid, ask = best_bid_ask(depth)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return float(bid)
    if ask is not None:
        return float(ask)
    return None


def buy_capacity(product: str, position: int) -> int:
    return max(0, LIMITS[product] - position)


def sell_capacity(product: str, position: int) -> int:
    return max(0, LIMITS[product] + position)


def scaled_lot(product: str, position: int, base_lot: int) -> int:
    pressure = abs(position) / LIMITS[product]
    if pressure <= 0.4:
        return base_lot
    scale = max(0.25, 1.0 - (pressure - 0.4) / 0.6 * 0.75)
    return max(1, int(round(base_lot * scale)))


def inventory_shift(product: str, position: int, max_ticks: int) -> int:
    if max_ticks <= 0:
        return 0
    return max(-max_ticks, min(max_ticks, round(-position / LIMITS[product] * max_ticks)))


def add_passive_quotes(
    orders: List[Order],
    product: str,
    depth: OrderDepth,
    position: int,
    buy_px: Optional[int],
    sell_px: Optional[int],
    lot: int,
) -> None:
    if buy_px is not None:
        buy_qty = min(lot, buy_capacity(product, position))
        if buy_qty > 0:
            orders.append(Order(product, int(buy_px), buy_qty))
    if sell_px is not None:
        sell_qty = min(lot, sell_capacity(product, position))
        if sell_qty > 0:
            orders.append(Order(product, int(sell_px), -sell_qty))


def quote_inside_spread(
    product: str,
    depth: OrderDepth,
    position: int,
    lot: int,
    shift: int = 0,
) -> List[Order]:
    orders: List[Order] = []
    bid, ask = best_bid_ask(depth)
    if bid is None and ask is None:
        return orders

    if bid is not None and ask is not None:
        spread = ask - bid
        if spread >= 2:
            buy_px = bid + 1 + shift
            sell_px = ask - 1 + shift
        else:
            buy_px = bid + shift
            sell_px = ask + shift
        if buy_px >= sell_px:
            sell_px = buy_px + 1
    elif bid is not None:
        buy_px = bid + 1 + shift
        sell_px = None
    else:
        buy_px = None
        sell_px = ask - 1 + shift

    add_passive_quotes(orders, product, depth, position, buy_px, sell_px, lot)
    return orders


def fair_value_quotes(
    product: str,
    depth: OrderDepth,
    position: int,
    fair: float,
    lot: int,
    half_width: int,
    aggressive_edge: int,
) -> List[Order]:
    orders: List[Order] = []
    bid, ask = best_bid_ask(depth)
    shift = inventory_shift(product, position, 3)

    if ask is not None and ask <= fair - aggressive_edge:
        qty = min(-depth.sell_orders[ask], buy_capacity(product, position), lot)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            position += qty

    if bid is not None and bid >= fair + aggressive_edge:
        qty = min(depth.buy_orders[bid], sell_capacity(product, position), lot)
        if qty > 0:
            orders.append(Order(product, bid, -qty))
            position -= qty

    target_bid = int(math.floor(fair - half_width + shift))
    target_ask = int(math.ceil(fair + half_width + shift))

    if bid is not None and ask is not None:
        spread = ask - bid
        if spread >= 2:
            buy_px = min(ask - 1, max(bid + 1, target_bid))
            sell_px = max(bid + 1, min(ask - 1, target_ask))
        else:
            buy_px = min(bid, target_bid)
            sell_px = max(ask, target_ask)
        if buy_px >= sell_px:
            buy_px = min(buy_px, sell_px - 1)
    elif bid is not None:
        buy_px = max(bid + 1, target_bid)
        sell_px = target_ask
    elif ask is not None:
        buy_px = target_bid
        sell_px = min(ask - 1, target_ask)
    else:
        return orders

    quote_lot = scaled_lot(product, position, lot)
    add_passive_quotes(orders, product, depth, position, buy_px, sell_px, quote_lot)
    return orders


def legacy_option_quotes(
    product: str,
    depth: OrderDepth,
    position: int,
    fair: float,
) -> List[Order]:
    lot, aggressive_edge, aggressive_lot, max_shift = OPTION_CONFIGS[product]
    orders: List[Order] = []
    bid, ask = best_bid_ask(depth)
    shift = inventory_shift(product, position, max_shift)

    if ask is not None and ask <= fair - aggressive_edge:
        qty = min(-depth.sell_orders[ask], buy_capacity(product, position), aggressive_lot)
        if qty > 0:
            orders.append(Order(product, ask, qty))
            position += qty

    if bid is not None and bid >= fair + aggressive_edge:
        qty = min(depth.buy_orders[bid], sell_capacity(product, position), aggressive_lot)
        if qty > 0:
            orders.append(Order(product, bid, -qty))
            position -= qty

    quote_lot = scaled_lot(product, position, lot)
    if bid is not None and ask is not None:
        spread = ask - bid
        if spread >= 3:
            buy_px = bid + 1 + shift
            sell_px = ask - 1 + shift
        elif spread == 2:
            buy_px = bid + 1 + shift
            sell_px = ask - 1 + shift
        else:
            buy_px = bid + shift
            sell_px = ask + shift
        if buy_px >= sell_px:
            sell_px = buy_px + 1
    elif bid is not None:
        buy_px = bid + 1 + shift
        sell_px = int(math.ceil(fair + 2 + shift))
    elif ask is not None:
        buy_px = int(math.floor(fair - 2 + shift))
        sell_px = ask - 1 + shift
    else:
        return orders

    add_passive_quotes(orders, product, depth, position, buy_px, sell_px, quote_lot)
    return orders


def update_counterparty_signal(st: State, state: TradingState) -> None:
    st.decay()

    for product, trades in state.market_trades.items():
        for trade in trades:
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            qty = getattr(trade, "quantity", 0)
            if product == VELVET:
                if buyer == VELVET_FLOW:
                    st.bump_flow(product, qty)
                if seller == VELVET_FLOW:
                    st.bump_flow(product, -qty)

    for product, trades in state.own_trades.items():
        for trade in trades:
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            qty = getattr(trade, "quantity", 0)
            if product == VELVET:
                if buyer == VELVET_FLOW and seller == "SUBMISSION":
                    st.bump_flow(product, qty)
                if seller == VELVET_FLOW and buyer == "SUBMISSION":
                    st.bump_flow(product, -qty)


class Trader:
    def run(self, state: TradingState):
        st = State.from_json(state.traderData or "")
        update_counterparty_signal(st, state)

        result: Dict[str, List[Order]] = {}

        for product in (HYDROGEL, VELVET, VOUCHER):
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            mid = mid_price(depth)
            if mid is not None:
                alpha = 0.12 if product == HYDROGEL else 0.18
                st.update_fair(product, mid, alpha)

        for product in OPTION_CONFIGS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            mid = mid_price(depth)
            if mid is not None:
                st.update_fair(product, mid, 0.15)

        if HYDROGEL in state.order_depths:
            hydro_pos = state.position.get(HYDROGEL, 0)
            hydro_shift = inventory_shift(HYDROGEL, hydro_pos, 2)
            hydro_lot = scaled_lot(HYDROGEL, hydro_pos, 24)
            result[HYDROGEL] = quote_inside_spread(
                HYDROGEL,
                state.order_depths[HYDROGEL],
                hydro_pos,
                hydro_lot,
                shift=hydro_shift,
            )

        if VELVET in state.order_depths:
            velvet_pos = state.position.get(VELVET, 0)
            velvet_mid = mid_price(state.order_depths[VELVET]) or DEFAULT_FAIRS[VELVET]
            base_fair = st.get_fair(VELVET, velvet_mid)

            # Mark 55 was the dominant counterparty in historical fills and showed
            # persistent directional flow, so we lean our fair in her direction.
            flow_bias = max(-3, min(3, round(st.flow.get(VELVET, 0.0) / 20.0)))
            velvet_fair = base_fair + flow_bias
            result[VELVET] = fair_value_quotes(
                VELVET,
                state.order_depths[VELVET],
                velvet_pos,
                velvet_fair,
                lot=36,
                half_width=1,
                aggressive_edge=3,
            )

        if VOUCHER in state.order_depths:
            voucher_pos = state.position.get(VOUCHER, 0)
            voucher_depth = state.order_depths[VOUCHER]
            voucher_mid = mid_price(voucher_depth)
            velvet_depth = state.order_depths.get(VELVET)
            velvet_fallback_mid = mid_price(velvet_depth) if velvet_depth is not None else None
            extract_fair = st.get_fair(
                VELVET,
                velvet_fallback_mid or DEFAULT_FAIRS[VELVET],
            )

            if voucher_mid is not None:
                observed_basis = voucher_mid - 10.0 * extract_fair
                st.voucher_basis = 0.08 * observed_basis + 0.92 * st.voucher_basis
                st.update_fair(VOUCHER, voucher_mid, 0.12)

            raw_voucher_fair = 10.0 * extract_fair + st.voucher_basis
            fallback_voucher_fair = st.get_fair(VOUCHER, raw_voucher_fair)
            voucher_fair = 0.65 * raw_voucher_fair + 0.35 * fallback_voucher_fair

            result[VOUCHER] = fair_value_quotes(
                VOUCHER,
                voucher_depth,
                voucher_pos,
                voucher_fair,
                lot=24,
                half_width=2,
                aggressive_edge=6,
            )

        for product in OPTION_CONFIGS:
            depth = state.order_depths.get(product)
            if depth is None:
                continue
            fair = st.get_fair(product, DEFAULT_FAIRS[product])
            position = state.position.get(product, 0)
            orders = legacy_option_quotes(product, depth, position, fair)
            if orders:
                result[product] = orders

        return result, 0, st.to_json()