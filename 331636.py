from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

class Trader:
    def run(self, state: TradingState):
        result = {}
        conversions = 0

        # Deserialize persistent state
        if state.traderData:
            data = json.loads(state.traderData)
        else:
            data = {}

        POSITION_LIMITS = {
            'INTARIAN_PEPPER_ROOT': 80,
            'ASH_COATED_OSMIUM': 80
        }

        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders: List[Order] = []

            if not order_depth.sell_orders or not order_depth.buy_orders:
                result[product] = orders
                continue

            pos_limit = POSITION_LIMITS.get(product, 20)
            current_pos = state.position.get(product, 0)

            sell_orders = sorted(order_depth.sell_orders.items())
            buy_orders  = sorted(order_depth.buy_orders.items(), reverse=True)
            best_ask, _ = sell_orders[0]
            best_bid, _ = buy_orders[0]
            mid = (best_ask + best_bid) / 2.0

            # ══════════════════════════════════════════
            # PEPPER ROOT → buy max immediately, HODL forever
            # ══════════════════════════════════════════
            if product == 'INTARIAN_PEPPER_ROOT':
                buy_room = pos_limit - current_pos

                # Always try to fill to position limit by market-taking
                if buy_room > 0:
                    remaining = buy_room
                    for ask, vol in sell_orders:
                        if remaining <= 0:
                            break
                        qty = min(-vol, remaining)
                        orders.append(Order(product, ask, qty))
                        remaining -= qty

                # NEVER sell — pepper trends +1000/day every day
                # Unrealized PnL accumulates and is counted in final score

            # ══════════════════════════════════════════
            # OSMIUM → pure mean-reversion market making
            # ══════════════════════════════════════════
            elif product == 'ASH_COATED_OSMIUM':
                key = f'{product}_ema'
                ema = data.get(key, mid)
                ema = 0.02 * mid + 0.98 * ema  # slower EMA, less reactive to noise
                ema = max(9975.0, min(10025.0, ema))
                data[key] = ema

                buy_room  = pos_limit - current_pos
                sell_room = pos_limit + current_pos
                skew = current_pos / pos_limit

    # Snipe only clearly mispriced orders (ask well below fair)
                for bid, vol in buy_orders:
                    if bid > ema + 6 and sell_room > 0:
                        qty = min(vol, sell_room)
                        orders.append(Order(product, bid, -qty))
                        sell_room -= qty

                # Market make: quote aggressively inside the spread
                # Spread is ~16, so quoting ±4 from fair captures ~8 per round trip
                skew_adj = round(skew * 4)
                bid_price = int(ema) - 4 - skew_adj
                ask_price = int(ema) + 4 - skew_adj

                # Hard limits: never cross the market
                bid_price = min(bid_price, best_bid + 1)  # +1 to be best bid
                ask_price = max(ask_price, best_ask - 1)  # -1 to be best ask

                if buy_room > 0:
                    orders.append(Order(product, bid_price, buy_room))
                if sell_room > 0:
                    orders.append(Order(product, ask_price, -sell_room))

            result[product] = orders

        trader_data = json.dumps(data)
        return result, conversions, trader_data