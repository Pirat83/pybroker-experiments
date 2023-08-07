import decimal
import os
from datetime import datetime, timedelta
from typing import Mapping

import numpy as np
import pandas as pd
import pandas_ta
from pandas import Series, DataFrame
from pybroker import Strategy, ExecContext, TestResult, Alpaca, StrategyConfig, PriceType

debug: bool = False
verbose: bool = False

start_date: datetime = datetime(2023, 8, 1)
end_date: datetime = datetime(2023, 8, 7)


def print_data_frame(data: [Series, DataFrame]):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


# noinspection SpellCheckingInspection
def before_exec(ctxs: Mapping[str, ExecContext]):
    dt = np.all([c.dt for c in ctxs.values()])
    if dt <= start_date - timedelta(days=1):
        return

    returns: Mapping[str, float] = {
        symbol: ctx.indicator('roc')[-1]
        for symbol, ctx in ctxs.items()
    }

    threshold = np.quantile(list(returns.values()), 0.5)
    match np.sign(threshold):
        case -1:
            signals = {key: value for key, value in returns.items() if value < threshold}
            for symbol, ctx in ctxs.items():
                if ctx.long_pos(symbol):
                    ctx.sell_all_shares()
                short_position = ctx.short_pos(symbol)
                if symbol in signals:
                    ctx.sell_fill_price = PriceType.OPEN
                    ctx.sell_shares = ctx.calc_target_shares(1 / len(signals))
                    ctx.stop_loss_pct = 2.75

                elif symbol not in signals and short_position:
                    if np.less(short_position.market_value, decimal.Decimal(0.04) * ctx.total_market_value):
                        ctx.cover_fill_price = PriceType.OPEN
                        ctx.cover_all_shares()
                    else:
                        ctx.cover_shares = short_position.shares / 2
        case 1:
            signals = {key: value for key, value in returns.items() if value > threshold}
            for symbol, ctx in ctxs.items():
                if ctx.short_positions(symbol):
                    ctx.cover_all_shares()
                long_position = ctx.long_pos(symbol)
                if symbol in signals:
                    ctx.buy_fill_price = PriceType.OPEN
                    ctx.buy_shares = ctx.calc_target_shares(1 / len(signals))
                    ctx.stop_loss_pct = 2.75

                elif symbol not in signals and long_position:
                    if np.less(long_position.market_value, decimal.Decimal(0.04) * ctx.total_market_value):
                        ctx.sell_fill_price = PriceType.OPEN
                        ctx.sell_all_shares()
                    else:
                        ctx.sell_shares = long_position.shares / 2
        case x if np.isnan(x):
            return
        case x if x == 0:
            return
        case _:
            return


def exec_fn(ctx: ExecContext):
    if verbose:
        print(f"{ctx.symbol:<5s} {ctx.dt} {ctx.bars:>5d}: "
              f"O:{ctx.open[-1]:>10.4f} H:{ctx.open[-1]:10.4f} L:{ctx.low[-1]:10.4f} C:{ctx.close[-1]:10.4f} "
              f"V:{ctx.volume[-1]:>10.0f} | "
              f"TE:{ctx.total_equity:>12.2f} TMA:{ctx.total_margin:>12.2f} TMV:{ctx.total_market_value:>12.2f} | "
              f"Cash:{ctx.cash:>12.2f} "
              f"Long:{ctx.long_pos(ctx.symbol).market_value if ctx.long_pos(ctx.symbol) else 0:>12.2f} "
              f"Short:{ctx.short_pos(ctx.symbol).market_value if ctx.short_pos(ctx.symbol) else 0:>12.2f}"
              )

    if ctx.dt <= start_date - timedelta(days=1):
        return


def main():
    warmup: int = 5
    import pybroker
    roc = pybroker.indicator('roc', lambda data: pandas_ta.roc(Series(data.close), length=warmup))

    strategy: Strategy = Strategy(
        Alpaca(os.getenv('ALPACA_KEY_ID'), os.getenv('ALPACA_SECRET')),
        start_date - timedelta(days=warmup * 2), end_date,
        StrategyConfig(initial_cash=10000, exit_on_last_bar=True)
    )
    strategy.set_before_exec(before_exec)
    strategy.add_execution(exec_fn, ['IYY', 'IWM', 'IVV'], indicators=[roc])

    result: TestResult = strategy.backtest(start_date - timedelta(days=warmup * 2), end_date, timeframe='1d')

    print_data_frame(result.portfolio)
    if debug:
        print_data_frame(result.orders)
        print_data_frame(result.positions)
        print_data_frame(result.trades)
        print_data_frame(result.metrics_df)


if __name__ == '__main__':
    main()
