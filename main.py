import decimal
import os
import pprint
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Mapping, Iterable, List

import numpy as np
import pandas as pd
import pandas_ta
import pybroker
import pytz
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas import Series, DataFrame
from pybroker import Strategy, ExecContext, TestResult, Alpaca, StrategyConfig, Position
from scipy.optimize import OptimizeResult
from skopt import Optimizer
from skopt.plots import plot_objective, plot_convergence, plot_evaluations
from skopt.space import Real, Integer

pybroker.enable_data_source_cache('skopt')
verbose: bool = True
debug: bool = True

start_date: datetime = datetime(2023, 1, 1, tzinfo=pytz.timezone('America/New_York')) - timedelta(days=1)
end_date: datetime = datetime(2023, 6, 30, tzinfo=pytz.timezone('America/New_York'))
# symbols: Iterable[str] = ['IYY', 'IWM', 'IVV']
# symbols: Iterable[str] = ['SPY', 'SPYV', 'SPYG', 'XLRE', 'XLY', 'XLI', 'XLF', 'XLB', 'XLK', 'XLC', 'XLE', 'XLP', 'XLV',
#                          'XLU']
symbols: Iterable[str] = ['QQQ', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META', 'AVGO', 'ASML', 'ORCL', 'CSCO',
                          'TXN', 'ADBE', 'CRM', 'NFLX', 'QCOM', 'IBM', 'SAP', 'INTC', 'INTU', 'AMD', 'SONY', 'ADP',
                          'AMAT', 'PYPL', 'BKNG', 'ADI', 'NOW', 'MU', 'ATVI', 'VMW', 'UBER', 'MELI', 'TEAM'
                          ]


def print_data_frame(data: [Series, DataFrame]):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


def backtest(
        roc_length: int = 8, quantile: float = .5,
        stop_loss_pct_long: float = 0.0, stop_loss_pct_short: float = 0.0,
        long_backoff: float = 0.5, short_backoff: float = 0.5,
) -> TestResult:
    def sell_long_position(ctx: ExecContext, symbol: str):
        position: Position = ctx.long_pos(symbol)
        if np.less(position.market_value, decimal.Decimal(0.04) * ctx.total_market_value):
            ctx.sell_fill_price = pybroker.PriceType.OPEN
            ctx.sell_all_shares()
        else:
            ctx.sell_shares = position.shares * Decimal(long_backoff)

    def cover_short_position(ctx: ExecContext, symbol: str):
        position: Position = ctx.short_pos(symbol)
        if np.less(position.market_value, decimal.Decimal(0.04) * ctx.total_market_value):
            ctx.cover_fill_price = pybroker.PriceType.OPEN
            ctx.cover_all_shares()
        else:
            ctx.cover_shares = position.shares * Decimal(short_backoff)

    # noinspection SpellCheckingInspection
    def before_exec(ctxs: Mapping[str, ExecContext]):
        margin_requirement: Decimal = Decimal(0.5)

        returns: Mapping[str, float] = {
            symbol: ctx.indicator('roc')[-1]
            for symbol, ctx in ctxs.items()
        }

        threshold = np.quantile(list(returns.values()), quantile)

        for symbol, ctx in ctxs.items():
            match np.sign(threshold):
                case -1:
                    signals = {key: value for key, value in returns.items() if value < threshold}
                    if ctx.long_pos(symbol):
                        sell_long_position(ctx, symbol)

                    if symbol in signals:
                        max_margin: Decimal = ctx.total_equity / margin_requirement - ctx.total_equity
                        available_margin: Decimal = max_margin - ctx.total_margin
                        ctx.sell_fill_price = pybroker.PriceType.OPEN
                        ctx.sell_shares = ctx.calc_target_shares(1 / len(signals), cash=float(available_margin))
                        ctx.stop_loss_pct = stop_loss_pct_short

                    elif symbol not in signals and ctx.short_pos(symbol):
                        cover_short_position(ctx, symbol)
                case 0:
                    return
                case 1:
                    signals = {key: value for key, value in returns.items() if value > threshold}
                    if ctx.short_pos(symbol):
                        cover_short_position(ctx, symbol)

                    if symbol in signals:
                        ctx.buy_fill_price = pybroker.PriceType.OPEN
                        ctx.buy_shares = ctx.calc_target_shares(1 / len(signals))
                        ctx.stop_loss_pct = stop_loss_pct_long

                    elif symbol not in signals and ctx.long_pos(symbol):
                        sell_long_position(ctx, symbol)

                case t if np.isnan(t):
                    raise NotImplementedError()
                case _:
                    raise NotImplementedError()

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
        if debug and ctx.bars == 6:
            print()

    def after_exec(ctx: Mapping[str, ExecContext]):
        print('after_exec')
        pprint.pprint(ctx)

    roc = pybroker.indicator('roc', lambda data: pandas_ta.roc(Series(data.close), length=roc_length))

    strategy: Strategy = Strategy(
        Alpaca(os.getenv('ALPACA_KEY_ID'), os.getenv('ALPACA_SECRET')), start_date, end_date,
        StrategyConfig(exit_on_last_bar=True)
    )

    strategy.set_before_exec(before_exec)
    strategy.add_execution(None, symbols, indicators=[roc])

    return strategy.backtest(timeframe='15m', warmup=roc_length)


def optimize() -> List[float]:
    search_space = [
        Integer(1, 32, name='roc_length'),
        Real(0, 1, name='quantile'),
        Real(0, 10, name='stop_loss_pct_long'),
        Real(0, 10, name='stop_loss_pct_short'),
        Real(0, 0.1, name='long_backoff'),
        Real(0, 0.1, name='short_backoff')
    ]
    optimizer = Optimizer(dimensions=search_space)
    x = optimizer.ask(n_points=12)
    test_results: List[TestResult] = Parallel(n_jobs=4)(
        delayed(backtest)(*params) for params in x
    )
    y = [-t.metrics.total_return_pct for t in test_results]
    optimizer.tell(x, y)

    optimization: OptimizeResult = optimizer.get_result()
    result: List[float] = optimization['x']
    print(f"Best optimization result:  {min(optimizer.yi)}")
    print(f"Best parameters:  {result}")

    dimensions = list([d.name for d in search_space])
    _ = plot_convergence(optimization, dimension_names=dimensions)
    _ = plot_evaluations(optimization, dimensions=dimensions)
    _ = plot_objective(optimization, dimensions=dimensions)
    plt.show()
    return result


def trade(
        roc_length: int = 8, quantile: float = .5,
        stop_loss_pct_long: float = 0.0, stop_loss_pct_short: float = 0.0,
        long_backoff: float = 0.5, short_backoff: float = 0.5,
) -> TestResult:
    return backtest(
        roc_length, quantile, stop_loss_pct_long, stop_loss_pct_short, long_backoff, short_backoff
    )


if __name__ == '__main__':
    optimum: List[float] = optimize()
    test_result: TestResult = trade(*optimum)
    print_data_frame(test_result.portfolio)
    print_data_frame(test_result.positions)
    print_data_frame(test_result.orders)
    print_data_frame(test_result.trades)
