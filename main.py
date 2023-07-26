import decimal
import os
from datetime import datetime, timedelta, date
from decimal import Decimal
from typing import Mapping, List, Union

import numpy as np
import pandas
import pandas as pd
import pandas_ta
import pybroker
import pytz
from dateutil import rrule
from joblib import Parallel, delayed
from pandas import Series, DataFrame
from pybroker import Strategy, ExecContext, TestResult, Alpaca, StrategyConfig, Position
from scipy.optimize import OptimizeResult
from skopt import Optimizer
from skopt.space import Integer

pybroker.enable_data_source_cache('skopt')
pybroker.enable_indicator_cache('roc')

verbose: bool = True
debug: bool = True

# symbols: Iterable[str] = ['IYY', 'IWM', 'IVV']
# symbols: Iterable[str] = ['SPY', 'SPYV', 'SPYG', 'XLRE', 'XLY', 'XLI', 'XLF', 'XLB', 'XLK', 'XLC', 'XLE', 'XLP', 'XLV', 'XLU']

min_roc_lengh: int = 1
max_roc_lengh: int = 12 * 3


def print_data_frame(data: Union[Series, DataFrame]):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


def optimize(start_date: datetime, end_date: datetime) -> List[float]:
    search_space = [
        Integer(min_roc_lengh, max_roc_lengh, name='roc_length'),
        Integer(75, 100, name='quantile'),
        Integer(1, 10, name='stop_loss_pct_long'),
        Integer(1, 10, name='stop_loss_pct_short'),
        Integer(10, 100, name='long_backoff'),
        Integer(10, 100, name='short_backoff')
    ]
    optimizer: Optimizer = Optimizer(dimensions=search_space)
    x: List[List] = optimizer.ask(n_points=12)
    test_results: List[TestResult] = Parallel(n_jobs=-1)(
        delayed(backtest)(start_date, end_date, *params) for params in x
    )
    y: List = [-t.metrics.total_return_pct for t in test_results]
    optimizer.tell(x, y)

    optimization: OptimizeResult = optimizer.get_result()
    result: List[float] = optimization['x']
    print(f"Best optimization result: {min(optimizer.yi)}")
    print(f"Best parameters: {result}")

    import skopt.plots
    dimensions = list([_.name for _ in search_space])
    _ = skopt.plots.plot_convergence(optimization, dimension_names=dimensions)
    _ = skopt.plots.plot_evaluations(optimization, dimensions=dimensions)
    _ = skopt.plots.plot_objective(optimization, dimensions=dimensions)

    import matplotlib.pyplot as plt
    plt.show()
    return result


def walk_forward(start_date: datetime, end_date: datetime, interval: timedelta) -> None:
    import csv
    os.makedirs('./data', exist_ok=True)

    for d in rrule.rrule(
            rrule.DAILY, dtstart=start_date.date(), until=end_date.date(),
            byweekday=[rrule.MO, rrule.TU, rrule.WE, rrule.TH, rrule.FR]
    ):
        print("Calculating strategies for: %s" % d.date())
        optimum: List[float] = optimize(d - interval, d)
        with open('./data/params.csv', 'a+', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow([d.date()] + optimum)


def backtest(
        start_date: datetime, end_date: datetime,
        roc_length: int, quantile: int,
        stop_loss_pct_long: int, stop_loss_pct_short: int,
        long_backoff: int, short_backoff: int,
) -> TestResult:
    optima: DataFrame = DataFrame(
        index=pandas.date_range(start_date, end_date, freq='D'),
        columns=['roc_length', 'quantile', 'stop_loss_pct_long', 'stop_loss_pct_short', 'long_backoff', 'short_backoff']
        # columns=['roc_length', 'quantile', 'long_backoff', 'short_backoff']
        # columns=['roc_length', 'quantile', 'long_backoff']
    )
    optima.index = optima.index.date

    optima['roc_length'] = roc_length
    optima['quantile'] = quantile
    optima['stop_loss_pct_long'] = stop_loss_pct_long
    optima['stop_loss_pct_short'] = stop_loss_pct_short
    optima['long_backoff'] = long_backoff
    optima['short_backoff'] = short_backoff

    result: TestResult = trade(start_date, end_date, optima)
    return result


def trade(
        start_date: datetime, end_date: datetime, optima: DataFrame
) -> TestResult:
    def sell_long_position(ctx: ExecContext, symbol: str):
        long_backoff: int = optima.loc[ctx.dt.date()]['long_backoff']

        position: Position = ctx.long_pos(symbol)
        ctx.sell_fill_price = pybroker.PriceType.OPEN
        if np.less(position.market_value, decimal.Decimal(0.05) * ctx.total_market_value):
            ctx.sell_all_shares()
        else:
            ctx.sell_shares = position.shares * Decimal(long_backoff / 100)

    def cover_short_position(ctx: ExecContext, symbol: str):
        short_backoff: int = optima.loc[ctx.dt.date()]['short_backoff']

        position: Position = ctx.short_pos(symbol)
        ctx.cover_fill_price = pybroker.PriceType.OPEN
        if np.less(position.market_value, decimal.Decimal(0.05) * ctx.total_market_value):
            ctx.cover_all_shares()
        else:
            ctx.cover_shares = position.shares * Decimal(short_backoff / 100)

    # noinspection SpellCheckingInspection
    def before_exec(ctxs: Mapping[str, ExecContext]):
        day: date = list([ctx.dt for symbol, ctx in ctxs.items()])[0].date()

        roc_length: int = int(optima.loc[day]['roc_length'])
        quantile: int = optima.loc[day]['quantile']
        stop_loss_pct_short: int = optima.loc[day]['stop_loss_pct_short']
        stop_loss_pct_long: int = optima.loc[day]['stop_loss_pct_long']

        returns: Mapping[str, int] = {
            symbol: ctx.indicator(f'roc_{roc_length}')[-1]
            for symbol, ctx in ctxs.items()
        }

        threshold = np.quantile(list(returns.values()), quantile / 100)

        for symbol, ctx in ctxs.items():
            match np.sign(threshold):
                case -1:
                    if ctx.long_pos(symbol):
                        sell_long_position(ctx, symbol)

                    signals = {key: value for key, value in returns.items() if value < threshold}
                    if symbol in signals:
                        margin_requirement: Decimal = Decimal(0.5)
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
                    if ctx.short_pos(symbol):
                        cover_short_position(ctx, symbol)

                    signals = {key: value for key, value in returns.items() if value > threshold}
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

    def after_exec(ctxs: Mapping[str, ExecContext]):
        print('after_exec')

    from collections import deque
    tickers = deque(pandas.read_html(
        'https://en.wikipedia.org/wiki/Nasdaq-100', attrs={'id': "constituents"}
    )[0]['Ticker'])
    tickers.remove('GEHC')

    indicators: List[pybroker.Indicator] = [
        pybroker.indicator(f'roc_{i}', lambda data, length: pandas_ta.roc(Series(data.close), length), length=i)
        for i in np.arange(min_roc_lengh, max_roc_lengh + 1)
    ]

    strategy: Strategy = Strategy(
        Alpaca(os.getenv('ALPACA_KEY_ID'), os.getenv('ALPACA_SECRET')),
        start_date, end_date,
        StrategyConfig(exit_on_last_bar=True)
    )
    # from pybroker import RandomSlippageModel
    # strategy.set_slippage_model(RandomSlippageModel(0.0, 0.01))

    strategy.set_before_exec(before_exec)
    strategy.add_execution(None, tickers, indicators=indicators)
    # strategy.set_after_exec(after_exec)

    from pybroker.common import Day
    return strategy.backtest(
        timeframe='30min',
        between_time=('9:30', '16:00'),
        days=[Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI],
        warmup=max_roc_lengh + 1
    )


def main():
    # exclusive when using Alpaca
    start_date: datetime = datetime(2022, 6, 1, tzinfo=pytz.timezone('America/New_York'))
    end_date: datetime = datetime(2022, 8, 2, tzinfo=pytz.timezone('America/New_York'))

    walk_forward(start_date, end_date, timedelta(weeks=2))

    optima: DataFrame = pandas.read_csv(
        './data/params.csv',
        names=['roc_length', 'quantile', 'stop_loss_pct_long', 'stop_loss_pct_short', 'long_backoff', 'short_backoff'],
        # names=['roc_length', 'quantile', 'long_backoff', 'short_backoff'],
        # names=['roc_length', 'quantile', 'long_backoff'],
        parse_dates=[0], index_col=0
    )
    optima.index = optima.index.date
    print_data_frame(optima)

    test_result: TestResult = trade(start_date, end_date, optima)
    print_data_frame(test_result.portfolio)
    print_data_frame(test_result.positions)
    print_data_frame(test_result.orders)
    print_data_frame(test_result.trades)


if __name__ == '__main__':
    main()
