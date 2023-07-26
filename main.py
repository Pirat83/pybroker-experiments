import os
from datetime import datetime
from typing import Mapping, Union, Iterable

import pandas
import pandas_ta
import pybroker
import pytz
from pandas import Series, DataFrame
from pybroker import Strategy, ExecContext, TestResult, Alpaca, StrategyConfig

pybroker.enable_data_source_cache('skopt')
pybroker.enable_indicator_cache('roc')

verbose: bool = True
debug: bool = True

roc_lenght = 60

# symbols: Iterable[str] = ['IYY', 'IWM', 'IVV']
symbols: Iterable[str] = ['SPY', 'SPYV', 'SPYG', 'XLRE', 'XLY', 'XLI', 'XLF', 'XLB', 'XLK', 'XLC', 'XLE', 'XLP', 'XLV', 'XLU']


def print_data_frame(data: Union[Series, DataFrame]):
    with pandas.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


def trade(
        start_date: datetime, end_date: datetime
) -> TestResult:
    def rank(ctxs: Mapping[str, ExecContext]):
        scores = {
            symbol: ctx.indicator('roc')[-1]
            for symbol, ctx in ctxs.items()
        }
        sorted_scores = sorted(
            scores.items(),
            key=lambda score: score[1],
            reverse=True
        )
        threshold = pybroker.param('rank_threshold')
        top_scores = sorted_scores[:threshold]
        top_symbols = [score[0] for score in top_scores]
        pybroker.param('top_symbols', top_symbols)

    def rotate(ctx: ExecContext):
        if ctx.long_pos():
            if ctx.symbol not in pybroker.param('top_symbols'):
                ctx.sell_all_shares()
        else:
            target_size = pybroker.param('target_size')
            ctx.buy_shares = ctx.calc_target_shares(target_size)
            ctx.score = ctx.indicator('roc')[-1]

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

    # from collections import deque
    # tickers = deque(pandas.read_html(
    #     'https://en.wikipedia.org/wiki/Nasdaq-100', attrs={'id': "constituents"}
    # )[0]['Ticker'])
    # tickers.remove('GEHC')

    roc = pybroker.indicator('roc', lambda data: pandas_ta.roc(Series(data.close), roc_lenght))
    config = StrategyConfig(max_long_positions=4, exit_on_last_bar=True)
    strategy = Strategy(
        Alpaca(os.getenv('ALPACA_KEY_ID'), os.getenv('ALPACA_SECRET')),
        start_date, end_date,
        config
    )
    pybroker.param('target_size', 1 / config.max_long_positions)
    pybroker.param('rank_threshold', 4)

    strategy.set_before_exec(rank)
    strategy.add_execution(rotate, symbols, indicators=[roc])

    from pybroker.common import Day
    return strategy.backtest(
        timeframe='1m',
        between_time=('9:30', '16:00'),
        days=[Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI],
        warmup=roc_lenght
    )


def main():
    # exclusive when using Alpaca
    start_date: datetime = datetime(2022, 12, 31, tzinfo=pytz.timezone('America/New_York'))
    end_date: datetime = datetime(2023, 7, 26, tzinfo=pytz.timezone('America/New_York'))

    test_result: TestResult = trade(start_date, end_date)
    print_data_frame(test_result.portfolio)
    # print_data_frame(test_result.positions)
    # print_data_frame(test_result.orders)
    # print_data_frame(test_result.trades)


if __name__ == '__main__':
    main()
