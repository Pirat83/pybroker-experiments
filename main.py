import heapq
from datetime import datetime, timedelta
from statistics import mean
from typing import Mapping

import pandas as pd
import pybroker
import scipy.stats as stats
import talib
from pandas import Series, DataFrame
from pybroker import Strategy, ExecContext, TestResult, StrategyConfig, Day

from extensions.alpaca import AlpacaAdjustedPricesDataSource

pybroker.disable_caches()

debug: bool = True
verbose: bool = False

start_date: datetime = datetime(2024, 7, 1)
end_date: datetime = datetime(2024, 7, 27)

n = 3

multiplier: int = 26
# warmup: int = 25 * 2
warmup: int = 5 * 2


def print_data_frame(data: [Series, DataFrame]):
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


def before_exec(ctxs: Mapping[str, ExecContext]):
    dt = {c.dt for c in ctxs.values()}
    dt = dt.pop()
    if dt < start_date:
        return

    returns_1 = [
        ctx.indicator('midpoint_roc_1')[-1]
        for ctx in ctxs.values()
    ]
    returns_5 = [
        ctx.indicator('midpoint_roc_5')[-1]
        for ctx in ctxs.values()
    ]

    pos_1: Mapping[str, float] = {
        symbol: stats.percentileofscore(returns_1, ctx.indicator('midpoint_roc_1')[-1], nan_policy='omit')
        for symbol, ctx in ctxs.items()
    }
    pos_5: Mapping[str, float] = {
        symbol: stats.percentileofscore(returns_5, ctx.indicator('midpoint_roc_5')[-1], nan_policy='omit')
        for symbol, ctx in ctxs.items()
    }

    scores: Mapping[str, float] = {symbol: mean([
        pos_1.get(symbol), pos_5.get(symbol)
    ]) for symbol, ctx in ctxs.items()}

    top_scores = heapq.nlargest(n, scores, key=scores.get)

    for symbol, ctx in ctxs.items():
        ctx.score = scores.get(symbol)
        if symbol in top_scores:
            ctx.buy_shares = ctx.calc_target_shares(1 / n)
        else:
            ctx.sell_all_shares()
    return None


def exec_fn(ctx: ExecContext):
    if ctx.dt < start_date:
        return
    if debug:
        print(f"{ctx.symbol:<5s} {ctx.dt} {ctx.bars:>5d}: Score:{ctx.score if ctx.score else 0:>12.2f}")
    if verbose:
        print(f"{ctx.symbol:<5s} {ctx.dt} {ctx.bars:>5d}: "
              f"O:{ctx.open[-1]:>10.4f} H:{ctx.open[-1]:10.4f} L:{ctx.low[-1]:10.4f} C:{ctx.close[-1]:10.4f} "
              f"V:{ctx.volume[-1]:>10.0f} | "
              f"TE:{ctx.total_equity:>12.2f} TMA:{ctx.total_margin:>12.2f} TMV:{ctx.total_market_value:>12.2f} | "
              f"Cash:{ctx.cash:>12.2f} "
              f"Long:{ctx.long_pos(ctx.symbol).market_value if ctx.long_pos(ctx.symbol) else 0:>12.2f} "
              f"Short:{ctx.short_pos(ctx.symbol).market_value if ctx.short_pos(ctx.symbol) else 0:>12.2f}"
              )


def midpoint_roc(data, length):
    mp = (data.high + data.low) / 2
    roc = talib.ROC(mp, length)
    return roc


def main():
    midpoint_roc_1 = pybroker.indicator('midpoint_roc_1', midpoint_roc, length=1 * multiplier)
    midpoint_roc_5 = pybroker.indicator('midpoint_roc_5', midpoint_roc, length=5 * multiplier)

    basic_etf_tickers = ['IYY', 'IWM', 'IVV']
    s_and_p_sector_etfs = [
        'SPY',  # S&P U.S. 500 ETF
        'XLC',  # S&P U.S. Communication Services ETF
        'XLY',  # S&P U.S. Consumer Discretionary ETF
        'XLP',  # S&P U.S. Consumer Staples ETF
        'XLE',  # S&P U.S. Energy ETF
        'XLF',  # S&P U.S. Financials ETF
        'XLV',  # S&P U.S. Health Care ETF
        'XLI',  # S&P U.S. Industrial ETF
        'XLB',  # S&P U.S. Basic Materials ETF
        'XLK',  # S&P U.S. Technology ETF
        'XLU',  # S&P U.S. Utilities ETF
    ]
    magnificent_7_tickers = [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'AMZN',  # Amazon
        'GOOGL',  # Alphabet (Class A)
        'GOOG',  # Alphabet (Class C)
        'META',  # Meta Platforms
        'NVDA',  # Nvidia
        'TSLA'  # Tesla
    ]

    strategy: Strategy = Strategy(
        AlpacaAdjustedPricesDataSource(),
        start_date - timedelta(days=warmup), end_date,
        StrategyConfig(exit_on_last_bar=True)
    )
    strategy.set_before_exec(before_exec)
    strategy.add_execution(exec_fn, s_and_p_sector_etfs, indicators=[midpoint_roc_1, midpoint_roc_5])

    result: TestResult = strategy.backtest(
        start_date - timedelta(days=warmup), end_date, '15m',
        ('9:30', '16:00'), [Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI],
    )

    if debug:
        print_data_frame(result.portfolio)
        print_data_frame(result.orders)
        print_data_frame(result.positions)
        print_data_frame(result.trades)
    print_data_frame(result.metrics_df)


if __name__ == '__main__':
    main()
