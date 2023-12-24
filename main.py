import math
import os
from datetime import datetime, timedelta
from typing import Union, Mapping

import pandas_ta
import pybroker
import pytz
from pandas import Series, DataFrame
from pybroker import Strategy, Day, TestResult, ExecContext
from pybroker.data import DataSource, Alpaca

pybroker.scope.disable_logging()
pybroker.scope.disable_progress_bar()
pybroker.cache.disable_caches()


def print_data_frame(data: [Series, DataFrame]):
    import pandas as pd
    with pd.option_context('expand_frame_repr', False, 'display.max_rows', None, 'display.max_columns', None):
        print(data)


class StrategyAdapter(Strategy):

    def __init__(
            self,
            data_source: Union[DataSource, DataFrame],
            start_date_time: datetime, end_date_time: datetime,
            time_frame: str,
            roc_length: int
    ):
        # The concrete timestamp of the first candle
        self.start_date_time = start_date_time
        # The concrete timestamp of the last candle
        self.end_date_time = end_date_time
        # Changed naming from Alpaca timeframe to time_frame # don't copy and paste
        # Keep timeframe in Alpaca and PyBroker code
        self.time_frame = time_frame
        # Symbols traded in this 'Strategy'
        self.symbols = ['IYY', 'IWM', 'IVV']

        # Since I have just indicators on the daily time_frame I am trying to 'translate' them into lower timeframes
        # One trading day has 390 trading minutes
        # On the hourly time_frame we can use a float / Decimal to multiply by 6.5
        # On the daily time_frame we don't need this
        # If this is a smart idea??? - I don't know

        # This values also need to be changed if we take extended hours into account
        # Extended hours might be important for the indicator calculations but there is not much volume so trading
        # extended hours is not a good idea for me now
        # This raises another issues - between_time is used to filter the indicators but also for trading
        # What happens If you want to use the indicators for your strategy but not to trade in the extended hours?
        # That might be an issue for the next year^^

        # noinspection PyProtectedMember
        from pybroker.data import _parse_alpaca_timeframe as parse_alpaca_timeframe, TimeFrameUnit
        amount, unit = parse_alpaca_timeframe(self.time_frame)
        match unit:
            case TimeFrameUnit.Day:
                self.multiplier = 1
            case TimeFrameUnit.Hour:
                if 6.5 / amount % 2 != 0:
                    raise NotImplementedError()
                else:
                    self.multiplier = int(6.5 / amount)
            case TimeFrameUnit.Minute:
                if 390 / amount % 2 != 0:
                    raise NotImplementedError()
                else:
                    self.multiplier = int(390 / amount)
            case _:
                raise NotImplementedError()

        # Calculate the warmup period in days for this Strategy (max of every indicator used in the strategy)
        # In this example it is easy because we just use one indicator
        self.warmup = roc_length
        # Indicators use the length of the daily timeframe by the time_frame multiplier
        # Time will show if this is a smart idea - don't use for extended hours, etc
        self.indicators = [
            pybroker.indicator(
                'roc', lambda data: pandas_ta.roc(Series(data.close), length=self.warmup * self.multiplier)
            )
        ]

        # Example 1:
        # if you have timezones, and you pass them everything seems to work - That's good
        # super().__init__(data_source, start_date_time, end_date_time)

        # Example 2:
        # If you have timezones, and you want to subtract a warmup period to make start trading at the exact start_date_time
        # Also end_date_time should be checked, but I did not implement it - so don't use my code please

        # Example 2a:
        # Then things do not work as expected when running this code:
        # def execute_backtest(self) -> TestResult:
        #     result: TestResult = self.backtest(
        #        self.start_date_time, self.end_date_time, self.time_frame, ('9:30', '16:00'),
        #        [Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI]
        #     )
        #    return result

        # Example 2b:
        # Run this code, and the backtesting is finished correctly
        # def execute_backtest(self) -> TestResult:
        #    result: TestResult = self.backtest(
        #        timeframe=self.time_frame, between_time=('9:30', '16:00'),
        #        days=[Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI]
        #    )
        #    return result
        super().__init__(data_source, start_date_time - timedelta(days=self.warmup * 2), end_date_time)

        self.set_before_exec(self._weight)
        self.add_execution(self._execute, self.symbols, None, self.indicators)

    @staticmethod
    def _portfolio_stats(ctx: ExecContext):
        print(f"{ctx.symbol:<5s} {ctx.dt} {ctx.bars:>5d}: "
              f"O:{ctx.open[-1]:>10.4f} H:{ctx.open[-1]:10.4f} L:{ctx.low[-1]:10.4f} C:{ctx.close[-1]:10.4f} "
              f"V:{ctx.volume[-1]:>10.0f} | "
              f"TE:{ctx.total_equity:>12.2f} TMA:{ctx.total_margin:>12.2f} TMV:{ctx.total_market_value:>12.2f} | "
              f"Cash:{ctx.cash:>12.2f} "
              f"Long:{ctx.long_pos(ctx.symbol).market_value if ctx.long_pos(ctx.symbol) else 0:>12.2f} "
              f"Short:{ctx.short_pos(ctx.symbol).market_value if ctx.short_pos(ctx.symbol) else 0:>12.2f}"
              )

    def _weight(self, ctxs: Mapping[str, ExecContext]):
        # Only trade if all ctxs are available.
        # Sometimes data is missing for less liquid assets.
        # and then you have just 2 of 3 ctxs - this messes up indicator calculation and weights
        # But this is another issue I have solved with TimeScaleDB timeseries interpolation.
        # Don't worry that's not part of my issues.
        dt = set([c.dt for c in ctxs.values()])
        if len(dt) != 1:
            raise ValueError()

        # All ctxs should have the same date now
        dt = dt.pop()

        # Since example 2a does not work like expected we need to filter manually the candles that we want to trade
        # https://github.com/edtechre/pybroker/issues/51
        # Drop timezone to pass weighting this corrupts your results if you use a non timezone:
        # https://github.com/edtechre/pybroker/issues/69
        if self.start_date_time and dt <= self.start_date_time.replace(tzinfo=None) - timedelta(minutes=1):
            return

        # Weighting - irrelevant
        scores: Mapping[str, float] = {
            symbol: ctx.indicator('roc')[-1]
            for symbol, ctx in ctxs.items()
        }

        for (symbol, score) in scores.items():
            if symbol in ctxs:
                ctxs[symbol].score = float(score)

    def _execute(self, ctx: ExecContext):
        self._portfolio_stats(ctx)
        # Since example 2a does not work like expected we need to filter manually the candles that we want to trade
        # https://github.com/edtechre/pybroker/issues/51
        # Drop timezone to pass execution this corrupts your results if you use a non timezone:
        # https://github.com/edtechre/pybroker/issues/69
        if self.start_date_time and ctx.dt <= self.start_date_time.replace(tzinfo=None) - timedelta(minutes=1):
            return

        if ctx.score is None or math.isnan(ctx.score):
            return
        elif ctx.score is not None and ctx.score <= 0:
            ctx.sell_all_shares()
        elif ctx.score is not None and ctx.score > 0:
            ctx.buy_shares = ctx.calc_target_shares(ctx.score)
        else:
            raise NotImplementedError()

    def execute_backtest(self) -> TestResult:
        # between_time 9:30, 16:00 must be 'US/Eastern' timezone not your local timezone not UTC only 'US/Eastern'
        # https://github.com/edtechre/pybroker/issues/51

        # Example 2a:
        # https://github.com/edtechre/pybroker/issues/51
        # The backtest should start and finish at the self.start_date and self.end_date but when we pass those parameters
        # an error gets raised
        #   File /.../pybroker-experiments/lib/python3.11/site-packages/pandas/core/ops/invalid.py", line 36, in invalid_comparison
        #     raise TypeError(f"Invalid comparison between dtype={left.dtype} and {typ}")
        # TypeError: Invalid comparison between dtype=datetime64[ns] and datetime

        # result: TestResult = self.backtest(
        #    self.start_date_time, self.end_date_time, self.time_frame, ('9:30', '16:00'),
        #    [Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI]
        # )

        # Example 2b:
        # https://github.com/edtechre/pybroker/issues/51
        # Don't pass self.start_date_time and self.end_date_time and filter manually in _weight and _execute
        result: TestResult = self.backtest(
            timeframe=self.time_frame, between_time=('9:30', '16:00'),
            days=[Day.MON, Day.TUES, Day.WEDS, Day.THURS, Day.FRI]
        )
        return result


def main():
    alpaca = Alpaca(os.getenv('ALPACA_KEY_ID'), os.getenv('ALPACA_SECRET'))
    # Align trading on the US/Eastern timezone
    # https://github.com/edtechre/pybroker/issues/69
    start_date_time: datetime = datetime(2023, 2, 1, 9, 30, 0, 0, pytz.timezone('US/Eastern'))
    end_date_time: datetime = datetime(2023, 2, 28, 16, 00, 0, 0, pytz.timezone('US/Eastern'))

    # Align trading on the Europe/Berlin timezone
    # https://github.com/edtechre/pybroker/issues/69
    # start_date_time: datetime = datetime(2023, 2, 1, 15, 30, 0, 0, pytz.timezone('Europe/Berlin'))
    # end_date_time: datetime = datetime(2023, 2, 28, 22, 0, 0, 0, pytz.timezone('Europe/Berlin'))

    # Align trading on the UTC timezone
    # https://github.com/edtechre/pybroker/issues/69
    # start_date_time: datetime = datetime(2023, 2, 1, 15, 30, 0, 0, pytz.timezone('UTC'))
    # end_date_time: datetime = datetime(2023, 2, 28, 22, 0, 0, 0, pytz.timezone('UTC'))
    for roc in [2, 3, 4, 5]:
        print('----------------------------------------------------------------------------------------------------')
        print(f'ROC:{roc}-----------------------------------------------------------------------------------------------')
        print('----------------------------------------------------------------------------------------------------')
        strategy = StrategyAdapter(alpaca, start_date_time, end_date_time, '15m', roc)
        result = strategy.execute_backtest()

        print_data_frame(result.orders)
        print_data_frame(result.positions)
        print_data_frame(result.trades)
        print_data_frame(result.metrics_df)


if __name__ == '__main__':
    main()
