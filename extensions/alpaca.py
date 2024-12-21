from datetime import datetime, time
from typing import Union, Iterable, Optional, Any
from zoneinfo import ZoneInfo

import pandas as pd
import pybroker
from alpaca.data import Adjustment
from pybroker import Alpaca, DataCol

pybroker.disable_caches()
est_timezone = ZoneInfo("America/New_York")


class AlpacaAdjustedPricesDataSource(Alpaca):

    def __init__(self):
        super().__init__('PKY8WINK7WPVMBOVTXNI', 'Dojy6ooxdmMn7jSG7m89CVvloCIraa3BirxGoZf0')

    def query(
            self,
            symbols: Union[str, Iterable[str]],
            start_date: Union[str, datetime],
            end_date: Union[str, datetime],
            timeframe: Optional[str] = "1d",
            adjust: Optional[Any] = Adjustment.ALL,
    ) -> pd.DataFrame:
        df = super().query(symbols, start_date, end_date, timeframe, adjust)
        return df

    def _fetch_data(
            self,
            symbols: frozenset[str],
            start_date: datetime,
            end_date: datetime,
            timeframe: Optional[str],
            adjust: Optional[Any] = Adjustment.ALL,
    ) -> pd.DataFrame:
        df = super()._fetch_data(symbols, start_date, end_date, timeframe, adjust)

        df = df.sort_values([DataCol.SYMBOL.value, DataCol.DATE.value])
        df = df.drop_duplicates()

        # TODO resampling does not work
        #   we are resampling with prices overnight where the stock exchange does not have open
        #   maybe query method might be a better place for resampling
        def resample_group(symbol, group):
            group = group.set_index(DataCol.DATE.value)
            # Alpaca Dataframe has the wrong timezone offset
            # It uses UTC and should have EST - Pybroker fixes this
            # Sice we want to manipulate the Dataframe - we need to restore the bug
            group = group.tz_convert('UTC')
            group = group.resample(timeframe).asfreq()
            group = group[group.index.dayofweek < 5]
            group = group.between_time(time(9, 30, tzinfo=est_timezone), time(16, 00, tzinfo=est_timezone))
            # And fix the timezone offset bug again
            group = group.tz_convert(est_timezone)
            group = group.apply(
                {
                    DataCol.SYMBOL.value: lambda x: symbol,
                    DataCol.OPEN.value: lambda x: x.ffill(),
                    DataCol.HIGH.value: lambda x: x.ffill(),
                    DataCol.LOW.value: lambda x: x.ffill(),
                    DataCol.CLOSE.value: lambda x: x.ffill(),
                    DataCol.VOLUME.value: lambda x: x.fillna(0),
                    DataCol.VWAP.value: lambda x: x.fillna(0)
                }
            )
            return group

        df = pd.concat([resample_group(symbol, group) for symbol, group in df.groupby(DataCol.SYMBOL.value)])
        df = df.reset_index()
        return df
