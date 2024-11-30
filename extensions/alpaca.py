from datetime import datetime
from typing import Union, Iterable, Optional, Any

import pandas as pd
import pybroker
from alpaca.data import Adjustment
from pybroker import Alpaca, DataCol

pybroker.disable_caches()


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

        def resample_group(group):
            group = group.set_index(DataCol.DATE.value)
            group = group.resample(timeframe).asfreq()
            group = group.ffill()
            return group

        df = pd.concat([resample_group(group) for symbol, group in df.groupby(DataCol.SYMBOL.value)])
        df = df.reset_index()
        return df
