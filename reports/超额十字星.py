"""
2016.05.12 方正证券 夜空中最亮的星：十字星形态的选股研究

（1）实体柱长度<h;
（2）上影线长度>(a*实体柱长度);
（3）下影线长度>(b*实体柱长度)。

"""
import multiprocessing
import pathlib

import pandas as pd
import polars as pl
from loguru import logger
from polars_ta.candles import real_body, upper_shadow, lower_shadow

from reports.utils import path_groupby_date

INPUT1_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_stock_minute")
INPUT2_PATH = pathlib.Path(r"D:\data\jqresearch\get_price_index_minute")

_ = (r"open", r"high", r"low", r"close", r"volume", r"excess")
(open, high, low, close, volume, excess) = (pl.col(i) for i in _)

_ = (r"R", r"R_index")
(R, R_index) = (pl.col(i) for i in _)

_DATE_ = "date"
_ASSET_ = "asset"


def func_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(by=[_DATE_])
    # ========================================
    df = df.with_columns(
        # TODO 偷懒了，没有用昨收价，因为还要复权
        R=close / open.first() - 1,
    )
    return df


def get_0_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_0_ts__asset)
    return df


def func_1_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.select(
        pl.col(_DATE_),
        pl.col(_ASSET_),
        excess=R - R_index
    )
    df = df.select(
        pl.col(_DATE_).first(),
        pl.col(_ASSET_).first(),
        open=excess.first(),
        high=excess.max(),
        low=excess.min(),
        close=excess.last(),
    )
    return df


def get_1_ts__asset(df: pl.DataFrame) -> pl.DataFrame:
    df = df.group_by(_ASSET_).map_groups(func_1_ts__asset)
    return df


def get_2_cl(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        real_body=real_body(open, high, low, close),
        upper_shadow=upper_shadow(open, high, low, close),
        lower_shadow=lower_shadow(open, high, low, close),
    )
    return df


def func_2files(idx_row):
    idx, row = idx_row
    logger.info(idx)
    df1 = pl.read_parquet(row['path_x']).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df2 = pl.read_parquet(row['path_y']).rename({"code": _ASSET_, "time": _DATE_, "money": "amount"})
    df1 = get_0_ts__asset(df1.filter(pl.col("paused") == 0))
    df2 = get_0_ts__asset(df2.filter(pl.col(_ASSET_) == "000001.XSHG"))
    dd = df1.join(df2, on=_DATE_, suffix='_index')
    d1 = get_1_ts__asset(dd)
    return d1


if __name__ == '__main__':
    f1 = path_groupby_date(INPUT1_PATH)
    f2 = path_groupby_date(INPUT2_PATH)
    ff = pd.merge(f1, f2, left_index=True, right_index=True, how='left')
    # 过滤日期
    ff = ff["2025-01":]

    logger.info("start")

    with multiprocessing.Pool(4) as pool:
        # pandas按月分组
        output = list(pool.map(func_2files, list(ff.iterrows())))
        # polars合并
        output = pl.concat(output)
        output = output.with_columns(pl.col(_DATE_).dt.truncate("1d"))
        output = get_2_cl(output)
        output.write_parquet("超额十字星.parquet")
        print(output.tail())

    logger.info("done")
