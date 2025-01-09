import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import polars as pl
import polars.selectors as cs
from expr_codegen.tool import codegen_exec
from loguru import logger

# 导入OPEN等特征
from sympy_define import *  # noqa


def _code_block_1():
    # 不能跳过停牌的相关信息。如成份股相关处理

    # 注意：收益没有减1，停牌时值为1。也没有平移
    ROCR = CLOSE / ts_delay(CLOSE, 1)

    # 不少成份股数据源每月底更新，而不是每天更新，所以需要用以下方法推算
    # 注意1：在成份股调整月，如果缺少调整日的权重信息当月后一段的数据不准确
    # 注意2：不在成份股的权重要为0，否则影响之后计算，所以停牌也得保留
    SSE50 = cs_scale(ts_cum_prod_by(ROCR, cs_fill_except_all_null(sz50)), 100)
    CSI300 = cs_scale(ts_cum_prod_by(ROCR, cs_fill_except_all_null(hs300)), 100)
    CSI500 = cs_scale(ts_cum_prod_by(ROCR, cs_fill_except_all_null(zz500)), 100)
    CSI1000 = cs_scale(ts_cum_prod_by(ROCR, cs_fill_except_all_null(zz1000)), 100)


def _code_block_2():
    # 跳过停牌的相关指标

    # 这里用未复权的价格更合适
    DOJI = four_price_doji(open, high, low, close)
    # 明日停牌
    NEXT_DOJI = ts_delay(DOJI, -1)

    # 远期收益率,由于平移过,含未来数据，只能用于打标签，不能用于训练
    _OC_01 = ts_delay(CLOSE, -1) / ts_delay(OPEN, -1)
    _CC_01 = ts_delay(CLOSE, -1) / CLOSE
    _CO_01 = ts_delay(OPEN, -1) / CLOSE
    _OO_01 = ts_delay(OPEN, -2) / ts_delay(OPEN, -1)

    _OO_02 = ts_delay(OPEN, -3) / ts_delay(OPEN, -1)
    _OO_05 = ts_delay(OPEN, -6) / ts_delay(OPEN, -1)
    _OO_10 = ts_delay(OPEN, -11) / ts_delay(OPEN, -1)

    # 一期收益率
    RETURN_OC_01 = _OC_01 - 1
    RETURN_CC_01 = _CC_01 - 1
    RETURN_CO_01 = _CO_01 - 1
    RETURN_OO_01 = _OO_01 - 1

    # 算术平均
    RETURN_OO_02 = (_OO_02 - 1) / 2
    RETURN_OO_05 = (_OO_05 - 1) / 5
    RETURN_OO_10 = (_OO_10 - 1) / 10

    # 几何平均
    RETURN_OO_02 = _OO_02 ** (1 / 2) - 1
    RETURN_OO_05 = _OO_05 ** (1 / 5) - 1
    RETURN_OO_10 = _OO_10 ** (1 / 10) - 1


# =======================================
# %% 生成因子


if __name__ == '__main__':
    # 由于读写多，推荐放到内存盘，加快速度
    INPUT_PATH = r'M:\preprocessing\data2.parquet'
    # 去除停牌后的基础数据
    OUTPUT_PATH = r'M:\preprocessing\delete2.parquet'

    logger.info('数据准备, {}', INPUT_PATH)
    df = pl.read_parquet(INPUT_PATH)
    df = df.rename({'time': 'date', 'code': 'asset', 'money': 'amount'})
    print(df.columns)

    df = df.with_columns([
        # 添加常数列，回归等场景用得上
        pl.lit(1, dtype=pl.Float32).alias('ONE'),
        # 成交均价，未复权
        (pl.col('amount') / pl.col('volume')).alias('vwap'),
        # 成交额与成交量对数处理
        pl.col('amount').log1p().alias('LOG_AMOUNT'),
        pl.col('volume').log1p().alias('LOG_VOLUME'),
        pl.col('market_cap').log1p().alias('LOG_MC'),
        pl.col('circulating_market_cap').log1p().alias('LOG_FC'),
    ])

    # 后复权
    df = df.with_columns([
        (pl.col(['open', 'high', 'low', 'close', 'vwap']) * pl.col('factor')).name.map(lambda x: x.upper()),
    ]).fill_nan(None)  # nan填充成null

    logger.info('数据准备完成')
    # =====================================

    df = codegen_exec(df, _code_block_1).filter(pl.col('paused') == 0)
    df = codegen_exec(df, _code_block_2)

    # 计算出来的结果需要进行部分修复，防止之后计算时出错
    df = df.with_columns(pl.col('NEXT_DOJI').fill_null(False))

    # 将计算结果中的inf都换成null
    df = df.with_columns(fill_nan(purify(cs.numeric())))

    logger.info('特征计算完成')

    # 推荐保存到内存盘中
    df.write_parquet(OUTPUT_PATH)
    logger.info('特征保存完成, {}', OUTPUT_PATH)
