import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================
import polars as pl  # noqa
import more_itertools
from expr_codegen.tool import codegen_exec

# 导入OPEN等特征
from sympy_define import *  # noqa
import polars as pl  # noqa


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # TODO 由于ts_decay_linear不支持null，暂时用ts_mean代替
    # from polars_ta.prefix.wq import ts_mean as ts_decay_linear  # noqa

    # adv{d} = average daily dollar volume for the past d days
    ADV5 = ts_mean(AMOUNT, 5)
    ADV10 = ts_mean(AMOUNT, 10)
    ADV15 = ts_mean(AMOUNT, 15)
    ADV20 = ts_mean(AMOUNT, 20)
    ADV30 = ts_mean(AMOUNT, 30)
    ADV40 = ts_mean(AMOUNT, 40)
    ADV50 = ts_mean(AMOUNT, 50)
    ADV60 = ts_mean(AMOUNT, 60)
    ADV81 = ts_mean(AMOUNT, 81)
    ADV120 = ts_mean(AMOUNT, 120)
    ADV150 = ts_mean(AMOUNT, 150)
    ADV180 = ts_mean(AMOUNT, 180)

    RETURNS = ts_returns(CLOSE, 1)


# 读取因子表达式
with open('transformer/alpha101_out.txt', 'r') as f:
    source1 = f.readlines()

# TODO 加载数据
# df = None
df = pl.read_parquet('data/data.parquet')

# 计算初始一批因子
df = codegen_exec(df, _code_block_)

# 所有因子一起计算，占用内存大
# df = codegen_exec(df, '\n'.join(source1))

# 101个因子并不多，但中间过程会有大量临时列，占用内存较大，所以分成几批减少内存
BATCH_SIZE = 30
for i, sources in enumerate(more_itertools.batched(source1, BATCH_SIZE)):
    print(f'batch {i}')
    df = codegen_exec(df, '\n'.join(sources), output_file='1_out.py')

# print(df.tail())
# df.write_parquet('alpha101_out.parquet')
