import os
import sys
from pathlib import Path

# 修改当前目录到上层目录，方便跨不同IDE中使用
pwd = str(Path(__file__).parents[1])
os.chdir(pwd)
sys.path.append(pwd)
print("pwd:", os.getcwd())
# ====================

from expr_codegen import codegen_exec


def _code_block_():
    # 因子编辑区，可利用IDE的智能提示在此区域编辑因子

    # 以下因子来原于qlib项目的Alpha158
    # https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

    # 标签
    LABEL0 = ts_delay(VWAP, -2) / ts_delay(VWAP, -1) - 1

    # 特征

    # kbar
    KMID = (CLOSE - OPEN) / OPEN
    KLEN = (HIGH - LOW) / OPEN
    KMID2 = (CLOSE - OPEN) / (HIGH - LOW + 1e-12)
    KUP = (HIGH - max_(OPEN, CLOSE)) / OPEN
    KUP2 = (HIGH - max_(OPEN, CLOSE)) / (HIGH - LOW + 1e-12)
    KLOW = (min_(OPEN, CLOSE) - LOW) / OPEN
    KLOW2 = (min_(OPEN, CLOSE) - LOW) / (HIGH - LOW + 1e-12)
    KSFT = (2 * CLOSE - HIGH - LOW) / OPEN
    KSFT2 = (2 * CLOSE - HIGH - LOW) / (HIGH - LOW + 1e-12)

    # price
    OPEN0 = OPEN / CLOSE
    HIGH0 = HIGH / CLOSE
    LOW0 = LOW / CLOSE

    # volume

    # rolling
    # BETA
    # RSQR
    # RESI


source2 = []
for i in (1, 2, 3, 4):
    source2.append(f'OPEN{i} = ts_delay(OPEN, {i}) / CLOSE')
    source2.append(f'HIGH{i} = ts_delay(HIGH, {i}) / CLOSE')
    source2.append(f'LOW{i} = ts_delay(LOW, {i}) / CLOSE')
    source2.append(f'VOLUME{i} = ts_delay(VOLUME, {i}) / (VOLUME + 1e-12)')

for i in (5, 10, 20, 30, 60):
    # 注意：qlib的roc与talib中的表示方法不同
    source2.append(f'ROC{i} = ts_delay(CLOSE, {i}) / CLOSE')
    source2.append(f'ROC{i} = ts_delay(CLOSE, {i}) / CLOSE')
    source2.append(f'MA{i} = ts_mean(CLOSE, {i}) / CLOSE')
    source2.append(f'STD{i} = ts_std_dev(CLOSE, {i}) / CLOSE')
    source2.append(f'MAX{i} = ts_max(CLOSE, {i}) / CLOSE')
    source2.append(f'MIN{i} = ts_min(CLOSE, {i}) / CLOSE')
    source2.append(f'QTLU{i} = ts_percentage(CLOSE, {i}, 0.8) / CLOSE')
    source2.append(f'QTLD{i} = ts_percentage(CLOSE, {i}, 0.2) / CLOSE')
    source2.append(f'RANK{i} = ts_rank(CLOSE, {i}) / CLOSE')
    source2.append(f'RSV{i} = ts_RSV(HIGH, LOW, CLOSE, {i})')
    source2.append(f'CORR{i} = ts_corr(CLOSE, log1p(VOLUME), {i})')
    source2.append(f'CNTP{i} = ts_mean(CLOSE > ts_delay(CLOSE, 1), {i})')
    source2.append(f'CNTN{i} = ts_mean(CLOSE < ts_delay(CLOSE, 1), {i})')
    source2.append(f'CNTD{i} = CNTP{i}-CNTN{i}')
    source2.append(
        f'SUMP{i} = ts_sum(max_(CLOSE - ts_delay(CLOSE, 1), 0), {i}) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), {i}) + 1e-12)')
    source2.append(
        f'SUMN{i} = ts_sum(max_(ts_delay(CLOSE, 1) - CLOSE, 0), {i}) / (ts_sum(abs_(CLOSE - ts_delay(CLOSE, 1)), {i}) + 1e-12)')
    source2.append(f'SUMD{i} = SUMP{i}-SUMN{i}')
    source2.append(f'VMA{i} = ts_mean(VOLUME, {i}) / (VOLUME + 1e-12)')
    source2.append(f'VSTD{i} = ts_std_dev(VOLUME, {i}) / (VOLUME + 1e-12)')
    source2.append(
        f'WVMA{i} = ts_std_dev(abs_(ts_returns(CLOSE, 1)) * VOLUME, {i}) / (ts_mean(abs_(ts_returns(CLOSE, 1)) * VOLUME, {i}) + 1e-12)')
    source2.append(
        f'VSUMP{i} = ts_sum(max_(VOLUME - ts_delay(VOLUME, 1), 0), {i}) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), {i}) + 1e-12)')
    source2.append(
        f'VSUMN{i} = ts_sum(max_(ts_delay(VOLUME, 1) - VOLUME, 0), {i}) / (ts_sum(abs_(VOLUME - ts_delay(VOLUME, 1)), {i}) + 1e-12)')
    source2.append(f'VSUMD{i} = VSUMP{i}-VSUMN{i}')

codegen_exec(None, _code_block_, '\n'.join(source2), output_file='codes/alpha158.py', over_null="partition_by")
