# TODO: 请在此文件中添加算子和因子
# TODO: 由于部分算子计算过慢，这里临时屏蔽了
import random


class RET_TYPE:
    # 是什么不重要
    # 只要addPrimitive中in_types, ret_type 与 PrimitiveSetTyped("MAIN", [], ret_type)中
    # 这三种type对应即可
    pass


# 改个名，因为从polars_ta中默认提取的annotation是Expr
# TODO 如果用使用其它库，这里可能要修改
Expr = RET_TYPE


def _random_int_():
    return random.choice([1, 3, 5, 10, 20, 40, 60])


def _random_float_():
    return random.choice([1.0, 2.0, 5.0])


def add_constants(pset):
    """添加常量"""
    # !!! 名字一定不能与其它名字重，上次与int一样，结果其它地方报错 [<class 'deap.gp.random_int'>]
    pset.addEphemeralConstant('_random_int_', _random_int_, int)
    pset.addEphemeralConstant('_random_float_', _random_float_, float)
    return pset


def add_operators_base(pset):
    """基础算子"""
    # 无法给一个算子定义多种类型，只好定义多个不同名算子，之后通过helper.py中的convert_inverse_prim修正
    for name in ['add', 'sub', 'mul', 'div']:
        pset.addPrimitive(dummy, [Expr, Expr], Expr, name=f'oo_{name}')
        pset.addPrimitive(dummy, [Expr, int], Expr, name=f'oi_{name}')
        pset.addPrimitive(dummy, [int, Expr], Expr, name=f'io_{name}')
        pset.addPrimitive(dummy, [Expr, float], Expr, name=f'of_{name}')
        pset.addPrimitive(dummy, [float, Expr], Expr, name=f'fo_{name}')

    for name in ['max', 'min']:
        pset.addPrimitive(dummy, [Expr, Expr], Expr, name=f'oo_{name}')

    pset.addPrimitive(dummy, [Expr], Expr, name='log')
    pset.addPrimitive(dummy, [Expr], Expr, name='sign')
    pset.addPrimitive(dummy, [Expr], Expr, name='abs_')

    return pset


def add_operators(pset):
    """添加算子"""
    pset = add_operators_base(pset)

    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_delay')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_delta')
    # pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_arg_max')
    # pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_arg_min')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_max')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_min')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_sum')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_mean')
    # TODO 等待修复
    # pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_decay_linear')
    # pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_product')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_std_dev')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_rank')
    # pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_corr')
    # pset.addPrimitive(dummy, [Expr, Expr, int], Expr, name='ts_covariance')

    # TODO 其它的`primitive`，可以从`gp/primitives.py`按需复制过来
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_scale')
    pset.addPrimitive(dummy, [Expr, int], Expr, name='ts_zscore')

    return pset


def add_factors(pset):
    pset.addTerminal(1, Expr, name='OPEN')
    pset.addTerminal(1, Expr, name='HIGH')
    pset.addTerminal(1, Expr, name='LOW')
    pset.addTerminal(1, Expr, name='CLOSE')
    # pset.addTerminal(1, Expr, name='VOLUME')
    # pset.addTerminal(1, Expr, name='AMOUNT')

    return pset


def dummy(*args):
    # 由于生成后的表达计算已经被map和evaluate接管，所以这里并没有用到，可随便定义
    print('dummy')
    return 1
