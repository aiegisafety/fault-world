"""按 run 的记录还原它当初用的环境。

历史数据（E1–E5）用的是 `fault_world_v2`，检查点里没有 "world" 字段；
E4 稳健性实验用参数化的 `fault_world_v3`，检查点里写了 "world"（变体名）。
两者的随机流不同，**不能互相替代**，必须按字段分派。
"""

import fault_world_v2 as V2
import fault_world_v3 as V3


def world_of(st):
    """从 state（或任何带 seed / world 字段的 dict）还原世界对象。"""
    variant = st.get("world")
    if variant:
        return V3.FaultWorldV3(seed=st["seed"], variant=variant)
    return V2.FaultWorldV2(seed=st["seed"])


def fmt_observations(world, obs, start=1):
    fn = V3.fmt_observations if isinstance(world, V3.FaultWorldV3) else V2.fmt_observations
    return fn(world, obs, start=start)


def restore_rng(world, rng_state):
    if rng_state is not None:
        world.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
    return world


def dump_rng(world):
    s = world.rng.getstate()
    return [s[0], list(s[1]), s[2]]
