"""T2 关键格重跑：30 个诊断 case + 分半.

改的是**测量**，不是设计：
  - 诊断 case 从 8 增加到 30（单 run 准确率的噪声是上一轮所有统计警告的根源）
  - 按下标奇偶分成互不重叠的两半：
      A（偶数下标）= 开发集，只用来产生**独立的基线协变量**
      B（奇数下标）= 评估集，效应只在这里报
    这样"门槛效应 vs 基线"的关系不再共享噪声（PLAN-REVISIONS R12 要求）

条件：严格性 L1、干预预算 6 次/run、门槛 {OFF, AUDIT}，同 seed 配对。

用法: python3 t2_replication.py [model ...]
"""

import json
import math
import os
import random
import statistics
import sys

from analyze_2x2 import score
from holm import holm, perm_p

BOOT = 5000
random.seed(20260825)
TAG = "__base__c30"


def ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    n = len(xs)
    ms = sorted(sum(random.choices(xs, k=n)) / n for _ in range(BOOT))
    return sum(xs) / n, ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT)]


def load(model):
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "data", "e1_2x2", "state", model.replace("/", "_") + TAG)
    out = {}
    if not os.path.isdir(sdir):
        return out
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".json"):
            st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
            if st.get("done"):
                out[(st["gate"], st["rep"])] = score(st)
    return out


def main():
    models = sys.argv[1:] or ["deepseek-v4-flash-0731", "glm-5.2",
                              "qwen3.7-flash-2026-07-15",
                              "qwen3.7-plus-2026-05-26"]
    print(f"{'model':<30}{'gate':<7}{'n':>4}{'可答数':>8}"
          f"{'诊断A(开发)':>20}{'诊断B(评估)':>20}{'审计通过':>16}")
    fam, per_model, moder = [], {}, []
    for model in models:
        data = load(model)
        if not data:
            continue
        for g in ("OFF", "AUDIT"):
            sub = [r for (gg, _), r in data.items() if gg == g]
            if not sub:
                continue
            na = statistics.mean(r["n_answerable"] for r in sub)
            a = ci([r["diag_clean_a"] for r in sub])
            b = ci([r["diag_clean_b"] for r in sub])
            au = ci([r["audit"]["n"] for r in sub])
            print(f"{model:<30}{g:<7}{len(sub):>4}{na:>8.1f}"
                  f"{f'{a[0]:.3f}[{a[1]:.2f},{a[2]:.2f}]':>20}"
                  f"{f'{b[0]:.3f}[{b[1]:.2f},{b[2]:.2f}]':>20}"
                  f"{f'{au[0]:.2f}':>16}")
        d_b = []
        for rep in range(40):
            o, u = data.get(("OFF", rep)), data.get(("AUDIT", rep))
            if not (o and u):
                continue
            if o["diag_clean_b"] is None or u["diag_clean_b"] is None:
                continue
            d_b.append(u["diag_clean_b"] - o["diag_clean_b"])
            if o["diag_clean_a"] is not None:
                # 独立协变量：基线取 OFF 在**开发集 A** 上的表现，
                # 效应取评估集 B 上的差 —— 两者不共享任何 case
                moder.append((o["diag_clean_a"], d_b[-1]))
        if d_b:
            per_model[model] = d_b
            p, m, n = perm_p(d_b)
            fam.append((f"gate effect @{model.split('-2026')[0]}", p, m, n))

    print("\n门槛效应（AUDIT − OFF，**评估集 B**，同 seed 配对）")
    for model, d in per_model.items():
        m, lo, hi = ci(d)
        print(f"  {model:<30} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(d)}")
    pooled = [x for d in per_model.values() for x in d]
    m, lo, hi = ci(pooled)
    p, _, n = perm_p(pooled)
    print(f"  {'合并':<30} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n} p={p:.4f}")

    if fam:
        res = holm(fam)
        print(f"\n=== 家族内 Holm 校正（m={len(res)}）===")
        print(f"{'对照':<34}{'Δ':>9}{'n':>5}{'p':>9}{'p_holm':>9}  结论")
        for lab, pp, adj, dd, nn in res:
            v = "显著" if adj < 0.05 else ("边缘" if adj < 0.10 else "不显著")
            print(f"{lab:<34}{dd:>+9.3f}{nn:>5}{pp:>9.4f}{adj:>9.4f}  {v}")

    # 调节检验：横轴 = OFF 在开发集 A 的准确率（独立），纵轴 = 评估集 B 上的门槛效应
    if len(moder) > 5:
        xs = [x for x, _ in moder]
        ys = [y for _, y in moder]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        r = (sum((x - mx) * (y - my) for x, y in moder) /
             math.sqrt(sum((x - mx) ** 2 for x in xs) *
                       sum((y - my) ** 2 for y in ys)))
        # 置换检验：打乱 y
        hits = 0
        for _ in range(3000):
            sh = ys[:]
            random.shuffle(sh)
            rr = (sum((x - mx) * (y - my) for x, y in zip(xs, sh)) /
                  math.sqrt(sum((x - mx) ** 2 for x in xs) *
                            sum((y - my) ** 2 for y in sh)))
            if abs(rr) >= abs(r):
                hits += 1
        print(f"\n调节检验（独立协变量，无共享噪声）")
        print(f"  基线(OFF@开发集A) vs 门槛效应(@评估集B): r={r:+.3f} "
              f"n={len(moder)} p={(hits + 1) / 3001:.4f}")
        for lo_, hi_ in ((0, 0.6), (0.6, 0.8), (0.8, 0.95), (0.95, 1.01)):
            sub = [y for x, y in moder if lo_ <= x < hi_]
            if len(sub) >= 5:
                mm, l, h = ci(sub)
                print(f"    基线 [{lo_},{hi_}): n={len(sub):3d} "
                      f"Δ={mm:+.3f} [{l:+.3f},{h:+.3f}]")


if __name__ == "__main__":
    main()
