"""R11：干预预算 × 行动门槛 的交互.

问题：T2 的门槛收益来自"审计通过的规则"，而 S2 显示预算才是决定
审计通过规则数的变量。那么**门槛的收益应当随预算增大而增大**——
预算 0 时审计集必然为空，门槛只会伤害；预算充足时门槛才有东西可用。

设计：预算 {0, 6, 21 次/run} × 门槛 {OFF, AUDIT}，严格性固定 L1，
每格 20 rep，全部条件共用同一批世界种子。
OFF 侧复用 EXP=S2 的数据（同 seed、同参数），只补跑 AUDIT 侧。

统计：
  - 每个预算档下的门槛效应 Δ = diag(AUDIT) − diag(OFF)，按 seed 配对
  - 交互 = 差的差 (DiD)：[Δ@高预算] − [Δ@预算0]，同样按 seed 配对
  - p 值用配对符号翻转置换检验，家族内 Holm 校正

用法: python3 bg_interaction.py [model ...]
"""

import json
import os
import random
import sys

from analyze_2x2 import score
from holm import holm, perm_p

BUDGETS = ["B0", "B2", "B7"]
PER_RUN = {"B0": 0, "B2": 6, "B7": 21}
BOOT = 5000
random.seed(20260825)


def ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    n = len(xs)
    ms = sorted(sum(random.choices(xs, k=n)) / n for _ in range(BOOT))
    return sum(xs) / n, ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT)]


def load(model):
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "e1_2x2", "state", model.replace("/", "_"))
    out = {}
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        if st.get("done") and st["strict"] in BUDGETS and st["gate"] in ("OFF", "AUDIT"):
            out[(st["strict"], st["gate"], st["rep"])] = score(st)
    return out


def gate_effect(data, budget, field):
    """同 seed 配对的 AUDIT − OFF"""
    d = []
    for rep in range(40):
        a = data.get((budget, "OFF", rep))
        b = data.get((budget, "AUDIT", rep))
        if a and b and field(a) is not None and field(b) is not None:
            d.append(field(b) - field(a))
    return d


def did(data, b_lo, b_hi, field):
    """差的差：[AUDIT−OFF]@b_hi − [AUDIT−OFF]@b_lo，按 seed 配对"""
    d = []
    for rep in range(40):
        cells = [data.get((b, g, rep)) for b in (b_lo, b_hi) for g in ("OFF", "AUDIT")]
        if any(c is None for c in cells):
            continue
        lo_off, lo_au, hi_off, hi_au = cells
        vals = [field(x) for x in (lo_off, lo_au, hi_off, hi_au)]
        if any(v is None for v in vals):
            continue
        d.append((vals[3] - vals[2]) - (vals[1] - vals[0]))
    return d


DIAG = lambda r: r["diag_acc_clean"]          # noqa: E731
AUDN = lambda r: r["audit"]["n"]              # noqa: E731


def main():
    models = sys.argv[1:] or ["deepseek-v4-flash-0731", "qwen3.7-plus"]
    print(f"{'model':<26}{'预算':<6}{'gate':<7}{'n':>4}"
          f"{'诊断(可答)':>22}{'审计通过规则':>20}")
    pooled = {}
    for model in models:
        data = load(model)
        pooled[model] = data
        for b in BUDGETS:
            for g in ("OFF", "AUDIT"):
                sub = [v for k, v in data.items() if k[0] == b and k[1] == g]
                if not sub:
                    continue
                m1, l1, h1 = ci([DIAG(r) for r in sub])
                m2, l2, h2 = ci([AUDN(r) for r in sub])
                print(f"{model:<26}{PER_RUN[b]:<6}{g:<7}{len(sub):>4}"
                      f"{f'{m1:.2f}[{l1:.2f},{h1:.2f}]':>22}"
                      f"{f'{m2:.2f}[{l2:.2f},{h2:.2f}]':>20}")

    fam = []
    print("\n各预算档下的门槛效应（AUDIT − OFF，诊断可答子集，合并模型）")
    for b in BUDGETS:
        d = []
        for model in models:
            d += gate_effect(pooled[model], b, DIAG)
        m, lo, hi = ci(d)
        p, _, n = perm_p(d)
        fam.append((f"gate@budget={PER_RUN[b]}", p, m, n))
        print(f"  预算 {PER_RUN[b]:>2}: Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")

    print("\n交互（差的差）")
    for b_hi in ("B2", "B7"):
        d = []
        for model in models:
            d += did(pooled[model], "B0", b_hi, DIAG)
        m, lo, hi = ci(d)
        p, _, n = perm_p(d)
        fam.append((f"DiD budget 0→{PER_RUN[b_hi]}", p, m, n))
        print(f"  预算 0→{PER_RUN[b_hi]:>2}: DiD={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}")

    res = holm(fam)
    print(f"\n=== 家族内 Holm 校正（m={len(res)}）===")
    print(f"{'对照':<26}{'Δ':>9}{'n':>5}{'p':>9}{'p_holm':>9}  结论")
    for lab, p, adj, d, n in res:
        v = "显著" if adj < 0.05 else ("边缘" if adj < 0.10 else "不显著")
        print(f"{lab:<26}{d:>+9.3f}{n:>5}{p:>9.4f}{adj:>9.4f}  {v}")


if __name__ == "__main__":
    main()
