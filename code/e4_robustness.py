"""E4 稳健性：换环境参数后，T2 的门槛效应还在不在.

变体（见 fault_world_v3.VARIANTS）：
  base   K=3 M=4 观察共现 0.77/0.74（与历史实验等价）
  weak   K=3 M=4 观察共现 0.65/0.62（相关性不诱人）
  strong K=3 M=4 观察共现 0.86/0.83（几乎与真规则难分）
  rich   K=4 M=2（真规则多、巧合少）
  poor   K=2 M=6（真规则少、巧合多）

条件固定为预算 6 次/run（B2）× 门槛 {OFF, AUDIT}，同 seed 配对。
输出各变体的门槛效应 Δ = diag_clean(AUDIT) − diag_clean(OFF)，
并做家族内 Holm 校正。

用法: python3 e4_robustness.py [model]
"""

import json
import os
import random
import sys

from analyze_2x2 import score
from holm import holm, perm_p

VARIANTS = ["base", "weak", "strong", "rich", "poor"]
BOOT = 5000
random.seed(20260825)


def ci(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), float("nan"), float("nan")
    n = len(xs)
    ms = sorted(sum(random.choices(xs, k=n)) / n for _ in range(BOOT))
    return sum(xs) / n, ms[int(0.025 * BOOT)], ms[int(0.975 * BOOT)]


def load(model, variant):
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                        "data", "e1_2x2", "state",
                        f"{model.replace('/', '_')}__{variant}")
    out = {}
    if not os.path.isdir(sdir):
        return out
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        if st.get("done"):
            out[(st["gate"], st["rep"])] = score(st)
    return out


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-plus-2026-05-26"
    print(f"model = {model}\n")
    print(f"{'变体':<8}{'gate':<7}{'n':>3}{'信念数':>18}{'recall':>18}"
          f"{'巧合占比':>18}{'审计通过':>18}{'诊断(可答)':>20}")
    fam, effects = [], {}
    for v in VARIANTS:
        data = load(model, v)
        if not data:
            continue
        for g in ("OFF", "AUDIT"):
            sub = [r for (gg, _), r in data.items() if gg == g]
            if not sub:
                continue
            cells = []
            for f in (lambda r: r["belief"]["n"],
                      lambda r: r["belief"]["recall"],
                      lambda r: r["belief"]["coinc"],
                      lambda r: r["audit"]["n"],
                      lambda r: r["diag_acc_clean"]):
                m, lo, hi = ci([f(r) for r in sub])
                cells.append(f"{m:.2f}[{lo:.2f},{hi:.2f}]")
            print(f"{v:<8}{g:<7}{len(sub):>3}" +
                  "".join(f"{c:>18}" for c in cells[:4]) + f"{cells[4]:>20}")
        d = []
        for rep in range(40):
            a, b = data.get(("OFF", rep)), data.get(("AUDIT", rep))
            if a and b and a["diag_acc_clean"] is not None \
                    and b["diag_acc_clean"] is not None:
                d.append(b["diag_acc_clean"] - a["diag_acc_clean"])
        if d:
            effects[v] = d
            p, mm, n = perm_p(d)
            fam.append((f"gate effect @{v}", p, mm, n))

    print("\n各变体的门槛效应（AUDIT − OFF，诊断可答子集，同 seed 配对）")
    for v, d in effects.items():
        m, lo, hi = ci(d)
        print(f"  {v:<8} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={len(d)}")

    if fam:
        res = holm(fam)
        print(f"\n=== 家族内 Holm 校正（m={len(res)}）===")
        print(f"{'对照':<24}{'Δ':>9}{'n':>5}{'p':>9}{'p_holm':>9}  结论")
        for lab, p, adj, d, n in res:
            verdict = "显著" if adj < 0.05 else ("边缘" if adj < 0.10 else "不显著")
            print(f"{lab:<24}{d:>+9.3f}{n:>5}{p:>9.4f}{adj:>9.4f}  {verdict}")
        pooled = [x for d in effects.values() for x in d]
        m, lo, hi = ci(pooled)
        p, _, n = perm_p(pooled)
        print(f"\n合并全部变体: Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n} p={p:.4f}")


if __name__ == "__main__":
    main()
