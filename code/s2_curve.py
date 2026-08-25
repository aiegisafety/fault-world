"""S2：干预预算阶梯（严格性固定 L1，门槛固定 OFF）.

修正 T3 里"预算固定 6 且总被用满 → 预算是天花板不是自由变量"的问题。
预算取每 episode 0/1/2/4/7 次（每 run 共 0/3/6/12/21 次），
跨越用不满与用得满两侧。

关注：
  used/budget  预算是否真的成为约束
  coinc        更多干预能否压制迷信（这是"有真实代价的严格性"）
  recall       代价在哪一端出现
  audit_n      能被 harness 审计通过的规则数（行动层可用的东西）
  diag_clean   下游诊断（oracle 可答子集）

用法: python3 s2_curve.py [model ...]
"""

import json
import os
import random
import sys

from analyze_2x2 import score
from audit_gate import parse_interventions

LEVELS = ["B0", "B1", "B2", "B4", "B7"]
PER_RUN_BUDGET = {"B0": 0, "B1": 3, "B2": 6, "B4": 12, "B7": 21}
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
    rows = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        if st.get("done") and st["strict"] in LEVELS:
            r = score(st)
            r["explored"] = len({c for c, _ in parse_interventions(st)})
            r["used_frac"] = (r["n_iv"] / PER_RUN_BUDGET[st["strict"]]
                              if PER_RUN_BUDGET[st["strict"]] else None)
            rows.append(r)
    return rows


METRICS = [("n_iv", lambda r: r["n_iv"], "interventions used"),
           ("used_frac", lambda r: r["used_frac"], "used / budget"),
           ("belief_n", lambda r: r["belief"]["n"], "candidate rules"),
           ("recall", lambda r: r["belief"]["recall"], "true-rule recall"),
           ("coinc", lambda r: r["belief"]["coinc"], "coincidences /4"),
           ("audit_n", lambda r: r["audit"]["n"], "audited rules"),
           ("diag", lambda r: r["diag_acc_clean"], "diagnosis (clean)")]


def main():
    models = sys.argv[1:] or ["qwen3.8-27b", "deepseek-v4-flash-0731",
                              "qwen3.7-plus"]
    curves = {}
    for model in models:
        rows = load(model)
        print(f"\n{'=' * 110}\nmodel = {model}   n = {len(rows)}\n{'=' * 110}")
        print(f"{'预算':<5}{'n':>4}" + "".join(f"{lab:>21}" for _, _, lab in METRICS))
        curves[model] = {}
        for lv in LEVELS:
            sub = [r for r in rows if r["strict"] == lv]
            if not sub:
                continue
            cells, means = [], {}
            for key, fn, _ in METRICS:
                m, lo, hi = ci([fn(r) for r in sub])
                means[key] = (m, lo, hi)
                cells.append("n/a" if m != m else f"{m:.2f}[{lo:.2f},{hi:.2f}]")
            curves[model][lv] = means
            print(f"{lv:<5}{len(sub):>4}" + "".join(f"{c:>21}" for c in cells))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "s2_curves.json")
    json.dump(curves, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\ncurves ->", out)
    try:
        plot(curves)
    except Exception as e:                                  # noqa: BLE001
        print("绘图跳过:", repr(e)[:120])


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = [m for m in METRICS if m[0] != "used_frac"]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4))
    xs = range(len(LEVELS))
    for ax, (key, _, lab) in zip(axes, keys):
        for model, c in curves.items():
            ys = [c[lv][key][0] if lv in c else float("nan") for lv in LEVELS]
            lo = [c[lv][key][1] if lv in c else float("nan") for lv in LEVELS]
            hi = [c[lv][key][2] if lv in c else float("nan") for lv in LEVELS]
            ax.plot(xs, ys, marker="o", label=model.split("-2026")[0])
            ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{PER_RUN_BUDGET[l]}" for l in LEVELS])
        ax.set_title(lab)
        ax.set_xlabel("intervention budget per run")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "s2_curves.png")
    fig.savefig(p, dpi=140)
    print("figure ->", p)


if __name__ == "__main__":
    main()
