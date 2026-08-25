"""T3（非单调性 / 探索退化）：6 档严格性阶梯的曲线.

问题：过严格的 agent 是不是不只是"发现得少"，而是**停止探索**——
候选假设数断崖式下降、干预也不做了。

阶梯（提示词层面，L1≈原 LOW，L3≈原 HIGH）：
  L0 只要看着可能就写   L1 有合理迹象即写   L2 证据基本一致才写
  L3 确信是因果才写     L4 必须干预验证过    L5 必须双向干预且差异明显

输出：每档的候选假设数 / 真规则 recall / 巧合采纳 / 干预次数 /
      探索广度（干预覆盖的不同组件数），均带 bootstrap 95% CI，
      并画成 PNG。

用法: python3 t3_curve.py <model> [<model> ...]
"""

import json
import os
import random
import sys

from analyze_2x2 import score
from audit_gate import parse_interventions

LEVELS = [f"L{i}" for i in range(6)]
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
            # 探索广度：干预覆盖了几个不同组件
            r["explored"] = len({c for c, _ in parse_interventions(st)})
            rows.append(r)
    return rows


METRICS = [("belief_n", lambda r: r["belief"]["n"], "candidate rules"),
           ("recall", lambda r: r["belief"]["recall"], "true-rule recall"),
           ("coinc", lambda r: r["belief"]["coinc"], "coincidences adopted /4"),
           ("n_iv", lambda r: r["n_iv"], "interventions used"),
           ("explored", lambda r: r["explored"], "components explored")]


def main():
    models = sys.argv[1:] or ["qwen3.7-flash-2026-07-15", "glm-5.2",
                              "qwen3.7-max-2026-06-08"]
    curves = {}
    for model in models:
        rows = load(model)
        print(f"\n{'=' * 96}\nmodel = {model}   n = {len(rows)}\n{'=' * 96}")
        print(f"{'档位':<6}{'n':>4}" + "".join(f"{lab:>22}" for _, _, lab in METRICS))
        curves[model] = {}
        for lv in LEVELS:
            sub = [r for r in rows if r["strict"] == lv]
            if not sub:
                continue
            cells, means = [], {}
            for key, fn, _ in METRICS:
                m, lo, hi = ci([fn(r) for r in sub])
                means[key] = (m, lo, hi)
                cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
            curves[model][lv] = means
            print(f"{lv:<6}{len(sub):>4}" + "".join(f"{c:>22}" for c in cells))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "data", "t3_curves.json")
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

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 3.6))
    xs = range(len(LEVELS))
    for ax, (key, _, lab) in zip(axes, METRICS):
        for model, c in curves.items():
            ys = [c[lv][key][0] if lv in c else float("nan") for lv in LEVELS]
            lo = [c[lv][key][1] if lv in c else float("nan") for lv in LEVELS]
            hi = [c[lv][key][2] if lv in c else float("nan") for lv in LEVELS]
            ax.plot(xs, ys, marker="o", label=model.split("-2026")[0])
            ax.fill_between(xs, lo, hi, alpha=0.15)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(LEVELS)
        ax.set_title(lab)
        ax.set_xlabel("strictness")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "t3_curves.png")
    fig.savefig(p, dpi=140)
    print("figure ->", p)


if __name__ == "__main__":
    main()
