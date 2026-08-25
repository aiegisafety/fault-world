"""E1/E2 统计分析：效应量 + bootstrap 置信区间 + 配对对照.

设计上每个 rep 对应一个固定 seed 的世界，所有条件共用同一批世界，
因此条件之间是**配对**的，用配对 bootstrap 比独立比较更有力。

输出：
  1. 各模型 × 条件的描述统计（均值 + 95% bootstrap CI）
  2. 三组关键对照，每组给配对差值的均值、95% CI 与非零方向
     T1  严格性 LOW→HIGH：真规则 recall / 巧合采纳 / 干预次数
     T2  门槛 OFF→AUDIT、SELF→AUDIT：诊断准确率（oracle 可答子集）
     GH  Goodhart：伪 VERIFIED 数在 LOW vs HIGH

用法: python3 stats.py [model ...]
"""

import json
import os
import random
import sys

from analyze_2x2 import score

BOOT = 10000
random.seed(20260825)


def ci(xs, boot=BOOT):
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"),) * 3
    n = len(xs)
    means = sorted(sum(random.choices(xs, k=n)) / n for _ in range(boot))
    m = sum(xs) / n
    return m, means[int(0.025 * boot)], means[int(0.975 * boot)]


def paired_ci(pairs, boot=BOOT):
    """pairs = [(a, b)]，返回 b-a 的均值与 95% CI"""
    d = [b - a for a, b in pairs if a is not None and b is not None]
    if not d:
        return float("nan"), float("nan"), float("nan"), 0
    n = len(d)
    means = sorted(sum(random.choices(d, k=n)) / n for _ in range(boot))
    return sum(d) / n, means[int(0.025 * boot)], means[int(0.975 * boot)], n


def load(model):
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "e1_2x2", "state", model.replace("/", "_"))
    rows = []
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".json"):
            st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
            if st.get("done"):
                rows.append(score(st))
    return rows


def get(rows, strict, gate):
    return {r["rep"]: r for r in rows
            if r["strict"] == strict and r["gate"] == gate}


def pair(rows, ca, cb, field):
    a, b = get(rows, *ca), get(rows, *cb)
    out = []
    for rep in sorted(set(a) & set(b)):
        out.append((field(a[rep]), field(b[rep])))
    return out


F = {
    "recall": lambda r: r["belief"]["recall"],
    "coinc": lambda r: r["belief"]["coinc"],
    "n_rules": lambda r: r["belief"]["n"],
    "audit_recall": lambda r: r["audit"]["recall"],
    "audit_coinc": lambda r: r["audit"]["coinc"],
    "fake_ver": lambda r: r["fake_ver"],
    "n_iv": lambda r: r["n_iv"],
    "diag_clean": lambda r: r["diag_acc_clean"],
    "diag_all": lambda r: r["diag_acc"],
}


def main():
    models = sys.argv[1:] or ["qwen3.8-27b", "deepseek-v4-flash-0731"]
    for model in models:
        rows = load(model)
        print(f"\n{'=' * 88}\nmodel = {model}   n_runs = {len(rows)}\n{'=' * 88}")
        print(f"{'条件':<12}{'n':>4}{'信念数':>16}{'recall':>16}"
              f"{'巧合/4':>16}{'伪VER':>14}{'诊断(可答)':>18}")
        for strict in ("LOW", "HIGH"):
            for gate in ("OFF", "SELF", "AUDIT"):
                sub = [r for r in rows
                       if r["strict"] == strict and r["gate"] == gate]
                if not sub:
                    continue
                cells = []
                for k in ("n_rules", "recall", "coinc", "fake_ver", "diag_clean"):
                    m, lo, hi = ci([F[k](r) for r in sub])
                    cells.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
                name = f"{strict}/{gate}"
                print(f"{name:<12}{len(sub):>4}  " +
                      "".join(f"{c:>20}" for c in cells[:3]) +
                      f"{cells[3]:>18}{cells[4]:>20}")

        print("\n配对对照（差值均值 [95% CI]，CI 不含 0 即方向稳定）")
        contrasts = [
            ("T1 严格性 LOW→HIGH @OFF", ("LOW", "OFF"), ("HIGH", "OFF"),
             ["recall", "coinc", "n_iv", "audit_recall"]),
            ("T1 严格性 LOW→HIGH @AUDIT", ("LOW", "AUDIT"), ("HIGH", "AUDIT"),
             ["recall", "coinc", "n_iv", "audit_recall"]),
            ("T2 门槛 OFF→AUDIT @LOW", ("LOW", "OFF"), ("LOW", "AUDIT"),
             ["diag_clean", "diag_all"]),
            ("T2 门槛 SELF→AUDIT @LOW", ("LOW", "SELF"), ("LOW", "AUDIT"),
             ["diag_clean", "diag_all"]),
            ("T2 门槛 OFF→SELF @LOW", ("LOW", "OFF"), ("LOW", "SELF"),
             ["diag_clean", "diag_all"]),
            ("T2 门槛 OFF→AUDIT @HIGH", ("HIGH", "OFF"), ("HIGH", "AUDIT"),
             ["diag_clean", "diag_all"]),
            ("GH Goodhart LOW→HIGH @OFF", ("LOW", "OFF"), ("HIGH", "OFF"),
             ["fake_ver"]),
        ]
        for label, ca, cb, fields in contrasts:
            print(f"  {label}")
            for k in fields:
                m, lo, hi, n = paired_ci(pair(rows, ca, cb, F[k]))
                flag = "" if (lo <= 0 <= hi) else "  <-- 方向稳定"
                print(f"      {k:<14} Δ={m:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}{flag}")


def pooled(models):
    """跨模型合并的配对对照（每个 run 与同 seed、同模型的对照条件配对）。"""
    print(f"\n{'=' * 88}\n跨模型合并 (models: {', '.join(models)})\n{'=' * 88}")
    allrows = {m: load(m) for m in models}
    contrasts = [
        ("T1 LOW→HIGH @OFF", ("LOW", "OFF"), ("HIGH", "OFF"),
         ["recall", "coinc", "audit_recall", "n_iv"]),
        ("T2 OFF→AUDIT @LOW", ("LOW", "OFF"), ("LOW", "AUDIT"), ["diag_clean"]),
        ("T2 SELF→AUDIT @LOW", ("LOW", "SELF"), ("LOW", "AUDIT"), ["diag_clean"]),
        ("T2 OFF→SELF @LOW", ("LOW", "OFF"), ("LOW", "SELF"), ["diag_clean"]),
    ]
    for label, ca, cb, fields in contrasts:
        print(f"  {label}")
        for k in fields:
            pairs = []
            for m in models:
                pairs += pair(allrows[m], ca, cb, F[k])
            mm, lo, hi, n = paired_ci(pairs)
            flag = "" if (lo <= 0 <= hi) else "  <-- 方向稳定"
            print(f"      {k:<14} Δ={mm:+.3f} [{lo:+.3f},{hi:+.3f}] n={n}{flag}")


if __name__ == "__main__":
    if sys.argv[1:2] == ["pooled"]:
        pooled(sys.argv[2:] or ["qwen3.8-27b", "deepseek-v4-flash-0731", "kimi-k3"])
    else:
        main()
