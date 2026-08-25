"""对全部配对对照做 Holm–Bonferroni 校正.

p 值用**配对符号翻转置换检验**（10000 次），不假设正态，也不需要 scipy：
零假设下每个配对差的符号等概率取正负，统计量取均值。

分三个家族分别校正（家族内校正，不跨家族）：
  F1  2×3 主实验（T1/T2）
  F2  T3 严格性阶梯的相邻对照
  F3  S2 预算阶梯的相邻对照

用法: python3 holm.py
"""

import functools
import random

import s2_curve
import t3_curve
from stats import F, load as load_2x2, pair

PERM = 3000
random.seed(20260825)


def perm_p(diffs):
    """双侧符号翻转置换 p 值。"""
    d = [x for x in diffs if x is not None]
    if not d:
        return float("nan"), float("nan"), 0
    n = len(d)
    obs = abs(sum(d) / n)
    hits = 0
    for _ in range(PERM):
        s = sum(x if random.random() < 0.5 else -x for x in d)
        if abs(s / n) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (PERM + 1), sum(d) / n, n


def holm(results):
    """results = [(label, p, delta, n)] -> 附加校正后 p 与是否通过 0.05"""
    order = sorted(range(len(results)), key=lambda i: results[i][1])
    m = len(results)
    out = [None] * m
    running = 0.0
    for rank, i in enumerate(order):
        label, p, d, n = results[i]
        adj = min(1.0, max(running, (m - rank) * p))
        running = adj
        out[i] = (label, p, adj, d, n)
    return out


CACHE = {}


def cached(loader, model, tag):
    key = (tag, model)
    if key not in CACHE:          # 每个模型只解析一次，否则重复读几百个大 JSON
        CACHE[key] = loader(model)
    return CACHE[key]


def diffs_2x2(models, ca, cb, key):
    d = []
    for m in models:
        for a, b in pair(cached(load_2x2, m, '2x2'), ca, cb, F[key]):
            if a is not None and b is not None:
                d.append(b - a)
    return d


def diffs_ladder(models, loader, lv_a, lv_b, getter):
    d = []
    for m in models:
        rows = cached(loader, m, loader.__module__)
        A = {r["rep"]: r for r in rows if r["strict"] == lv_a}
        B = {r["rep"]: r for r in rows if r["strict"] == lv_b}
        for rep in sorted(set(A) & set(B)):
            x, y = getter(A[rep]), getter(B[rep])
            if x is not None and y is not None:
                d.append(y - x)
    return d


def report(title, results):
    res = holm([(lab, p, d, n) for lab, p, d, n in results])
    print(f"\n=== {title}（家族内 Holm 校正，m={len(res)}）===")
    print(f"{'对照':<40}{'Δ':>9}{'n':>5}{'p':>9}{'p_holm':>9}  结论")
    for lab, p, adj, d, n in res:
        verdict = "显著" if adj < 0.05 else ("边缘" if adj < 0.10 else "不显著")
        print(f"{lab:<40}{d:>+9.3f}{n:>5}{p:>9.4f}{adj:>9.4f}  {verdict}")


def main():
    m2x2 = ["qwen3.8-27b", "deepseek-v4-flash-0731", "kimi-k3"]
    f1 = []
    for key in ("recall", "coinc", "n_iv", "audit_recall"):
        p, d, n = perm_p(diffs_2x2(m2x2, ("LOW", "OFF"), ("HIGH", "OFF"), key))
        f1.append((f"T1 LOW→HIGH @OFF : {key}", p, d, n))
    for ca, cb, lab in ((("LOW", "OFF"), ("LOW", "AUDIT"), "OFF→AUDIT"),
                        (("LOW", "SELF"), ("LOW", "AUDIT"), "SELF→AUDIT"),
                        (("LOW", "OFF"), ("LOW", "SELF"), "OFF→SELF")):
        p, d, n = perm_p(diffs_2x2(m2x2, ca, cb, "diag_clean"))
        f1.append((f"T2 {lab} @LOW : diag_clean", p, d, n))
    report("F1 · 2×3 主实验", f1)

    m_t3 = ["qwen3.7-flash-2026-07-15", "glm-5.2", "qwen3.7-max-2026-06-08"]
    f2 = []
    getters = {"recall": lambda r: r["belief"]["recall"],
               "cand": lambda r: r["belief"]["n"],
               "n_iv": lambda r: r["n_iv"]}
    for a, b in zip(t3_curve.LEVELS, t3_curve.LEVELS[1:]):
        for key, g in getters.items():
            p, d, n = perm_p(diffs_ladder(m_t3, t3_curve.load, a, b, g))
            f2.append((f"T3 {a}→{b} : {key}", p, d, n))
    report("F2 · T3 严格性阶梯相邻对照", f2)

    m_s2 = ["qwen3.8-27b", "deepseek-v4-flash-0731", "qwen3.7-plus"]
    f3 = []
    g_s2 = {"coinc": lambda r: r["belief"]["coinc"],
            "recall": lambda r: r["belief"]["recall"],
            "audit_n": lambda r: r["audit"]["n"],
            "diag": lambda r: r["diag_acc_clean"]}
    for a, b in zip(s2_curve.LEVELS, s2_curve.LEVELS[1:]):
        for key, g in g_s2.items():
            p, d, n = perm_p(diffs_ladder(m_s2, s2_curve.load, a, b, g))
            f3.append((f"S2 {a}→{b} : {key}", p, d, n))
    report("F3 · S2 预算阶梯相邻对照", f3)


if __name__ == "__main__":
    main()
