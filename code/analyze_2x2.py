"""2×3 结果汇总（严格性 × 行动门槛 OFF/SELF/AUDIT）。

要点：
- 信念集取**最后一个可解析且非空的回合**（模型偶尔返回空内容，
  不回退会被误记为"放弃全部规则"）。
- 诊断准确率同时报**全部 case** 与 **oracle 可答子集**。
  后者是必需的：真效应有 10% 概率不出现、两条真效应也可能同时出现，
  这类 case 本质无解，算进去只会稀释信号（实测 oracle 全 case 上限仅 0.81）。
- 「伪 VER」= 模型自报 VERIFIED 但通不过 harness 干预审计的规则数。

用法: python3 analyze_2x2.py <model>
"""

import json
import os
import statistics as stat
import sys

from audit_gate import acc_on, answerable, audit, oracle_accuracy, parse_interventions
from llm import parse_json
from worlds import world_of

GATES = ["OFF", "SELF", "AUDIT", "ON"]      # ON = 旧命名，等同 SELF


def rebuild(st):
    plan, empty = {}, []
    for entry in st["log"]:
        if "raw" not in entry:
            continue
        p = parse_json(entry["raw"]) or {}
        if p.get("rules"):
            plan = p
        else:
            empty.append(entry["ep"])
    return plan, empty


def score(st):
    w = world_of(st)
    true_set = {tuple(r) for r in w.true_rules}
    coinc_set = {tuple(r) for r in w.coincidences}
    plan, empty = rebuild(st)
    held, ver = set(), set()
    for r in plan.get("rules") or []:
        c, e = str(r.get("cause", "")).strip(), str(r.get("effect", "")).strip()
        if c in w.components and e in w.symptoms:
            held.add((c, e))
            if str(r.get("status", "")).upper() == "VERIFIED":
                ver.add((c, e))
    audited, _ = audit(held, parse_interventions(st))

    diag_raw = next((x["diagnosis"] for x in st["log"] if "diagnosis" in x), "")
    ans = [str(a).strip() for a in ((parse_json(diag_raw) or {}).get("answers") or [])]
    cases = st.get("cases", [])
    acc = (sum(1 for i, c in enumerate(cases)
               if i < len(ans) and ans[i] == c["answer"]) / len(cases)
           if cases else None)
    idx = answerable(w, cases) if cases else []
    # 分半：偶数下标 = 开发集 A，奇数下标 = 评估集 B。
    # A 只用于产生**独立的**基线协变量，B 用于报告效应，两者不重叠，
    # 这样"门槛效应 vs 基线"的关系就不再共享噪声（见 PLAN-REVISIONS R12）。
    idx_a = [i for i in idx if i % 2 == 0]
    idx_b = [i for i in idx if i % 2 == 1]
    ora = oracle_accuracy(w, cases)[0] if cases else None

    def s(rs):
        return {"n": len(rs), "recall": len(rs & true_set) / len(true_set),
                "coinc": len(rs & coinc_set),
                "fdr": (len(rs - true_set) / len(rs)) if rs else 0.0}

    return {"strict": st["strict"], "gate": st["gate"], "rep": st["rep"],
            "belief": s(held), "self_ver": s(ver), "audit": s(audited),
            "fake_ver": len(ver - audited),
            "diag_acc": acc, "diag_acc_clean": acc_on(cases, ans, idx),
            "diag_clean_a": acc_on(cases, ans, idx_a),
            "diag_clean_b": acc_on(cases, ans, idx_b),
            "n_answerable": len(idx), "n_a": len(idx_a), "n_b": len(idx_b),
            "n_cases": len(cases), "oracle_acc": ora,
            "n_iv": st["n_iv"], "empty_rounds": empty}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return stat.mean(xs) if xs else float("nan")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-flash"
    sdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "e1_2x2", "state", model.replace("/", "_"))
    rows = []
    for fn in sorted(os.listdir(sdir)):
        if fn.endswith(".json"):
            st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
            if st.get("done"):
                rows.append(score(st))
    print(f"model={model}  n_runs={len(rows)}\n")
    cols = ("条件", "n", "信念数", "信念recall", "巧合/4", "自报VER",
            "伪VER", "审计通过", "审计中真", "审计中巧合",
            "诊断(全)", "诊断(可答)", "oracle", "干预")
    print("".join(f"{c:>10}" for c in cols))
    print("-" * 140)
    for strict in ("LOW", "HIGH"):
        for gate in GATES:
            sub = [r for r in rows if r["strict"] == strict and r["gate"] == gate]
            if not sub:
                continue
            vals = [f"{strict}/{gate}", len(sub),
                    f"{mean(r['belief']['n'] for r in sub):.1f}",
                    f"{mean(r['belief']['recall'] for r in sub):.2f}",
                    f"{mean(r['belief']['coinc'] for r in sub):.2f}",
                    f"{mean(r['self_ver']['n'] for r in sub):.1f}",
                    f"{mean(r['fake_ver'] for r in sub):.2f}",
                    f"{mean(r['audit']['n'] for r in sub):.1f}",
                    f"{mean(r['audit']['recall'] for r in sub):.2f}",
                    f"{mean(r['audit']['coinc'] for r in sub):.2f}",
                    f"{mean(r['diag_acc'] for r in sub):.2f}",
                    f"{mean(r['diag_acc_clean'] for r in sub):.2f}",
                    f"{mean(r['oracle_acc'] for r in sub):.2f}",
                    f"{mean(r['n_iv'] for r in sub):.1f}"]
            print("".join(f"{v:>10}" for v in vals))
    print("\n逐 run 明细:")
    for r in sorted(rows, key=lambda x: (x["strict"], x["gate"], x["rep"])):
        print(f"  {r['strict']}/{r['gate']} rep{r['rep']}: "
              f"belief={r['belief']['n']}(rec {r['belief']['recall']:.2f}, "
              f"coinc {r['belief']['coinc']}) selfVER={r['self_ver']['n']} "
              f"fake={r['fake_ver']} audit={r['audit']['n']}"
              f"(true {r['audit']['recall']:.2f}, coinc {r['audit']['coinc']}) "
              f"diag={r['diag_acc']} clean={r['diag_acc_clean']} "
              f"iv={r['n_iv']} empty={r['empty_rounds']}")


if __name__ == "__main__":
    main()
