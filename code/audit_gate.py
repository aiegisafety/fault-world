"""E2 修正版：harness 审计的行动门槛 + 诊断任务的 oracle 上限.

试点发现模型自报的 VERIFIED 标签不可信（存在未做任何干预就贴标签的情况），
因此 T2 的行动门槛必须由 harness 判定，不能采信模型自述。

本脚本做三件事：

1. **审计**：从检查点里的真实干预记录重算每条规则的地位。
   判据（T=20 trials，真规则 P=0.90，巧合 P=0.50，基线 0.05）：
     - 做过 do(cause=ACTIVE)   且 count(effect) >= 14  → 支持
     - 做过 do(cause=INACTIVE) 且 count(effect) <= 6   → 支持
   两个阈值的单侧假阳性率均为 P(X>=14 | p=0.5) = 0.058。
   未做过相关干预 = 无依据，无论模型标了什么。

2. **oracle 上限**：用真规则集在同一批诊断 case 上算算法上限，
   给 LLM 的诊断准确率一个可解读的天花板（试点缺这个，准确率差异无法解读）。

3. **重放**：对每个已完成 run，用**同一批 case、同一信念集**重跑最后一次诊断，
   但只允许使用审计通过的规则（gate=AUDIT）。
   这是 run 内对照：与已有的 OFF / SELF 结果同源，比跨 run 比较干净得多。

用法: python3 audit_gate.py <model> [budget_seconds]
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from llm import call, parse_json
from worlds import world_of

T = 20
ACTIVE_MIN = 14      # do(c=1) 下效应出现次数下限
INACTIVE_MAX = 6     # do(c=0) 下效应出现次数上限

IV_LINE = re.compile(r"force\s+(\S+)\s+(ACTIVE|INACTIVE)\s*->\s*(.*)")


def parse_interventions(st):
    """-> {(component, state): {symptom: count}}"""
    out = {}
    for entry in st["log"]:
        for line in entry.get("executed", []):
            m = IV_LINE.search(line)
            if not m:
                continue
            comp, state, rest = m.group(1), m.group(2), m.group(3)
            counts = {}
            for part in rest.split(","):
                if ":" in part:
                    k, v = part.split(":", 1)
                    counts[k.strip()] = int(v.split("/")[0])
            out[(comp, state)] = counts
    return out


def audit(rules, ivs):
    """返回 (通过审计的规则集, 每条规则的判定理由)"""
    ok, why = set(), {}
    for (c, e) in rules:
        act = ivs.get((c, "ACTIVE"), {}).get(e)
        ina = ivs.get((c, "INACTIVE"), {}).get(e)
        if act is not None and act >= ACTIVE_MIN:
            ok.add((c, e)); why[(c, e)] = f"do({c}=1): {act}/{T} >= {ACTIVE_MIN}"
        elif ina is not None and ina <= INACTIVE_MAX:
            ok.add((c, e)); why[(c, e)] = f"do({c}=0): {ina}/{T} <= {INACTIVE_MAX}"
        elif act is None and ina is None:
            why[(c, e)] = "从未干预过该组件"
        else:
            why[(c, e)] = f"干预不支持 (act={act}, ina={ina})"
    return ok, why


def answerable(world, cases):
    """oracle 能唯一作答的 case 下标（真效应恰好出现一个且指向正确答案）。
    其余 case 本质上无解（真效应以 0.10 概率缺失，或两条真效应同时出现），
    把它们算进准确率只会稀释信号。"""
    eff2cause = {e: c for c, e in world.true_rules}
    idx = []
    for i, case in enumerate(cases):
        hits = {eff2cause[s] for s in case["symptoms"] if s in eff2cause}
        if len(hits) == 1 and next(iter(hits)) == case["answer"]:
            idx.append(i)
    return idx


def acc_on(cases, answers, idx):
    if not idx:
        return None
    ok = sum(1 for i in idx
             if i < len(answers) and answers[i] == cases[i]["answer"])
    return ok / len(idx)


def oracle_accuracy(world, cases):
    """给定真规则集的算法上限：症状里出现哪条真规则的效应，就报哪个组件。
    多个真效应同时出现时按 case 视为不可判（记 0.5 的期望没有意义，直接记错）。"""
    eff2cause = {e: c for c, e in world.true_rules}
    correct, ambiguous = 0, 0
    for case in cases:
        hits = {eff2cause[s] for s in case["symptoms"] if s in eff2cause}
        if len(hits) == 1 and hits.pop() == case["answer"]:
            correct += 1
        elif len(hits) > 1:
            ambiguous += 1
    return correct / len(cases), ambiguous / len(cases)


def replay(st, model):
    """用审计后的规则集重跑诊断（同一批 case）。"""
    world = world_of(st)
    plan = {}
    for entry in st["log"]:
        p = parse_json(entry.get("raw", "")) or {}
        if p.get("rules"):
            plan = p
    held, self_ver = set(), set()
    for r in plan.get("rules") or []:
        c, e = str(r.get("cause", "")).strip(), str(r.get("effect", "")).strip()
        if c in world.components and e in world.symptoms:
            held.add((c, e))
            if str(r.get("status", "")).upper() == "VERIFIED":
                self_ver.add((c, e))
    ivs = parse_interventions(st)
    audited, why = audit(held, ivs)
    cases = st["cases"]

    rule_txt = ("\n".join(f"  {c} causes {e}" for c, e in sorted(audited))
                or "  (none — no rule passed the audit)")
    case_txt = "\n".join(
        f"  case {i + 1}: symptoms present = {', '.join(c['symptoms']) or 'none'}"
        for i, c in enumerate(cases))
    msgs = [
        {"role": "system", "content":
         "You are a diagnostic engineer. You answer with exactly the JSON "
         "object requested, nothing else."},
        {"role": "user", "content":
         f"A machine has components: {', '.join(world.components)}.\n"
         f"Possible symptoms: {', '.join(world.symptoms)}.\n\n"
         "An independent audit of your experiments confirmed exactly these "
         f"causal rules:\n{rule_txt}\n\n"
         "Diagnostic task. In each case exactly one component was activated "
         f"and caused what you see; name that component.\n{case_txt}\n\n"
         "You may rely only on the audited rules above.\n\n"
         'Reply with ONLY this JSON object:\n'
         '{"answers": ["<component for case 1>", "..."]}'},
    ]
    raw = call(model, msgs)
    ans = [str(a).strip() for a in ((parse_json(raw) or {}).get("answers") or [])]
    acc = sum(1 for i, c in enumerate(cases)
              if i < len(ans) and ans[i] == c["answer"]) / len(cases)
    ora, amb = oracle_accuracy(world, cases)
    idx = answerable(world, cases)
    # 原始 run（gate=OFF 或 SELF）在同一批 case 上的答案，做 run 内对照
    orig_raw = next((x["diagnosis"] for x in st["log"] if "diagnosis" in x), "")
    orig_ans = [str(a).strip()
                for a in ((parse_json(orig_raw) or {}).get("answers") or [])]
    true_set = {tuple(r) for r in world.true_rules}
    coinc_set = {tuple(r) for r in world.coincidences}
    return {
        "strict": st["strict"], "gate_orig": st["gate"], "rep": st["rep"],
        "n_held": len(held),
        "self_verified": len(self_ver),
        "self_verified_coinc": len(self_ver & coinc_set),
        "self_verified_unaudited": len(self_ver - audited),
        "audited": len(audited),
        "audited_true": len(audited & true_set),
        "audited_coinc": len(audited & coinc_set),
        "diag_acc_audit_gate": acc,
        "oracle_acc": ora, "oracle_ambiguous": amb,
        "n_answerable": len(idx),
        "acc_clean_audit": acc_on(cases, ans, idx),
        "acc_clean_orig": acc_on(cases, orig_ans, idx),
        "why": {f"{c}->{e}": w for (c, e), w in why.items()},
        "raw": raw,
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "glm-4.5-air"
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "e1_2x2")
    sdir = os.path.join(root, "state", model.replace("/", "_"))
    out = os.path.join(root, f"audit_{model.replace('/', '_')}.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            if "error" not in r:      # 失败的重放不算已完成
                done.add((r["strict"], r["gate_orig"], r["rep"]))
    states = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        if st.get("done") and (st["strict"], st["gate"], st["rep"]) not in done:
            states.append(st)
    print(f"{len(states)} runs to replay", flush=True)
    t0 = time.time()

    def work(st):
        if time.time() - t0 > budget:
            return None
        time.sleep(1.5 * (st["rep"] % 3))   # 错开请求，规避账户级速率限制
        try:
            return replay(st, model)
        except Exception as e:                              # noqa: BLE001
            return {"strict": st["strict"], "gate_orig": st["gate"],
                    "rep": st["rep"], "error": repr(e)[:200]}

    with ThreadPoolExecutor(max_workers=int(os.environ.get("WORKERS", "4"))) as ex, \
            open(out, "a", encoding="utf-8") as f:
        for row in ex.map(work, states):
            if row is None:
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps({k: v for k, v in row.items()
                              if k not in ("why", "raw")}, ensure_ascii=False),
                  flush=True)
    print("saved:", out)


if __name__ == "__main__":
    main()
