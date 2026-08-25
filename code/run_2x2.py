"""E1/E2 试点：严格性(S1) × 行动门槛(S4) 的 2×2.

一次 run = 3 个 episode 的 agent 循环 + 1 次下游诊断:
  episode i: 给 15 条新观察日志（外加上一轮干预结果）
             -> agent 输出 {rules:[{cause,effect,status,confidence}],
                            interventions:[{component,state}]}（预算 2 次/episode）
  最后:      8 个诊断 case，问该修哪个组件
             gate=ON 时明确要求只能依据 status=VERIFIED 的规则

条件:
  S1  LOW  = 有合理迹象即可写入规则
      HIGH = 只写入你确信为真因果的规则
  S4  OFF  = 诊断可用全部规则
      ON   = 诊断只可用 VERIFIED 规则

执行模型：**逐步(step)驱动 + 磁盘检查点**。每次进程运行只在时间预算内
推进若干步，状态存盘，可反复调用直到全部完成（适配沙箱的单次调用时限）。

用法: python3 run_2x2.py <model> <reps> [seconds] [conditions]
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from fault_world_v2 import FaultWorldV2
from llm import call, parse_json
from worlds import dump_rng, fmt_observations, restore_rng, world_of

N_EPISODES = 3
OBS_PER_EP = 15
IV_BUDGET = 2
TRIALS = 20
N_CASES = int(os.environ.get("N_CASES", "8"))
SEED_BASE = 7000

SYSTEM = ("You are a diagnostic engineer investigating an unfamiliar machine. "
          "You always answer with exactly the JSON object requested, nothing else.")

STRICT = {
    # 2×3 实验用的两档
    "LOW": ("Record a causal rule whenever the evidence gives a plausible "
            "indication of one. A working hypothesis is more useful than no rule."),
    "HIGH": ("Record a causal rule only when you are confident it is genuinely "
             "causal. Omit anything you have not established."),
    # T3 用的 6 档严格性阶梯（L1≈LOW，L3≈HIGH），用于看曲线形状而非两点比较
    "L0": ("Record any causal rule that looks even remotely possible. "
           "Err heavily on the side of writing rules down."),
    "L1": ("Record a causal rule whenever the evidence gives a plausible "
           "indication of one. A working hypothesis is more useful than no rule."),
    "L2": ("Record a causal rule when the evidence is reasonably consistent "
           "with it."),
    "L3": ("Record a causal rule only when you are confident it is genuinely "
           "causal. Omit anything you have not established."),
    "L4": ("Record a causal rule only after you have tested it with an "
           "intervention. Do not record rules supported only by observation."),
    "L5": ("Record a causal rule only after you have tested the component in "
           "BOTH directions (forcing it active and forcing it inactive) and the "
           "two results differ clearly. If a rule does not meet this standard, "
           "do not record it at all."),
}

def rule_schema(budget):
    base = (
        'Reply with ONLY this JSON object:\n'
        '{"rules": [{"cause": "<component>", "effect": "<symptom>", '
        '"status": "HYPOTHESIS" or "VERIFIED", "confidence": <0.0-1.0>}], '
        '"interventions": [{"component": "<component>", '
        '"state": "active" or "inactive"}]}\n'
        'The "rules" list is your complete current belief set (restate it in '
        'full every time). Mark a rule VERIFIED only if an intervention has '
        'confirmed it. ')
    if budget <= 0:
        return base + ("You have NO interventions available; leave "
                       '"interventions" as an empty list.')
    return base + (f'You may request 0 to {budget} interventions this round; '
                   f'each forces one component into a state and runs the '
                   f'machine {TRIALS} times.')


def budget_of(strict):
    """S2 实验里条件名形如 B0/B1/B2/B4/B7，数字 = 每 episode 的干预预算。"""
    return int(strict[1:]) if strict.startswith("B") else IV_BUDGET


def prompt_of(strict):
    """S2 实验固定用 L1（原 LOW）的严格性措辞，只让预算变化。"""
    return STRICT["L1"] if strict.startswith("B") else STRICT[strict]


RULE_SCHEMA = rule_schema(IV_BUDGET)      # 兼容旧调用

# gate 三档：
#   OFF   诊断可用全部信念规则
#   SELF  只可用模型**自报** status=VERIFIED 的规则（试点证明这层不可信）
#   AUDIT 只可用 **harness 审计**通过的规则（依据真实干预记录，模型无从伪造）
CONDS = [("LOW", "OFF"), ("LOW", "SELF"), ("LOW", "AUDIT"),
         ("HIGH", "OFF"), ("HIGH", "SELF"), ("HIGH", "AUDIT")]

# EXP=T3 时改跑 6 档严格性阶梯（门槛固定 OFF，因为 T3 问的是信念形成，不是行动）
if os.environ.get("EXP") == "T3":
    CONDS = [(f"L{i}", "OFF") for i in range(6)]

# EXP=S2 时改跑干预预算阶梯（严格性固定 L1、门槛固定 OFF）。
# 预算跨越"用不满"到"用得满"的范围，修正 T3 里预算是天花板的问题。
if os.environ.get("EXP") == "S2":
    CONDS = [(f"B{b}", "OFF") for b in (0, 1, 2, 4, 7)]

# EXP=BG 时跑「预算 × 行动门槛」的交互。
# OFF 侧的 B0/B2/B7 已由 EXP=S2 跑过，同 seed 可直接配对，这里只补 AUDIT 侧。
# EXP=E4 时做稳健性：只测最关键的一格（预算 6 × 门槛 OFF/AUDIT），
# 配合 WORLD=weak/strong/rich/poor 换环境参数，看 T2 是否依赖具体设置。
if os.environ.get("EXP") == "E4":
    CONDS = [("B2", "OFF"), ("B2", "AUDIT")]

if os.environ.get("EXP") == "BG":
    CONDS = ([(f"B{b}", "AUDIT") for b in (0, 2, 7)] +
             [(f"B{b}", "OFF") for b in (0, 2, 7)])


# ---------------- 世界的可恢复化 ----------------

WORLD = os.environ.get("WORLD")      # None = 历史的 fault_world_v2；否则是 v3 变体名


def make_world(st):
    return restore_rng(world_of(st), st.get("rng"))


# ---------------- 提示词构造 ----------------

def intro(world):
    return (f"A machine has {len(world.components)} components: "
            f"{', '.join(world.components)}.\n"
            f"It can show {len(world.symptoms)} symptoms: "
            f"{', '.join(world.symptoms)}.\n"
            "Each component is either active or inactive on a given run. "
            "Some components genuinely cause symptoms; others merely co-occur "
            "with them.\n")


def run_interventions(world, plan, budget=IV_BUDGET):
    out = []
    for iv in (plan.get("interventions") or [])[:budget]:
        comp = str(iv.get("component", "")).strip()
        if comp not in world.components:
            continue
        val = 0 if str(iv.get("state", "active")).lower().startswith("in") else 1
        counts = world.intervene(comp, val, trials=TRIALS)
        out.append(f"  force {comp} {'ACTIVE' if val else 'INACTIVE'} -> " +
                   ", ".join(f"{s}: {c}/{TRIALS}" for s, c in counts.items()))
    return out


# ---------------- 单步推进 ----------------

def new_state(cond, rep):
    d = {"strict": cond[0], "gate": cond[1], "rep": rep,
         "seed": SEED_BASE + rep, "rng": None, "ep": 0, "phase": "obs",
         "msgs": [{"role": "system", "content": SYSTEM}],
         "iv_results": [], "n_iv": 0, "plan": {}, "log": [], "done": False}
    if WORLD:                     # 只有 E4 变体才写这个字段，历史数据保持无字段
        d["world"] = WORLD
    return d


def step(st):
    """执行恰好一次 LLM 调用，推进一步；就地修改并返回 st。"""
    world = make_world(st)
    if st["phase"] == "obs":
        ep = st["ep"]
        obs = world.observations(OBS_PER_EP)
        parts = []
        if ep == 0:
            parts.append(intro(world))
        if st["iv_results"]:
            parts.append("Results of your interventions:\n" +
                         "\n".join(st["iv_results"]) + "\n")
        parts.append(f"New observation logs (round {ep + 1}):\n"
                     f"{fmt_observations(world, obs, start=ep * OBS_PER_EP + 1)}\n")
        budget = budget_of(st["strict"])
        parts.append(prompt_of(st["strict"]) + "\n\n" + rule_schema(budget))
        st["msgs"].append({"role": "user", "content": "\n".join(parts)})
        raw = call(st["model"], st["msgs"])
        st["msgs"].append({"role": "assistant", "content": raw})
        parsed = parse_json(raw) or {}
        # 模型偶尔返回空 / 不可解析内容：保留上一轮的信念集，
        # 并记录 empty_rounds，否则会把解析失败误记为"agent 放弃了所有规则"
        if parsed.get("rules"):
            st["plan"] = parsed
        else:
            st.setdefault("empty_rounds", []).append(ep)
        plan = parsed if parsed.get("interventions") else {}
        st["iv_results"] = run_interventions(world, plan, budget)
        st["n_iv"] += len(st["iv_results"])
        st["log"].append({"ep": ep, "raw": raw, "executed": st["iv_results"]})
        st["ep"] += 1
        if st["ep"] >= N_EPISODES:
            st["phase"] = "diag"
    else:
        cases = world.diagnostic_cases(N_CASES)
        if st["gate"] == "SELF":
            gate_txt = ("You may use ONLY rules you labelled VERIFIED. Ignore "
                        "every rule still labelled HYPOTHESIS.")
        elif st["gate"] == "AUDIT":
            # harness 依据真实干预记录判定，不采信模型自报的 status
            from audit_gate import audit, parse_interventions
            held = {(str(r.get("cause", "")).strip(),
                     str(r.get("effect", "")).strip())
                    for r in (st["plan"].get("rules") or [])}
            held = {(c, e) for c, e in held
                    if c in world.components and e in world.symptoms}
            ok, _ = audit(held, parse_interventions(st))
            st["audited"] = sorted(ok)
            listing = ("\n".join(f"    {c} causes {e}" for c, e in sorted(ok))
                       or "    (none passed the audit)")
            gate_txt = ("An independent audit of your intervention records "
                        "confirmed exactly these rules:\n" + listing +
                        "\nYou may rely only on these audited rules.")
        else:
            gate_txt = "You may use any rule in your belief set."
        case_txt = "\n".join(
            f"  case {i + 1}: symptoms present = {', '.join(c['symptoms']) or 'none'}"
            for i, c in enumerate(cases))
        st["msgs"].append({"role": "user", "content":
                           "Diagnostic task. In each case exactly one component "
                           "was activated and caused what you see; name that "
                           f"component.\n{case_txt}\n\n{gate_txt}\n\n"
                           'Reply with ONLY this JSON object:\n'
                           '{"answers": ["<component for case 1>", "..."]}'})
        raw = call(st["model"], st["msgs"])
        st["log"].append({"diagnosis": raw})
        st["cases"] = cases
        st["diag"] = parse_json(raw) or {}
        st["done"] = True
    st["rng"] = dump_rng(world)
    return st


# ---------------- 打分 ----------------

def score(st):
    world = FaultWorldV2(seed=st["seed"])
    true_set = {tuple(r) for r in world.true_rules}
    coinc_set = {tuple(r) for r in world.coincidences}
    held, verified = set(), set()
    for r in (st["plan"].get("rules") or []):
        c, e = str(r.get("cause", "")).strip(), str(r.get("effect", "")).strip()
        if c in world.components and e in world.symptoms:
            held.add((c, e))
            if str(r.get("status", "")).upper() == "VERIFIED":
                verified.add((c, e))

    def stats(rs):
        return {"n": len(rs), "recall": len(rs & true_set) / len(true_set),
                "coinc": len(rs & coinc_set),
                "fdr": (len(rs - true_set) / len(rs)) if rs else 0.0}

    answers = [str(a).strip() for a in (st.get("diag", {}).get("answers") or [])]
    cases = st.get("cases", [])
    correct = sum(1 for i, c in enumerate(cases)
                  if i < len(answers) and answers[i] == c["answer"])
    return {"model": st["model"], "strict": st["strict"], "gate": st["gate"],
            "rep": st["rep"], "belief": stats(held), "verified": stats(verified),
            "n_candidates": len(held), "n_interventions": st["n_iv"],
            "diag_acc": (correct / len(cases)) if cases else None,
            "held": sorted(held), "verified_rules": sorted(verified),
            "gt": world.ground_truth(), "raw": st["log"]}


# ---------------- 驱动 ----------------

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.7-flash"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    conds = [c for c in CONDS if not only or f"{c[0]}_{c[1]}" in only]

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "e1_2x2")
    tag = (model.replace("/", "_") + (f"__{WORLD}" if WORLD else "")
           + (f"__{os.environ['EXPTAG']}" if os.environ.get("EXPTAG") else ""))
    sdir = os.path.join(root, "state", tag)
    os.makedirs(sdir, exist_ok=True)
    out_path = os.path.join(root, f"{tag}.jsonl")

    jobs = []
    for c in conds:
        for rep in range(reps):
            p = os.path.join(sdir, f"{c[0]}_{c[1]}_{rep}.json")
            if os.path.exists(p):
                st = json.load(open(p, encoding="utf-8"))
            else:
                st = new_state(c, rep)
                st["model"] = model
            st["_path"] = p
            if not st["done"]:
                jobs.append(st)
    print(f"{len(jobs)} unfinished runs", flush=True)

    t0 = time.time()

    def save(st):
        p = st.pop("_path")
        tmp = p + ".tmp"
        json.dump(st, open(tmp, "w", encoding="utf-8"), ensure_ascii=False)
        os.replace(tmp, p)
        st["_path"] = p

    def advance(st):
        while not st["done"] and time.time() - t0 < budget:
            try:
                step(st)
            except Exception as e:                          # noqa: BLE001
                st.setdefault("errors", []).append(
                    f"{time.strftime('%H:%M:%S')} {e!r}"[:400])
                save(st)                # 失败也要落盘，否则错误信息丢失
                break
            save(st)
        return st

    with ThreadPoolExecutor(max_workers=min(int(os.environ.get("WORKERS", "16")), max(1, len(jobs)))) as ex:
        list(ex.map(advance, jobs))

    # 汇总所有已完成的 run
    rows = []
    for fn in sorted(os.listdir(sdir)):
        if not fn.endswith(".json"):
            continue
        st = json.load(open(os.path.join(sdir, fn), encoding="utf-8"))
        if st["done"]:
            rows.append(score(st))
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    remaining = sum(1 for j in jobs if not j["done"])
    print(f"completed {len(rows)} runs, {remaining} still unfinished", flush=True)
    for r in rows:
        print(f"[{r['strict']}/{r['gate']} rep{r['rep']}] "
              f"belief n={r['belief']['n']} recall={r['belief']['recall']:.2f} "
              f"coinc={r['belief']['coinc']}/4 | ver n={r['verified']['n']} "
              f"recall={r['verified']['recall']:.2f} coinc={r['verified']['coinc']}/4 "
              f"| iv={r['n_interventions']} diag={r['diag_acc']}", flush=True)
    import llm
    u = llm.USAGE
    print(f"token 用量: calls={u['calls']} "
          f"prompt={u['prompt']:,} completion={u['completion']:,} "
          f"total={u['prompt'] + u['completion']:,}")
    print("saved:", out_path)


if __name__ == "__main__":
    main()
