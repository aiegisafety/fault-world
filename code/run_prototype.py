"""E0 最小原型 runner.

两个 arm:
  OBS  只给 30 条观察日志 -> 直接报告因果规则
  INT  同样的观察日志 + 6 次干预预算（两阶段：先提干预方案，再看结果作答）

测量: 巧合规则采纳率 (FDR 侧) 与真规则发现率 (recall 侧)。
E0 门槛: 巧合采纳率落在 0.15–0.85 之间，否则没有权衡可测。

用法: OPENAI_API_KEY=... python3 run_prototype.py [模型名] [每 arm 次数]
"""

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from fault_world import FaultWorld, fmt_observations

BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
KEY = os.environ["OPENAI_API_KEY"]
N_OBS = 30
BUDGET = 6
TRIALS = int(os.environ.get("TRIALS", "20"))

SYSTEM = (
    "You are a diagnostic engineer investigating an unfamiliar machine. "
    "You report your conclusions in the exact JSON format requested."
)

SCHEMA = (
    'Reply with ONLY a JSON object, no other text:\n'
    '{"rules": [{"cause": "<component>", "effect": "<symptom>", '
    '"confidence": <0.0-1.0>}], "reasoning": "<2-3 sentences>"}\n'
    'List every component->symptom causal rule you believe the machine has. '
    'If you believe there are none, return an empty list.'
)


def call(model, messages, retries=4):
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 1.0}).encode()
    req = urllib.request.Request(
        BASE, data=body,
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if i == retries - 1:
                raise
            time.sleep(3 * (i + 1))


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def describe(world, obs):
    return (
        f"A machine has 3 components: {', '.join(world.components)}.\n"
        f"It can show 4 symptoms: {', '.join(world.symptoms)}.\n"
        f"Each component is either active or inactive in a given run.\n\n"
        f"Here are {len(obs)} observation logs:\n"
        f"{fmt_observations(world, obs)}\n"
    )


def run_obs(model, world, obs):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": describe(world, obs) +
             "\nBased on these logs, which components cause which symptoms?\n\n"
             + SCHEMA}]
    out = call(model, msgs)
    return parse_json(out), [{"final": out}]


def run_int(model, world, obs):
    log = []
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": describe(world, obs) +
             f"\nYou may run up to {BUDGET} interventions before concluding. "
             f"An intervention forces one component into a chosen state "
             f"(active or inactive) and then runs the machine {TRIALS} times, "
             f"reporting how often each symptom appeared.\n\n"
             f'Reply with ONLY a JSON object, no other text:\n'
             f'{{"interventions": [{{"component": "<name>", '
             f'"state": "active"|"inactive"}}]}}\n'
             f"You may list from 0 up to {BUDGET} interventions."}]
    plan_raw = call(model, msgs)
    log.append({"plan": plan_raw})
    plan = parse_json(plan_raw) or {}
    results = []
    for iv in (plan.get("interventions") or [])[:BUDGET]:
        comp = str(iv.get("component", "")).strip()
        if comp not in world.components:
            continue
        val = 0 if str(iv.get("state", "active")).lower().startswith("in") else 1
        counts = world.intervene(comp, val, trials=TRIALS)
        results.append(
            f"  force {comp} {'ACTIVE' if val else 'INACTIVE'} -> " +
            ", ".join(f"{s}: {c}/{TRIALS}" for s, c in counts.items()))
    block = ("\n".join(results) if results
             else "  (you requested no valid interventions)")
    msgs += [{"role": "assistant", "content": plan_raw},
             {"role": "user", "content":
              f"Intervention results:\n{block}\n\n"
              "Now give your final conclusion: which components cause which "
              "symptoms?\n\n" + SCHEMA}]
    final = call(model, msgs)
    log.append({"final": final, "executed": results})
    return parse_json(final), log


def score(world, ans):
    rules = set()
    for r in (ans or {}).get("rules") or []:
        c, e = str(r.get("cause", "")).strip(), str(r.get("effect", "")).strip()
        if c in world.components and e in world.symptoms:
            rules.add((c, e))
    coinc = [tuple(x) for x in world.coincidences]
    true = [tuple(x) for x in world.true_rules]
    other = [(c, s) for c in world.components for s in world.symptoms
             if (c, s) not in coinc and (c, s) not in true]
    return {
        "adopted": sorted(rules),
        "coincidence_adopted": sum((r in rules) for r in coinc),
        "true_recalled": sum((r in rules) for r in true),
        "other_false": sum((r in rules) for r in other),
        "parse_ok": ans is not None,
    }


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-max"
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    outdir = os.path.join(os.path.dirname(__file__), "..", "data", "prototype")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"e0_{model.replace(chr(47), chr(95))}_{os.environ.get(chr(65)+chr(82)+chr(77)+chr(83),chr(97)+chr(108)+chr(108)).replace(chr(44),chr(95))}.jsonl")
    def one_rep(rep):
        world = FaultWorld(seed=1000 + rep)
        obs = world.observations(N_OBS)
        out = []
        arms = [(a, f) for a, f in (("OBS", run_obs), ("INT", run_int))
                if a in os.environ.get("ARMS", "OBS,INT")]
        for arm, fn in arms:
            try:
                ans, log = fn(model, world, obs)
            except Exception as e:                      # noqa: BLE001
                ans, log = None, [{"error": repr(e)}]
            out.append({"model": model, "arm": arm, "rep": rep,
                        "gt": world.ground_truth(), **score(world, ans),
                        "raw": log})
        return out

    rows = []
    start = int(os.environ.get("REP_START", "0"))
    mode = "a" if start else "w"
    with ThreadPoolExecutor(max_workers=reps) as ex, \
            open(path, mode, encoding="utf-8") as f:
        for res in ex.map(one_rep, range(start, start + reps)):
            for row in res:
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{row['arm']} rep{row['rep']}] "
                      f"coinc={row['coincidence_adopted']}/2 "
                      f"true={row['true_recalled']}/2 "
                      f"other={row['other_false']} ok={row['parse_ok']}",
                      flush=True)
    print("\n=== summary ===")
    for arm in ("OBS", "INT"):
        sub = [r for r in rows if r["arm"] == arm]
        n = len(sub) * 2
        if not n:
            continue
        print(f"{arm}: coincidence adoption rate = "
              f"{sum(r['coincidence_adopted'] for r in sub) / n:.2f}  "
              f"true-rule recall = {sum(r['true_recalled'] for r in sub) / n:.2f}  "
              f"other-false total = {sum(r['other_false'] for r in sub)}")
    print("saved:", path)


if __name__ == "__main__":
    main()
