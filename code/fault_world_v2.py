"""Fault World v2 —— 全尺寸环境（Paper 1 E1/E2 用）.

6 组件 / 8 症状 / K=3 条真规则 / M=4 条巧合关联 / 1 个纯噪声组件 / 1 个纯噪声症状。
假设空间 6×8 = 48 条（v1 只有 12 条），足以测 T3 的"停止探索"。

因果图（agent 不可见）:
    c0 -> s0        真规则, P=0.90, 基线 0.05
    c1 -> s1        真规则
    c2 -> s2        真规则
    Z1 -> c3, s3, s4   巧合: (c3,s3) (c3,s4)   观察共现 ≈0.78 / 0.74
    Z2 -> c4, s5, s6   巧合: (c4,s5) (c4,s6)
    c5              纯噪声组件（与任何症状无关）
    s7              纯噪声症状（基线 0.5）

关键不变量: 观察数据不能区分真规则与巧合；只有 do() 能。
"""

import json
import random
import string

TRUE_P = 0.90
BASE_P = 0.05
C_Z_P = 0.90              # 组件被隐变量驱动的强度
SYM_Z_P = (0.85, 0.80)    # 两个巧合症状被隐变量驱动的强度
N_COMP = 6
N_SYM = 8


def _sym(rng, n=3):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


class FaultWorldV2:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seed = seed
        names = set()
        while len(names) < N_COMP + N_SYM:
            names.add(_sym(self.rng))
        names = sorted(names)
        self.rng.shuffle(names)
        self.components = names[:N_COMP]
        self.symptoms = names[N_COMP:]
        c, s = self.components, self.symptoms
        self.true_rules = [(c[0], s[0]), (c[1], s[1]), (c[2], s[2])]
        self.coincidences = [(c[3], s[3]), (c[3], s[4]),
                             (c[4], s[5]), (c[4], s[6])]
        self.noise_component = c[5]
        self.noise_symptom = s[7]

    # ---------- 采样 ----------
    def _flip(self, p):
        return 1 if self.rng.random() < p else 0

    def _noisy(self, z, p):
        return z if self.rng.random() < p else 1 - z

    def episode(self, forced=None):
        forced = forced or {}
        c, s = self.components, self.symptoms
        z1, z2 = self._flip(0.5), self._flip(0.5)
        state = {c[0]: self._flip(0.5), c[1]: self._flip(0.5),
                 c[2]: self._flip(0.5), c[5]: self._flip(0.5),
                 c[3]: self._noisy(z1, C_Z_P), c[4]: self._noisy(z2, C_Z_P)}
        for k, v in forced.items():
            state[k] = v                       # do(): 切断入边
        sym = {
            s[0]: self._flip(TRUE_P if state[c[0]] else BASE_P),
            s[1]: self._flip(TRUE_P if state[c[1]] else BASE_P),
            s[2]: self._flip(TRUE_P if state[c[2]] else BASE_P),
            s[3]: self._noisy(z1, SYM_Z_P[0]),
            s[4]: self._noisy(z1, SYM_Z_P[1]),
            s[5]: self._noisy(z2, SYM_Z_P[0]),
            s[6]: self._noisy(z2, SYM_Z_P[1]),
            s[7]: self._flip(0.5),
        }
        return state, sym

    def observations(self, n):
        return [self.episode() for _ in range(n)]

    def intervene(self, component, value, trials=20):
        counts = {x: 0 for x in self.symptoms}
        for _ in range(trials):
            _, sy = self.episode(forced={component: value})
            for x in self.symptoms:
                counts[x] += sy[x]
        return counts

    # ---------- 下游诊断任务 ----------
    def diagnostic_cases(self, n=8):
        """每个 case: 只激活一个真因组件（其余组件正常随机），
        问 agent 该修哪个组件。正确答案 = 被激活的真因。
        迷信 agent 会被隐变量制造的 s3–s6 带偏。"""
        cases = []
        causes = [r[0] for r in self.true_rules]
        for i in range(n):
            target = causes[i % len(causes)]
            forced = {x: 0 for x in self.components}
            forced[target] = 1
            st, sy = self.episode(forced=forced)
            present = [x for x in self.symptoms if sy[x]]
            cases.append({"symptoms": present, "answer": target})
        return cases

    def ground_truth(self):
        return {"seed": self.seed, "components": self.components,
                "symptoms": self.symptoms,
                "true_rules": self.true_rules,
                "coincidences": self.coincidences,
                "noise_component": self.noise_component,
                "noise_symptom": self.noise_symptom}


def fmt_observations(world, obs, start=1):
    lines = []
    for i, (state, sym) in enumerate(obs, start):
        act = [c for c in world.components if state[c]] or ["none"]
        pre = [s for s in world.symptoms if sym[s]] or ["none"]
        lines.append(f"  #{i:03d}  active: {', '.join(act):<26} "
                     f"symptoms: {', '.join(pre)}")
    return "\n".join(lines)


if __name__ == "__main__":
    import collections
    N = 20000
    w = FaultWorldV2(0)
    agg = collections.Counter()
    for _ in range(N):
        st, sy = w.episode()
        for cmp_, s_ in w.true_rules + w.coincidences:
            if st[cmp_]:
                agg[(cmp_, s_, "n")] += 1
                agg[(cmp_, s_, "k")] += sy[s_]
    print("--- 观察共现 P(S|C=1) ---")
    for r in w.true_rules:
        print("  true ", r, round(agg[(r[0], r[1], "k")] / agg[(r[0], r[1], "n")], 3))
    for r in w.coincidences:
        print("  coinc", r, round(agg[(r[0], r[1], "k")] / agg[(r[0], r[1], "n")], 3))
    print("--- 干预 P(S|do(C=1)) ---")
    for r in w.true_rules + w.coincidences:
        cnt = w.intervene(r[0], 1, trials=N)
        print("  ", r, round(cnt[r[1]] / N, 3))
    print(json.dumps(w.ground_truth(), ensure_ascii=False))
