"""Fault World v3 —— 参数化环境，用于 E4 稳健性检验.

v2 把 K=3 真规则 / M=4 巧合 / 共现强度写死了。
v3 把这些变成参数，用来回答"结论是否依赖具体环境设置"。

结构：
  K 个真因组件      c_i -> s_i          （P=true_p，无因基线 base_p）
  n_conf 个混杂组件  每个由一个隐变量 Z 驱动，同时驱动 2 个症状
                    -> 每个混杂组件贡献 2 条巧合关联，M = 2 * n_conf
  1 个纯噪声组件、1 个纯噪声症状

共现强度由 sym_z 控制：观察 P(S|C=1) ≈ c_z*sz + (1-c_z)*(1-sz)。
干预时 P(S|do(C=1)) 恒为 0.5，与 sym_z 无关——这是环境的核心不变量。

变体见 VARIANTS。
"""

import random
import string

VARIANTS = {
    # 基线：与 fault_world_v2 等价（K=3, M=4, 观察共现 0.78/0.74）
    "base": dict(k=3, n_conf=2, sym_z=(0.85, 0.80), c_z=0.90),
    # 弱巧合：观察共现 ≈0.62/0.58，相关性本来就不诱人
    "weak": dict(k=3, n_conf=2, sym_z=(0.70, 0.65), c_z=0.90),
    # 强巧合：观察共现 ≈0.86/0.83，几乎与真规则难以区分
    "strong": dict(k=3, n_conf=2, sym_z=(0.95, 0.92), c_z=0.90),
    # 真规则多、巧合少：K=4, M=2
    "rich": dict(k=4, n_conf=1, sym_z=(0.85, 0.80), c_z=0.90),
    # 真规则少、巧合多：K=2, M=6
    "poor": dict(k=2, n_conf=3, sym_z=(0.85, 0.80), c_z=0.90),
}

TRUE_P = 0.90
BASE_P = 0.05


def _sym(rng, n=3):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


class FaultWorldV3:
    def __init__(self, seed, variant="base"):
        cfg = VARIANTS[variant]
        self.variant = variant
        self.k = cfg["k"]
        self.n_conf = cfg["n_conf"]
        self.sym_z = cfg["sym_z"]
        self.c_z = cfg["c_z"]
        self.seed = seed
        self.rng = random.Random(seed)

        n_comp = self.k + self.n_conf + 1
        n_sym = self.k + 2 * self.n_conf + 1
        names = set()
        while len(names) < n_comp + n_sym:
            names.add(_sym(self.rng))
        names = sorted(names)
        self.rng.shuffle(names)
        self.components = names[:n_comp]
        self.symptoms = names[n_comp:]

        self.true_causes = self.components[:self.k]
        self.conf_components = self.components[self.k:self.k + self.n_conf]
        self.noise_component = self.components[-1]
        self.true_effects = self.symptoms[:self.k]
        self.conf_symptoms = [self.symptoms[self.k + 2 * i: self.k + 2 * i + 2]
                              for i in range(self.n_conf)]
        self.noise_symptom = self.symptoms[-1]

        self.true_rules = list(zip(self.true_causes, self.true_effects))
        self.coincidences = [(c, s) for c, pair in
                             zip(self.conf_components, self.conf_symptoms)
                             for s in pair]

    def _flip(self, p):
        return 1 if self.rng.random() < p else 0

    def _noisy(self, z, p):
        return z if self.rng.random() < p else 1 - z

    def episode(self, forced=None):
        forced = forced or {}
        zs = [self._flip(0.5) for _ in range(self.n_conf)]
        state = {c: self._flip(0.5) for c in self.true_causes}
        state[self.noise_component] = self._flip(0.5)
        for i, c in enumerate(self.conf_components):
            state[c] = self._noisy(zs[i], self.c_z)
        for kk, v in forced.items():
            state[kk] = v                      # do(): 切断入边
        sym = {}
        for c, s in self.true_rules:
            sym[s] = self._flip(TRUE_P if state[c] else BASE_P)
        for i, pair in enumerate(self.conf_symptoms):
            for j, s in enumerate(pair):
                sym[s] = self._noisy(zs[i], self.sym_z[j])
        sym[self.noise_symptom] = self._flip(0.5)
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

    def diagnostic_cases(self, n=8):
        cases = []
        for i in range(n):
            target = self.true_causes[i % self.k]
            forced = {x: 0 for x in self.components}
            forced[target] = 1
            _, sy = self.episode(forced=forced)
            cases.append({"symptoms": [x for x in self.symptoms if sy[x]],
                          "answer": target})
        return cases

    def ground_truth(self):
        return {"seed": self.seed, "variant": self.variant,
                "components": self.components, "symptoms": self.symptoms,
                "true_rules": self.true_rules, "coincidences": self.coincidences,
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
    N = 20000
    for name in VARIANTS:
        w = FaultWorldV3(0, name)
        obs_p, do_p = [], []
        agg = {}
        for _ in range(N):
            st, sy = w.episode()
            for c, s in w.coincidences:
                if st[c]:
                    a = agg.setdefault((c, s), [0, 0])
                    a[0] += 1
                    a[1] += sy[s]
        obs_p = [a[1] / a[0] for a in agg.values()]
        cnt = w.intervene(w.coincidences[0][0], 1, trials=N)
        print(f"{name:<7} K={w.k} M={len(w.coincidences)} "
              f"观察共现={[round(x, 3) for x in obs_p]} "
              f"干预 P(S|do)={round(cnt[w.coincidences[0][1]] / N, 3)}")
