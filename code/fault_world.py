"""Mini Fault World — E0 最小原型环境.

3 个组件 / 4 个症状 / 2 条真规则 / 2 条巧合关联。
符号每次实验重新随机化（无语义，排除训练记忆污染）。

生成模型（隐变量 Z 制造巧合）:
    Z ~ Bern(0.5)                       隐藏共因，agent 不可见
    A ~ Bern(0.5), B ~ Bern(0.5)        两个真实原因，独立
    C = Z  w.p. 0.90                    被 Z 驱动 -> 与 S3/S4 相关但无因果
    S1 = 0.90 if A else 0.05            真规则 1: A -> S1
    S2 = 0.90 if B else 0.05            真规则 2: B -> S2
    S3 = Z  w.p. 0.85                   巧合 1: C ~ S3   (P(S3|C)≈0.78)
    S4 = Z  w.p. 0.80                   巧合 2: C ~ S4   (P(S4|C)≈0.74)

关键性质: 观察数据无法区分真规则与巧合关联；
只有 do(C=1) 才会暴露 S3/S4 掉回基线 0.5。
"""

import json
import random
import string


TRUE_P = 0.90       # 真因果的复现概率
BASE_P = 0.05       # 无因时的症状基线
C_Z_P = 0.90        # C 被 Z 驱动的强度
S3_Z_P = 0.85
S4_Z_P = 0.80


def _sym(rng, n=3):
    return "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(n))


class FaultWorld:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.seed = seed
        names = set()
        while len(names) < 7:
            names.add(_sym(self.rng))
        names = sorted(names)
        self.rng.shuffle(names)
        self.components = names[:3]          # A, B, C
        self.symptoms = names[3:]            # S1, S2, S3, S4
        self.A, self.B, self.C = self.components
        self.S1, self.S2, self.S3, self.S4 = self.symptoms
        self.true_rules = [(self.A, self.S1), (self.B, self.S2)]
        self.coincidences = [(self.C, self.S3), (self.C, self.S4)]

    def _flip(self, p):
        return 1 if self.rng.random() < p else 0

    def _noisy(self, z, p):
        """以概率 p 复制 z，否则取反。"""
        return z if self.rng.random() < p else 1 - z

    def episode(self, forced=None):
        """采样一个 episode。forced = {component: 0/1} 表示 do() 干预。"""
        forced = forced or {}
        z = self._flip(0.5)
        state = {
            self.A: self._flip(0.5),
            self.B: self._flip(0.5),
            self.C: self._noisy(z, C_Z_P),
        }
        for k, v in forced.items():
            state[k] = v                      # do(): 切断 C 与 Z 的连边
        sym = {
            self.S1: self._flip(TRUE_P if state[self.A] else BASE_P),
            self.S2: self._flip(TRUE_P if state[self.B] else BASE_P),
            self.S3: self._noisy(z, S3_Z_P),
            self.S4: self._noisy(z, S4_Z_P),
        }
        return state, sym

    def observations(self, n):
        return [self.episode() for _ in range(n)]

    def intervene(self, component, value, trials=5):
        """执行一次干预，重复 trials 次，返回各症状出现次数。"""
        counts = {s: 0 for s in self.symptoms}
        for _ in range(trials):
            _, sym = self.episode(forced={component: value})
            for s in self.symptoms:
                counts[s] += sym[s]
        return counts

    def ground_truth(self):
        return {
            "seed": self.seed,
            "components": self.components,
            "symptoms": self.symptoms,
            "true_rules": self.true_rules,
            "coincidences": self.coincidences,
        }


def fmt_observations(world, obs):
    lines = []
    for i, (state, sym) in enumerate(obs, 1):
        act = [c for c in world.components if state[c]] or ["none"]
        pre = [s for s in world.symptoms if sym[s]] or ["none"]
        lines.append(f"  #{i:02d}  active_components: {', '.join(act):<14} "
                     f"symptoms_present: {', '.join(pre)}")
    return "\n".join(lines)


if __name__ == "__main__":
    # 环境自检：验证观察相关性与干预效应确实如设计
    import collections
    agg = collections.Counter()
    N = 20000
    w = FaultWorld(0)
    for _ in range(N):
        st, sy = w.episode()
        if st[w.C]:
            agg["C"] += 1
            agg["S3|C"] += sy[w.S3]
            agg["S4|C"] += sy[w.S4]
        if st[w.A]:
            agg["A"] += 1
            agg["S1|A"] += sy[w.S1]
    print("observational  P(S3|C=1) =", round(agg["S3|C"] / agg["C"], 3))
    print("observational  P(S4|C=1) =", round(agg["S4|C"] / agg["C"], 3))
    print("observational  P(S1|A=1) =", round(agg["S1|A"] / agg["A"], 3))
    ic = w.intervene(w.C, 1, trials=N)
    ia = w.intervene(w.A, 1, trials=N)
    print("interventional P(S3|do C=1) =", round(ic[w.S3] / N, 3))
    print("interventional P(S4|do C=1) =", round(ic[w.S4] / N, 3))
    print("interventional P(S1|do A=1) =", round(ia[w.S1] / N, 3))
    print(json.dumps(w.ground_truth(), ensure_ascii=False))
