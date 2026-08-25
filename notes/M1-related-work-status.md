# M1 · Prior art 核查（进行中）

日期：2026-08-25 ｜ 状态：**第一轮检索完成，原文精读未做**

规则：**只有本人确认过 arXiv 页面或摘要的条目才写进这里。**
计划文件里那份未核实的列表不再作为依据。

---

## 一、已确认存在的工作

| ID | 标题 | 与本文的关系 |
|---|---|---|
| [2607.10059](https://arxiv.org/abs/2607.10059) | AgentAbstain: Do LLM Agents Know When Not to Act? | 弃权 benchmark。263 对配对任务、42 个沙盒、17 个前沿模型。**关键结论：弃权能力与一般任务能力基本独立。** |
| [2606.28733](https://arxiv.org/abs/2606.28733) | Agentic Abstention: Do Agents Know When to Stop Instead of Act? | 把弃权当作序贯决策；13 个 agent 系统、2 万 8 千个任务。 |
| [2605.09330](https://arxiv.org/abs/2605.09330) | The Trap of Trajectory: Spurious Correlations in Agentic Memory | agentic memory 里的虚假相关会自我强化（行动产生的轨迹又写回记忆）。 |
| [2605.26029](https://arxiv.org/abs/2605.26029) | CausaLab: A Scalable Environment for Interactive Causal Discovery | **合成 SCM 实验室 + 预算内干预**。发现预测准确率与机制还原的持续落差（GPT-5.2-high 92% 任务准确率 vs 0.471 all-edge F1）。 |
| [2604.08401](https://arxiv.org/abs/2604.08401) | Verify Before You Commit: Faithful Reasoning via Self-Auditing (SAVeR) | **在行动提交前对内部信念状态强制验证**。 |
| [2607.09195](https://arxiv.org/abs/2607.09195) | Toward Auditable AI Scientists: A Hypothesis Evolution Protocol | 把假设的生成/评估/演化做成显式可审计操作。 |
| [2606.27409](https://arxiv.org/abs/2606.27409) | Delayed Verification Destabilizes Multi-Agent LLM Belief | 验证延迟期间错误信念在 agent 网络中传播。 |
| [2605.23414](https://arxiv.org/abs/2605.23414) | Epistemic Calibration for LLM-Based Multi-Agent Systems | 多 agent 的认知校准。 |

**第三轮补充核实（2026-08-25）：**

| ID | 标题 | 判断 |
|---|---|---|
| [2605.16346](https://arxiv.org/abs/2605.16346) | PropGuard: Safeguarding LLM-MAS via Propagation-Aware Exploration and Remediation | **存在**。做的是多 agent 系统里**恶意指令**的传播检测（时空图 + inspector agent），不是认知地位传播。**与 Paper 1 无关，与 Paper 2 相关但主题不同。** |
| [2606.24535](https://arxiv.org/abs/2606.24535) | Governed Shared Memory for Multi-Agent LLM Systems | **存在**。形式化 fleet-memory 问题，四种失效模式（未授权泄漏、陈旧传播、矛盾持存、**provenance collapse**），给出 scoped retrieval / temporal supersession / provenance tracking 等系统级原语。**这是 Paper 2 的直接邻居。** |

**仍未核实、因此不得引用**：Honest Lying 2605.29463、Evidence Tracing 综述 2606.04990、
Peters & Chin-Yee 2025 (RSOS)。三者都只与 Paper 2 有关，Paper 1 不需要。

**CausaLab 正文未读**（免费额度与时间限制）。本文对它的差异声明因此限定为可从摘要确认的部分：
CausaLab 采样**随机 SCM** 并评估图/结构方程的还原保真度；
Fault World **固定混杂结构**，使得 M 条虚假关联由构造给定，
从而可以把"虚假规则采纳数"本身作为因变量。这是可从公开摘要核实的差异，
**不声称 CausaLab 没有混杂结构**。

---

## 二、撞车评估（这是关键，不要含糊）

### 🔴 环境不再是贡献

**CausaLab 已经做了"合成 SCM 实验室 + 预算内干预 + 随机化以防训练记忆污染"。**
Fault World 的设计动机与它高度重合。本文**不能再把环境当作主要贡献**。

保留的差异：Fault World 显式构造了**观察上不可区分、干预时不复现的巧合关联**，
并把"巧合采纳数"作为一个独立因变量；CausaLab 测的是图/结构方程的还原保真度。
但这是程度差异，不是范畴差异。**§1 的贡献列表必须改写。**

### 🔴 "行动前验证"这个机制也已被占位

SAVeR（2604.08401）明确做"行动提交前对内部信念状态强制验证"，
HEP（2607.09195）把假设演化做成可审计操作。
**本文 T2 的"审计门槛"作为机制不是新的。**

保留的差异——而且这正是本文真正剩下的东西：
1. **本文测的是这类机制的代价与失效条件**，不是提出机制。
   预算为 0 时门槛把准确率打低 0.315；基线 ≥0.80 时收益归零。
   提出机制的论文通常只报"加了更好"。
2. **本文证明模型自报的地位标签不可信**，因此这类机制必须由 harness 审计。
   SAVeR 是"self-auditing"——**这恰恰是本文数据显示不可靠的那一层**。
   这是一个可以直接指向已有工作的、可检验的批评。

### 🟡 弃权文献：本文的定位仍然成立，但要换说法

AgentAbstain 的结论"弃权能力与一般任务能力基本独立"与本文的调节结果
（门槛收益随基线升高而消失）是**互补而非重复**。
本文原来的说法"没有人测过谨慎的代价"需要弱化为：
现有弃权 benchmark 测的是"该弃权时是否弃权"，
**没有把弃权/严格性的收益与它在发现侧的损失放在同一坐标系里比较**。

### 🟢 仍未见占位

- **迷信与发现是同一机制、抑制其一必抑制其二**（T1 的核心）——本轮未找到直接占位工作。
- **认知地位标签的自报不可靠性**及其对门槛机制的后果——未找到直接占位工作。
- **行动门槛的效应符号随验证预算翻转**——未找到直接占位工作。

---

## 三、对论文定位的修改建议

原定位："我们提出 Fault World 环境 + 审计门槛机制，并证明它有效。"
→ 两个组件都已被占位，这个定位站不住。

**建议新定位：**

> 本文不提出新机制。**已有工作提出了大量"行动前验证/自审计/可审计假设演化"的机制，
> 但都在报告它们有效；本文测量它们的代价与失效条件。**
> 结论：(a) 认知严格性同时压制迷信与发现，代价依模型；
> (b) 自报的认知地位标签不可信，因此 self-auditing 这一层不可靠；
> (c) 审计门槛的收益仅在系统既有验证预算、且本来表现不佳时出现，
> 无预算时它是有害的（Δ=−0.315）。

这个定位更弱，但**更难被抢跑**，而且直接与 SAVeR 这类工作对话。

---

## 三点五、第二轮精读的修正（2026-08-25 晚）

读了 SAVeR 与 HEP 的完整摘要后，**撞车比第一轮判断的要轻，但另有两处新撞车**。

### SAVeR（2604.08401，**ACL 2026 Main**）实际做的是什么

> "enforces verification over internal belief states within the agent before action commitment…
> adversarial auditing to localize violations and repair through constraint-guided minimal
> interventions under verifiable acceptance criteria… improves reasoning faithfulness
> **while preserving competitive end-task performance**."

三点关键差异，本文的定位因此**可以站住**：

1. **验证的基底不同。** SAVeR 审的是**内部逻辑/证据一致性**（faithfulness），
   由模型自身的对抗审计完成；本文审的是**外部干预证据**（do 操作的实测结果），
   由 harness 判定，模型无从伪造。
2. **SAVeR 自己只声称"保持"下游表现，没有声称提升**——
   这与本文"高基线时门槛收益归零"的结果**一致**，可以直接引用互证。
3. **SAVeR 属于"self-auditing"家族，而本文的数据正好显示模型自报的地位标签不可信**
   （伪 VERIFIED；`OFF→SELF` 门槛无可检测收益）。这是一个有数据支撑、
   可直接指向该家族的可检验批评。

### HEP（2607.09195）的一个反向发现，必须在 Discussion 里处理

HEP 报告"**base LLM 越强，越能充分利用该协议**"。
本文的调节结果方向相反：**基线越高，门槛收益越小**（r=−0.537）。
两者测的量不同（协议利用度 vs 下游准确率增益），但这个张力必须写出来，
不能装作不存在。这也是本文与 HEP 对话的抓手。

### 🔴 新撞车 1：「不要相信 agent 的自述，要用干预来测」已被明确提出

[2605.27567](https://arxiv.org/abs/2605.27567) *Why LLMs Fail at Causal Discovery
and How Interventional Agents Escape* 明确主张：
"progress should be measured by **interventional outcomes against a fixed, hidden SCM,
not by the agent's own narrative**"。

本文这条方法学主张的**一般形式已被占位**。
本文剩下的是更窄更硬的版本：**模型自报的 `VERIFIED` 标签存在系统性伪标
（原因组件从未被干预过），且建立在自报标签上的行动门槛没有可检测收益**——
这是一个针对具体机制的定量结果，不是方法学呼吁。措辞必须改到这个粒度。

### 🔴 新撞车 2：弃权–有用性的权衡本身已被广泛研究

QA 场景下"过度弃权降低有用性"是成熟话题（cascade、I-CALM、
Knowing When to Quit 等均有量化）。
**因此"没有人测过谨慎的代价"这句话是错的，必须删除。**

本文 T1 的正确说法是：
> 弃权文献量化的是"本可回答的问题上少答了多少"；
> 本文量化的是**在交互式因果发现中，压制虚假因果信念是否同时压制了真规则的发现**。
> 代价的落点不同——一个在已知问题的回答率上，一个在新规则的发现率上。

---

## 四、下一步（M1 未完成）

1. ~~精读 SAVeR、HEP 摘要~~ **已完成**（见 §3.5）。CausaLab 仍需读正文，
   确认它是否也构造了"观察不可分、干预可分"的巧合关联。
2. 核实剩余 5 篇未确认条目（PropGuard 2605.16346、Governed Shared Memory 2606.24535、
   Honest Lying 2605.29463、Evidence Tracing 2606.04990、Peters & Chin-Yee 2025 RSOS）。
3. ~~补检索"谨慎的代价"与"LLM agent 迷信"两条线~~ **已完成**，结论见 §3.5 的两处新撞车。
4. 精读 [2605.27567](https://arxiv.org/abs/2605.27567)，确认本文的"伪 VERIFIED"结果
   与它的主张的确切边界。
5. 写完 §2 后回头改 §1 的贡献列表与 Abstract 的第一句。

**在这五步完成之前不得投稿。**

---

## 五、M1 之后对论文价值的重新评估（诚实版）

经两轮检索，本文**没有一个组件是全新的**：

| 组件 | 状态 |
|---|---|
| 合成 SCM + 预算内干预环境 | CausaLab、2605.27567 已占位 |
| 行动前验证 / 可审计假设演化 | SAVeR（ACL 2026）、HEP 已占位 |
| "别信自述，用干预测" | 2605.27567 已占位（一般形式） |
| 弃权 / 谨慎的有用性代价 | QA 场景已被广泛量化 |

**剩下的是三条"条件与代价"型结果**，均未找到占位：

1. **行动门槛的效应符号随验证预算翻转**（无预算时 Δ=−0.315，p_holm=.0017）。
2. **门槛收益被未加门槛时的基线强烈调节**（独立协变量，r=−0.537，p=0.0007）；
   基线 ≥0.80 时收益归零。
3. **模型自报 VERIFIED 存在系统性伪标，基于自报标签的门槛无可检测收益。**

**这决定了论文的体量：它是一篇"已有机制何时失效"的负面/条件性结果论文，
不是一篇提出机制的论文。** 这类论文在 arXiv 上完全站得住，
对工程实践也有直接价值（别在没有验证预算的系统上加 verified-only 门槛），
但不要指望它是一篇旗舰工作。

**如果 Stein 想要更强的贡献**，最短路径是把第 1、2 条做扎实：
换更强的模型、开思维链、把预算 × 门槛交互用 30-case 协议重测，
并把"何时该加门槛"做成一个可操作的判据。那会是一篇有人愿意引用的工程论文。
