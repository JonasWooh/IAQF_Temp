# IAQF 2026 项目整体总结（基于当前代码与输出）

> 生成时间：2026-02-24（基于当前 `project_layer` 代码、配置与 `output_project_layer` 实际产物）
>
> 目的：给出一个可直接用于论文撰写/项目复盘的“整体、详细、当前状态”总结，而不是仅描述设计意图。

---

## 1. 项目定位（一句话）

本项目以 **2023 年 3 月 USDC 脱锚（SVB 事件）** 为自然实验窗口，研究 **BTC 在不同计价货币（USD / USDC / USDT）及不同中心化交易所（BinanceUS / Coinbase）中的价格偏离与流动性动力学**，并进一步分析 **价格摩擦（Gas）** 与 **制度性约束（周末/兑换暂停）** 如何影响传导路径，同时用反事实情景将结果映射到 **GENIUS Act 稳定币监管语境**。

---

## 2. 研究问题（当前版本）

项目目前围绕以下几类问题展开：

1. **现象层**：USDC 脱锚期间是否出现显著的跨币种基差（Basis）与跨所价差（Premium）放大？
2. **机制层**：这种价格偏离是否伴随市场微观结构恶化（流动性压力上升）？
3. **动态层**：Depeg 冲击如何通过 CMLSI（综合流动性压力指数）传导到 Basis / Premium（VAR/IRF/FEVD）？
4. **归因层**：价格摩擦（Gas）与制度性约束（Weekend / Conversion Pause）谁对传导变化更重要？
5. **稳健性层**：结论是否依赖 CMLSI 指数构造方式（Stable vs Adaptive）？
6. **政策映射层**：若冲击幅度、尾部风险、持续时间下降（GENIUS Act 相关情景），响应路径会如何变化？

---

## 3. 样本与数据范围（当前实现）

### 3.1 样本设计

- **Base asset**：BTC
- **Quote currencies**：USD / USDC / USDT
- **Exchanges**：BINANCEUS、COINBASE
- **时间窗口（UTC）**：2023-03-01 至 2023-03-21
- **阶段划分（regimes）**：
  - Normal：2023-03-01 ~ 2023-03-09
  - Crisis：2023-03-10 ~ 2023-03-13
  - Recovery：2023-03-14 ~ 2023-03-21

### 3.2 原始与中间数据

- 高频 Tick 数据（盘口级）
- Ethereum `base_fee_gwei`（Gas Fee）
- ETL 后统一到 **1 分钟级** parquet
- 主样本表：`project_layer/output_project_layer/master_features_1min.parquet`

### 3.3 当前数据可用性（已核查）

主样本目前可用的关键列包括：

- `Depeg_bps`
- `CMLSI`
- `Basis_bps`
- `Premium_bps`
- `Basis_USDC_USDT_BN`
- `Basis_USDC_USDT_BN_bps`

当前仍缺失（数据源可用性问题，不是代码逻辑缺失）：

- `Basis_USDC_USDT_CB`（Coinbase 侧对应列）

说明：

- 这 **不会影响核心 VAR/VARX 主模型**（主模型不使用该列）
- 但会影响部分 EDA 的“跨所稳定币相对定价对比”的完整性

---

## 4. 变量角色与研究逻辑（当前版本）

项目主线可以概括为：

`Depeg（冲击） -> CMLSI（机制状态） -> Basis / Premium（结果变量）`

### 4.1 核心变量角色

#### A. 冲击变量（Shock）
- `Depeg_bps`
  - USDC 相对 1 美元脱锚程度（bps）
  - 代表事件冲击强度

#### B. 机制变量（Mechanism / State）
- `CMLSI`
  - 基于微观结构特征（spread/depth/OBI + 扩展特征）构造的综合流动性压力指数
  - 用于压缩多维流动性恶化信息，进入 VAR 系统

#### C. 结果变量（Outcomes）
- `Basis_bps`
  - BTC/USD 与 BTC/USDC 的跨币种定价偏离
- `Premium_bps`
  - BinanceUS vs Coinbase 的 BTC/USD 跨所价差（市场割裂）

#### D. 外生摩擦变量（Exogenous Frictions）
- `base_fee_gwei`（Gas Fee）
- `is_weekend`
- `conversion_pause`

### 4.2 为什么要引入 CMLSI

危机期间流动性恶化并非单一维度（spread 变宽、depth 下降、OBI 失衡、波动上升），若直接把所有特征都放入 VAR：

- 维度过高
- 共线性强
- 样本窗口短（尤其 crisis）

因此用 PCA 构造低维综合压力状态变量（CMLSI）是合理且必要的。

### 4.3 为什么 CMLSI 采用 Normal fit + OOS transform

当前实现中，CMLSI 构造遵循：

- **仅在 Normal 期拟合** `Scaler + PCA`
- 再对全样本（分阶段）做 OOS transform

动机：

- 避免用危机样本反向定义指数（减少前视偏差）
- 让 Crisis/Recovery 的 CMLSI 更像“偏离正常状态的压力度量”

---

## 5. 方法流程（当前实现）

### 5.1 数据处理与特征构造

1. Tick -> 1 分钟 OHLC 与微观结构特征（spread / depth / OBI）
2. 多交易所/多交易对对齐到统一时间轴
3. 构造研究变量：
   - `Depeg_bps`
   - `Basis_bps`
   - `Premium_bps`
   - 稳定币相对基差相关列
4. 构造扩展微观结构特征（如 `|ret|`、`range`、`rel_spread`、`log_depth`）
5. PCA 构造 CMLSI

### 5.2 EDA（现象层）

目的不是“证明因果”，而是确认 stylized facts：

- 危机是否真实存在（Depeg 时间线）
- 流动性是否蒸发（spread/depth/CMLSI）
- 跨币种与跨所偏离是否放大（Basis/Premium）
- Gas 是否在危机期显著抬升（支持后续 VARX）

### 5.3 VAR / VARX（动态传导 + 摩擦归因）

#### 核心 VAR（按 regime）
- 内生变量：`[Depeg_bps, CMLSI, Basis_bps, Premium_bps]`
- 阶段切片：Normal / Crisis / Recovery
- ADF 检验后必要差分（危机期当前常见差分：`Depeg_bps`、`Basis_bps`）
- 输出：
  - IRF（冲击响应）
  - FEVD（方差分解）

#### VARX（摩擦归因）
- 在 crisis 期对 VAR 进行外生扩展：
  - Gas（价格摩擦）
  - Gas + Weekend（制度性约束）
  - Gas + Conversion Pause（制度性约束）

### 5.4 诊断与经济约束

- VAR 稳定性与残差白噪声检验（Ljung-Box）
- `Basis_bps` 的 AR(1) 均值回归强度对比（Normal / Crisis / Recovery）
- 交易成本边界图（用于无套利区间直观验证）

### 5.5 反事实模拟（政策映射）

基于 crisis 期 IRF 做情景变换：

- A：冲击幅度缩放（Magnitude scaling）
- B：尾部截断（Tail truncation / winsorize）
- C：持续时间衰减（Duration damp）

用途：

- 将 2023 事件分析映射到 “若稳定币机制更稳健，市场响应会如何变化” 的政策相关情景。

### 5.6 CMLSI 高级稳健性（可选分支）

模块：`CMLSI_test_advanced`

包括：

- 协方差结构变化检验
- 特征值与子空间角检验
- RMT（Marchenko-Pastur）信号/噪声判断
- Rolling PCA 构造 Adaptive CMLSI
- Stable vs Adaptive 的 VAR/IRF/FEVD 对比

---

## 6. 当前输出产物概览（已存在）

### 6.1 主要图表（`project_layer/output_project_layer/figures/`）

当前已存在（部分）：

- `01_depeg_timeline.png`
- `02_liquidity_evaporation.png`
- `03_cross_exchange_spread.png`
- `04_cmlsi_loadings.png`
- `05_cmlsi_decomposition.png`
- `06_implied_irf_genius_act.png`
- `07_fragmentation_irf.png`
- `08_pca_variance_comparison.png`
- `09_cross_currency_basis.png`
- `10_stablecoin_premium.png`
- `11_fevd_depeg_contribution.png`
- `12_regime_irf_comparison.png`
- `14_triple_regime_irf_depeg_to_basis.png`
- `15_basis_irf.png`
- `16_triple_regime_irf_full.png`
- `17_ar1_mean_reversion.png`
- `17_gas_fee_comparison_baseline_vs_varx.png`
- `18_transaction_cost_bounds.png`
- `19_gas_fee_eda.png`
- `20_institutional_friction_comparison.png`
- `21_nested_model_comparison.png`
- `21_var_hypotheses_transmission.png`（新增）
- `22_var_hypotheses_regime_slicing.png`（新增）

### 6.2 文本/报告类产物（`project_layer/output_project_layer/`）

- `depeg_eda_report.txt`
- `gas_fee_eda_report.txt`
- `var_diagnostics.txt`
- `var_gas_comparison_report.txt`
- `var_institutional_report.txt`
- `ar1_mean_reversion.txt`
- `nested_model_comparison_report.txt`
- `var_hypotheses_report.txt`（新增）
- `var_hypotheses_metrics.json`（新增）
- `genius_act_simulation/simulation_report.txt`

### 6.3 CMLSI 高级稳健性产物（`CMLSI_test_advanced/`）

- `structure_tests_report.txt / .json`
- `pca_rmt_report.txt`
- `economic_interpretation_report.txt`
- `CMLSI_robustness_full_statistics.md`
- 相关图表（01~04）

---

## 7. 基于当前输出的实证判断（核心）

> 这一节是“当前项目到底已经回答到什么程度”的结论摘要。

### 7.1 已有较强证据支持的结论

#### 结论 A：危机窗口与分阶段（regime）设定是合理的

证据：

- `Depeg_bps` 在 crisis 期显著放大（均值、波动、尾部都明显上升）
- Gas 在 crisis 期显著抬升（均值与尾部提高）
- CMLSI 结构变化检验显示 Normal 与 Crisis 协方差/PCA 结构明显不同

当前判断：

- `regime shift` 不是叙事假设，而是得到数据支持的建模选择

#### 结论 B：Depeg 冲击在 crisis 期会显著传导到 CMLSI / Basis / Premium

证据（来自新增假设检验补充）：

- `Depeg -> CMLSI` 峰值点 CI 不跨 0
- `CMLSI -> Basis` 峰值点 CI 不跨 0
- `CMLSI -> Premium` 峰值点 CI 不跨 0

当前判断：

- 危机期动态传导存在
- “冲击 -> 流动性压力 -> 市场偏离”链条中的关键边有实证支持（至少 reduced-form 层面）

#### 结论 C：危机期传导机制显著强于正常期，且全样本平均模型会掩盖危机机制

证据：

- `Depeg -> Basis_bps` 的 crisis 响应峰值与累计响应远高于 normal/recovery
- `Depeg -> Premium_bps` 在 crisis 期也更强、更持久
- full-sample VAR 与 regime VAR 在多个响应上的形状相关度不高

当前判断：

- 分阶段估计是必要的，不是“展示型切片”

#### 结论 D：危机期套利修复速度显著下降（市场分割具有经济意义）

证据（AR1 报告）：

- Normal：`rho ≈ 0.6292`
- Crisis：`rho ≈ 0.9992`
- Recovery：`rho ≈ 0.9422`

当前判断：

- Crisis 期 `Basis` 均值回归几乎失效，支持市场割裂/套利修复受阻的叙事

### 7.2 有证据但需要谨慎措辞的结论

#### 结论 E：CMLSI 是显著机制变量之一，但“主要中介通道”证据较弱

证据（新增图 21 / 报告）：

- `Depeg -> CMLSI` 与 `CMLSI -> Basis/Premium` 均有响应
- 但“含/不含 CMLSI”时，`Depeg -> Basis/Premium` 的 IRF 形状和幅度变化不大（相关性高）

当前判断：

- CMLSI 很可能是重要状态变量 / 并行机制表征
- 但不足以证明其为主导中介通道
- 这意味着：**Depeg 可能还通过信用/可赎回性预期与制度约束等直接渠道影响 Basis/Premium**

#### 结论 F：制度性约束与价格摩擦有边际解释力，但当前证据仍是初步的

证据（VARX 与嵌套模型）：

- `+gas`、`+weekend`、`+pause` 后，`Depeg -> Premium` 的 IRF / FEVD 会变化，但量级不大
- `nested model` 显示 institution block 对 `Depeg -> Basis` 的衰减比当前信用代理 block 更明显（在当前代理定义下）

当前判断：

- 可以说“存在边际影响/初步证据”
- 不宜过度写成“制度摩擦主导一切结果变量”

### 7.3 稳健性层的当前判断

#### 结论 G：CMLSI 构造存在结构错配风险，但主传导结论对指数构造总体稳健

证据（CMLSI advanced 模块）：

- 结构变化检验显示 Normal vs Crisis 的 PCA 结构确实变化
- RMT（Marchenko-Pastur）显示显著信号 PC 约 2 个（主线使用 3 PC 为实务折中）
- Stable vs Adaptive CMLSI 下：
  - `Depeg -> Basis` IRF 差异极小
  - `Depeg -> Premium` 有轻微差异，但量级有限

当前判断：

- 主线结论对 CMLSI 构造方式具有较好稳健性
- 同时也说明“结构变化”这一事实本身值得在论文中讨论

---

## 8. 当前项目的“证据强度分层”判断（建议用于写作）

### 8.1 强（可以较明确表述）

- 危机窗口中 USDC 脱锚、Gas 抬升、市场分割与流动性恶化共同出现
- Depeg 冲击与 CMLSI / Basis / Premium 的动态响应在 crisis 期存在
- 三阶段机制明显异质（regime heterogeneity）
- 分阶段估计优于全样本平均估计（至少在机制解释上）

### 8.2 中（可表述，但要保守）

- CMLSI 是传导中的重要状态变量之一
- 制度/价格摩擦对传导变化有边际贡献
- CMLSI 构造对主结论总体不敏感（Stable vs Adaptive）

### 8.3 弱（仅能作为方向性/情景性陈述）

- “CMLSI 是主要中介通道”
- “制度摩擦主导所有结果变量”
- “GENIUS Act 的精确定量效果”

---

## 9. 当前项目的主要局限（务必写入论文）

1. **识别层面**：当前以 reduced-form VAR/VARX 为主，不是结构因果识别
   - 已做 Cholesky 顺序轻量稳健性，但仍不能替代 SVAR/外部工具变量

2. **样本窗口层面**：crisis 期样本较短
   - 某些 IRF 峰值出现在 horizon 末端，更应解释为“窗口内尚未回归完成”

3. **代理变量层面**：信用/可赎回性预期代理仍较粗糙
   - 当前 credit block 主要依赖 `USDC-USDT` 相对基差代理
   - 跨所信用预期对比（如 Coinbase 对应列）不完整

4. **政策映射层面**：反事实模拟是情景分析，不是政策效果估计
   - 适合“方向性政策含义”，不宜做精确 welfare 或 causal policy claim

5. **文档与产物同步层面**：项目经历多次迭代，需确保论文最终引用的是最新图与最新报告

---

## 10. 当前版本相对早期版本的关键改进（已完成）

以下是近期已经完成并影响结果可信度的修复/增强：

1. **修复反事实模拟 IRF 口径**
   - 对差分变量的响应在模拟图中先恢复到 level 再绘图

2. **补齐特征工程缺失列生成逻辑**
   - 支持生成 `Basis_USD_USDT_BN` 与 `Basis_USDC_USDT_CB`（若原始列可用）

3. **修复 EDA fallback 不可达问题**
   - `master` 缺失时的回退路径不再被前置异常阻断

4. **修复 `fixed_lag` 名不副实问题**
   - 现在可真正固定 VAR 滞后阶数（而非继续 AIC 选阶）

5. **新增显式假设检验与可视化（重大增强）**
   - `var_hypotheses_report.txt`
   - `21_var_hypotheses_transmission.png`
   - `22_var_hypotheses_regime_slicing.png`
   - 明确回答：
     - 传导路径证据
     - 三阶段差异量化
     - regime slicing 必要性
     - Cholesky 顺序轻量稳健性

6. **新增嵌套模型比较（M0~M3）**
   - 开始量化“信用/制度约束并行直接渠道”的边际吸收能力

---

## 11. 目前最合适的论文口径（建议）

### 11.1 主结论口径（建议）

可写为：

> 在 2023 年 3 月 USDC 脱锚危机窗口内，BTC 跨计价货币与跨交易所市场出现显著的价格偏离与流动性压力上升。动态模型结果表明，Depeg 冲击与 CMLSI、Basis、Premium 的响应具有显著的阶段异质性（regime heterogeneity），且危机期传导更强、更持久。分阶段估计比全样本平均估计更能识别危机机制。

### 11.2 机制口径（建议）

可写为：

> CMLSI 是显著的流动性传导状态变量之一；然而，启发式检验显示在移除 CMLSI 后 Depeg→Basis/Premium 的 reduced-form IRF 变化有限，说明流动性压力并非唯一或主导传导渠道。稳定币信用/可赎回性预期与制度性约束可能构成并行的直接渠道。

### 11.3 政策口径（建议）

可写为：

> 基于危机期 IRF 的反事实情景模拟提供了与稳定币监管相关的方向性映射（例如冲击幅度、尾部风险与持续时间下降时的潜在市场响应路径变化），但不构成结构性政策效果估计。

---

## 12. 下一步建议（按优先级）

### P1（高优先级，直接提升论文质量）

1. **强化反事实模拟报告量化**
   - 为 `simulation_report.txt` 增加 `peak / cum_abs / half-life` 的情景对比指标
   - 让政策映射从“有图”升级为“有数”

2. **统一图号与图注**
   - 当前存在 `17`、`19` 等图号在不同模块重复使用的情况
   - 建议论文版重编号并固定图注口径

3. **把关键结论写成表格**
   - 特别是：
   - regime 差异量化
   - nested model（信用 vs 制度 block）增量解释

### P2（中优先级，提升机制解释强度）

4. **增强信用/可赎回性预期代理（credit block）**
   - 引入更多 USDC 自身微观结构代理（若 master 中加入）
   - 加入更细的法币通道可用性代理（如 banking hours）

5. **完善交易成本边界文本报告**
   - 当前有图（`18_transaction_cost_bounds.png`），建议补文本量化摘要

### P3（研究扩展）

6. **更强识别（SVAR / LP with interactions / event-time local projections）**
   - 若时间允许，可将“机制”从 reduced-form 提升到更接近结构识别

---

## 13. 总体评价（当前状态）

### 13.1 对比赛提交的适配性

当前项目已经具备：

- 明确的研究问题与现实动机（稳定币监管语境）
- 合理且可执行的数据与变量设计
- 较完整的实证证据链（EDA -> VAR/VARX -> 诊断 -> 稳健性 -> 反事实）
- 对关键不足（传导路径/分阶段动机）进行过补充量化与可视化

因此，**作为 IAQF 学生竞赛项目是有竞争力的**。

### 13.2 最准确的项目定位（建议）

最准确、最稳妥的定位是：

> 一个以 USDC 脱锚事件为自然实验窗口的 **高频市场微观结构 + 动态传导 + 摩擦归因 + 政策情景映射** 项目；其证据强项在于 regime-specific 的动态响应与市场割裂刻画，弱项在于结构因果识别与政策效果的精确定量。

---

## 14. 关键输出路径（便于后续写作引用）

### 主样本与配置
- `project_layer/output_project_layer/master_features_1min.parquet`
- `project_layer/settings.yaml`

### 关键图（新增重点）
- `project_layer/output_project_layer/figures/21_var_hypotheses_transmission.png`
- `project_layer/output_project_layer/figures/22_var_hypotheses_regime_slicing.png`

### 关键报告（建议优先引用）
- `project_layer/output_project_layer/var_hypotheses_report.txt`
- `project_layer/output_project_layer/nested_model_comparison_report.txt`
- `project_layer/output_project_layer/var_diagnostics.txt`
- `project_layer/output_project_layer/var_gas_comparison_report.txt`
- `project_layer/output_project_layer/var_institutional_report.txt`
- `project_layer/output_project_layer/ar1_mean_reversion.txt`
- `project_layer/output_project_layer/CMLSI_test_advanced/CMLSI_robustness_full_statistics.md`
- `project_layer/output_project_layer/CMLSI_test_advanced/pca_rmt_report.txt`

---

## 15. 附：一句话版本（适合口头汇报）

本项目利用 2023 年 3 月 USDC 脱锚事件，构建了一个以 Depeg、CMLSI、Basis、Premium 为核心的分阶段 VAR/VARX 框架，系统刻画了危机期跨币种与跨所市场割裂的动态传导，并给出价格摩擦、制度约束与监管情景的方向性证据；当前 strongest claim 在于危机期机制异质性与市场割裂动态，弱项在于“主导中介”和政策效果的结构性识别。

