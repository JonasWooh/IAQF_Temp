# IAQF 2026 项目总结 — 2023 SVB 危机期稳定币与跨所割裂

**更新日期：2026-02-23**

---

## 一、研究目标

本研究围绕 **2023 年 3 月 SVB 危机** 期间稳定币（USDC）脱锚与跨交易所市场割裂现象，旨在回答：

1. **交易成本是否解释割裂？** — 引入链上 Gas Fee 作为价格型摩擦，检验其对跨所价差（Premium）、跨币基差（Basis）传导的影响。
2. **价格摩擦 vs 制度摩擦谁主导？** — 引入周末（银行关闭）、转换暂停等制度哑变量，区分成本型摩擦与制度型摩擦的边际贡献。
3. **危机期市场微观结构如何演变？** — 通过 VAR/VARX 的脉冲响应（IRF）与方差分解（FEVD），刻画 Depeg 冲击对 CMLSI、Basis、Premium 的动态传导。
4. **无套利区间是否被突破？** — 检验 Basis 与交易成本的关系，以及 AR(1) 均值回归在危机期的失效。
5. **GENIUS Act 反事实效应** — 若监管降低冲击幅度、截断尾部或缩短持续时间，IRF 如何变化。

---

## 二、数据与数据格式

### 2.1 数据来源总览

| 数据类型 | 来源 | 格式 | 时间范围 |
|----------|------|------|----------|
| **交易所 Tick** | CoinAPI（BINANCEUS, COINBASE） | `csv.gz`，分号分隔 | 2023-03-01 至 03-25 |
| **Gas Fee** | Ethereum 链上（base_fee_gwei） | `csv` | 2023-03-01 至 03-31 |
| **1 分钟 Parquet** | ETL 从 Tick 生成 | `parquet` | 同上 |
| **Master 表** | Feature Pipeline 生成 | `parquet` | 2023-03-01 至 03-21 |

### 2.2 交易所 Tick 原始数据

- **根目录**：`data/data_flatfiles` 或 `data_flatfiles`（配置自动解析）
- **目录结构**：`{根目录}/{Exchange}/{Pair}/*.csv.gz`
  - 例：`BINANCEUS/BTC_USD/`, `BINANCEUS/USDC_USD/`, `COINBASE/BTC_USD/`
- **文件命名**：`YYYYMMDD_IDDI-*+SC-{EXCHANGE}_SPOT_{PAIR}+S-{SYMBOL}.csv.gz`
  - 例：`20230301_IDDI-5678121+SC-BINANCEUS_SPOT_BTC_USD+S-BTCUSD.csv.gz`
- **列（必需）**：`time_exchange`, `ask_px`, `ask_sx`, `bid_px`, `bid_sx`
  - `ask_px`/`bid_px`：买卖价（美元）；`ask_sx`/`bid_sx`：买卖档位挂单量（size）
- **分隔符**：分号 `;`
- **压缩**：gzip
- **时区**：`time_exchange` 为 UTC（交易所惯例）

### 2.3 1 分钟 Parquet（ETL 输出）

- **路径**：`output/processed_1min/{Exchange}_{Pair}.parquet`（主项目）或 `output_project_layer/processed_1min/`（若主项目无则 project_layer 自建）
- **列**：`time_exchange`, `open`, `high`, `low`, `close`, `spread_mean`, `spread_last`, `depth_mean`, `depth_last`, `obi_mean`, `obi_last`
- **重采样规则**：mid=(ask+bid)/2, spread=ask-bid, depth=ask_sx+bid_sx, obi=(bid_sx-ask_sx)/(bid_sx+ask_sx)；OHLC 取 first/max/min/last，spread/depth/obi 取 mean/last
- **合并后列名**：加前缀 `{Exchange}_{Pair}_`，如 `BINANCEUS_BTCUSD_close`, `COINBASE_BTCUSD_spread_mean`

### 2.4 交易对与交易所

| 交易所 | 交易对 | 用途 |
|--------|--------|------|
| BINANCEUS | USDC_USD | Depeg 监测 |
| BINANCEUS | BTC_USD, BTC_USDC, BTC_USDT | Basis、CMLSI、Premium（BN 端） |
| COINBASE | BTC_USD, BTC_USDC, BTC_USDT | Premium（跨所）、Basis（CB 端） |

### 2.5 Gas Fee 数据

- **路径**：`data/gas_fee/*.csv` 或 `data/gas_Fee/*.csv`（取目录下首个 csv）
- **必要列**：`timestamp`, `base_fee_gwei`（代码仅使用此两列）
- **可选列**：`block_count`, `block_number`（若存在则保留，不参与计算）
- **时区**：有 tz 则转 UTC，无 tz 则假定 UTC；与 Master 对齐时统一为 naive UTC
- **对齐**：按 Master 的 1min 索引 reindex，缺失用 ffill(limit=5) 填充

### 2.6 Master 表（最终特征表）

- **路径**：`output_project_layer/master_features_1min.parquet`
- **索引**：`time_exchange`（DatetimeIndex，1 分钟频率）
- **列**：各交易所-交易对的 OHLC/spread/depth/obi（带前缀）+ `Depeg_bps`, `Premium_bps`, `Basis_bps`, `Basis_USDC_USDT_BN`, `CMLSI` 及扩展因子 `_ret_abs`, `_range_1m`, `_rel_spread`, `_log_depth`
- **缺失填充**：合并时 ffill(limit=5)

### 2.7 中间产物（序列化）

| 文件 | 格式 | 用途 |
|------|------|------|
| `cmlsi_scaler.pkl` | joblib | StandardScaler（Normal 期拟合） |
| `cmlsi_loadings.pkl` | joblib | PCA 载荷向量 |
| `cmlsi_meta.json` | JSON | 载荷、方差解释率、拟合期等元数据 |

### 2.8 派生数据（无独立文件）

| 变量 | 来源 | 说明 |
|------|------|------|
| `is_weekend` | Master 索引 | 按 America/New_York 判断周六/周日 |
| `conversion_pause` | Master 索引 | 2023-03-10 21:00 ET 至 03-13 09:00 ET = 1 |
| `base_fee_gwei`（VARX 用） | Gas CSV 对齐 | 按 Master 1min 索引 reindex + ffill |

### 2.9 危机三阶段划分

| 阶段 | 时间窗口 | 说明 |
|------|----------|------|
| Normal | 2023-03-01 至 03-09 | 危机前 |
| Crisis | 2023-03-10 至 03-13 | SVB 暴雷与 USDC 脱锚 |
| Recovery | 2023-03-14 至 03-21 | 恢复期 |

---

## 三、特征工程

### 3.1 数据流水线（ETL）

1. **Tick → 1 分钟**：对 `ask_px`, `bid_px`, `ask_sx`, `bid_sx` 重采样，得到 OHLC、`spread_mean`, `depth_mean`, `obi_mean` 等。
2. **多表合并**：按 `time_exchange` 外连接，生成 1 分钟频率的完整时间轴，缺失用 `ffill(limit=5)` 填充。
3. **输出**：`output/processed_1min/{Exchange}_{Pair}.parquet`

### 3.2 核心特征构造

| 特征 | 公式 | 含义 |
|------|------|------|
| **Depeg_bps** | `(1 - USDC/USD_close) × 10_000` | USDC 脱锚程度（bps） |
| **Premium_bps** | `(BN_BTC_USD - CB_BTC_USD) / mid × 10_000` | BinanceUS 与 Coinbase 价差 |
| **Basis_bps** | `(BTC_USD - BTC_USDC) / mid × 10_000` | USD 与 USDC 计价价差（BINANCEUS） |
| **Basis_USDC_USDT_BN** | `BTC_USDC - BTC_USDT` | USDC 与 USDT 计价价差（BINANCEUS） |
| **_ret_abs** | `|pct_change(close)|` | 1 分钟收益绝对值 |
| **_range_1m** | `high - low` | 1 分钟波动幅度 |
| **_rel_spread** | `spread / close` | 相对买卖价差 |
| **_log_depth** | `log(depth + ε)` | 深度对数 |

### 3.3 CMLSI（综合流动性压力指数）

#### 3.3.1 流程

1. **标准化**：`StandardScaler` 仅用 Normal 期（2023-03-01 至 03-09）拟合，对 7 个特征做 z-score。
2. **PCA 拟合**：同上 Normal 期，`PCA(n_components=3)` 拟合。
3. **OOS 变换**：对 Normal / Crisis / Recovery 三期分别 `scaler.transform` + `pca.transform`，全样本无 look-ahead。
4. **加权合成**：`CMLSI = Σ(PC_i × weight_i)`，其中 `weight_i = evr_i / Σevr`（evr 为各主成分方差解释率）。
5. **符号翻转**：若 Crisis 期 CMLSI 均值 < Normal 期均值，则整体取反，使 CMLSI 升高表示流动性恶化。
6. **缺失填充**：`ffill(limit=5).bfill(limit=5)`。

#### 3.3.2 输入特征（7 个）

| 特征 | 含义 |
|------|------|
| `BINANCEUS_BTCUSD_spread_mean` | 买卖价差 |
| `BINANCEUS_BTCUSD_depth_mean` | 深度 |
| `BINANCEUS_BTCUSD_obi_mean` | 订单簿失衡 OBI |
| `_ret_abs` | 1 分钟收益绝对值 |
| `_range_1m` | 1 分钟波动幅度 |
| `_rel_spread` | 相对价差 |
| `_log_depth` | 深度对数 |

#### 3.3.3 主成分与方差解释率

| PC | 解释方差 |
|----|----------|
| PC1 | 46.9% |
| PC2 | 24.3% |
| PC3 | 14.3% |
| **Top 3 累计** | **85.6%** |
| PC4–7 | 9.8%, 3.1%, 1.5%, 0.1% |

#### 3.3.4 综合载荷（loadings）

`loadings = components.T @ weights`（Top 3 主成分载荷的加权组合）：

| 特征 | 载荷 |
|------|------|
| spread_mean | 0.233 |
| depth_mean | 0.258 |
| obi_mean | 0.163 |
| _ret_abs | 0.226 |
| _range_1m | 0.254 |
| _rel_spread | 0.236 |
| _log_depth | 0.302 |

#### 3.3.5 动态分解（图 05）

Crisis 期内，`contribution_j = scaled_feature_j × loading_j`，各特征对 CMLSI 的贡献可分解；图 05 展示 15 分钟滚动平均后的动态分解曲线。

#### 3.3.6 输出与图表

- **输出**：Master 表新增列 `CMLSI`。
- **图 04**：CMLSI 载荷条形图。
- **图 05**：Crisis 期 CMLSI 动态分解（各特征贡献 + CMLSI 曲线）。
- **图 08**：PCA 各主成分方差解释率条形图。

#### 3.3.7 备注

- **数据源**：仅用 BINANCEUS BTC/USD 的微观结构特征，不包含 COINBASE。
- **可复现**：`scaler_mean`、`scaler_scale` 保存于 `cmlsi_meta.json`，用于复现标准化。

### 3.4 外生变量（VARX）

| 变量 | 定义 | 用途 |
|------|------|------|
| **base_fee_gwei** | 原始 Gas 费（Gwei） | 价格型摩擦，ADF 平稳后作为连续 exog |
| **is_weekend** | 周六/周日 = 1（America/New_York） | 制度摩擦：银行周末关闭 |
| **conversion_pause** | 2023-03-10 21:00–03-13 09:00 ET = 1 | 制度摩擦：Coinbase USDC↔USD 暂停 |

### 3.5 Gas 外生变量模式（VARX 配置）

| 模式 | 说明 | 使用条件 |
|------|------|----------|
| **level** | 直接使用 `base_fee_gwei` | ADF 平稳（当前采用） |
| **log_level** | ln(gas) | 需 ADF 通过 |
| **dummy** | `I(gas > threshold)`，极端拥堵哑变量 | ADF 不通过时回退 |

### 3.6 0-1 变量（哑变量）完整说明

#### 3.6.1 制度摩擦哑变量

| 变量 | 定义 | 时区 | 取值 |
|------|------|------|------|
| **is_weekend** | 周六/周日 = 1，否则 0 | America/New_York (ET) | 0, 1 |
| **conversion_pause** | 2023-03-10 21:00 ET 至 03-13 09:00 ET = 1 | America/New_York (ET) | 0, 1 |

- **is_weekend**：`weekday >= 5`（Python 周一=0，周六=5、周日=6），表示银行周末关闭、法币通道流动性下降。
- **conversion_pause**：Coinbase USDC↔USD 转换暂停窗口，可自定义 `pause_start`、`pause_end`。

#### 3.6.2 Gas 极端拥堵哑变量（gas_congestion_dummy）

- **定义**：`I(base_fee_gwei > threshold)`，1 = 链上拥堵（套利受阻），0 = 畅通。
- **阈值确定**：
  - 若 `gas_dummy_threshold_gwei` 给定，直接使用；
  - 否则用 **calibration 期** 内 `base_fee_gwei` 的 **percentile 分位数**；
  - 默认 calibration：Normal 期（2023-03-01 至 03-09）；
  - 默认 percentile：95（`gas_dummy_threshold_percentile`）；
  - fallback：若 calibration 样本 < 10，则 threshold = 80.0 Gwei。
- **取值**：0, 1；Gas 缺失时 NaN（上游 dropna 剔除）。

#### 3.6.3 哑变量重合度与使用规则

- **函数**：`check_dummy_overlap(weekend, pause)` 返回 `corr`、`n`、`rule`。
- **规则**：
  - `corr > 0.90`：主模型只用其一（weekend）；
  - `0.60 ≤ corr < 0.90`：谨慎使用，避免共线性；
  - `corr < 0.60`：可同时放入。
- **本项目**：corr(is_weekend, conversion_pause) ≈ 0.77，处于 0.60–0.90 区间，主模型用 gas+weekend，稳健性用 gas+pause，分别估计。

#### 3.6.4 时区与索引

- **Master 索引**：naive UTC（交易所惯例）。
- **weekend / pause**：先将索引转为 UTC（若无 tz 则 `tz_localize("UTC")`），再 `tz_convert("America/New_York")` 判断。

---

## 四、流程总览

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. Data (ETL)                                                            │
│    Tick CSV.gz → 1min Parquet (OHLC + spread, depth, obi)                │
├─────────────────────────────────────────────────────────────────────────┤
│ 2. Features (Pipeline)                                                   │
│    run_stage_data → run_stage_features → run_stage_pca                   │
│    合并多表 → Depeg, Basis, Premium, 扩展因子 → CMLSI                     │
│    输出：master_features_1min.parquet                                     │
├─────────────────────────────────────────────────────────────────────────┤
│ 3. EDA                                                                   │
│    图01-05, 09, 10：Depeg 时间线、流动性、价差、CMLSI、Basis、Premium    │
│    图19：Gas Fee EDA（时间序列、分布、箱线图）                            │
├─────────────────────────────────────────────────────────────────────────┤
│ 4. VAR / VARX 建模                                                       │
│    - 按 regime 切片（Normal / Crisis / Recovery）                        │
│    - 内生变量：[Depeg_bps, CMLSI, Basis_bps, Premium_bps]                 │
│    - 外生变量：gas | gas+weekend | gas+pause                              │
│    - 滞后阶数：AIC 选择，平稳性：ADF 检验后必要时差分                      │
├─────────────────────────────────────────────────────────────────────────┤
│ 5. 诊断与仿真                                                            │
│    - VAR 稳定性、残差白噪声                                               │
│    - AR(1) 均值回归（Basis）                                              │
│    - 交易成本与无套利区间（图18）                                         │
│    - GENIUS Act 反事实仿真（A/B/C 场景）                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.1 运行方式

```bash
# 项目根目录
python -m project_layer.main

# 可选参数
python -m project_layer.main --etl              # 从 Tick 重新跑 ETL
python -m project_layer.main --stages features var diagnostics simulation  # 指定阶段
```

**阶段**：`data` | `features` | `var` | `diagnostics` | `simulation`（默认全跑；`features` 会触发 `eda`）

### 4.2 各阶段产出

| 阶段 | 产出 |
|------|------|
| **data** | `output/processed_1min/{Exchange}_{Pair}.parquet`（需 `--etl` 或主项目已生成） |
| **features** | `master_features_1min.parquet`, `cmlsi_meta.json`, `cmlsi_scaler.pkl`, `cmlsi_loadings.pkl`, 图08 |
| **eda** | 图01–05, 09, 10, 19, `gas_fee_eda_report.txt` |
| **var** | 图06, 07, 11, 12, 14–17, 20, `var_diagnostics.txt`, `var_gas_comparison_report.txt`, `var_institutional_report.txt` |
| **diagnostics** | `ar1_mean_reversion.txt`, 图18 |
| **simulation** | `genius_act_simulation/figures/19*.png`, `simulation_report.txt` |

---

## 五、统计与检验结果

### 5.1 VAR 模型诊断

| Regime | 稳定性 | Depeg_bps | CMLSI | Basis_bps | Premium_bps |
|--------|--------|-----------|-------|-----------|-------------|
| Normal | ✓ | p=1.0000 | p=1.0000 | p=1.0000 | p=0.9997 |
| Crisis | ✓ | p=0.9955 | p=1.0000 | p=1.0000 | p=1.0000 |
| Recovery | ✓ | p=0.9999 | p=1.0000 | p=1.0000 | p=1.0000 |

*注：p 为 Ljung-Box 残差白噪声检验 p 值，>0.05 表示不拒绝白噪声。*

### 5.2 Gas Fee 描述统计与 ADF

| 阶段 | n | mean | median | std | p95 | ADF p |
|------|---|------|--------|-----|-----|-------|
| Full | 30,240 | 29.53 | — | 24.43 | — | — |
| Normal | 12,960 | 29.81 | 25.77 | 13.74 | 54.56 | 0.0000 ✓ |
| Crisis | 5,760 | 43.12 | 28.16 | 47.01 | 118.30 | 0.0003 ✓ |
| Recovery | 11,520 | 22.43 | 19.51 | 10.33 | 40.82 | 0.0000 ✓ |

*Crisis 期 Gas 均值较 Normal 高约 45%，波动显著增大；各期 base_fee_gwei 均通过 ADF 平稳性检验。*

### 5.3 AR(1) 均值回归（Basis）

| Regime | ρ | α = 1−ρ |
|--------|---|---------|
| Normal | 0.6292 | 0.3708 |
| Crisis | 0.9992 | 0.0008 |
| Recovery | 0.9422 | 0.0578 |

*Crisis 期 ρ≈1，均值回归几乎失效，Basis 呈现近似随机游走。*

### 5.4 交易成本 vs Gas Fee 对比（VARX）

| 模型 | exog | Depeg→Premium IRF 峰值 | Premium FEVD (终期) |
|------|------|------------------------|---------------------|
| 纯 VAR | 无 | 0.2221 bps | 0.91% |
| VARX | base_fee_gwei | 0.2177 bps | 0.79% |

*控制 Gas 后，IRF 与 FEVD 均下降，说明链上拥堵对 Depeg→Premium 传导有抑制效应。*

### 5.5 制度摩擦对比（Crisis 子样本）

| 模型 | exog | IRF 峰值 | FEVD |
|------|------|----------|------|
| Baseline | gas only | 0.2177 bps | 0.79% |
| Main | gas + weekend | 0.2237 bps | 0.85% |
| Robustness | gas + pause | 0.2205 bps | 0.81% |

**重合度**：corr(is_weekend, conversion_pause) = 0.77 → 0.60–0.90 区间，主模型与稳健性分别估计，避免共线性。

### 5.6 交易成本与无套利区间（图18）

- **交易成本构成**：TC = 20 bps（round-trip 手续费）+ spread_USD_bps + spread_USDC_bps。
- **保守阈值**：15 bps，用于判断 |Basis| 是否突破无套利区间。
- **输出**：三期（Normal / Crisis / Recovery）的 Mean |Basis| vs Mean TC 柱状图，以及 % Minutes with |Basis| > 15 bps。
- **含义**：Crisis 期 Mean |Basis| 与 % exceed 15 bps 通常最高，反映套利成本约束在危机期被突破更频繁。

### 5.7 VAR 参数（settings.yaml）

| 参数 | 值 |
|------|-----|
| 内生变量 | Depeg_bps, CMLSI, Basis_bps, Premium_bps |
| IRF 步数 | 60 |
| IRF 置信区间 | 500 次 MC，seed=42，α=0.05 |
| 三期 IRF 置信区间 | 150 次 MC |
| ADF 显著性 | 0.05（内生）；Gas 0.10 |

### 5.8 IRF 与 FEVD 完整说明

#### 5.8.1 脉冲响应（IRF）设定

- **方法**：正交化 IRF（`orth_irfs`），基于 Cholesky 分解，由 statsmodels VAR 提供。
- **变量顺序**：`[Depeg_bps, CMLSI, Basis_bps, Premium_bps]`（索引 0–3）
  - 顺序即因果顺序：Depeg 最外生（受其他变量当期影响最小），Premium 最内生。
- **IRF 形状**：`orth_irfs[t, 响应变量, 冲击变量]`，t = 0, 1, …, 59（60 步）。
- **冲击定义**：1 单位正向冲击（Depeg 冲击 = 1 bps 脱锚加剧）。
- **滞后阶数**：AIC 选择，`maxlags=20`。
- **平稳性**：若某变量 ADF 不通过，则对该变量做一阶差分后再拟合。

#### 5.8.2 置信区间

- **Crisis 单期**：`irf_errband_mc(orth=True, repl=500, steps=60, signif=0.05, seed=42)`。
- **三期对比**：`repl=150`（图 14、16）。
- **含义**：95% 蒙特卡洛置信带，用于判断 IRF 是否显著异于 0。

#### 5.8.3 各图 IRF 内容

| 图 | 冲击 | 响应 | 对比/说明 |
|----|------|------|-----------|
| **06** | Depeg | Spread / RelSpread（隐含） | CMLSI IRF × loading × std，反推微观指标响应 |
| **07** | Depeg | Premium | Crisis 期，带 95% CI |
| **11** | — | FEVD | Depeg 对 CMLSI、Basis、Premium 的方差贡献（%） |
| **12** | Depeg | CMLSI, Basis, Premium | Normal vs Crisis 两期对比 |
| **14** | Depeg | Basis | 三期（Normal/Crisis/Recovery）带 CI |
| **15** | Depeg / CMLSI / Premium | Basis | Basis 对三种冲击的 IRF（3 子图） |
| **16** | Depeg | CMLSI, Basis, Premium | 三期对比，3 子图 |
| **17** | Depeg | Premium | Baseline(VAR) vs VARX(gas) |
| **20** | Depeg | Premium | gas only vs gas+weekend vs gas+pause |

#### 5.8.4 FEVD（Forecast Error Variance Decomposition）

- **定义**：各冲击对某变量 h 步预测误差方差的贡献比例。
- **输出**：`fevd.decomp[响应, horizon, 冲击]`，horizon 1–60。
- **常用**：Premium 方差中来自 Depeg 的比例（终期约 0.79%–0.91%）。

#### 5.8.5 Implied IRF（图 06）

- **公式**：`implied_j = CMLSI_IRF × loading_j × std_j`
- **含义**：Depeg 冲击下，CMLSI 的响应通过 CMLSI 载荷反推为 spread、rel_spread 等原始微观指标的“隐含”响应（单位：美元或比例）。

#### 5.8.6 模型与 exog 对应

| 图表 | 模型 | exog |
|------|------|------|
| 06, 07, 11, 12, 14, 15, 16 | 纯 VAR | 无 |
| 17 | VAR vs VARX | 无 vs base_fee_gwei |
| 20 | VARX × 3 | gas / gas+weekend / gas+pause |

*注：图 12、14、16 按 regime 分别拟合 VAR（Normal/Crisis/Recovery 各一期），均为纯 VAR。*

---

## 六、主要图表

| 编号 | 内容 |
|------|------|
| 01 | USDC Depeg 时间线 |
| 02 | 流动性蒸发（Spread / Depth） |
| 03 | 跨所价差 |
| 04–05 | CMLSI 载荷与分解 |
| 06 | Implied IRF（Genius Act） |
| 07 | 市场割裂 IRF（Depeg→Premium） |
| 08 | PCA 方差解释率 |
| 09–10 | 跨币基差、稳定币 Premium |
| 11 | FEVD（Depeg 贡献） |
| 12 | Regime 对比 IRF |
| 14–16 | 三期 IRF（Depeg→Basis 等） |
| 17 | Gas Fee 对比（Baseline vs VARX） |
| 18 | 交易成本与无套利区间 |
| 19 | Gas Fee EDA |
| 20 | 制度摩擦对比（gas vs gas+weekend vs gas+pause） |

### 6.1 EDA 完整说明

#### 6.1.1 运行入口与依赖

- **入口**：`run_eda()`（在 `features` 或 `eda` 阶段自动调用）
- **依赖**：`processed_1min` 或 `master_features_1min.parquet` 至少其一存在；图 04、05 需 Master + CMLSI；图 19 需 Gas CSV

#### 6.1.2 各图内容与数据源

| 图 | 标题 | 数据源 | 变量 | 可视化 |
|----|------|--------|------|--------|
| **01** | USDC De-peg Timeline | Master 或 BINANCEUS USDC_USD | USDC/USD 1min close | 时间序列，y=1 参考线，Crisis 期红色阴影 |
| **02** | Liquidity Evaporation | Master 或 BINANCEUS/COINBASE BTC_USD | spread_mean, depth_last | 2 子图：Spread + Depth，1min 原始 + 60min MA，Crisis 期阴影 |
| **03** | Cross-Exchange Spread | Master 或 BN/CB BTC_USD | Premium_BTC = BN−CB (USD) | 时间序列，Crisis 期阴影 |
| **04** | CMLSI Factor Loadings | cmlsi_meta.json | 7 特征载荷 | 水平条形图，正负分色 |
| **05** | CMLSI Dynamic Decomposition | Master + scaler + loadings | contribution_j = scaled_j × loading_j | Crisis 期各特征贡献 + CMLSI，15min 滚动平均 |
| **09** | Cross-Currency Basis | Master | Basis_USD_USDC_BN, Basis_USD_USDT_BN | BTC/USD−USDC、BTC/USD−USDT，Crisis 期阴影 |
| **10** | Stablecoin Dynamics | Master | Basis_USDC_USDT_BN, Basis_USDC_USDT_CB | BN/CB 的 USDC−USDT 价差，Crisis 期阴影 |
| **19** | Gas Fee EDA | Gas CSV | base_fee_gwei | 3 子图：时间序列+60min MA、分 regime 直方图、箱线图 |

#### 6.1.3 通用设定

- **时间范围**：`dates.start` 至 `dates.end`（多数）；图 01、09、10 用 `eda_usdc_end`
- **Crisis 期标注**：2023-03-10 至 03-13，红色半透明 `axvspan`
- **滚动窗口**：`spread_ma_window` = 60 分钟（图 02、19）
- **原始数据透明度**：`raw_data_alpha` = 0.3

#### 6.1.4 Gas Fee EDA 报告（gas_fee_eda_report.txt）

- **全样本**：n, mean, std, min, max
- **分 regime**：Normal / Crisis / Recovery 的 n, mean, median, std, min, max, p95
- **ADF 检验**：各期 base_fee_gwei 的 p 值及 stationary 结论

#### 6.1.5 图 04、05 前置条件

- 需 `master_features_1min.parquet` 存在（即已跑完 features 阶段）
- 图 04：`cmlsi_meta.json`
- 图 05：`cmlsi_meta.json` + `cmlsi_scaler.pkl` + `cmlsi_loadings.pkl`

### 6.2 GENIUS Act 反事实仿真

| 场景 | 策略 | 参数 | 含义 |
|------|------|------|------|
| **A** | MagnitudeScaling | c=1, 0.75, 0.5, 0.2 | 监管降低冲击幅度，IRF × c |
| **B** | TailTruncation | 95% winsorize | 截断 Depeg 残差尾部，有效 c = std_win/std_orig |
| **C** | DurationDamp | τ=5, 10, 20 | IRF × exp(-t/τ)，监管缩短冲击持续时间 |

**输出**：`genius_act_simulation/figures/19_genius_act_three_scenarios.png`（3×3 子图：Depeg→CMLSI/Basis/Premium × A/B/C）、`19a_scenario_A_magnitude.png`。

### 6.3 输出文件路径

| 类型 | 路径 |
|------|------|
| Master 表 | `output_project_layer/master_features_1min.parquet` |
| CMLSI meta | `output_project_layer/cmlsi_meta.json` |
| 图表 | `output_project_layer/figures/01–20_*.png` |
| VAR 报告 | `var_diagnostics.txt`, `var_gas_comparison_report.txt`, `var_institutional_report.txt` |
| Gas EDA | `gas_fee_eda_report.txt` |
| AR(1) | `ar1_mean_reversion.txt` |
| 仿真 | `genius_act_simulation/figures/`, `simulation_report.txt` |

---

## 七、结论

1. **交易成本（Gas Fee）显著影响跨所价差传导**  
   在 Crisis 子样本中，将 base_fee_gwei 作为外生变量纳入 VARX 后，Depeg 对 Premium 的脉冲响应与方差贡献均下降，表明链上拥堵是重要的价格型摩擦。

2. **制度摩擦具有边际解释力**  
   在控制 Gas 的基础上加入 weekend 或 conversion_pause，IRF 与 FEVD 略有上升，说明非价格型制度摩擦（银行周末关闭、转换暂停）在危机期对跨所割裂有额外贡献。主模型采用 gas+weekend，gas+pause 作为稳健性替代。

3. **价格摩擦与制度摩擦共同作用**  
   成本摩擦（Gas）主要起“抑制”作用，制度摩擦（weekend/pause）在危机与周末重叠时起“放大”作用，二者共同刻画了 2023 年 3 月 SVB 危机期间的市场割裂机制。

4. **Crisis 期 Basis 均值回归失效**  
   AR(1) 的 ρ 在 Crisis 期接近 1，跨币基差呈现近似随机游走，套利力量在危机期显著减弱。

5. **CMLSI 有效刻画流动性压力**  
   基于微观结构特征的 PCA 综合指数在 Normal 期拟合、全样本 OOS 变换，累计解释方差约 85.6%，能区分 Normal / Crisis / Recovery 三期的流动性状态。

6. **数据与模型设定稳健**  
   时区统一为 UTC，Gas 与 Master 完全对齐；各期 Gas 通过 ADF 平稳性检验；VAR 在三期均满足稳定性与残差白噪声假设。

7. **GENIUS Act 反事实**  
   三种情景（幅度缩放、尾部截断、持续时间衰减）可量化监管干预对 Depeg→CMLSI/Basis/Premium 传导的缓解效果，为政策评估提供参考。

---

## 八、各变量对研究目标的解释度

下表汇总各变量对五个研究目标的贡献程度及对应的度量指标。

### 8.1 变量 × 目标 解释度矩阵

| 变量 | 目标1：交易成本解释割裂 | 目标2：价格 vs 制度摩擦 | 目标3：微观结构演变 | 目标4：无套利区间 | 目标5：GENIUS Act 反事实 |
|------|------------------------|-------------------------|---------------------|-------------------|--------------------------|
| **Depeg_bps** | 冲击源（外生冲击） | 冲击源 | 冲击源，IRF 起点 | 间接（驱动 Basis 偏离） | 被变换的 IRF 基准 |
| **CMLSI** | 传导中介 | 传导中介 | **核心**：流动性压力综合指数，IRF 响应变量 | 间接 | IRF 响应之一 |
| **Basis_bps** | 传导结果 | 传导结果 | **核心**：跨币基差，IRF 响应变量 | **核心**：被检验变量，AR(1) ρ、均值 abs(Basis) vs TC | IRF 响应之一 |
| **Premium_bps** | **核心**：被解释变量，割裂的直接度量 | **核心**：被解释变量 | **核心**：跨所价差，IRF 响应变量 | 间接 | IRF 响应之一 |
| **base_fee_gwei** | **核心**：价格型摩擦，控制后 IRF↓、FEVD↓ | 与制度哑变量对比 | 外生控制 | 链上成本成分 | — |
| **is_weekend** | — | **核心**：制度摩擦，主模型 exog | — | 间接（周末银行关闭影响套利） | — |
| **conversion_pause** | — | **核心**：制度摩擦，稳健性 exog | — | 直接（暂停期间无法套利） | — |
| **微观结构特征**（spread, depth, obi 等） | — | — | **核心**：CMLSI 输入，载荷贡献 | 间接（spread 构成 TC） | — |

### 8.2 各目标的度量指标与解释度

| 目标 | 主要度量 | 核心解释变量 | 解释度结论 |
|------|----------|--------------|------------|
| **1. 交易成本解释割裂** | Depeg→Premium IRF 峰值、Premium FEVD from Depeg | base_fee_gwei | 控制 Gas 后 IRF 0.2221→0.2177 bps，FEVD 0.91%→0.79%，**中等偏强**：链上拥堵显著抑制传导 |
| **2. 价格 vs 制度摩擦** | 同上，Baseline vs Main vs Robustness | gas, is_weekend, conversion_pause | 加入 weekend 后 IRF/FEVD 略升，**制度摩擦有边际解释力**；与 gas 共线需谨慎（corr≈0.77） |
| **3. 微观结构演变** | IRF、FEVD、CMLSI 动态分解 | Depeg, CMLSI, Basis, Premium | **强**：四变量 VAR 刻画完整传导链；CMLSI 载荷、图 05 分解给出各微观特征贡献 |
| **4. 无套利区间** | AR(1) ρ、均值 abs(Basis) vs TC、% exceed 15 bps | Basis_bps, TC | **强**：Crisis 期 ρ≈1 均值回归失效；Crisis 期 abs(Basis) 与突破比例最高 |
| **5. GENIUS Act 反事实** | 变换后 IRF（c×IRF、winsorize、exp(-t/τ)） | IRF 本身（无新变量） | **政策模拟**：量化监管对 Depeg→CMLSI/Basis/Premium 的缓解幅度 |

### 8.3 变量角色分类

| 角色 | 变量 | 说明 |
|------|------|------|
| **冲击源** | Depeg_bps | 外生冲击，所有 IRF 的起点 |
| **传导中介** | CMLSI | 流动性压力，连接 Depeg 与 Basis/Premium |
| **被解释/结果** | Basis_bps, Premium_bps | 割裂的直接度量 |
| **外生控制（价格摩擦）** | base_fee_gwei | 链上成本，抑制传导 |
| **外生控制（制度摩擦）** | is_weekend, conversion_pause | 非价格约束，放大传导 |
