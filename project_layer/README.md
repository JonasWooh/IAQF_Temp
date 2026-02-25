# IAQF 2026 - Project Layer

复合型架构：管道 + 策略 + 单例配置，在软件工程与量化研究的交汇点实现层次模型。

## 架构概览

| 模块 | 模式 | 职责 |
|------|------|------|
| Config | Singleton | 载入 `settings.yaml`，提供 `get_params()` 接口，全局唯一配置源 |
| Data | ETL Job | 纯函数式：CSV → Parquet，不保存状态 |
| Features | Stateful Pipeline | `fit()` 仅限 Normal 期，`transform()` 全样本，消除 Look-ahead Bias |
| Models | Strategy | 根据 regime (normal/crisis/recovery) 返回 VAR 估计对象 |
| Diagnostics | Observer | 监听模型输出，生成文本报告和稳定性图表 |
| Simulation | Strategy | GENIUS Act 情景：A 幅度缩减 / B 尾部截断 / C 持续时间缩短 |
| Usage | Facade | 用户只需 `Runner.start()`，屏蔽内部复杂交互 |

## 目录结构

```
project_layer/
├── settings.yaml       # 全局配置 (15 bps、日期切片等)
├── config/             # Singleton 配置
├── data/               # ETL 纯函数
├── features/           # Pipeline + PCA (fit/transform)
├── models/             # VAR Strategy (by regime)
├── diagnostics/        # Observer 报告
├── simulation/         # GENIUS Act Strategy (A/B/C)
├── usage/              # Facade Runner
├── main.py             # 入口
└── requirements.txt
```

## 运行方式

```bash
# 在项目根目录 (competition_materials) 执行
python -m project_layer.main

# 指定阶段
python -m project_layer.main --stages features var simulation

# 从 Tick 重新跑 ETL（默认使用主项目 processed_1min）
python -m project_layer.main --etl
```

## GENIUS Act 情景切换

在 `simulation/runner.py` 中通过 `strategy_name` 切换：

- `A`, `A_075`, `A_050`, `A_020`：幅度缩放 (c × IRF)
- `B`：尾部截断 (95% winsorize)
- `C_5`, `C_10`, `C_20`：持续时间缩短 (exp(-t/τ))

调用 `run_simulation("A_050")` 即可得到对应反事实图表。
