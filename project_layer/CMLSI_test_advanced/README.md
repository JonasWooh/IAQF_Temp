# CMLSI Advanced Robustness Tests

结构化稳健性检验，用于回应「Normal 期协方差结构在 Crisis 期是否适用」的质疑。

## 运行方式

```bash
# 项目根目录
python -m project_layer.main --stages cmlsi_robustness
```

或单独运行：

```python
from project_layer.CMLSI_test_advanced import run_full_cmlsi_robustness
run_full_cmlsi_robustness()
```

## 四步流程

### Step 1：证明结构变了

- **协方差矩阵差异**：`||Σ_N - Σ_C||_F`（Frobenius 范数）
- **特征值分布**：Crisis/Normal 特征值比
- **PCA 子空间角度**：`θ = arccos(|v_N^T v_C|)`，主方向旋转

### Step 2：构造两个指数

- **Stable CMLSI**：Normal 期拟合，OOS 变换（测「冲击幅度」）
- **Adaptive CMLSI**：Rolling PCA（24h 窗口），测「结构性崩溃」

### Step 3：VAR 中比较

- IRF：Depeg → CMLSI, Basis, Premium
- FEVD：各变量方差分解

### Step 4：经济解释

- Stable index → 冲击幅度
- Adaptive index → 结构性崩溃
- 若 IRF/FEVD 差异大 → 指数构造影响结论；若相似 → 对指数构造稳健

## 输出

| 文件 | 说明 |
|------|------|
| `structure_tests_report.txt` | 结构检验结果 |
| `structure_tests_report.json` | 同上（JSON） |
| `01_structure_tests.png` | 结构检验图 |
| `02_rolling_pca.png` | Rolling PCA 的 PC1 EVR 与载荷漂移 |
| `03_stable_vs_adaptive_cmlsi.png` | Stable vs Adaptive 时间序列 |
| `04_var_stable_vs_adaptive.png` | VAR IRF/FEVD 对比 |
| `cmlsi_adaptive.csv` | Adaptive CMLSI 序列 |
| `rolling_pca_metrics.csv` | Rolling PCA 指标 |
| `var_comparison_stable_vs_adaptive.json` | VAR 对比结果 |
| `economic_interpretation_report.txt` | 经济解释报告 |
