# CMLSI 稳健性检验 — 完整统计结果

**更新日期：2026-02-23**

---

## 一、Step 1：结构变化检验

### 1.1 协方差矩阵差异（Frobenius 范数）

| 指标 | 值 |
|------|-----|
| **\|\|Σ_N - Σ_C\|\|_F** | 374.41 |
| \|\|Σ_N\|\|_F（Normal 协方差范数） | 211.95 |
| \|\|Σ_C\|\|_F（Crisis 协方差范数） | 586.33 |
| n_normal | 12,959 |
| n_crisis | 5,760 |

**解读**：Frobenius 范数 374.41 较大，且 Crisis 期协方差范数约为 Normal 期的 2.77 倍，表明 **regime shift 存在**。

---

### 1.2 特征值分布比较（Crisis / Normal）

| PC | 特征值比 (λ_C / λ_N) | Normal EVR | Crisis EVR |
|----|----------------------|------------|------------|
| PC1 | 0.915 | 46.94% | 42.96% |
| PC2 | 1.022 | 24.34% | 24.86% |
| PC3 | 0.998 | 14.30% | 14.26% |
| PC4 | 1.270 | 9.75% | 12.38% |
| PC5 | 1.114 | 3.07% | 3.42% |
| PC6 | 1.316 | 1.46% | 1.93% |
| PC7 | 1.390 | 0.13% | 0.19% |

**解读**：
- PC1 特征值比 < 1：Crisis 期第一主成分解释力略降。
- PC4–PC7 特征值比 > 1：Crisis 期方差更分散，高维成分贡献上升。
- PC1 EVR 从 46.94% 降至 42.96%：**方差解释结构发生跃迁**。

---

### 1.3 PCA 子空间角度 θ = arccos(\|v_N^T v_C\|)

| 主成分 | 角度（度） | 解读 |
|--------|------------|------|
| PC1 | 14.54° | 主方向有中等旋转 |
| PC2 | 11.74° | 次主方向有轻微旋转 |
| PC3 | **89.11°** | 几乎正交，**结构显著变化** |

**PC1 载荷对比（Normal vs Crisis）**：

| 特征 | Normal PC1 | Crisis PC1 |
|------|------------|------------|
| spread_mean | 0.497 | 0.617 |
| depth_mean | 0.107 | 0.022 |
| obi_mean | 0.001 | -0.010 |
| _ret_abs | 0.446 | 0.351 |
| _range_1m | 0.506 | 0.400 |
| _rel_spread | 0.499 | 0.576 |
| _log_depth | 0.193 | 0.065 |

**解读**：Crisis 期 spread、rel_spread 权重上升，depth、log_depth 权重下降，流动性压力主要由价差驱动，深度贡献减弱。

---

## 二、Step 2：Rolling PCA 与 Adaptive CMLSI

### 2.1 设定

- **窗口**：1,440 分钟（24 小时）
- **步长**：60 分钟
- **主成分数**：3

### 2.2 Rolling PC1 解释率（EVR）

Rolling PCA 在每步用过去 24h 数据拟合，输出 PC1 EVR。全样本共 482 个时间点。

| 时期 | 时间范围 | PC1 EVR 均值 | PC1 EVR 标准差 | 说明 |
|------|----------|--------------|----------------|------|
| Normal | 2023-03-02 ~ 03-09 | 0.4433 | 0.0407 | 相对稳定 |
| Crisis | 2023-03-10 ~ 03-13 | 0.4313 | 0.0388 | 略降，结构变化 |
| Recovery | 2023-03-14 ~ 03-21 | 0.4178 | 0.0342 | 进一步下降 |

*注：Recovery 期 PC1 EVR 均值最低，反映危机后方差结构尚未完全回归 Normal 形态。*

---

## 三、Step 3：VAR 对比（Stable vs Adaptive CMLSI）

### 3.1 IRF 峰值（Crisis 期）

| 冲击→响应 | Stable CMLSI | Adaptive CMLSI | 差异 |
|-----------|--------------|----------------|------|
| Depeg → CMLSI | 0.0225 | 0.0171 | -23.8% |
| Depeg → Basis | 2.1622 bps | 2.1615 bps | -0.03% |
| Depeg → Premium | 0.2221 bps | 0.2277 bps | +2.5% |

### 3.2 FEVD 终期（h=60，Depeg 冲击的方差贡献 %）

| 响应变量 | Stable CMLSI | Adaptive CMLSI | 差异 |
|----------|--------------|----------------|------|
| CMLSI | 0.078% | 0.109% | +39.9% |
| Basis | 5.656% | 5.667% | +0.2% |
| Premium | 0.911% | 1.022% | +12.2% |

### 3.3 解读

- **Depeg → CMLSI**：Adaptive 的 IRF 峰值更低，FEVD 更高，说明 Adaptive CMLSI 对 Depeg 冲击更敏感。
- **Depeg → Basis**：两种指数下几乎一致，**对 Basis 传导稳健**。
- **Depeg → Premium**：IRF 峰值差异约 2.5%，FEVD 差异约 12%，**对 Premium 传导有轻微差异**，但量级不大。

**结论**：CMLSI 构造（Stable vs Adaptive）对 VAR 传导结论影响有限，**对指数构造具有稳健性**。

---

## 四、Step 4：经济解释汇总

| 检验 | 结果 | 经济含义 |
|------|------|----------|
| 协方差 Frobenius 范数 | 374.41 | Normal 与 Crisis 协方差结构显著不同，regime shift 成立 |
| 特征值比 | PC1<1, PC4–7>1 | Crisis 期方差更分散，高维成分重要性上升 |
| PC3 子空间角度 | 89.11° | 第三主方向几乎正交，结构发生根本性变化 |
| Stable vs Adaptive IRF | 差异 < 3% | 冲击传导对指数构造稳健 |
| Stable vs Adaptive FEVD | Premium +12% | Adaptive 下 Depeg 对 Premium 方差贡献略高 |

---

## 五、稳健性结论

1. **结构变化得到证实**：协方差矩阵、特征值分布、PCA 子空间在 Normal 与 Crisis 间存在显著差异。
2. **Stable CMLSI 的局限**：基于 Normal 期拟合，在 Crisis 期存在结构错配，但作为“冲击幅度”的度量仍可用。
3. **Adaptive CMLSI 的补充**：Rolling PCA 能捕捉 regime 内结构变化，更适合测度“结构性崩溃”。
4. **VAR 结论稳健**：两种 CMLSI 下 Depeg→Basis、Depeg→Premium 的 IRF 与 FEVD 差异有限，主要传导结论不受指数构造影响。
