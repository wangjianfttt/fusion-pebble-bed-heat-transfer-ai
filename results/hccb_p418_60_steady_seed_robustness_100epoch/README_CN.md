# 稳态模型三次独立初值结果

- 工况划分：`interleaved_all_ranges`
- 随机种子：20260717, 20260718, 20260719
- 三次训练使用完全相同的OpenFOAM场、训练/验证/独立预测工况和归一化量。
- 响应面是确定性方法，不做没有物理意义的随机重复。

| 模型 | 指标 | 均值 | 样本标准差 | 最小值 | 最大值 |
|---|---|---:|---:|---:|---:|
| Data-only PINN | Solid-$T$ nRMSE  | 0.329 | 0.314 | 0.112 | 0.689 |
| Data-only PINN | Max-$T$ p95 K | 116 | 40.7 | 73.4 | 154 |
| Data-only PINN | Pressure p95 Pa | 13.7 | 10.9 | 4.67 | 25.7 |
| Data-only PINN | Wall-heat p95 \% | 39.8 | 17.1 | 29.3 | 59.6 |
| Data-only PINN | Energy difference \% | 1.59e+03 | 122 | 1.47e+03 | 1.72e+03 |
| Physics PINN | Solid-$T$ nRMSE  | 0.454 | 0.469 | 0.167 | 0.995 |
| Physics PINN | Max-$T$ p95 K | 103 | 22.2 | 80.8 | 125 |
| Physics PINN | Pressure p95 Pa | 16.1 | 13.7 | 6.9 | 31.8 |
| Physics PINN | Wall-heat p95 \% | 39.4 | 20.8 | 24.5 | 63.2 |
| Physics PINN | Energy difference \% | 1.18e+03 | 184 | 977 | 1.33e+03 |
| Graph operator | Solid-$T$ nRMSE  | 0.0725 | 1.44e-03 | 0.0709 | 0.0738 |
| Graph operator | Max-$T$ p95 K | 27.1 | 8.9 | 21.8 | 37.3 |
| Graph operator | Pressure p95 Pa | 3.13 | 0.117 | 3.02 | 3.25 |
| Graph operator | Wall-heat p95 \% | 87 | 10.5 | 78.3 | 98.7 |
| Graph operator | Energy difference \% | 1.55e+03 | 33.1 | 1.53e+03 | 1.59e+03 |
| Transolver | Solid-$T$ nRMSE  | 0.0675 | 2.62e-03 | 0.0647 | 0.0699 |
| Transolver | Max-$T$ p95 K | 11.9 | 1.2 | 10.6 | 13 |
| Transolver | Pressure p95 Pa | 2.14 | 0.138 | 2.03 | 2.29 |
| Transolver | Wall-heat p95 \% | 73.5 | 31.8 | 44.6 | 108 |
| Transolver | Energy difference \% | 1.32e+03 | 63.1 | 1.25e+03 | 1.36e+03 |
