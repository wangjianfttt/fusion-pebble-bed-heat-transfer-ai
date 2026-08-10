# P418 氦气直接输运物性模块

该模块把项目已登记的两个氦气输运关联式直接写入 OpenFOAM 13：

- 黏度：`mu = 0.4646 T^0.66 x 1e-6`；
- 导热系数：`k = 0.1448 (T/273)^0.68 [1 + 2.5e-3 (p/1e6)^1.17 (T/273)^-1.85]`。

它不改变关联式、常数、状态方程、热容、网格或边界条件。目的只是避免
`UniformTable2` 在压力稍微越出表格边界时直接终止计算。密度仍由现有
`perfectGas` 状态方程计算。

## 编译与逐点检查

```bash
source /opt/openfoam13/etc/bashrc
wmake libso solver_extensions/hccbHeliumTransport
wmake solver_extensions/hccbHeliumTransport/check
hccbHeliumTransportCheck
```

检查程序只计算 12 个 `(p,T)` 点的 `mu` 和 `kappa`，不读取网格，也不启动
流动或传热求解器。实际算例需要在 `controlDict` 的 `libs` 中加载
`libhccbHeliumTransport.so`，并把流体 `physicalProperties` 的 transport 改为
`hccbHelium`；示例见 `physicalProperties.example`。
