# P418 三维球床换热计算采用的物理参数

本表列出进入60组稳态端点、12组温度阶跃和瞬态能量方程的物理参数。神经网络层数、隐藏宽度和注意力头数不是球床物理参数，另列在模型结构文件中。

| 编号 | 物理量 | 采用值或关系式 | 单位 | 在计算中的用途 | 来源 |
|---|---|---|---|---|---|
| P048 | 颗粒直径 | 1 | mm | 定义源球床的1 mm颗粒 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P049 | 源球床目标孔隙率 | 0.397 | dimensionless | 定义堆积生成目标；实际三角网格孔隙率另行计算 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P050 | 初始大球床尺寸 | 25dp x 25dp x 10dp | dimensionless | 先生成25dp x 25dp x 10dp大球床 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P390 | 文献截取区域尺寸 | 12.5dp x 12.5dp x 10dp; inlet channel=10dp; outlet extension=10dp | dimensionless_geometry | 从大球床截取12.5dp x 12.5dp x 10dp区域 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P404 | 网格颗粒直径修正 | 1 | percent_diameter_reduction | 网格生成前将颗粒直径缩小1%，去除点接触 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P423 | 靠冷却壁的截取方式 | wall-adjacent and laterally centred crop; one retained outer face=constant-temperature wall; remaining transverse faces=symmetry boundaries | dimensionless_geometry_and_boundary | 采用一侧靠壁、横向居中的文献截取方式 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P418 | 60组入口速度、入口温度和颗粒发热率 | Re_p_AVE<1.8;87 percent within +/-30 percent;60 cases from u_in=0.05,0.10,0.15,0.20,0.25 m/s x T_in=300,500,700,900 K x phi=4.85,6.85,8.85 MW/m3 | mixed_support_envelope | 定义全部计算工况 | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P425 | 冷却壁温度 | 635 | K | 固定为635 K | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P426 | 工作压力 | 0.12 | MPa | 出口绝对压力固定为0.12 MPa | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P427 | 计算域和边界类型 | bed=12.5dp x 12.5dp x 10dp;inlet_extension=10dp;outlet_extension=10dp;one_constant_temperature_wall;remaining_transverse_faces=symmetry | mixed_geometry_boundary | 采用一面恒温冷却壁，其余横向面为对称边界；当前局部球床尺寸另行说明 | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P070 | 氦气动力黏度 | mu=0.4646*T_K^0.66*1e-6 | Pa s | 随局部温度更新 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P071 | 氦气导热系数 | lambda_f=0.1448*(T_K/273)^0.68*(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85) | W/m/K | 随局部温度和压力更新 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P388 | 氦气定压比热 | 5200 | J/kg/K | 流体能量方程 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P389 | 氦气密度 | rho_f=480.19*p_MPa/T_K | kg/m3 | 随局部温度和绝对压力更新 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P092 | Li4SiO4颗粒导热系数 | 1.42 | W/m/K | 颗粒导热方程 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P403 | Li4SiO4颗粒密度 | 1526.4 | kg/m3 | 稳态和瞬态固体能量方程 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P406 | EU参考球比热关系 | Cp=(-5.33e-7*T_C^2+0.001925*T_C+1.238)*1000 | J/kg/K | 只用于60组稳态端点；不进入温度阶跃计算 | [Discrete element method for effective thermal conductivity of packed pebbles accounting for the Smoluchowski effect](https://doi.org/10.1016/j.fusengdes.2018.01.013) |
| P428 | 纯Li4SiO4焓增量关系 | H_T_minus_H_298_J_mol=-17156+73.694*T_K+0.103210*T_K^2-4163115/T_K | J/mol | 定义瞬态固体储热关系 | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P429 | 纯Li4SiO4比热关系 | Cp_molar=73.694+0.206420*T_K+4163115/T_K^2 | J/mol/K | 温度阶跃OpenFOAM和PINN瞬态储热项 | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P430 | 纯Li4SiO4摩尔质量换算 | Li=6.94;O=15.999;Si=28.085;M_Li4SiO4=119.841 | g/mol | 把P429从J/mol/K换算为J/kg/K | [Abridged Standard Atomic Weights 2024](https://www.ciaaw.org/abridged-atomic-weights.htm) |
| P431 | 纯Li4SiO4二级相变温度 | Tc1=938;Tc2=996 | K | 标出平滑比热关系未解析的相变温区 | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P424 | 氦气物性表温度范围 | 300 to 1000 | K | 保证物性表覆盖300--1000 K | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |

## 使用范围

- 60 组工况严格采用 P418 中的 5 个入口速度、4 个入口温度和 3 个颗粒发热率的全部组合。
- P048--P050、P390、P404和P423共同定义源球床生成过程：先生成大球床，再截取靠冷却壁的文献区域，并在网格前把颗粒直径缩小1%以消除点接触。
- 当前精细局部域是在上述文献区域中进一步截取的数值计算域；其尺寸、保留颗粒数和三角网格孔隙率都是本次网格的计算结果，不回写成文献参数。
- 当前计算域是局部致密球床，不等同于 P427 中的完整 12.5dp x 12.5dp x 10dp 区域；这里只沿用文献给出的边界类型。
- P406 来自 EU reference Li4SiO4，仅用于生成60组稳态端点；稳态温度场不依赖热容。
- 12组温度阶跃在建算例时会把P406替换为P428--P431给出的纯Li4SiO4高温量热关系，OpenFOAM和PINN使用同一比热。
- P428--P429是298--1300 K的平滑关系。它不解析P431在938 K和996 K附近的尖锐热容异常，因此接近该温区的结果必须单独标明。
- P430采用天然同位素组成的摩尔质量；若以后改为富集锂颗粒，需要换成实测或明确给定的同位素组成。
