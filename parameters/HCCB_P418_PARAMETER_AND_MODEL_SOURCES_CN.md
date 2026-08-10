# HCCB P418 参数、方程和模型来源总表

这份总表把球床物理输入、有限体积方程、实际OpenFOAM算例、热阶跃和人工智能模型的来源放在一起。网络层数、学习率和时间步等数值设置与球床物理参数分开。

## 结论先说

- 当前P418换热计算使用`22`项物理输入，全部在文献参数表中标为已摘录，数值、单位和来源标题逐项一致。
- 这些物理输入进入`31`处方程或边界条件；60个OpenFOAM稳态工况和12条热阶跃没有增加新的物理参数。
- 颗粒直径、装填目标孔隙率、大球床尺寸、截取区域、近壁位置和网格用粒径也已与来源表逐项对照。
- 已保存并核对`2`篇直接面向HCCB球床的PINN研究。它们证明这一方法确实已用于增殖球床，但没有公开网络层数、宽度、学习率、批量大小和随机种子，因此这些设置不能从原文照搬或自行补写。
- PINN、Transformer、图神经算子和扩散模型的层数、学习率、批量等属于计算设置，不作为球床物性或运行参数。
- 仓库其他旧研究分支中仍有公开资料未给出的几何或实验坐标，但这些内容没有进入当前P418换热计算。

## 1. 球床换热物理输入

| 编号 | 物理量 | 采用值或关系 | 单位 | 来源 |
|---|---|---|---|---|
| P048 | 颗粒直径 | 1 | mm | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P049 | 源球床目标孔隙率 | 0.397 | dimensionless | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P050 | 初始大球床尺寸 | 25dp x 25dp x 10dp | dimensionless | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P390 | 文献截取区域尺寸 | 12.5dp x 12.5dp x 10dp; inlet channel=10dp; outlet extension=10dp | dimensionless_geometry | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P404 | 网格颗粒直径修正 | 1 | percent_diameter_reduction | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P423 | 靠冷却壁的截取方式 | wall-adjacent and laterally centred crop; one retained outer face=constant-temperature wall; remaining transverse faces=symmetry boundaries | dimensionless_geometry_and_boundary | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P418 | 60组入口速度、入口温度和颗粒发热率 | Re_p_AVE<1.8;87 percent within +/-30 percent;60 cases from u_in=0.05,0.10,0.15,0.20,0.25 m/s x T_in=300,500,700,900 K x phi=4.85,6.85,8.85 MW/m3 | mixed_support_envelope | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P425 | 冷却壁温度 | 635 | K | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P426 | 工作压力 | 0.12 | MPa | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P427 | 计算域和边界类型 | bed=12.5dp x 12.5dp x 10dp;inlet_extension=10dp;outlet_extension=10dp;one_constant_temperature_wall;remaining_transverse_faces=symmetry | mixed_geometry_boundary | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |
| P070 | 氦气动力黏度 | mu=0.4646*T_K^0.66*1e-6 | Pa s | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P071 | 氦气导热系数 | lambda_f=0.1448*(T_K/273)^0.68*(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85) | W/m/K | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P388 | 氦气定压比热 | 5200 | J/kg/K | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P389 | 氦气密度 | rho_f=480.19*p_MPa/T_K | kg/m3 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P092 | Li4SiO4颗粒导热系数 | 1.42 | W/m/K | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://www.sciencedirect.com/science/article/abs/pii/S1359431124007828) |
| P403 | Li4SiO4颗粒密度 | 1526.4 | kg/m3 | [Macroscopic transport characteristics in packed bed with a multi-physics inversion strategy for solid breeding blanket of HCCB TBM](https://doi.org/10.1016/j.applthermaleng.2024.123114) |
| P406 | EU参考球比热关系 | Cp=(-5.33e-7*T_C^2+0.001925*T_C+1.238)*1000 | J/kg/K | [Discrete element method for effective thermal conductivity of packed pebbles accounting for the Smoluchowski effect](https://doi.org/10.1016/j.fusengdes.2018.01.013) |
| P428 | 纯Li4SiO4焓增量关系 | H_T_minus_H_298_J_mol=-17156+73.694*T_K+0.103210*T_K^2-4163115/T_K | J/mol | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P429 | 纯Li4SiO4比热关系 | Cp_molar=73.694+0.206420*T_K+4163115/T_K^2 | J/mol/K | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P430 | 纯Li4SiO4摩尔质量换算 | Li=6.94;O=15.999;Si=28.085;M_Li4SiO4=119.841 | g/mol | [Abridged Standard Atomic Weights 2024](https://www.ciaaw.org/abridged-atomic-weights.htm) |
| P431 | 纯Li4SiO4二级相变温度 | Tc1=938;Tc2=996 | K | [Enthalpy heat capacity second-order transitions and enthalpy of fusion of Li4SiO4 by high-temperature calorimetry](https://doi.org/10.1016/S0040-6031(96)02996-6) |
| P424 | 氦气物性表温度范围 | 300 to 1000 | K | [Pore-scale simulation on flow and heat transfer characteristics in packed beds with internal heat sources at low Reynolds numbers](https://doi.org/10.1016/j.ijheatmasstransfer.2023.124325) |

## 2. 输入量进入的方程

| 物理量或方程 | 文献参数 | 程序位置 | 在模型中的作用 |
|---|---|---|---|
| 颗粒直径 | P048 | code/hccb_p418_source_contract.py | 定义固定网格的长度尺度 |
| 源球床目标孔隙率 | P049 | code/hccb_p418_source_contract.py | 说明堆积来源 |
| 初始大球床尺寸 | P050 | code/build_hccb_source_sequence_lammps_packing.py | 定义颗粒排列的生成母域 |
| 文献截取区域尺寸 | P390 | code/check_hccb_source_sequence_lammps_packing.py | 定义正式颗粒排列来源区域 |
| 颗粒网格直径 | P404 | code/check_hccb_source_sequence_lammps_packing.py | 保持颗粒物理直径与网格处理相互区分 |
| 靠壁截取和横向边界 | P423 | code/check_hccb_source_sequence_lammps_packing.py | 定义颗粒排列相对冷却壁的位置 |
| 入口速度—入口温度—体积发热率矩阵 | P418 | code/build_hccb_dense_cht_p418_matrix.py | 稳态网络的三个工况输入以及瞬态源/目标工况输入 |
| 冷却壁温度 | P425 | code/hccb_p418_regional_cht_adapter.py | 物理边界输入 |
| 出口绝对压力 | P426 | code/hccb_p418_regional_cht_adapter.py | 物理边界输入 |
| 计算域与边界类型 | P427 | code/hccb_p418_regional_cht_adapter.py | 边界类别和区域图边类型 |
| 密度 | P389 | code/hccb_source_backed_thermophysical.py | 质量方程和流体储热项 |
| 动力黏度 | P070 | code/hccb_source_backed_thermophysical.py | 动量方程及压降物理关系 |
| 导热系数 | P071 | code/hccb_source_backed_thermophysical.py | 流体导热通量和能量方程 |
| 定压比热 | P388 | code/hccb_source_backed_thermophysical.py | 流体焓和瞬态储热项 |
| 颗粒导热系数 | P092 | code/hccb_source_backed_thermophysical.py | 颗粒导热通量和固体能量方程 |
| 颗粒密度 | P403 | code/hccb_source_backed_thermophysical.py | 固体瞬态储热项 |
| 稳态算例字典比热 | P406 | code/build_hccb_pore_resolved_openfoam_steady_case.py | 只用于满足稳态OpenFOAM材料字典 |
| 纯Li4SiO4焓关系 | P428 | code/hccb_source_backed_thermophysical.py | 定义热阶跃初态和储热关系 |
| 纯Li4SiO4比热 | P429;P430 | code/hccb_source_backed_thermophysical.py | 固体瞬态储热项 |
| 二级相变温度 | P431 | code/hccb_source_backed_thermophysical.py | 标记模型适用温区 |
| 氦气物性表温度范围 | P424 | code/build_hccb_openfoam_helium_property_table.py | 限制训练和预测温度范围 |
| 质量守恒 | P070;P389 | code/hccb_p418_regional_cht_adapter.py;code/hccb_p418_conservative_mixed_operator.py | 区域质量收支损失和质量流输出 |
| 流体能量守恒 | P071;P388;P389 | code/hccb_p418_regional_cht_adapter.py | 区域流体能量损失和热流输出 |
| 固体能量守恒 | P092;P418 | code/hccb_p418_regional_cht_adapter.py | 区域固体能量损失和颗粒热流输出 |
| 流固界面温度与热流连续 | P071;P092 | code/hccb_p418_regional_cht_adapter.py | 界面温差与热流互反损失 |
| 流体储热 | P388;P389 | code/hccb_p418_transient_regional_physics.py | OpenFOAM-13守恒焓方程的储热与压力功 |
| 固体储热 | P403;P428;P429;P430;P431 | code/hccb_p418_transient_regional_physics.py | OpenFOAM固体内能方程的储热项 |
| 连续性 | P389 | code/hccb_p418_fully_coupled_transient_physics.py | 全耦合模型的连续性关系 |
| 动量 | P070;P389 | code/hccb_p418_fully_coupled_transient_physics.py;code/hccb_steady_momentum_residual.py | 全耦合模型的三分量动量关系 |
| 总焓与动能 | P071;P388;P389 | code/hccb_p418_fully_coupled_transient_physics.py;code/hccb_p418_transient_regional_physics.py | 全耦合模型的流体能量关系 |
| 面质量流一致性 | P389 | code/hccb_p418_fully_coupled_transient_physics.py;code/openfoam13_face_flux_reconstruction.py | 面质量流预测与单元状态之间的一致性关系 |

## 3. 实际计算工况

- OpenFOAM稳态算例：`60`组，对应P418的`5 x 4 x 3`完整工况矩阵。
- 60个算例的入口速度、入口温度、颗粒发热率、壁温、压力和材料物性已逐个与来源表对照。
- 当前精细局部域为`3.923 x 4.242 x 3.49 dp`，包含`125`个颗粒片段，三角网格孔隙率为`0.386791`。这些是从文献规定的大球床和截取方式计算得到的局部几何，不作为新增物性。
- 热阶跃：`12`条，每一个起点和终点都是上述60个P418稳态工况之一。
- 实际网格孔隙率是固定颗粒堆积和表面三角化后的几何计算结果，不冒充文献常数。

## 4. 本领域已有的直接PINN工作

| 已有工作 | 解决的问题 | 原文给出的训练信息 | 原文未给出的设置 | 在本研究中的用法 |
|---|---|---|---|---|
| [HCCB球床有效渗透率和导热系数反演](https://www.cetjournal.it/index.php/cet/article/view/CET24114068) | 二维坐标PINN，由孔隙尺度压力、速度和温度场反演宏观参数 | 2000 outlet velocity points and 10000 packed-bed pressure/temperature points. A coordinate fully connected PINN maps (x,y) to (u,v,p,T), treats effective permeability and effective thermal conductivity as trainable quantities, uses tanh and Adam, and runs for 60000 epochs. | network depth、network width、learning rate、batch size、random seed | 只支持坐标PINN对照和物理约束思路；不作为当前三维模型精度、网络规模或新物性的来源。 |
| [低流速球床有效扩散特性反演](https://www.sciencedirect.com/science/article/pii/S0017931025003114) | 二维坐标PINN，由孔隙尺度浓度场反演有效扩散系数 | 10000 packed-bed concentration points and 400 inlet/outlet boundary points. A coordinate fully connected PINN maps (y,z) to concentration, treats the effective diffusion coefficient as trainable, uses tanh, Adam, normalized variables and 60000 epochs. | network depth、network width、learning rate、batch size、random seed | 只支持坐标PINN对照和物理约束思路；不作为当前三维模型精度、网络规模或新物性的来源。 |

这两篇工作都处理二维宏观反演。当前研究增加三维流固共轭换热、完整热阶跃、热点位置和跨装填预测，因此不能把已有二维结果当作当前三维模型的验证结果。

## 5. 人工智能和传统降阶模型

| 模型 | 在本研究中的作用 | 主要来源 |
|---|---|---|
| 经典坐标PINN | 稳态物理约束对照模型 | [Physics-Informed Neural Operator for Learning Partial Differential Equations](https://doi.org/10.1145/3648506) |
| RIGNO式区域图神经算子 | 非结构流体--颗粒区域稳态预测 | [RIGNO: A Graph-based Framework for Robust and Accurate Operator Learning for PDEs on Arbitrary Domains](https://papers.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html) |
| Transolver | 比较长距离区域热耦合的Transformer | [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://proceedings.mlr.press/v235/wu24r.html) |
| 时间Transformer | 预测出口温度、壁面热功率等完整时间曲线 | [Attention Is All You Need](https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |
| 区域图--Transformer时空算子 | 预测12条真实热阶跃的三维温度场 | [RIGNO: A Graph-based Framework for Robust and Accurate Operator Learning for PDEs on Arbitrary Domains](https://papers.nips.cc/paper_files/paper/2025/hash/dcb91f43033bb1d367d1848806dee98d-Abstract-Conference.html); [Learning Mesh-Based Simulation with Graph Networks](https://iclr.cc/virtual/2021/spotlight/3542); [Transolver: A Fast Transformer Solver for PDEs on General Geometries](https://proceedings.mlr.press/v235/wu24r.html); [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html); [Training Transformers for Mesh-Based Simulations](https://arxiv.org/abs/2508.18051) |
| DPOT自回归去噪Transformer | 保留为今后多堆积、大样本预训练候选 | [DPOT: Auto-Regressive Denoising Operator Transformer for Large-Scale PDE Pre-Training](https://proceedings.mlr.press/v235/hao24d.html) |
| 体积加权DMDc | 传统线性瞬态降阶对照 | [Dynamic Mode Decomposition with Control](https://epubs.siam.org/doi/10.1137/15M1013857) |
| 快照POD低秩残差修正 | 使用少量空间模态修正确定性模型的温度误差 | [Turbulence and the dynamics of coherent structures. Part I: Coherent structures](https://www.ams.org/qam/1987-45-03/S0033-569X-1987-0910462-6/) |
| PDE-Refiner式扩散残差修正 | 修正图--Transformer剩余温度误差并给出不确定范围 | [PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html) |

## 6. 数值设置与物理参数的区别

`parameters/hccb_p418_model_numerical_settings.csv`共记录`76`项模型数值设置，全部明确标为不是球床物理参数。

- `data_derived`: 6项。
- `finite_volume_definition`: 1项。
- `measured_compute_setting`: 5项。
- `official_OpenFOAM13_software_constant`: 1项。
- `official_code_architecture`: 4项。
- `official_code_constant`: 1项。
- `official_code_training`: 15项。
- `predeclared_baseline`: 3项。
- `predeclared_numerical_scan`: 2项。
- `predeclared_selection_rule`: 2项。
- `problem_geometry`: 2项。
- `project_adaptation`: 2项。
- `published_algorithm`: 13项。
- `published_architecture`: 16项。
- `published_component_adaptation`: 3项。

## 7. 可重复生成

```bash
python3 code/build_hccb_p418_source_summary.py
```

机器可读结果保存在`results/hccb_p418_source_summary.json`。
