# P418数据整理入口说明

## 作用

`code/prepare_hccb_p418_model_data.py`只负责把OpenFOAM结果整理成模型输入，不运行PINN、Transformer、DMDc、POD或扩散模型。

它把三类数据分开处理：

1. 60个稳态共轭换热工况；
2. 12条固定流场热阶跃；
3. 12条全耦合流动--换热阶跃。

固定流场和全耦合曲线采用相同的12对P418文献工况。前者在阶跃期间固定目标速度、压力和面质量流，后者让速度、压力、面质量流以及流体和颗粒温度同时变化。两套结果不能混在一个数据文件中。

## 默认只检查

```bash
python3 code/prepare_hccb_p418_model_data.py
```

这条命令只读取完成标记和已有文件，并写出：

```text
results/hccb_p418_model_data_preparation/summary.json
```

它不会启动OpenFOAM或模型训练。

## 数据完成后的运行方式

每个阶段必须单独指定：

```bash
python3 code/prepare_hccb_p418_model_data.py --execute-stage steady
python3 code/prepare_hccb_p418_model_data.py --execute-stage fixed_hydrodynamics_thermal_steps
python3 code/prepare_hccb_p418_model_data.py --execute-stage fully_coupled_flow_heat_steps
```

程序在执行前会再次检查：

- 稳态是否达到`60/60`；
- 对应阶跃是否达到`12/12`；
- 共享网格、稳态数据索引、区域网格和面几何是否存在；
- 固定流场和全耦合曲线是否仍使用相同的12对文献端点；
- 运行命令中是否意外出现模型训练程序。

## 输出内容

稳态阶段生成共享网格、三维状态、区域状态、质量流、能量流和训练统计。固定流场和全耦合阶段均生成两类文件：

- 压降、入口/出口温度、质量流、焓流、壁面热量、颗粒最高温度等整体时间曲线；
- `Ux、Uy、Uz、p、T`区域状态和有方向的内部面、边界面质量流。

每个区域节点还保存一组不随时间变化的结构信息：

- 三维区域中心坐标；
- 区域体积；
- 流体或颗粒区域标记；
- 区域内与入口、出口、冷却壁、对称面和流固界面相邻的细网格体积占比。

这五类边界相邻体积占比直接由P425/P427边界定义和当前OpenFOAM网格计算。程序先标出与各类边界相邻的细网格，再按细网格体积聚合到区域节点；它不是边界面积比例。它们告诉图网络边界条件和流固换热关系作用在哪里，不是新增的材料参数，也不是神经网络拟合出来的数。

全耦合曲线使用`fully_coupled_flow_heat_response`作为整体时间曲线类型，使用`fully_coupled_flow_heat`作为区域三维序列类型。这样可以直接比较固定流场近似与完整流热耦合的差别。

本数据整理入口没有增加任何球床物性或边界参数。

## 训练与测试的物理范围

`code/analyze_hccb_p418_step_split_coverage.py`同时读取12条阶跃划分和`results/hccb_p418_inlet_dimensionless_envelope/inlet_dimensionless_conditions.csv`。后者由P048、P068、P070、P071、P073、P388、P418和P426中的颗粒直径、氦气物性、工作压力及文献工况计算，不含拟合物性。

程序对每条曲线分别记录源端和目标端的入口颗粒Re、Pr和Pe，并报告：

- 训练、检查和测试端点各自的无量纲范围；
- 测试曲线到最近训练曲线的无量纲距离；
- 测试曲线是否超出训练集的Re、Pr或Pe范围；
- 同一源--目标端点对是否跨越训练、检查和测试数据组。

当前12条曲线覆盖入口颗粒Re `0.078--1.44`、Pr `0.660--0.675`和Pe `0.051--0.973`。这些是入口状态量，不代替三维孔道内的局部Re，也不等同于原文未给出具体空间平均公式的`Re_p,AVE`。

## 高流速工况组合的独立测试

原12条曲线的严格端点对分离测试主要位于低入口Re范围，因此另设6条高流速工况组合曲线。它们仍然只使用P418的`5×4×3`工况端点，覆盖速度、温度和颗粒发热率三类双向阶跃，入口颗粒Re最高约为`2.40`，Pe最高约为`1.62`。

这6条曲线不增加训练数据，不参与归一化、网络选择、训练轮数选择、损失权重选择、POD基底或扩散模型训练。主模型完全确定后，才在这些曲线上进行一次独立预测。由于部分稳态端点也出现在原12条曲线中，文中将其称为“高流速工况组合测试”，不写成“所有端点均未见过的外推”。

计划文件为：

- `parameters/hccb_p418_high_re_independent_step_plan.json`；
- `parameters/hccb_p418_high_re_independent_fully_coupled_step_plan.json`。

默认运行入口`code/run_hccb_p418_high_re_independent_steps.sh`只打印计划，不启动OpenFOAM或训练。工作站迁移暂停标记存在时，即使给出执行开关也会停止。
