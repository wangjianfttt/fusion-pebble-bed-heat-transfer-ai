# 球床流动与换热实验数据填写说明

本目录只提供空白表头，没有填写或生成任何实验值。取得真实球床实验数据后，按以下五张表填写。

1. `experiment_conditions.csv`：材料、颗粒装填、实验类型和实验/文献来源。只有存在完全对应的数值工况时才填写`model_condition_id`；没有同条件计算时留空。
2. `sensor_layout.csv`：传感器测量量、三维坐标、测量方法、标定编号、`model_observable`和`sensor_response_model`。前者说明比较哪个模型量，后者说明这个模型量怎样代表真实仪器读数。整体流量、总功率等积分测量的坐标留空；空间热电偶必须填写三个坐标。
3. `steady_measurements.csv`：稳态平均值、平均时间段和同单位标准不确定度。
4. `transient_measurements.csv`：热阶跃长表。每一行是一个时刻、一个传感器、一个测量量，可保留任意数量的出口和内部热电偶。
5. `calibration_records.csv`：流量计、压力传感器、热电偶和功率测量的标定信息。

可用量和单位定义在`parameters/hccb_p418_experimental_data_schema.json`。数据填写后运行：

```bash
python3 code/validate_hccb_p418_experimental_data.py \
  --data-root experimental_data_templates \
  --output results/hccb_p418_experimental_data_validation.json
```

程序只检查表格结构、单位、编号关系和不确定度是否完整，不会把其他论文的仪器误差自动添加为训练噪声。真实数据进入模型前，还需要按传感器坐标从OpenFOAM和PINN场中提取对应位置的预测值。

热电偶读数不能自动等同于最近颗粒温度或最近气体温度，必须明确填写传感器响应表示：

- 内部气体或颗粒床热电偶使用`fluid_temperature`或`solid_temperature`时，可填写`nearest_regional_phase_temperature`。这只表示最近同相区域近似，不包含探头直径、接触热阻、探头造成的局部装填扰动和时间响应。
- 压力接口使用`fluid_absolute_pressure`时，可填写`nearest_regional_fluid_pressure`。
- 压降、出口温度、流量或总热量可填写`direct_integral_or_boundary_quantity`。
- 如果数值模型确实建立了探头实体、接触和动态响应，可填写`explicit_sensor_body`。当前比较程序会保留这一声明，但不会以最近区域值冒充尚未实现的仪器模型。

实验标准不确定度只用于衡量模型与测量的差异，不会覆盖计算温度场，也不会转换成人工训练噪声。

填写并检查完成后，可直接运行：

```bash
bash code/run_hccb_p418_experimental_comparison.sh
```

只有传感器表明确声明最近区域近似时，程序才对空间测点寻找同相区域网格中的最近点，并在结果中保留传感器到该网格点的距离；压降、出口温度、流量和热量使用对应的边界积分量。输出为`model_experiment_comparison.csv`。残差定义为模型值减实验值，只有本实验记录了非零标准不确定度时才计算二者之比。瞬态接口也已实现：只在实验时刻位于模型输出时间范围内时，对模型曲线做线性取值；不向时间范围外推。当前12条正式热阶跃尚未完成，所以仍不会生成瞬态对照数字。
