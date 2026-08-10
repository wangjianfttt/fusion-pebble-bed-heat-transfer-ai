# P418物理参数、原文、方程和代码对应表

这份表只回答四件事：每个物理量取什么值、原文在哪里、进入哪条方程、由哪段程序读取。22项物理参数均来自已登记文献，没有为了训练PINN、图--Transformer或扩散模型另填球床物性。

## 资料覆盖情况

- 当前22/22项参数的来源文件可在项目中直接读取。
- 22项参数的来源文件均可在项目中直接读取。
- P430可在本地保存的CIAAW 2024官方网页中直接复核。
- P428的全部焓系数来自Kleykamp 1996 ScienceDirect出版社公开摘要，P429为其解析导数；没有把元数据文件写成论文全文。本地另存Kleykamp 2000 J-STAGE公开论文；该文表1给出Li4SiO4在298 K和1100 K时的比热为182.1和304.2 J/(mol K)，与P429计算值四舍五入一致。
- P431的精确938 K和996 K来自同一出版社摘要。Kleykamp 2000公开全文进一步给出648--683 °C和713--735 °C两段相变影响区及900和630 J/mol额外焓吸收；这些数据只用于判断计算温度是否进入平滑比热关系不充分的温区，不凭空假设热容峰形。FZKA5515官方报告又给出约940 K和998 K，Asou等独立研究组的出版社摘要还报告约885 K、930 K和985 K热容异常。
- P406只用于满足稳态OpenFOAM材料字典格式，不进入12条正式热阶跃。正式瞬态储热使用P428--P431。

## 逐项对应

|编号|物理量|采用值或关系|原文位置和本地资料|进入的方程或边界|主要程序|
|---|---|---|---|---|---|
|P048|颗粒直径|1 [mm]|论文第2页及提取文本第88--89行：Li4SiO4颗粒直径dp=1 mm<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|颗粒直径|`code/hccb_p418_source_contract.py`|
|P049|源球床目标孔隙率|0.397 [dimensionless]|论文第2页及提取文本第92--96行：孔隙率39.7%<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|源球床目标孔隙率|`code/hccb_p418_source_contract.py`|
|P050|初始大球床尺寸|25dp x 25dp x 10dp [dimensionless]|第4页对应文本第420--449行：初始25dp×25dp×10dp球床<br>`logs/pdf_text/hccb_tbm_macroscopic_transport_2024.clean.txt`|初始大球床尺寸|`code/build_hccb_source_sequence_lammps_packing.py`|
|P390|文献截取区域尺寸|12.5dp x 12.5dp x 10dp; inlet channel=10dp; outlet extension=10dp [dimensionless_geometry]|论文第2页及提取文本第92--96行：12.5dp×12.5dp×10dp球床及10dp入口和出口延伸段<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|文献截取区域尺寸|`code/check_hccb_source_sequence_lammps_packing.py`|
|P404|网格颗粒直径修正|1 [percent_diameter_reduction]|第5页对应文本第636--654行：网格前将每个颗粒直径缩小1%<br>`logs/pdf_text/hccb_tbm_macroscopic_transport_2024.clean.txt`|颗粒网格直径|`code/check_hccb_source_sequence_lammps_packing.py`|
|P423|靠冷却壁的截取方式|wall-adjacent and laterally centred crop; one retained outer face=constant-temperature wall; remaining transverse faces=symmetry boundaries [dimensionless_geometry_and_boundary]|论文第3页图2及提取文本第103--114行：一侧恒温冷却壁、其余横向面为对称边界<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|靠壁截取和横向边界|`code/check_hccb_source_sequence_lammps_packing.py`|
|P418|60组入口速度、入口温度和颗粒发热率|Re_p_AVE<1.8;87 percent within +/-30 percent;60 cases from u_in=0.05,0.10,0.15,0.20,0.25 m/s x T_in=300,500,700,900 K x phi=4.85,6.85,8.85 MW/m3 [mixed_support_envelope]|第5--6页对应文本第368--377和433--441行：5×4×3共60组速度、温度和发热率<br>`literature/extracted/ijhmt2023_low_re_internal_heat/full_text.txt`|入口速度—入口温度—体积发热率矩阵；固体能量守恒|`code/build_hccb_dense_cht_p418_matrix.py`<br>`code/hccb_p418_regional_cht_adapter.py`|
|P425|冷却壁温度|635 [K]|第6页对应文本第433--437行：冷却壁温度635 K<br>`literature/extracted/ijhmt2023_low_re_internal_heat/full_text.txt`|冷却壁温度|`code/hccb_p418_regional_cht_adapter.py`|
|P426|工作压力|0.12 [MPa]|第6页对应文本第433--441行：工作压力0.12 MPa<br>`literature/extracted/ijhmt2023_low_re_internal_heat/full_text.txt`|出口绝对压力|`code/hccb_p418_regional_cht_adapter.py`|
|P427|计算域和边界类型|bed=12.5dp x 12.5dp x 10dp;inlet_extension=10dp;outlet_extension=10dp;one_constant_temperature_wall;remaining_transverse_faces=symmetry [mixed_geometry_boundary]|第4--6页对应文本第340--352和433--441行：计算域、入口出口延伸段和恒温壁面<br>`literature/extracted/ijhmt2023_low_re_internal_heat/full_text.txt`|计算域与边界类型|`code/hccb_p418_regional_cht_adapter.py`|
|P070|氦气动力黏度|mu=0.4646*T_K^0.66*1e-6 [Pa s]|论文第3页表3及提取文本第139--145行：氦气动力黏度关系<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|动力黏度；质量守恒；动量|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_regional_cht_adapter.py`<br>`code/hccb_p418_conservative_mixed_operator.py`<br>`code/hccb_p418_fully_coupled_transient_physics.py`<br>`code/hccb_steady_momentum_residual.py`|
|P071|氦气导热系数|lambda_f=0.1448*(T_K/273)^0.68*(1+2.5e-3*p_MPa^1.17*(T_K/273)^-1.85) [W/m/K]|论文第3页表3及提取文本第139--145行：氦气导热系数关系<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|导热系数；流体能量守恒；流固界面温度与热流连续；总焓与动能|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_regional_cht_adapter.py`<br>`code/hccb_p418_fully_coupled_transient_physics.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P388|氦气定压比热|5200 [J/kg/K]|论文第3页表3及提取文本第139--145行：氦气定压比热5200 J/(kg K)<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|定压比热；流体能量守恒；流体储热；总焓与动能|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_regional_cht_adapter.py`<br>`code/hccb_p418_transient_regional_physics.py`<br>`code/hccb_p418_fully_coupled_transient_physics.py`|
|P389|氦气密度|rho_f=480.19*p_MPa/T_K [kg/m3]|论文第3页表3及提取文本第139--145行：氦气密度关系<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|密度；质量守恒；流体能量守恒；流体储热；连续性；动量；总焓与动能；面质量流一致性|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_regional_cht_adapter.py`<br>`code/hccb_p418_conservative_mixed_operator.py`<br>`code/hccb_p418_transient_regional_physics.py`<br>`code/hccb_p418_fully_coupled_transient_physics.py`<br>`code/hccb_steady_momentum_residual.py`<br>`code/openfoam13_face_flux_reconstruction.py`|
|P092|Li4SiO4颗粒导热系数|1.42 [W/m/K]|论文第3页表3及提取文本第146--148行：Li4SiO4导热系数1.42 W/(m K)<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|颗粒导热系数；固体能量守恒；流固界面温度与热流连续|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_regional_cht_adapter.py`|
|P403|Li4SiO4颗粒密度|1526.4 [kg/m3]|论文第3页表3及提取文本第146--148行：Li4SiO4密度1526.4 kg/m3<br>`literature/raw/thermophysical_sources/Wang_2024_CET_PINN_macroscopic_transport.pdf`<br>`logs/pdf_text/Wang_2024_CET_PINN_macroscopic_transport.txt`|颗粒密度；固体储热|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P406|EU参考球比热关系|Cp=(-5.33e-7*T_C^2+0.001925*T_C+1.238)*1000 [J/kg/K]|论文第5页表3及提取文本第416--425行：EU参考增殖剂比热关系<br>`literature/raw/ceramic_breeder_heat_transfer/Moscardini_2018_TDEM_Smoluchowski.pdf`<br>`literature/raw/ceramic_breeder_heat_transfer/Moscardini_2018_TDEM_Smoluchowski.txt`|稳态算例字典比热|`code/build_hccb_pore_resolved_openfoam_steady_case.py`|
|P428|纯Li4SiO4焓增量关系|H_T_minus_H_298_J_mol=-17156+73.694*T_K+0.103210*T_K^2-4163115/T_K [J/mol]|ScienceDirect公开摘要给出298--1300 K平滑焓关系及全部系数；J-STAGE论文第2页说明焓拟合形式和量热温区并给出独立比热端点<br>`literature/raw/thermophysical_sources/Kleykamp_1996_Crossref_metadata.json`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_High_temperature_calorimetry_review.pdf`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_Li4SiO4_enthalpy_Cp_evidence.txt`|纯Li4SiO4焓关系；固体储热|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P429|纯Li4SiO4比热关系|Cp_molar=73.694+0.206420*T_K+4163115/T_K^2 [J/mol/K]|ScienceDirect公开摘要给出焓关系并明确比热由求导得到；J-STAGE论文表1给出298 K时182.1和1100 K时304.2 J/(mol K)<br>`literature/raw/thermophysical_sources/Kleykamp_1996_Crossref_metadata.json`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_High_temperature_calorimetry_review.pdf`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_Li4SiO4_enthalpy_Cp_evidence.txt`|纯Li4SiO4比热；固体储热|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P430|纯Li4SiO4摩尔质量换算|Li=6.94;O=15.999;Si=28.085;M_Li4SiO4=119.841 [g/mol]|官方网页第95--120行附近：Li=6.94±0.06、O=15.999±0.001、Si=28.085±0.001<br>`literature/raw/thermophysical_sources/CIAAW_abridged_atomic_weights_2024.html`|纯Li4SiO4比热；固体储热|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P431|纯Li4SiO4二级相变温度|Tc1=938;Tc2=996 [K]|ScienceDirect公开摘要给出938 K和996 K；J-STAGE全文第3页/印刷102页给出648--683 °C和713--735 °C两段温区及900/630 J mol^-1额外焓吸收；FZKA5515第115页/印刷107页给出约940.15 K和998.15 K；Asou等独立量热摘要报告约885 K、930 K和985 K热容异常<br>`literature/raw/thermophysical_sources/Kleykamp_1996_Crossref_metadata.json`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_High_temperature_calorimetry_review.pdf`<br>`literature/raw/thermophysical_sources/Kleykamp_2000_Li4SiO4_enthalpy_Cp_evidence.txt`<br>`literature/raw/thermophysical_sources/FZKA5515_fusion_annual_report_1994_1995.pdf`<br>`literature/raw/thermophysical_sources/FZKA5515_Li4SiO4_calorimetry_page115.txt`<br>`literature/raw/thermophysical_sources/Asou_1992_Li4SiO4_ScienceDirect_abstract_evidence.txt`|二级相变温度；固体储热|`code/hccb_source_backed_thermophysical.py`<br>`code/hccb_p418_transient_regional_physics.py`|
|P424|氦气物性表温度范围|300 to 1000 [K]|第2页对应文本第183--209行：氦冷固态增殖包层温度范围300--1000 K及温变物性必要性<br>`literature/extracted/ijhmt2023_low_re_internal_heat/full_text.txt`|氦气物性表温度范围|`code/build_hccb_openfoam_helium_property_table.py`|

## 使用时需要注意的物理边界

1. P048--P050和P390描述文献球床及其截取区域；当前精细局部网格来自该区域内部，不能写成完整12.5dp×12.5dp×10dp几何的原样复现。
2. P404的1%缩径只用于网格消除点接触，因此当前模型解析氦气对流、气固导热和颗粒内发热，不包含真实受压颗粒接触导热。
3. P428--P429是298--1300 K的平滑纯Li4SiO4量热关系。Kleykamp 2000公开论文说明热容由平滑焓曲线求导，并指出该方法适用于没有二级相变的温区。该文还给出648--683 °C和713--735 °C两段影响区及900和630 J/mol额外焓吸收，但没有给出唯一解析峰形，因此当前程序只统计温度场进入这些原文温区的程度，不修改OpenFOAM比热。P431的精确938 K和996 K取自Kleykamp 1996出版社摘要；FZKA5515给出约940 K和998 K，Asou等独立研究又在约885 K、930 K和985 K观察到热容异常。
4. P430按天然同位素组成的简化标准原子量换算。若研究对象改为明确富集6Li的颗粒，应使用材料实际同位素组成重新计算摩尔质量。
5. 神经网络层数、隐藏维数、学习率和扩散步数属于数值设置，不在本表中冒充材料参数。

## 自动检查

```bash
python3 code/verify_hccb_p418_parameter_evidence_files.py \
  --output results/hccb_p418_parameter_evidence/summary.json
python3 code/build_hccb_p418_parameter_evidence_summary.py
```
