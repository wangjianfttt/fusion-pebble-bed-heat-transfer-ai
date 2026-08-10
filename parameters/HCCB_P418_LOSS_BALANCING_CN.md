# P418全耦合图-Transformer的损失权重方法

## 1. 为什么要比较损失权重

全耦合模型同时预测流体速度、压力、流体温度、颗粒温度以及面质量流。训练误差分成三组：

1. 状态数据：速度、压力和温度；
2. 面质量流数据：内部面和边界面质量流；
3. 物理方程：连续性、三分量动量、流体能量、颗粒能量、流固界面热流、流固界面温差、内部面质量流一致性和边界面质量流一致性，共八类误差；动量项内部同时包含三个方向。

各物理量先使用训练曲线计算的尺度变成无量纲量，同一组内部再取平均。即便如此，三组误差随训练下降的速度仍可能不同。若一直采用固定权重，某一组误差可能长期控制梯度；若自适应权重设置不合适，也可能把网络推向数值上容易下降、但工程量预测较差的方向。因此这里不预先宣称某种方法更好，而是在完全相同的数据、网络和随机初值下进行比较。

损失权重只是神经网络训练设置。它不改变颗粒直径、孔隙率、氦气物性、Li4SiO4物性、入口温度、入口速度、颗粒发热率或任何边界条件。本方法没有增加新的球床物理参数。

## 2. 固定权重基准

三组无量纲误差分别记为：

`L_state`、`L_flux`和`L_physics`。

固定权重基准为：

`L_train = L_state + L_flux + L_physics`

三组权重均为1。这是无量纲处理后的中性基准，不代表三类物理量具有相同实验重要性。

## 3. ReLoBRaLo自适应权重

ReLoBRaLo来自Bischof和Kraus的多目标物理信息深度学习方法。对第`i`组误差，以以前某一步`t'`为参照的相对权重为：

`lambda_hat_i(t,t') = 3 exp[L_i(t)/(T L_i(t'))] / sum_j exp[L_j(t)/(T L_j(t'))]`

其中`T`控制权重对相对误差变化的敏感程度，三组权重之和保持为3。正式实现采用论文公式(11)和作者公开程序中的更新形式：

`lambda_i(t) = rho alpha lambda_i(t-1) + (1-rho) alpha lambda_hat_i(t,0) + (1-alpha) lambda_hat_i(t,t-1)`

`rho`由期望值为`E[rho]`的伯努利分布产生，用于在初始误差与上一训练步之间随机回看。随机数状态和当前权重均写入训练断点，因此停电续算后会得到与不中断训练相同的后续权重序列。

作者程序在前两次更新中依次使用`alpha=1`和`alpha=0`，随后才使用论文设置的`alpha`。本项目保持这一顺序，没有自行改写公式。

## 4. 正式比较的四种候选

| 编号 | 方法 | T | alpha | E[rho] | 来源 |
|---|---|---:|---:|---:|---|
| fixed_equal_dimensionless | 固定等权 | 不适用 | 不适用 | 不适用 | 无量纲三组等权基准 |
| relobralo_burgers_table_viii | ReLoBRaLo | 0.1 | 0.999 | 0.9999 | Bischof和Kraus，表VIII，Burgers设置 |
| relobralo_kirchhoff_table_viii | ReLoBRaLo | 0.01 | 0.999 | 0.9999 | Bischof和Kraus，表VIII，Kirchhoff设置 |
| relobralo_helmholtz_table_viii | ReLoBRaLo | 1e-5 | 0.99 | 0.99 | Bischof和Kraus，表VIII，Helmholtz设置 |

这些数值来自原论文，不是根据P418测试曲线反向试出来的。它们只是候选设置。哪一种适合当前流动-换热问题，必须由P418检查曲线决定。

主要来源：

- Bischof R, Kraus M A. Multi-Objective Loss Balancing for Physics-Informed Deep Learning. arXiv:2110.09813；Computer Methods in Applied Mechanics and Engineering 439 (2025) 117914.
- 作者程序：https://github.com/rbischof/relative_balancing ，记录版本`b3c76d2bed7c6bebb2e2628575008a04858472cf`。
- Wang S, Teng Y, Perdikaris P. Understanding and mitigating gradient pathologies in physics-informed neural networks. SIAM Journal on Scientific Computing 43 (2021) A3055-A3081.
- Chen Z, Badrinarayanan V, Lee C Y, Rabinovich A. GradNorm. Proceedings of Machine Learning Research 80 (2018) 794-803.

Wang等人的梯度统计方法和GradNorm用于说明多目标训练中梯度不平衡是一个实际问题。本轮只实现ReLoBRaLo与固定权重比较，避免在仅有12条物理曲线时同时扩展过多训练方法。

## 5. 两阶段运行顺序

### 第一阶段：确定权重方法

四种候选使用完全相同的：

- 训练、检查和独立测试曲线编号；
- 网络结构；
- 初始随机种子；
- 训练轮数和学习率；
- 训练曲线归一化；
- 八类物理关系及其尺度。

训练期间只读取训练曲线，每轮只用检查曲线选择网络状态。四种方法统一使用：

`S_validation = (L_state + L_flux + L_physics) / 3`

这个分数不使用各方法自己的动态权重，因此不会因为某种方法主动压低某组权重而获得表面优势。四种方法在这一阶段都不读取独立测试曲线。

程序把每种方法的结果保存为`selection_summary.json`。选择程序检查四份结果没有测试指标，并且数据、划分、网络、方程、随机种子和归一化完全相同，然后选取检查分数最低的方法。

### 第二阶段：独立测试

只有被选中的一种方法才能继续执行`final`阶段。程序先核对第一阶段结果、文件校验值和候选编号没有变化，随后才读取独立测试曲线。最终结果另存为`final_summary.json`，不会覆盖第一阶段依据。

这样可避免看过四种方法的测试结果后再选择其中最好的一种。

## 6. 正式运行入口

先只打印运行命令，不启动训练：

```bash
python3 code/run_hccb_p418_loss_balancing_protocol.py plan \
  --dataset-index <完整全耦合曲线目录>/dataset_index.json \
  --splits parameters/hccb_p418_step_response_splits.json \
  --split-name <数据划分名称> \
  --residual-geometry <有限体积区域几何文件> \
  --output-root <结果目录>
```

第一阶段：

```bash
python3 code/run_hccb_p418_loss_balancing_protocol.py selection \
  --dataset-index <完整全耦合曲线目录>/dataset_index.json \
  --splits parameters/hccb_p418_step_response_splits.json \
  --split-name <数据划分名称> \
  --residual-geometry <有限体积区域几何文件> \
  --output-root <结果目录>
```

确认`selected_loss_balancing_method.json`后，执行一次独立测试：

```bash
python3 code/run_hccb_p418_loss_balancing_protocol.py final \
  --dataset-index <完整全耦合曲线目录>/dataset_index.json \
  --splits parameters/hccb_p418_step_response_splits.json \
  --split-name <数据划分名称> \
  --residual-geometry <有限体积区域几何文件> \
  --output-root <结果目录>
```

正式训练仍须等待12条完整流动-换热阶跃曲线。当前轻量测试只证明公式、断点续算、方法选择和测试曲线隔离能够按预定流程工作，不代表已经得到模型精度结论。
