# P418球床流动换热论文复现说明

这份说明只针对当前准备投国际传热传质类期刊的 P418 论文，不包含项目里早期的其他研究分支。

## 论文结果分成四步

1. 用 OpenFOAM 13 计算三维颗粒床孔隙流动、颗粒导热和流固共轭换热。
2. 从三维结果中提取速度、压力、流体和颗粒温度、压降、壁面换热、热点温度，以及质量和能量收支。
3. 在完全相同的数据划分下比较响应面、DMDc、PINN、图网络、Transformer 和温度残差修正。
4. 从正式结果重新生成表格、图片、英文正文和中文便读版；精简补充材料仅在编辑明确要求时生成。

这四步相互分开。复现入口默认只做检查，不会自动启动 OpenFOAM，也不会自动训练模型。

## 已记录的计算环境

- 操作系统：Ubuntu 22.04，Linux 6.8.0-124-generic，x86_64
- OpenFOAM：OpenFOAM Foundation 13
- 多区域求解入口：OpenFOAM 13 自带的 `foamMultiRun`
- MPI：Open MPI 4.1.2
- 编译器：GCC 11.4.0
- Python：3.10.12
- 每个正式 OpenFOAM 工况：32 个 MPI 进程
- Python 包版本：根目录 `requirements-p418.txt`

氦气物性直接计算模块位于 `solver_extensions/hccbHeliumTransport/`。它直接
计算已经登记的 P070 黏度关联式和 P071 导热系数关联式，避免有限压力表插值，
没有改变关联式及其系数。该目录同时包含逐点对照检查程序。

机器可读版本保存在 `reproducibility/p418_environment.json`。这些内容来自已经完成工况的工作站环境记录，不是根据经验补写的。

Python环境使用Python 3.10。先从PyTorch官方CUDA 13.0软件源安装已经登记的
PyTorch版本，再安装其余固定版本：

```bash
python3.10 -m venv .venv-p418
source .venv-p418/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-p418.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一条命令应输出PyTorch `2.12.1+cu130`、CUDA `13.0`和`True`，之后才启动
CUDA训练。只做CPU后处理时不要求CUDA设备。

## 复现命令

在项目根目录运行：

```bash
# 只检查参数、程序、数据完成数和论文路线，不求解、不训练
bash scripts/reproduce_p418_paper.sh preflight

# 只重新生成文件及SHA-256清单
bash scripts/reproduce_p418_paper.sh manifest

# 逐文件核对后生成可公开的小型代码复现包
bash scripts/reproduce_p418_paper.sh archive

# 正式OpenFOAM结果齐全后，重新整理训练数据和工程量
bash scripts/reproduce_p418_paper.sh postprocess

# 所有正式结果齐全后，重建论文数值、图表和正文PDF
bash scripts/reproduce_p418_paper.sh paper

# 仅在编辑明确要求补充材料时使用
BUILD_SUPPLEMENT=1 bash scripts/reproduce_p418_paper.sh paper
```

正式求解单独保留为显式命令：

```bash
make p418-formal-plan
make p418-formal-run
```

第一条只显示计算顺序，第二条才会启动或续算，因此不能在普通复现检查中误触发。

## 文件放在哪里

- 文献参数、方程输入和模型设置：`parameters/`
- 正式 OpenFOAM 工况：根目录下 P418 稳态、瞬态和独立装填工况目录
- 整理后的数据和数值结果：`results/`
- 画图程序：`code/plot_hccb_p418_*.py`
- 氦气物性直接计算模块及其检查程序：`solver_extensions/hccbHeliumTransport/`
- 论文图片：`figures/`
- 正文和备用补充材料源码：`manuscript/`
- 逐文件 SHA-256 和复现文件说明：`results/hccb_p418_reproducibility_manifest/`

三维 OpenFOAM 原始场很大，不复制进小型代码包。最终公开数据时，需要公开原始场，或者明确给出原始场的长期存储位置和校验值。

小型包中会直接包含 `results/hccb_p418_public_figure_data/`，里面是已去除
本机路径的轻量作图数据。它可以独立重画60工况物理响应、9工况独立装填
对比和5种模型×5种数据划分对比三张定量图。几何模型渲染、三维云图和
最终瞬态预测图需要较大的几何或预测数组，将放入带DOI的处理数据包，不硬塞进
这个小型代码包。

当前公开范围记录在
`results/hccb_p418_public_data_release_preflight/`。其中会明确区分已经可以公开的
轻量数据和仍在等待最终模型结果的瞬态预测数组。P418新论文的DOI和许可协议目前
明确写为待确定，不会沿用之前氚释放论文的DOI。

`archive`模式会生成
`results/hccb_p418_reproducibility_manifest/p418_reproduction_source.tar.gz`
及其机器可读记录。相同文件重复打包会得到完全相同的压缩包；包内不含符号
链接、模型检查点或OpenFOAM原始时间目录，可直接作为后续GitHub Release或
Zenodo代码附件的基础。

## 结果使用原则

缺少正式计算时，不用预期结果或示意数值补位。`code/check_hccb_p418_final_scientific_requirements.py` 会列出尚未完成的计算，并阻止不完整结果被记录为最终论文。
