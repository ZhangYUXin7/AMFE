# 仓库结构说明（中文）

本文档面向后续需要继续做实验、调试模型、接入真实数据集或扩展训练流程的开发者，说明当前仓库里重要目录和关键文件分别负责什么，以及它们如何组成完整的四尺度检测主链。

## 1. 仓库主线
当前仓库的模型主线是：

```text
输入图像
-> AMFEBackbone
-> AMFNeck
-> Ultralytics Detect Head
-> 预测 / loss / backward
```

其中：
- `AMFEBackbone` 输出 `F2 / F3 / F4 / F5`
- `AMFNeck` 输出 `N2 / N3 / N4 / N5`
- Detect strides 固定为 `[4, 8, 16, 32]`

当前仓库已经打通：
- 模型构建
- synthetic forward/loss/backward/optimizer step
- 本地 YOLO 数据集校验
- Ultralytics trainer bridge

## 2. 目录说明

### 2.1 `amfe/`
这是项目的主 Python 包。

主要职责：
- 暴露模型构建 API
- 提供训练桥接逻辑
- 提供 Ultralytics 兼容层
- 提供数据工具

关键文件：
- `amfe/__init__.py`：包级导出入口
- `amfe/training.py`：训练桥接
- `amfe/ultralytics_compat.py`：Detect 和 loss 的兼容导入层

### 2.2 `amfe/models/`
这是模型实现核心目录。

主要职责：
- 定义 backbone、neck、detector
- 提供 YAML 到模型对象的构建逻辑
- 封装当前四尺度检测器的主装配关系

关键文件：
- `amfe/models/backbone/`：backbone 子模块
- `amfe/models/neck.py`：neck 主体
- `amfe/models/detector.py`：整网装配
- `amfe/models/registry.py`：配置加载与模型构建入口
- `amfe/models/common.py`：共享基础模块

### 2.3 `amfe/models/backbone/`
这是 AMFEBackbone 的核心实现目录。

主要职责：
- 实现 `LEM`
- 实现 `DPSStem`
- 实现 `MSB`
- 实现 `ADB / DEB`
- 实现 `LGCB`
- 实现 `MBFM / SRAFMBFM`
- 在总装文件中输出 `F2 / F3 / F4 / F5`

关键文件：
- `lem.py`
- `dps_stem.py`
- `msb.py`
- `adb.py`
- `lgcb.py`
- `mbfm.py`
- `amfe_backbone.py`

如果后续改 backbone，优先阅读和修改这个目录。

### 2.4 `amfe/data/`
这是数据相关工具目录。

主要职责：
- 校验本地 YOLO 数据集路径和类别信息
- 将原始 VisDrone 数据转换成 YOLO 检测格式
- 校验转换后的数据集完整性

关键文件：
- `local_dataset.py`
- `visdrone_conversion.py`

### 2.5 `configs/`
这是配置目录。

子目录职责：
- `configs/model/`：模型结构和损失超参数配置
- `configs/train/`：训练启动配置
- `configs/data/`：数据集路径与类别配置

重要说明：
- 当前项目依赖显式 YAML 配置
- 不应把隐藏路径或实验条件硬编码进模型或训练代码

### 2.6 `tools/`
这是命令行工具入口目录。

关键文件：
- `tools/train.py`
  - synthetic smoke
  - 仅检查数据集
  - 启动真实训练
- `tools/validate_dataset.py`
  - 数据集配置校验入口

### 2.7 `tests/`
这是核心测试目录。

关键文件：
- `tests/test_backbone.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

当前测试主要覆盖：
- backbone 子模块
- neck 子模块
- 四尺度 shape 合同
- detector forward/loss/backward 主链
- YAML 构建与 stride 初始化

### 2.8 `docs/`
这是文档目录。

作用：
- 记录仓库结构说明
- 记录模型结构和工作约定
- 用于降低后续交接和实验迭代成本

## 3. 当前模型装配关系

### 3.1 Backbone 层级
`amfe/models/backbone/amfe_backbone.py` 是 backbone 总装文件。

其内部顺序为：
- `LEM`
- `DPSStem`
- `F2` 独立浅层分支
- `MSB`
- `ADB`
- `LGCB`
- `SRAFMBFM`

输出：
- `F2, F3, F4, F5`

### 3.2 Neck 层级
`amfe/models/neck.py` 是 neck 总装文件。

其内部顺序为：
- `CAF` 统一通道
- `TDSF` 自顶向下融合
- `BURF` 自底向上 refine

输入：
- `F2, F3, F4, F5`

输出：
- `N2, N3, N4, N5`

### 3.3 Detector 层级
`amfe/models/detector.py` 负责将 backbone、neck 和 Ultralytics Detect 真正装配在一起。

主要职责：
- 构建 `AMFEBackbone`
- 构建 `AMFNeck`
- 构建 Ultralytics `Detect`
- 推断并校验 detect strides
- 包装 Ultralytics `v8DetectionLoss`
- 暴露 `forward_features()`、`forward()`、`loss()`

## 4. 关键文件该怎么读
如果你第一次接手这个仓库，推荐阅读顺序如下：

1. `README.md`
2. `AGENTS.md`
3. `configs/model/amfe_amf_yolo.yaml`
4. `tools/train.py`
5. `amfe/models/detector.py`
6. `amfe/models/backbone/amfe_backbone.py`
7. `amfe/models/neck.py`
8. `amfe/models/backbone/` 其他子模块
9. `tests/test_backbone.py`
10. `tests/test_neck.py`
11. `tests/test_detector.py`

这样可以先建立系统视角，再下钻到各个模块细节和测试保护网。

## 5. 后续修改时优先看哪些文件

### 5.1 如果要改 backbone
优先看：
- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/backbone/*.py`
- `tests/test_backbone.py`

### 5.2 如果要改 neck
优先看：
- `amfe/models/neck.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

### 5.3 如果要改 detector 装配或 Detect 输入
优先看：
- `amfe/models/detector.py`
- `configs/model/*.yaml`
- `tests/test_detector.py`

### 5.4 如果要改训练入口
优先看：
- `tools/train.py`
- `amfe/training.py`
- `configs/train/*.yaml`

### 5.5 如果要接入或校验本地数据集
优先看：
- `configs/data/*.yaml`
- `amfe/data/local_dataset.py`
- `tools/validate_dataset.py`

## 6. 你最需要记住的几件事
- 当前仓库已经是四尺度检测，不是旧的三尺度版本
- `F2` 是独立浅层检测分支，不并入 `MBFM`
- `MSB` 当前是 YOLO 风格实现，不是 ResNet50
- Detect Head 和 loss 复用 Ultralytics，不要默认重写
- 任何结构改动都要同步更新配置、测试和文档

## 7. 推荐的最低回归命令
```bash
python -m pytest tests/test_backbone.py tests/test_neck.py tests/test_detector.py
```

这是当前仓库最重要的一组快速回归测试。
