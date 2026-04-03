# AMFE

面向研究实验的保守实现版 AMFE 检测器，当前代码已经完成四尺度检测主链：

- Backbone：`LEM -> DPSStem -> F2 分支 + MSB + ADB + LGCB + SRAFMBFM`
- Neck：`CAF + TDSF + BURF`
- Head：Ultralytics `Detect`
- 检测尺度：`N2 / N3 / N4 / N5`
- Detect strides：`[4, 8, 16, 32]`

当前仓库强调：
- 代码可运行
- shape 合同明确
- 可接入 Ultralytics 训练 / loss / validator 主链
- 本地数据集路径显式、可验证
- 测试覆盖 synthetic smoke path

## 当前状态
仓库当前支持两类验证路径：

1. Synthetic smoke path
- 用随机张量完成模型构建、forward、loss、backward 和一步 optimizer step
- 适合快速验证模型结构和训练主链是否连通

2. 本地真实数据路径
- 通过项目内工具校验本地 YOLO 数据集
- 通过项目内 trainer bridge 启动 Ultralytics 训练流程
- 当前仓库内置了本地 VisDrone YOLO 数据的 smoke config

注意：当前仓库实现的是四尺度版本，不再是旧文档中的三尺度 `F3/F4/F5 -> N3/N4/N5`。

## 安装
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 为什么使用 `opencv-python-headless`
部分无界面环境会因为缺少 GUI 依赖而无法导入标准 OpenCV。仓库当前优先使用 `opencv-python-headless`。此外，`amfe/ultralytics_compat.py` 还提供了一个极小的 `cv2` fallback stub，用于依赖受限环境下的 smoke test。

## 模型构建方式
```python
from amfe.models import build_amfe_detector, build_model_from_yaml

model = build_amfe_detector(num_classes=3)
# 或
model = build_model_from_yaml("configs/model/amfe_amf_yolo.yaml")
```

## 当前模型结构

### Backbone
当前 backbone 由以下模块组成：

- `LEM`
  - 输出：`[B, 32, H, W]`
- `DPSStem`
  - 输出共享浅层特征 `S2`：`[B, 64, H/4, W/4]`
- 独立 `F2` 分支
  - `Conv1x1(64 -> 256) + DWConv3x3(256 -> 256)`
  - 输出 `F2`：`[B, 256, H/4, W/4]`
- `MSB`
  - 输出：
    - `C3`：`[B, 256, H/8, W/8]`
    - `C4`：`[B, 512, H/16, W/16]`
    - `C5`：`[B, 512, H/32, W/32]`
- `ADB`
  - 输出：
    - `D3`：`[B, 128, H/8, W/8]`
    - `D4`：`[B, 256, H/16, W/16]`
- `LGCB`
  - 输出：
    - `G3`：`[B, 256, H/8, W/8]`
    - `G4`：`[B, 256, H/16, W/16]`
    - `G5`：`[B, 256, H/32, W/32]`
- `SRAFMBFM`
  - 输出：
    - `F3`：`[B, 256, H/8, W/8]`
    - `F4`：`[B, 512, H/16, W/16]`
    - `F5`：`[B, 512, H/32, W/32]`

Backbone 输出顺序固定为：
- `F2, F3, F4, F5`

### Neck
当前 neck 输入是四尺度特征：
- `F2, F3, F4, F5`

主要步骤：
- `CAF`：先把各层对齐到 `256` 通道
- `TDSF`：执行自顶向下选择性融合
- `BURF`：执行自底向上 refinement

Neck 输出顺序固定为：
- `N2, N3, N4, N5`

输出合同：
- `N2`：`[B, 256, H/4, W/4]`
- `N3`：`[B, 256, H/8, W/8]`
- `N4`：`[B, 256, H/16, W/16]`
- `N5`：`[B, 256, H/32, W/32]`

### Detect Head
当前 head 直接复用 Ultralytics `Detect`：
- 输入：`N2, N3, N4, N5`
- 每层输入通道：`256`
- strides：`[4, 8, 16, 32]`
- loss：Ultralytics `v8DetectionLoss`

## 关键 Shape 合同
对输入 `[B, 3, 640, 640]`，当前实现应满足：

### Backbone outputs
- `F2`：`[B, 256, 160, 160]`
- `F3`：`[B, 256, 80, 80]`
- `F4`：`[B, 512, 40, 40]`
- `F5`：`[B, 512, 20, 20]`

### Neck outputs
- `N2`：`[B, 256, 160, 160]`
- `N3`：`[B, 256, 80, 80]`
- `N4`：`[B, 256, 40, 40]`
- `N5`：`[B, 256, 20, 20]`

### Detect strides
- `[4, 8, 16, 32]`

## 主要代码入口

### 模型相关
- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/neck.py`
- `amfe/models/detector.py`
- `amfe/models/registry.py`

### 训练相关
- `amfe/training.py`
- `tools/train.py`

### 数据相关
- `amfe/data/local_dataset.py`
- `amfe/data/visdrone_conversion.py`
- `tools/validate_dataset.py`

## 配置说明

### 模型配置
模型配置位于 `configs/model/`，当前重点字段包括：
- `num_classes`
- `in_channels`
- `neck_channels`
- `msb_variant`
- `lem_channels`
- `mbfm_gate_reduction`
- `tdsf_spg_reduction`
- `tdsf_dpg_kernels`
- `detect_feature_strides`
- `stride_init_image_size`
- `loss_hyperparameters`

默认模型配置：
- [`configs/model/amfe_amf_yolo.yaml`](D:/code/AMFE/configs/model/amfe_amf_yolo.yaml)

VisDrone 10 类配置：
- [`configs/model/amfe_amf_yolo_visdrone.yaml`](D:/code/AMFE/configs/model/amfe_amf_yolo_visdrone.yaml)

### 训练配置
训练启动配置位于 `configs/train/`：
- `train_default.yaml`：默认训练配置
- `train_visdrone_smoke.yaml`：本地 VisDrone smoke run
- `train_visdrone_full.yaml`：更长周期训练配置

### 数据配置
数据配置位于 `configs/data/`：
- `visdrone_local.yaml`：本地 VisDrone YOLO 数据路径
- `dataset_example.yaml`：示例数据配置

## 常用命令

### 1. 运行 synthetic smoke
```bash
.venv\Scripts\python.exe tools/train.py --config configs/train/train_default.yaml --synthetic-smoke
```

### 2. 仅校验本地数据集
```bash
.venv\Scripts\python.exe tools/validate_dataset.py --data configs/data/visdrone_local.yaml --model-config configs/model/amfe_amf_yolo_visdrone.yaml
```

### 3. 启动本地 VisDrone smoke training
```bash
.venv\Scripts\python.exe tools/train.py --config configs/train/train_visdrone_smoke.yaml
```

## 本地数据路径说明
当前仓库的本地 VisDrone YOLO 数据配置在：
- [`configs/data/visdrone_local.yaml`](D:/code/AMFE/configs/data/visdrone_local.yaml)

默认期望数据集根目录：
- `D:/code/AMFE/true_datasets/visdrone_yolo`

典型结构：
```text
true_datasets/visdrone_yolo/
├─ dataset.yaml
├─ images/
│  ├─ train/
│  └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

## 测试
运行核心测试：
```bash
.venv\Scripts\python.exe -m pytest tests/test_backbone.py tests/test_neck.py tests/test_detector.py
```

当前测试覆盖：
- backbone 子模块 forward
- `F2` 独立分支 forward
- backbone 四尺度输出合同
- neck 四输入四输出合同
- detector forward smoke
- detector strides 校验
- synthetic loss / backward / optimizer step
- YAML 构建测试

## 开发注意事项
- 当前代码事实优先于旧设计文档
- 当前实现是四尺度，不是三尺度
- `F2` 是独立浅层检测分支，不并入 `MBFM`
- `MSB` 当前是 YOLO 风格实现，不是 ResNet50
- 任何结构变更都应同步更新：
  - `amfe/models/backbone/amfe_backbone.py`
  - `amfe/models/neck.py`
  - `amfe/models/detector.py`
  - `configs/model/*.yaml`
  - `tests/test_backbone.py`
  - `tests/test_neck.py`
  - `tests/test_detector.py`

## 说明
如果后续模型结构、训练路径或数据路径发生明显变化，请同步更新 README 和 AGENTS.md，保证文档描述的是当前仓库里真正存在并能运行的代码。
