# AMFE/AMF 当前实现规格说明

## 1. 文档定位
本文件描述当前仓库中已经落地的 AMFE/AMF 检测器实现规格，用于指导后续开发、调试、训练接入和文档维护。

如果旧版设计说明、历史讨论或更早的三尺度方案与当前代码冲突，优先以当前实现为准；如需回退或重构，必须显式说明变更目标和影响范围。

## 2. 当前实现总览
当前仓库实现的是一个四尺度目标检测模型，主链如下：

```text
Input
-> LEM
-> DPSStem
-> F2 独立浅层分支
-> MSB + ADB + LGCB + SRAFMBFM
-> F2 / F3 / F4 / F5
-> AMFNeck
-> N2 / N3 / N4 / N5
-> Ultralytics Detect Head
```

设计原则：
- Detect Head 直接复用 Ultralytics `Detect`
- loss 直接复用 Ultralytics `v8DetectionLoss`
- 结构创新主要集中在 backbone 和 neck
- 当前版本优先保证结构稳定、shape 正确、可训练、可测试

## 3. 当前模型结构

### 3.1 Backbone
Backbone 入口定义于 `amfe/models/backbone/amfe_backbone.py`。

#### 3.1.1 LEM
- 输入：`[B, 3, H, W]`
- 输出：`[B, 32, H, W]`
- 作用：轻量低层增强，不做图像复原，不改变空间尺寸

#### 3.1.2 DPSStem
- 输入：`[B, 32, H, W]`
- 输出 `S2`：`[B, 64, H/4, W/4]`
- 作用：以较保守的方式完成早期下采样，保留更多小目标细节

#### 3.1.3 F2 独立浅层检测分支
- 输入：`S2`
- 结构：`Conv1x1(64 -> 256) + DWConv3x3(256 -> 256)`
- 输出 `F2`：`[B, 256, H/4, W/4]`
- 说明：当前版本中 `F2` 不进入 `MBFM`，而是作为独立浅层检测特征提供给 neck

#### 3.1.4 MSB
- 输入：`S2`
- 输出：
  - `C3`：`[B, 256, H/8, W/8]`
  - `C4`：`[B, 512, H/16, W/16]`
  - `C5`：`[B, 512, H/32, W/32]`
- 当前实现说明：
  - 采用 YOLOv8 风格的 stage 结构
  - 内部使用项目本地 `C2fBlock` 和 `SPPFLite`
  - 当前实现不是 ResNet50

#### 3.1.5 ADB
- 输入：`S2`
- 输出：
  - `D3`：`[B, 128, H/8, W/8]`
  - `D4`：`[B, 256, H/16, W/16]`
- 作用：提供局部细节补偿分支

#### 3.1.6 LGCB
- 输入：`C3, C4, C5`
- 输出：
  - `G3`：`[B, 256, H/8, W/8]`
  - `G4`：`[B, 256, H/16, W/16]`
  - `G5`：`[B, 256, H/32, W/32]`
- 作用：从深层语义特征中提取轻量上下文信息

#### 3.1.7 SRAFMBFM
- `F3 = MBFM3(C3, D3, G3)` -> `[B, 256, H/8, W/8]`
- `F4 = MBFM4(C4, D4, G4)` -> `[B, 512, H/16, W/16]`
- `F5 = MBFM5(C5, None, G5)` -> `[B, 512, H/32, W/32]`
- 作用：融合语义、细节和上下文分支，保留语义主分支作为锚点

#### 3.1.8 Backbone 输出顺序
Backbone 输出顺序固定为：
- `F2, F3, F4, F5`

### 3.2 Neck
Neck 定义于 `amfe/models/neck.py`。

#### 3.2.1 输入顺序
Neck 输入顺序固定为：
- `F2, F3, F4, F5`

#### 3.2.2 CAF
- `CAF2(F2) -> L2`
- `CAF3(F3) -> L3`
- `CAF4(F4) -> L4`
- `CAF5(F5) -> L5`
- 作用：统一各尺度通道到 `256`

#### 3.2.3 Top-Down Selective Fusion
- `TD4 = TDSF(L4, Up(L5))`
- `TD3 = TDSF(L3, Up(TD4))`
- `N2 = TDSF(L2, Up(TD3))`
- 作用：高层语义引导低层特征融合

#### 3.2.4 Bottom-Up Refinement Fusion
- `N3 = BURF(N2, TD3)`
- `N4 = BURF(N3, TD4)`
- `N5 = BURF(N4, L5)`
- 作用：低层信息反向补充高层融合结果

#### 3.2.5 Neck 输出顺序
Neck 输出顺序固定为：
- `N2, N3, N4, N5`

### 3.3 Detect Head
Detect Head 装配在 `amfe/models/detector.py`。

当前实现：
- 直接使用 Ultralytics `Detect`
- 输入特征：`N2, N3, N4, N5`
- 每层输入通道：`256`
- strides：`[4, 8, 16, 32]`
- loss：Ultralytics `v8DetectionLoss`

除非任务明确要求，不要重写 Detect Head 原理或 loss 原理。优先改 wrapper、配置和特征路由。

## 4. Shape 合同
对输入 `img = [1, 3, 640, 640]`，当前实现应满足：

### 4.1 Backbone outputs
- `F2`：`[1, 256, 160, 160]`
- `F3`：`[1, 256, 80, 80]`
- `F4`：`[1, 512, 40, 40]`
- `F5`：`[1, 512, 20, 20]`

### 4.2 Neck outputs
- `N2`：`[1, 256, 160, 160]`
- `N3`：`[1, 256, 80, 80]`
- `N4`：`[1, 256, 40, 40]`
- `N5`：`[1, 256, 20, 20]`

### 4.3 Detect strides
- `[4, 8, 16, 32]`

### 4.4 固定顺序要求
以下顺序必须保持一致：
- backbone 输出：`F2, F3, F4, F5`
- neck 输出：`N2, N3, N4, N5`
- Detect 输入：`N2, N3, N4, N5`

## 5. 主要模块命名
当前仓库保留并使用以下核心命名：

### 5.1 Backbone 相关
- `AMFEBackbone`
- `LEM`
- `DPSStem`
- `MSB`
- `ADB`
- `DEB`
- `LGCB`
- `MBFM`
- `CDG`
- `SRAFMBFM`

### 5.2 Neck 相关
- `AMFNeck`
- `CAF`
- `TDSF`
- `BURF`
- `SPG`
- `DPG`

当前实现中没有实际落地 `MCA` 独立模块。如未来引入，必须明确它的职责、输入输出和测试。

## 6. 配置约定

### 6.1 模型配置
模型配置位于 `configs/model/`，采用 `model:` 顶层映射。

当前关键字段：
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

### 6.2 训练配置
训练配置位于 `configs/train/`，主要包含：
- `model_config`
- `data`
- `training`

`tools/train.py` 会将这些字段翻译为 Ultralytics trainer overrides。

### 6.3 数据配置
数据配置位于 `configs/data/`。当前仓库要求本地数据路径显式声明，不允许在训练代码中硬编码隐藏路径假设。

## 7. 模型构建与训练接口

### 7.1 模型构建入口
- `amfe.models.build_amfe_detector(...)`
- `amfe.models.build_model_from_yaml(path)`
- `amfe.models.build_model_from_config(mapping)`

### 7.2 主模型类
- `AMFEYOLODetectionModel`
- `forward_features(x)` 返回 `N2, N3, N4, N5`
- `forward(x)` 兼容 Ultralytics 风格的预测/损失入口
- `loss(batch)` 期望 batch 至少包含：
  - `img`
  - `batch_idx`
  - `cls`
  - `bboxes`

### 7.3 训练桥接
- `amfe/training.py` 中实现 `AMFEDetectionTrainer`
- 作用：将当前 AMFE detector 接入 Ultralytics `DetectionTrainer`
- 行为：
  - train start 输出模型参数量和 FLOPs
  - 允许从 checkpoint resume
  - 保持 Ultralytics trainer 的训练主链兼容性

## 8. 数据与工具路径

### 8.1 数据工具
- `amfe/data/local_dataset.py`
  - 本地 YOLO 数据集路径解析与一致性校验
- `amfe/data/visdrone_conversion.py`
  - 原始 VisDrone 转换与转换后数据集校验

### 8.2 工具脚本
- `tools/train.py`
  - synthetic smoke
  - 仅检查数据集
  - 真实训练入口
- `tools/validate_dataset.py`
  - 本地数据集配置校验入口

## 9. 测试要求
对任何有意义的模型结构改动，至少要保持以下测试有效：

1. backbone 子模块 forward 测试
2. backbone 集成 shape 测试
3. neck 子模块与 neck 集成测试
4. detector forward smoke test
5. YAML 构建测试
6. synthetic loss / backward / optimizer step 测试

当前核心测试文件：
- `tests/test_backbone.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

推荐最低回归命令：

```bash
python -m pytest tests/test_backbone.py tests/test_neck.py tests/test_detector.py
```

## 10. 允许与禁止的改动边界

### 10.1 允许的方向
- 在保持主链稳定前提下修复 shape、接口、配置、训练桥接和测试
- 对 backbone 或 neck 做保守增强，但必须同步维护配置、测试和文档
- 对 Detect 的输入层数、路由和 wrapper 做适配性修改

### 10.2 不建议默认执行的改动
除非任务明确要求，否则不要默认进行以下行为：
- 重写 Ultralytics Detect Head
- 重写 detection loss
- 引入新的大型 attention/transformer 主结构
- 把 `F2` 强行并回 `MBFM`
- 静默改变 feature order 或 detect stride
- 在不更新测试的情况下改动 shape 合同

## 11. 当前实现与旧方案的主要差异
以下旧说法不再适用于当前仓库：
- 仅三尺度检测 `F3/F4/F5 -> N3/N4/N5`
- `MSB` 基于 ResNet50 的实现假设
- `F2` 尚未接入主链

当前代码事实是：
- 四尺度检测已落地
- `F2` 为独立浅层检测分支
- `MSB` 为 YOLO 风格语义主干
- neck 和 Detect 均已扩展到四尺度

## 12. 文档维护规则
当实际代码发生以下变化时，必须同步更新本文件：
- feature 数量变化
- feature 顺序变化
- channel 合同变化
- detect strides 变化
- 训练入口或配置 schema 变化
- 测试要求变化

本文件应始终描述当前仓库中真正存在并可运行的实现。
