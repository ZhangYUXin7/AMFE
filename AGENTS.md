# AGENTS.md

## 用途
本文件描述 AMFE 仓库的当前代码事实，用于指导后续开发、调试、评审和文档维护。

如果 `AMFE_AMF_Codex_Implementation_Spec.md`、README、历史讨论与当前代码实现冲突，必须明确指出冲突，并优先以当前代码为准；只有在用户明确要求回退或重新对齐设计文档时，才按文档目标做结构调整。

## 当前仓库状态
仓库当前实现的是一个可运行的四尺度 AMFE 检测器，并已经接入 Ultralytics 组件。

当前主链：
- Backbone：`LEM -> DPSStem -> F2 分支 + MSB + ADB + LGCB + SRAFMBFM`
- Neck：`CAF + TDSF + BURF`
- Head：Ultralytics `Detect`
- 检测尺度：`N2 / N3 / N4 / N5`
- Detect strides：`[4, 8, 16, 32]`

当前仓库支持：
- 通过 Python 或 YAML 直接构建模型
- synthetic forward / loss / backward smoke test
- 本地 YOLO 数据集校验与摘要输出
- 通过项目内 trainer bridge 接入 Ultralytics 训练流程

## 仓库结构

### 核心包
- `amfe/models/backbone/`：backbone 模块与特征融合模块
- `amfe/models/neck.py`：AMFNeck 及其子模块
- `amfe/models/detector.py`：整网装配、stride 初始化、loss 包装
- `amfe/models/registry.py`：YAML 加载与模型构建入口
- `amfe/training.py`：Ultralytics 训练桥接与复杂度日志输出
- `amfe/ultralytics_compat.py`：`Detect` 与 `v8DetectionLoss` 兼容导入层，附带 headless OpenCV fallback
- `amfe/data/local_dataset.py`：本地 YOLO 数据集解析与校验
- `amfe/data/visdrone_conversion.py`：原始 VisDrone 数据转换与转换后数据集校验

### 工具脚本
- `tools/train.py`：synthetic smoke、仅检查数据、或真实训练启动入口
- `tools/validate_dataset.py`：本地 YOLO 数据集配置校验入口

### 配置目录
- `configs/model/`：AMFE 模型配置
- `configs/data/`：数据集配置
- `configs/train/`：训练启动配置

### 测试目录
- `tests/test_backbone.py`：backbone 与 backbone 子模块测试
- `tests/test_neck.py`：neck 子模块与 neck 集成测试
- `tests/test_detector.py`：整网 forward、stride、YAML、loss、backward 测试

## 当前模型结构

### Backbone
核心定义在 `amfe/models/backbone/amfe_backbone.py`。

实际特征流如下：

1. `LEM`
- 输入：`[B, 3, H, W]`
- 输出：`[B, 32, H, W]`

2. `DPSStem`
- 输入：`[B, 32, H, W]`
- 输出 `S2`：`[B, 64, H/4, W/4]`

3. 独立浅层检测分支
- `f2_branch = Conv1x1(64 -> 256) + DWConv3x3(256 -> 256)`
- 输出 `F2`：`[B, 256, H/4, W/4]`
- 当前实现中，`F2` 不经过 `MBFM`

4. 语义分支 `MSB`
- 输入：`S2`
- 输出：
  - `C3`：`[B, 256, H/8, W/8]`
  - `C4`：`[B, 512, H/16, W/16]`
  - `C5`：`[B, 512, H/32, W/32]`
- 当前实现是 YOLOv8 风格的 stage stack，内部使用项目本地 `C2fBlock` 和 `SPPFLite`
- 当前实现不是 ResNet50

5. 细节分支 `ADB`
- 输入：`S2`
- 输出：
  - `D3`：`[B, 128, H/8, W/8]`
  - `D4`：`[B, 256, H/16, W/16]`

6. 上下文分支 `LGCB`
- 输入：`C3, C4, C5`
- 输出：
  - `G3`：`[B, 256, H/8, W/8]`
  - `G4`：`[B, 256, H/16, W/16]`
  - `G5`：`[B, 256, H/32, W/32]`

7. 融合模块 `SRAFMBFM`
- `F3 = MBFM3(C3, D3, G3)` -> `[B, 256, H/8, W/8]`
- `F4 = MBFM4(C4, D4, G4)` -> `[B, 512, H/16, W/16]`
- `F5 = MBFM5(C5, None, G5)` -> `[B, 512, H/32, W/32]`

Backbone 输出顺序固定为：
- `return F2, F3, F4, F5`

### Neck
核心定义在 `amfe/models/neck.py`。

输入顺序固定为：
- `F2, F3, F4, F5`

通道对齐：
- `CAF2(F2) -> L2: 256 channels`
- `CAF3(F3) -> L3: 256 channels`
- `CAF4(F4) -> L4: 256 channels`
- `CAF5(F5) -> L5: 256 channels`

自顶向下路径：
- `TD4 = TDSF(L4, Up(L5))`
- `TD3 = TDSF(L3, Up(TD4))`
- `N2 = TDSF(L2, Up(TD3))`

自底向上路径：
- `N3 = BURF(N2, TD3)`
- `N4 = BURF(N3, TD4)`
- `N5 = BURF(N4, L5)`

Neck 输出顺序固定为：
- `return N2, N3, N4, N5`

Neck 输出合同：
- `N2`：`[B, 256, H/4, W/4]`
- `N3`：`[B, 256, H/8, W/8]`
- `N4`：`[B, 256, H/16, W/16]`
- `N5`：`[B, 256, H/32, W/32]`

### Detect Head
核心装配在 `amfe/models/detector.py`。

当前设计：
- 直接使用 Ultralytics `Detect`
- 输入特征：`N2, N3, N4, N5`
- 每层输入通道：`256`
- 配置 stride：`[4, 8, 16, 32]`
- loss：Ultralytics `v8DetectionLoss`

除非任务明确要求，不要重写 head 或 loss。优先修改 wrapper、配置或特征路由，而不是改 Detect 原理。

## 关键 Shape 合同
对输入 `[1, 3, 640, 640]`，期望特征层级如下：

### Backbone outputs
- `F2`：`[1, 256, 160, 160]`
- `F3`：`[1, 256, 80, 80]`
- `F4`：`[1, 512, 40, 40]`
- `F5`：`[1, 512, 20, 20]`

### Neck outputs
- `N2`：`[1, 256, 160, 160]`
- `N3`：`[1, 256, 80, 80]`
- `N4`：`[1, 256, 40, 40]`
- `N5`：`[1, 256, 20, 20]`

### Detect strides
- `[4, 8, 16, 32]`

这些合同在代码和测试中都有显式约束。只要修改通道数、尺度数、顺序或 stride，就必须同步更新实现、配置、stride 初始化和测试。

## 对外构建与训练 API

### 模型构建入口
- `amfe.models.build_amfe_detector(...)`
- `amfe.models.build_model_from_yaml(path)`
- `amfe.models.build_model_from_config(mapping)`

### 主模型类
- `AMFEYOLODetectionModel`
- `forward_features(x)` 返回 `N2, N3, N4, N5`
- `forward(x)` 按 Ultralytics 风格路由到预测或 loss
- `loss(batch)` 期望最小 batch key：`img`、`batch_idx`、`cls`、`bboxes`

### 训练桥接
- `AMFEDetectionTrainer` 位于 `amfe/training.py`
- 负责把自定义模型接入 Ultralytics `DetectionTrainer`
- 训练开始时会通过 Ultralytics `get_num_params` 和 `get_flops` 记录模型复杂度
- 支持 resume，并保留一组明确的 override keys

### 工具入口
- synthetic smoke：
  - `python tools/train.py --config configs/train/train_default.yaml --synthetic-smoke`
- 数据集校验：
  - `python tools/validate_dataset.py --data configs/data/visdrone_local.yaml --model-config configs/model/amfe_amf_yolo_visdrone.yaml`
- 真实训练：
  - `python tools/train.py --config configs/train/train_visdrone_smoke.yaml`

## 配置约定

### 模型配置
当前模型配置位于 `configs/model/`，采用 `model:` 顶层映射。

重要字段：
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

### 训练配置
训练启动配置同时引用模型配置和数据集配置：
- `model_config: ...`
- `data: ...`
- `training: ...`

`tools/train.py` 负责将这些字段翻译成 Ultralytics trainer overrides。

### 数据集配置
仓库当前要求使用 `configs/data/` 下显式的数据集 YAML。不要把隐藏的数据路径假设写死在模型代码或训练代码中。

## 测试与验证要求
任何有意义的模型改动或数据路径改动都必须带测试。

最低覆盖要求：
- backbone / neck 子模块 forward 测试
- backbone 集成 shape 测试
- neck 集成 shape 测试
- detector forward smoke test
- detector stride 与 YAML 构建测试
- synthetic loss / backward / optimizer-step 测试

核心测试运行命令：
- `python -m pytest tests/test_backbone.py tests/test_neck.py tests/test_detector.py`

如果修改了以下任一项：
- 特征层数量
- 特征顺序
- 通道合同
- detect strides
- 模型配置 schema
- 训练 batch 合同

必须在同一个变更里同步更新测试。

## 后续开发工作规则

### 以当前代码为准，不盲从旧设计说明
仓库里仍然存在较早期的设计文档，但当前实现已经明确演化为：
- 四尺度检测，而不是三尺度
- `F2/F3/F4/F5 -> N2/N3/N4/N5`
- YOLO 风格语义主干，而不是 ResNet50

如果旧文档与实现冲突，必须在回复或提交说明里点明这一点。

### 保持显式合同稳定
除非任务明确要求改变，否则保持以下合同稳定：
- backbone 输出顺序：`F2, F3, F4, F5`
- neck 输出顺序：`N2, N3, N4, N5`
- detect 输入顺序：`N2, N3, N4, N5`
- detect strides：`[4, 8, 16, 32]`

### 结构变更必须联动检查
只要改模型结构，就至少一起检查这些文件：
- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/neck.py`
- `amfe/models/detector.py`
- `configs/model/*.yaml`
- `tests/test_backbone.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

### 保守复用 Ultralytics 组件
优先复用已有 Ultralytics 组件：
- `Detect`
- `v8DetectionLoss`
- `DetectionTrainer`
- 数据集解析工具

除非任务明确要求，否则不要重写 Ultralytics 内部实现。

### 对无效假设快速失败
当前代码风格是显式检查并尽早报错。需要继续保持：
- tensor 维度检查
- 通道数检查
- add / fuse 前空间尺寸对齐检查
- stride 整除检查
- batch key 合同检查
- dataset / model config 一致性检查

不要引入静默 fallback 或偷偷修正 shape/config 的逻辑。

### 数据路径必须显式
仓库支持本地数据集和本地转换工具。不要假设：
- 运行时有网络
- 自动下载数据集
- 工作区外部存在隐式状态

## 数据工具说明
仓库当前有两类数据工具：

1. 本地 YOLO 数据集校验
- `amfe/data/local_dataset.py`
- 校验解析后的 train/val 图像与标签目录
- 交叉检查数据集 `nc` / `channels` 与模型配置是否一致

2. 原始 VisDrone 转换与校验
- `amfe/data/visdrone_conversion.py`
- 将扁平 VisDrone 原始标注转换为 YOLO 检测格式
- 写出 `dataset.yaml`
- 校验转换后 labels 与 split 结构

这些工具默认采用保守策略：遇到格式错误直接抛异常，不做静默修复。

## 环境说明
- 期望在仓库根目录使用 `.venv/` 虚拟环境
- Ultralytics 是运行时依赖
- headless 环境优先使用 `opencv-python-headless`
- `amfe/ultralytics_compat.py` 只在真实 OpenCV 因缺少 GUI 依赖而导入失败时，才安装最小 `cv2` stub

## 维护本文件的规则
只要仓库的实际实现发生明显变化，就要同步更新本文件。本文件应该比项目规格文档更短、更偏操作指南，并始终描述当前真正存在的代码，而不是历史目标。
