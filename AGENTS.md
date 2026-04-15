# AGENTS.md

## 用途
本文件描述 AMFE 仓库当前真实代码状态，用于指导后续开发、调试、评审和文档维护。

如果 `AMFE_AMF_Codex_Implementation_Spec.md`、README、历史讨论与当前实现冲突，必须明确指出冲突，并优先以当前代码为准。当前仓库已经从旧四尺度实现正式迁移到新的三尺度实现，因此任何仍描述 `F5/N5/stride32` 主链的文档都属于过时信息。

## 当前仓库状态
仓库当前实现的是一个可运行的三尺度 AMFE 检测器，并继续复用 Ultralytics 组件。

当前主链：
- Backbone：`optional LEM(identity by default) -> DPSStem -> MSB(C2/C3/C4) + ADB(D2/D3) + SemanticDetailFusion(F2/F3) + RFBLite(F4e)`
- Neck：三尺度 `CAF + TDSF + gated BURF`
- Head：Ultralytics `Detect`
- 检测尺度：`N2 / N3 / N4`
- Detect strides：`[4, 8, 16]`

当前仓库支持：
- 通过 Python 或 YAML 直接构建模型
- synthetic forward / loss / backward smoke test
- 本地 YOLO 数据集校验与摘要输出
- 通过项目内 trainer bridge 接入 Ultralytics 训练流程
- 训练结束后可选的 synthetic inference FPS benchmark 输出

## 仓库结构

### 核心包
- `amfe/models/backbone/`：backbone 模块、三尺度融合模块、`RFBLite`
- `amfe/models/neck.py`：三尺度 `AMFNeck` 及其子模块
- `amfe/models/detector.py`：整网装配、stride 初始化、loss 包装
- `amfe/models/registry.py`：YAML 加载与模型构建入口
- `amfe/training.py`：Ultralytics 训练桥接与复杂度日志输出
- `amfe/evaluation.py`：已训练 checkpoint 的独立验证与指标汇总
- `amfe/ultralytics_compat.py`：`Detect` 与 `v8DetectionLoss` 兼容导入层，附带 headless OpenCV fallback
- `amfe/data/local_dataset.py`：本地 YOLO 数据集解析与校验
- `amfe/data/visdrone_conversion.py`：原始 VisDrone 数据转换与转换后数据集校验

### 工具脚本
- `tools/train.py`：synthetic smoke、仅检查数据、或真实训练启动入口
- `tools/eval.py`：使用训练好的 `best.pt/last.pt` 做独立验证与可选 FPS benchmark
- `tools/validate_dataset.py`：本地 YOLO 数据集配置校验入口

### 配置目录
- `configs/model/`：AMFE 模型配置
- `configs/data/`：数据集配置
- `configs/train/`：训练启动配置

### 测试目录
- `tests/test_backbone.py`：backbone 与三尺度融合模块测试
- `tests/test_neck.py`：neck 子模块与 neck 集成测试
- `tests/test_detector.py`：整网 forward、stride、YAML、loss、backward 测试

## 当前模型结构

### Backbone
核心定义在 `amfe/models/backbone/amfe_backbone.py`。

实际特征流如下：

1. `LEM`
- 输入：`[B, 3, H, W]`
- 默认：`use_lem: false`，此时 `self.lem = Identity()`
- 兼容模式：`use_lem: true` 时才启用真实 `LEM`

2. `DPSStem`
- 输入：LEM 输出
- 输出 `S2`：`[B, 64, H/4, W/4]`

3. 语义主干 `MSB`
- 输入：`S2`
- 输出：
  - `C2`：`[B, 256, H/4, W/4]`
  - `C3`：`[B, 256, H/8, W/8]`
  - `C4`：`[B, 512, H/16, W/16]`
- 当前实现是 YOLOv8 风格 stage stack
- 当前实现不再生成 `C5`

4. 细节分支 `ADB`
- 输入：`S2`
- 输出：
  - `D2`：`[B, 128, H/4, W/4]`
  - `D3`：`[B, 128, H/8, W/8]`
- 当前实现不再生成 `D4`

5. 浅中层融合 `SemanticDetailFusion`
- `F2 = Fuse(C2, D2)` -> `[B, 256, H/4, W/4]`
- `F3 = Fuse(C3, D3)` -> `[B, 256, H/8, W/8]`
- 新主链不再依赖 `LGCB -> G3/G4/G5 -> SRAFMBFM`

6. 深层语义增强尾部 `RFBLite`
- `F4e = RFBLite(C4)` -> `[B, rfb_channels, H/16, W/16]`
- 默认 `rfb_channels = 512`
- 新主链不再保留 `F5`

Backbone 输出顺序固定为：
- `return F2, F3, F4e`

### Neck
核心定义在 `amfe/models/neck.py`。

输入顺序固定为：
- `F2, F3, F4e`

通道对齐：
- `CAF2(F2) -> L2: 256 channels`
- `CAF3(F3) -> L3: 256 channels`
- `CAF4(F4e) -> L4: 256 channels`

自顶向下路径：
- `TD3 = TDSF(L3, Up(L4))`
- `N2 = TDSF(L2, Up(TD3))`

自底向上路径：
- `N3 = BURF(N2, TD3)`
- `N4 = BURF(N3, L4)`

当前 `BURF` 逻辑：
- `lower_down = downsample(lower)`
- `spatial_prior = DPG(lower_down)`，用于筛选低层上送的空间细节
- `channel_prior = SPG(higher)`，用于校正高层语义通道
- `refine(lower_down * spatial_prior + higher * channel_prior)` 输出 bottom-up refined feature
- 若 `downsample(lower)` 后与 `higher` 空间尺寸不一致，显式抛错

Neck 输出顺序固定为：
- `return N2, N3, N4`

Neck 输出合同：
- `N2`：`[B, 256, H/4, W/4]`
- `N3`：`[B, 256, H/8, W/8]`
- `N4`：`[B, 256, H/16, W/16]`

### Detect Head
核心装配在 `amfe/models/detector.py`。

当前设计：
- 直接使用 Ultralytics `Detect`
- 输入特征：`N2, N3, N4`
- 每层输入通道：`256`
- 配置 stride：`[4, 8, 16]`
- loss：Ultralytics `v8DetectionLoss`

除非任务明确要求，不要重写 head 或 loss。优先修改 wrapper、配置或特征路由，而不是改 Detect 原理。

## 关键 Shape 合同
对输入 `[1, 3, 640, 640]`，期望特征层级如下：

### Backbone outputs
- `F2`：`[1, 256, 160, 160]`
- `F3`：`[1, 256, 80, 80]`
- `F4e`：`[1, 512, 40, 40]`

### Neck outputs
- `N2`：`[1, 256, 160, 160]`
- `N3`：`[1, 256, 80, 80]`
- `N4`：`[1, 256, 40, 40]`

### Detect strides
- `[4, 8, 16]`

这些合同在代码和测试中都有显式约束。只要修改通道数、尺度数、顺序或 stride，就必须同步更新实现、配置、stride 初始化和测试。

## 对外构建与训练 API

### 模型构建入口
- `amfe.models.build_amfe_detector(...)`
- `amfe.models.build_model_from_yaml(path)`
- `amfe.models.build_model_from_config(mapping)`

### 主模型类
- `AMFEYOLODetectionModel`
- `forward_features(x)` 返回 `N2, N3, N4`
- `forward(x)` 按 Ultralytics 风格路由到预测或 loss
- `loss(batch)` 期望最小 batch key：`img`、`batch_idx`、`cls`、`bboxes`

### 训练桥接
- `AMFEDetectionTrainer` 位于 `amfe/training.py`
- 负责把自定义模型接入 Ultralytics `DetectionTrainer`
- 训练开始时会通过 Ultralytics `get_num_params` 和 `get_flops` 记录模型复杂度
- 训练结束时可按 `training.fps_benchmark` 配置输出推理延迟/FPS 摘要
- 支持 resume，但不能直接复用旧四尺度 checkpoint

### 工具入口
- synthetic smoke：
  - `python tools/train.py --config configs/train/train_default.yaml --synthetic-smoke`
- 数据集校验：
  - `python tools/validate_dataset.py --data configs/data/visdrone_local.yaml --model-config configs/model/amfe_amf_yolo_visdrone.yaml`
- 真实训练：
  - `python tools/train.py --config configs/train/train_visdrone_smoke.yaml`
- 独立验证已训练权重：
  - `python tools/eval.py --config configs/train/train_visdrone_smoke.yaml --weights runs/phase_e/visdrone_3scale_smoke/weights/best.pt`
  - `tools/eval.py` 使用 Ultralytics 原生 `DetectionValidator`，输出 `precision`、`recall`、`mAP50`、`mAP50-95` 等标准验证指标
  - 独立验证开始时会额外打印模型复杂度摘要：`params/M` 与 `FLOPs/G`
  - 可选 `--plots` / `--save-json` 仍沿用 Ultralytics 原生行为

## 配置约定

### 模型配置
当前模型配置位于 `configs/model/`，采用 `model:` 顶层映射。

重要字段：
- `num_classes`
- `in_channels`
- `neck_channels`
- `msb_variant`
- `use_lem`
- `lem_channels`
- `fusion_gate_reduction`
- `rfb_channels`
- `rfb_expand_ratio`
- `rfb_dilations`
- `tdsf_spg_reduction`
- `tdsf_dpg_kernels`
- `detect_feature_strides`
- `stride_init_image_size`
- `loss_hyperparameters`

说明：
- `mbfm_gate_reduction` 仍可被 registry 兼容读取，但属于 legacy config alias
- 新默认配置使用 `fusion_gate_reduction`

### 训练配置
训练启动配置同时引用模型配置和数据集配置：
- `model_config: ...`
- `data: ...`
- `training: ...`

`tools/train.py` 负责将这些字段翻译成 Ultralytics trainer overrides。
若 `training.fps_benchmark` 启用，训练结束后会在当前设备上用 synthetic 输入执行一次推理测速。

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
仓库已经从旧四尺度实现迁移到：
- 三尺度检测，而不是四尺度
- `F2/F3/F4e -> N2/N3/N4`
- 最深层语义增强使用 `RFBLite`
- 新主链不再包含 `F5/N5/stride32`

如果旧文档仍描述 `LGCB + SRAFMBFM + F5/N5` 主链，必须明确指出那是历史实现。

### 保持显式合同稳定
除非任务明确要求改变，否则保持以下合同稳定：
- backbone 输出顺序：`F2, F3, F4e`
- neck 输出顺序：`N2, N3, N4`
- detect 输入顺序：`N2, N3, N4`
- detect strides：`[4, 8, 16]`

### 结构变更必须联动检查
只要改模型结构，就至少一起检查这些文件：
- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/backbone/adb.py`
- `amfe/models/backbone/msb.py`
- `amfe/models/backbone/mbfm.py`
- `amfe/models/backbone/rfb.py`
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
- fuse 前空间尺寸对齐检查
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

## 兼容性说明
以下模块当前属于 legacy compatibility only：
- `amfe/models/backbone/lgcb.py`
- `amfe/models/backbone/SRAFMBFM` 相关旧接口
- `SPPFLite` 类定义

这些定义可保留给旧 import 路径，但不属于新 backbone 主链。

## 维护本文件的规则
只要仓库的实际实现发生明显变化，就要同步更新本文件。本文件应该比项目规格文档更短、更偏操作指南，并始终描述当前真正存在的代码，而不是历史目标。
