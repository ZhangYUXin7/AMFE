# 仓库结构说明（中文）

本文档面向后续需要继续做实验、调试模型、接入真实数据集或扩展训练流程的研究者，说明当前仓库中**真正重要**的目录与文件分别负责什么，以及它们如何共同构成完整的训练执行链路。

> 当前仓库是一个偏研究实现、但工程风格相对保守的目标检测代码库。它已经打通了最小可运行链路：
>
> **AMFE-Backbone -> AMF-Neck -> Ultralytics Detect Head**
>
> 同时具备基于合成数据的 forward / loss / backward / optimizer step 烟雾测试能力。

---

## 1. 仓库总览

### 1.1 这个仓库实现了什么

这个仓库实现的是一个面向复杂天气与小目标检测场景的实验型检测模型工程骨架，目标是将论文式结构拆解为可测试、可训练、可后续接入真实数据的 Python / PyTorch 代码。

当前实现重点包括：

- **AMFE-Backbone**：负责从输入图像中提取多尺度主干特征，输出 `F3 / F4 / F5`。
- **AMF-Neck**：负责对 `F3 / F4 / F5` 做通道对齐、Top-Down selective fusion、Bottom-Up refinement，输出 `N3 / N4 / N5`。
- **Ultralytics Detect Head**：复用 Ultralytics 原生 `Detect` 头，不重新设计检测头与损失。
- **最小训练链路**：通过合成 batch 验证模型构建、loss 计算、反向传播和参数更新是否能跑通。

从工程状态看，它不是一个已经完整接好真实数据训练的大型训练框架，而是一个**“结构已经清晰、训练主链路已打通、真实数据流程仍待后续扩展”**的研究实验基础代码库。

### 1.2 整体架构主线是什么

仓库的模型主线非常明确：

**输入图像 -> AMFE-Backbone -> AMF-Neck -> Ultralytics Detect Head -> 检测损失 / 预测输出**

更细一点的主干内部流向如下：

- 输入先进入 `LEM` 做轻量低层增强。
- 然后进入 `DPSStem` 做更保守的早期下采样。
- 接着由 `MSB` 抽取语义特征，`ADB` 补充细节分支，`LGCB` 提供全局上下文。
- `MBFM` 将语义、细节、上下文融合为主干最终输出 `F3 / F4 / F5`。
- `AMFNeck` 接收 `F3 / F4 / F5`，先做 `CAF` 通道对齐，再走 `TDSF` 和 `BURF` 完成自顶向下与自底向上的融合，生成 `N3 / N4 / N5`。
- `Detect` 头直接接收 `N3 / N4 / N5`，并复用 Ultralytics 的 `v8DetectionLoss` 进行训练。

### 1.3 当前仓库的实现边界

阅读仓库时需要注意当前阶段的边界：

- 已完成：模型模块化实现、配置加载、模型构建、合成数据训练烟雾测试、pytest 测试覆盖。
- 未完成最终版：真实数据集训练调度、完整实验管理、正式训练入口与 Ultralytics 全套 trainer 深度对接。
- 因此，`tools/train.py` 当前更准确地说是**最小训练/验证入口**，主要用于证明训练主链路可运行，而不是完整实验平台。

---

## 2. 按文件夹说明作用

下面只解释当前仓库里真正重要的目录，不浪费篇幅描述无关紧要的小目录。

### 2.1 `amfe/`

**通常放什么：**
- 顶层 Python 包入口。
- 对外导出的公共 API。
- 与第三方框架兼容的适配代码。

**职责：**
- 作为整个项目的 Python 包根目录。
- 将内部模型构建接口统一暴露给外部调用者。
- 放置与 Ultralytics 兼容相关的辅助模块。

**分类：**
- 工具函数
- 检测头接入
- 主模型包入口

**重点说明：**
- `amfe/__init__.py` 是包级导出入口。
- `amfe/ultralytics_compat.py` 是 Ultralytics 兼容层，属于**兼容性/辅助性文件**，不是核心模型逻辑本体。

### 2.2 `amfe/models/`

**通常放什么：**
- 模型主模块。
- 模型注册/构建函数。
- 各个子结构之间的连接代码。

**职责：**
- 汇总 backbone、neck、detector 的实现。
- 负责 YAML 配置到模型对象的构建转换。
- 负责把 AMFE 主干、AMF neck 与 Ultralytics Detect 接起来。

**分类：**
- 主干网络
- Neck
- 检测头接入
- 多分支融合
- 工具函数

**重点说明：**
- `detector.py` 是“系统装配文件”，在整个仓库中地位非常关键。
- `registry.py` 是配置读取与模型构建入口。
- `common.py` 是共享基础模块库。

### 2.3 `amfe/models/backbone/`

**通常放什么：**
- AMFE-Backbone 相关的所有子模块实现。
- 论文中各个 block 的对应代码。

**职责：**
- 实现主干网络的完整内部结构。
- 将 `LEM / DPSStem / MSB / ADB / LGCB / MBFM` 串成最终 backbone。
- 输出 `F3 / F4 / F5` 给 neck。

**分类：**
- 主干网络
- 多分支融合

**重点说明：**
- 这是后续修改 backbone 时最核心的目录。
- 几乎所有与 AMFE-Backbone 设计有关的实验改动都应优先从这里开始。

### 2.4 `configs/`

**通常放什么：**
- 模型配置 YAML。
- 训练配置 YAML。
- 数据集描述 YAML。

**职责：**
- 把“实验参数”从代码中分离出来。
- 为后续真实训练提供统一的配置来源。

**分类：**
- 配置文件

**子目录说明：**
- `configs/model/`：模型结构与超参数配置。
- `configs/train/`：训练入口读取的训练超参数配置。
- `configs/data/`：数据集路径与类别映射配置。

**重点说明：**
- 这些文件是后续切换实验设置时的首选修改点。
- 其中 `dataset_example.yaml` 明确是**占位/示例配置**，不是正式数据训练的最终版本。

### 2.5 `tools/`

**通常放什么：**
- 命令行训练入口。
- 辅助脚本。

**职责：**
- 作为运行训练或烟雾测试的入口。
- 串起配置读取、模型构建、batch 准备、loss、backward、optimizer step。

**分类：**
- 训练入口

**重点说明：**
- 当前只有 `tools/train.py`。
- 它目前更偏向**验证训练链路是否跑通**，而不是完整真实数据训练平台。

### 2.6 `tests/`

**通常放什么：**
- 针对 backbone、neck、detector 的单元测试与烟雾测试。

**职责：**
- 验证模块 shape contract。
- 验证端到端 forward 是否可运行。
- 验证 loss / backward / optimizer step 是否已打通。

**分类：**
- 测试文件

**重点说明：**
- 这里的测试基本都使用 synthetic tensor，不依赖真实数据下载。
- 对研究型迭代非常重要，因为它们能快速告诉你结构改动是否破坏了张量接口。

### 2.7 `docs/`

**通常放什么：**
- 项目说明文档。
- 结构说明、设计说明、使用文档。

**职责：**
- 承载对仓库结构、模块职责和阅读路径的解释。

**分类：**
- 文档

**重点说明：**
- 当前这份 `docs/repo_structure_explained_zh.md` 就是新增的中文结构说明文档。
- 该目录在当前仓库中不是模型运行所必需，但对后续协作、交接和实验延续非常有价值。

---

## 3. 按文件说明作用（重点文件逐个解释）

下面只解释当前仓库中真正承担关键职责的文件。为了便于阅读，我把它们分成“核心模型文件”和“辅助/兼容/入口文件”两组。

### 3.1 核心模型文件

#### 3.1.1 `amfe/models/backbone/amfe_backbone.py`

**文件用途：**
- 定义完整的 `AMFEBackbone`。
- 负责把 backbone 内部子模块按固定顺序组装起来。

**主要类 / 内容：**
- `BackboneOutputChannels`：定义 `F3 / F4 / F5` 的标准通道数。
- `AMFEBackbone`：主干网络主体。

**系统角色：**
- 它是 backbone 的总装文件，相当于“主干总入口”。
- 接收输入图像，输出 `F3 / F4 / F5`。

**与三段式主线的关系：**
- 直接对应 **AMFE-Backbone**。
- 它的输出是 `AMF-Neck` 的输入。
- 它不直接处理 Detect Head，但决定了 neck 的输入通道和尺度契约。

#### 3.1.2 `amfe/models/backbone/lem.py`

**文件用途：**
- 实现 `LEM` 低层增强模块。

**主要类 / 内容：**
- `LEM`

**系统角色：**
- 位于 backbone 最前端，对输入图像进行轻量增强。
- 不改变整体任务形式，但改善进入 stem 之前的底层特征表达。

**与三段式主线的关系：**
- 属于 **AMFE-Backbone** 内部第一步。
- 不直接接触 neck 或 detect head。

#### 3.1.3 `amfe/models/backbone/dps_stem.py`

**文件用途：**
- 实现 `DPSStem`，替代更激进的早期下采样。

**主要类 / 内容：**
- `DPSStem`

**系统角色：**
- 产生共享浅层特征 `S2`。
- 为后续 `MSB` 和 `ADB` 提供共同输入。

**与三段式主线的关系：**
- 属于 **AMFE-Backbone** 的 early stem。
- 是 backbone 内部多分支分流之前的公共入口。

#### 3.1.4 `amfe/models/backbone/msb.py`

**文件用途：**
- 实现 `MSB` 主语义分支。

**主要类 / 内容：**
- `TorchvisionCompatibleBottleneck`：在没有 torchvision 时提供兼容 block。
- `MSB`：基于 ResNet-50 stage 布局提取 `C2 / C3 / C4 / C5`。
- `load_torchvision_resnet50_weights()`：可选地加载 torchvision ResNet-50 权重。

**系统角色：**
- 负责 backbone 中的主语义路径，是语义特征提取主力。
- 输出深层语义特征给 `LGCB` 和 `MBFM`。

**与三段式主线的关系：**
- 属于 **AMFE-Backbone** 的核心语义支路。
- 对 neck 和 detect head 的影响是间接的：它决定 backbone 输出的主语义基础。

#### 3.1.5 `amfe/models/backbone/adb.py`

**文件用途：**
- 实现辅助细节分支 `ADB`，并在内部定义 `DEB`。

**主要类 / 内容：**
- `DEB`：细节增强 block。
- `ADB`：从浅层共享特征中提取 `D3 / D4`。

**系统角色：**
- 为语义主支路提供局部细节补偿。
- 它不直接输出最终检测特征，而是服务于 `MBFM` 的多分支融合。

**与三段式主线的关系：**
- 属于 **AMFE-Backbone** 内部的 detail branch。
- 与 neck 和 detect head 没有直接连接。

#### 3.1.6 `amfe/models/backbone/lgcb.py`

**文件用途：**
- 实现 `LGCB` 全局上下文模块。

**主要类 / 内容：**
- `LGCB`

**系统角色：**
- 从深层语义特征 `C5` 中提取轻量全局上下文，产生 `G3 / G4 / G5`。
- 为 `MBFM` 提供上下文支撑。

**与三段式主线的关系：**
- 属于 **AMFE-Backbone** 内部 context branch。
- 与 neck / detect head 的关系也是间接关系。

#### 3.1.7 `amfe/models/backbone/mbfm.py`

**文件用途：**
- 实现多分支融合模块 `MBFM` 和内部 gating 模块 `CDG`。

**主要类 / 内容：**
- `CDG`
- `MBFM`

**系统角色：**
- 把 semantic / detail / context 三路特征整合成 backbone 最终输出。
- 决定 `F3 / F4 / F5` 的融合方式。

**与三段式主线的关系：**
- 仍属于 **AMFE-Backbone**，但已经是主干输出前的最后融合阶段。
- 它的结果直接决定 `AMF-Neck` 的输入质量与通道结构。

#### 3.1.8 `amfe/models/neck.py`

**文件用途：**
- 实现完整的 `AMFNeck` 以及内部 neck 子模块。

**主要类 / 内容：**
- `NeckOutputChannels`
- `CAF`：通道对齐模块。
- `TDSF`：Top-Down Selective Fusion。
- `BURF`：Bottom-Up Refinement Fusion。
- `AMFNeck`：neck 总装模块。

**系统角色：**
- 接收 `F3 / F4 / F5`。
- 先对齐通道，再进行 top-down selective fusion，最后执行 bottom-up refinement。
- 输出 `N3 / N4 / N5`。

**与三段式主线的关系：**
- 直接对应 **AMF-Neck**。
- 上接 `AMFE-Backbone`，下接 `Ultralytics Detect Head`。
- 是主干特征与检测头之间最重要的中间桥梁。

#### 3.1.9 `amfe/models/detector.py`

**文件用途：**
- 定义完整 detector 装配逻辑。
- 将 backbone、neck、Ultralytics Detect head 连接成最终模型。

**主要类 / 内容：**
- `LossHyperparameters`
- `AMFEModelConfig`
- `AMFEYOLODetectionModel`
- `AMFEDetector`（别名）
- `build_amfe_detector()`

**系统角色：**
- 整个系统最关键的“组装层”。
- 负责：
  - 创建 `AMFEBackbone`
  - 创建 `AMFNeck`
  - 创建 Ultralytics `Detect`
  - 初始化 stride
  - 组织 `loss()` 逻辑
  - 暴露 `forward()`、`forward_features()`、`predict()`

**与三段式主线的关系：**
- 它把三段主线显式连接为：
  **AMFE-Backbone -> AMF-Neck -> Ultralytics Detect Head**。
- 如果没有这个文件，仓库中的 backbone 和 neck 只是孤立模块，无法形成完整 detector。

### 3.2 配置与构建文件

#### 3.2.1 `amfe/models/registry.py`

**文件用途：**
- 负责 YAML 配置读取与模型构建。

**主要类 / 函数：**
- `load_yaml_config()`
- `build_model_from_config()`
- `build_model_from_yaml()`

**系统角色：**
- 它是“配置 -> Python dict -> 模型对象”的转换层。
- 让训练入口、测试代码不需要手写 detector 初始化细节。

**与三段式主线的关系：**
- 不属于 backbone/neck/head 本体。
- 但它决定三段主线如何被实例化。

#### 3.2.2 `configs/model/amfe_amf_yolo.yaml`

**文件用途：**
- 定义模型相关超参数。

**主要配置：**
- `num_classes`
- `in_channels`
- `neck_channels`
- `stride_init_image_size`
- `loss_hyperparameters`

**系统角色：**
- 是 detector 构建时最直接读取的模型配置文件。

**与三段式主线的关系：**
- 间接控制三段式主线的构建参数，尤其影响 Detect head 的类别数和 neck 输出宽度。

#### 3.2.3 `configs/train/train_default.yaml`

**文件用途：**
- 定义训练入口所需的训练参数。

**主要配置：**
- `model_config`
- `data`
- `training` 下的学习率、batch、设备、synthetic smoke 等

**系统角色：**
- 是 `tools/train.py` 默认读取的总入口配置。

**与三段式主线的关系：**
- 不改变模型结构本身。
- 它控制三段式主线在训练时如何被调用、在哪个设备上运行、用什么合成 batch 设置进行烟雾测试。

#### 3.2.4 `configs/data/dataset_example.yaml`

**文件用途：**
- 提供 Ultralytics 风格数据集 YAML 示例。

**主要配置：**
- `path`
- `train`
- `val`
- `names`

**系统角色：**
- 作为真实数据训练前的占位模板。
- 用于告诉后续开发者正式数据集配置应该长什么样。

**与三段式主线的关系：**
- 不参与模型结构。
- 只在将来接入真实训练时起到数据来源声明作用。

**额外提醒：**
- 这个文件目前明显不是最终版本，因为里面使用的是占位路径 `/absolute/path/to/dataset_root`。

### 3.3 训练入口与辅助文件

#### 3.3.1 `tools/train.py`

**文件用途：**
- 当前仓库的训练/烟雾测试命令行入口。

**主要函数：**
- `parse_args()`
- `synthetic_batch()`
- `run_synthetic_smoke()`
- `validate_real_run_inputs()`
- `main()`

**系统角色：**
- 串起完整执行流程：读取配置、构建模型、生成合成 batch、计算 loss、反向传播、执行 optimizer step。
- 如果不走 synthetic smoke，则只验证真实训练所需输入配置是否存在，而不会真正启动完整真实训练。

**与三段式主线的关系：**
- 它不实现三段结构本身。
- 但它是三段主线被真正跑起来的命令入口。

**额外提醒：**
- 这是**最小入口脚本**，不是完整 trainer。
- 后续若要正式接真实数据训练，这个文件大概率需要继续扩展。

#### 3.3.2 `amfe/models/common.py`

**文件用途：**
- 放共享基础层和通用检查函数。

**主要类 / 函数：**
- `ConvBNAct`
- `ResidualProjection`
- `DepthwiseSeparableConv`
- `ensure_feature_channels()`

**系统角色：**
- 作为 backbone 和 neck 的公共基础库。
- 负责减少重复代码，并强化 shape / channel contract 检查。

**与三段式主线的关系：**
- 不直接对应三段结构中的某一段。
- 但它为 backbone 和 neck 提供基础组件，属于重要辅助文件。

#### 3.3.3 `amfe/ultralytics_compat.py`

**文件用途：**
- 提供与 Ultralytics 的兼容导入层。

**主要内容：**
- 导入 `Detect`
- 导入 `v8DetectionLoss`
- 必要时安装极简 `cv2` stub

**系统角色：**
- 保证某些 headless / 依赖不完整环境下，测试和最小训练链路仍然能 import Ultralytics。

**与三段式主线的关系：**
- 直接服务于 **Ultralytics Detect Head** 的接入。
- 同时服务于 loss 的复用。

**额外提醒：**
- 这是典型的**兼容性辅助文件**，不是核心模型设计文件。
- 如果后续环境稳定、依赖完整，它的重要性会低于核心模型文件。

### 3.4 包导出与元信息文件

#### 3.4.1 `amfe/models/__init__.py` 与 `amfe/__init__.py`

**文件用途：**
- 统一导出公共 API。

**系统角色：**
- 让外部使用者可以更方便地导入 `AMFEBackbone`、`AMFNeck`、`build_amfe_detector` 等。

**与三段式主线的关系：**
- 不参与模型内部计算。
- 但定义了项目外部如何访问这些核心模块。

#### 3.4.2 `pyproject.toml` 与 `requirements.txt`

**文件用途：**
- 声明项目依赖、构建信息、pytest 配置。

**系统角色：**
- 保证仓库可安装、可测试。
- 声明对 `torch`、`torchvision`、`ultralytics`、`PyYAML` 等依赖的要求。

**与三段式主线的关系：**
- 不属于模型实现。
- 但没有这些文件，整套 backbone / neck / detect head 的代码很难在一致环境中运行。

### 3.5 测试文件

#### 3.5.1 `tests/test_backbone.py`

**文件用途：**
- 测试 backbone 子模块与整体 backbone 的 shape contract。

**主要验证内容：**
- `LEM` 输出 shape
- `DPSStem` 输出 shape
- `DEB` 输出 shape
- `LGCB` 输出 shape
- `MBFM` 输出 shape
- `AMFEBackbone` 输出 `F3 / F4 / F5` 的 shape

**系统角色：**
- 是 backbone 实验迭代时最先应该跑的保护网。

#### 3.5.2 `tests/test_neck.py`

**文件用途：**
- 测试 `AMFNeck` 的输出 shape。

**主要验证内容：**
- 输入 `(F3, F4, F5)` 后，是否正确得到 `(N3, N4, N5)`。

**系统角色：**
- 保证 neck 的输入输出接口稳定。

#### 3.5.3 `tests/test_detector.py`

**文件用途：**
- 测试 detector 级别的端到端行为。

**主要验证内容：**
- detector forward smoke test
- `forward_features()` 的 feature shape
- `loss + backward + optimizer step` 是否可运行
- YAML 构建路径是否可工作

**系统角色：**
- 是当前仓库最接近真实训练链路的自动化验证文件。

---

## 4. 训练执行链路说明

下面按“从命令启动到参数更新”顺序说明一次训练是如何跑起来的。当前仓库最标准的路径是：

```bash
python tools/train.py --config configs/train/train_default.yaml --synthetic-smoke
```

### 4.1 配置文件如何被读取

1. 命令行入口从 `tools/train.py` 的 `main()` 开始。
2. `parse_args()` 解析 `--config` 和 `--synthetic-smoke` 参数。
3. `main()` 调用 `load_yaml_config(args.config)` 读取 `configs/train/train_default.yaml`。
4. 训练配置中最关键的两个字段是：
   - `model_config`：指向 `configs/model/amfe_amf_yolo.yaml`
   - `data`：指向 `configs/data/dataset_example.yaml`
5. 如果当前走 `--synthetic-smoke`，后续主要使用训练配置和模型配置；数据集配置只在真实训练校验模式下检查。

### 4.2 模型如何被构建

1. `run_synthetic_smoke(config)` 被调用。
2. 其中执行 `build_model_from_yaml(config["model_config"])`。
3. `build_model_from_yaml()` 位于 `amfe/models/registry.py`，内部先调用 `load_yaml_config()` 读取模型 YAML。
4. 随后 `build_model_from_config()` 将 dict 转成 `AMFEModelConfig`。
5. 最终实例化 `AMFEYOLODetectionModel`。

这一步完成以后，一个完整的 detector 就被创建出来了，其中已经包含：

- `self.backbone = AMFEBackbone(...)`
- `self.neck = AMFNeck(...)`
- `self.detect = Detect(...)`

### 4.3 backbone 如何前向

在 `AMFEYOLODetectionModel.forward_features(x)` 中，会先调用：

```text
self.backbone(x)
```

其内部流程为：

1. 输入图像进入 `LEM`。
2. 进入 `DPSStem` 形成浅层共享特征 `S2`。
3. `MSB` 从 `S2` 提取语义特征 `C2 / C3 / C4 / C5`。
4. `ADB` 从 `S2` 提取细节特征 `D3 / D4`。
5. `LGCB` 从深层特征构造上下文特征 `G3 / G4 / G5`。
6. `MBFM` 分别融合这些分支，输出最终 backbone 特征：
   - `F3`
   - `F4`
   - `F5`

这些输出满足固定 stride 契约：
- `F3`: stride 8
- `F4`: stride 16
- `F5`: stride 32

### 4.4 neck 如何前向

`forward_features(x)` 在拿到 backbone 输出后，会继续调用：

```text
self.neck((F3, F4, F5))
```

其流程为：

1. `CAF` 分别将 `F3 / F4 / F5` 对齐到统一通道宽度。
2. 用 `TDSF` 做 top-down selective fusion：
   - 先由高层 `F5` 融合到中层
   - 再由中层继续融合到低层
3. 用 `BURF` 做 bottom-up refinement：
   - 从低层往上逐级细化
4. 最终得到：
   - `N3`
   - `N4`
   - `N5`

这些输出是 Detect head 的直接输入。

### 4.5 detect head 如何接入

在 `AMFEYOLODetectionModel.forward(x)` 中：

1. 先执行 `forward_features(x)` 得到 `(N3, N4, N5)`。
2. 然后执行：

```text
self.detect(list(features))
```

这里的 `self.detect` 来自 Ultralytics 原生 `Detect`。

额外还有一步很关键：

- 在模型初始化时，`_initialize_detect_head()` 会构造一个 dummy image。
- 它先跑一次 `forward_features(dummy)`。
- 再根据输出 feature map 的尺寸推导 stride。
- 然后把 stride 写入 `self.detect.stride`，并调用 `bias_init()`。

也就是说，当前仓库不是手工硬编码 Detect stride，而是通过一次实际 feature forward 推断出头部 stride。

### 4.6 loss 如何计算

在训练路径中，`run_synthetic_smoke()` 不会直接手写损失，而是调用：

```text
total_loss, loss_items = model.loss(batch)
```

其内部逻辑是：

1. `AMFEYOLODetectionModel.loss()` 先检查 batch 中是否包含：
   - `img`
   - `batch_idx`
   - `cls`
   - `bboxes`
2. 如果 `self.criterion` 还没有初始化，则通过 `init_criterion()` 创建 `v8DetectionLoss(self)`。
3. 若没有外部传入预测结果，则先执行 `self.forward(batch["img"])`。
4. 把预测结果和 batch 一起送入 Ultralytics 的 `v8DetectionLoss`。
5. 返回：
   - `loss_vector.sum()` 作为总损失
   - `loss_items` 作为损失分量

因此当前仓库的 loss 计算核心是**直接复用 Ultralytics**，而不是自定义检测损失。

### 4.7 backward 如何执行

在 `run_synthetic_smoke()` 中：

1. 先创建 SGD optimizer。
2. 调用 `optimizer.zero_grad(set_to_none=True)` 清空梯度。
3. 通过 `model.loss(batch)` 得到 `total_loss`。
4. 调用：

```text
total_loss.backward()
```

这一步会沿着：

**Detect Head -> AMF-Neck -> AMFE-Backbone**

的反方向，把梯度回传到整个模型参数。

### 4.8 optimizer step 如何完成

在 backward 完成后，`run_synthetic_smoke()` 继续调用：

```text
optimizer.step()
```

这会按照 optimizer 中配置的学习率、momentum、weight decay 更新参数。

当前默认使用的是：
- `torch.optim.SGD`
- 学习率来自 `training.lr0`
- `weight_decay` 来自训练配置

### 4.9 当前真实训练链路的限制

需要明确说明：

- 当前 `tools/train.py` 在非 synthetic 模式下，**不会真正启动完整真实数据训练**。
- 它只会检查：
  - 数据集 YAML 是否存在
  - 数据集根路径是否存在
- 然后抛出一个明确的 `RuntimeError`，说明真实数据驱动训练会在后续阶段接入。

这说明：

- 当前仓库已经打通了**训练机制的最小闭环**。
- 但尚未形成**完整的正式数据集训练框架**。

---

## 5. 测试体系说明

当前测试体系比较精简，但结构上是清晰的，主要分成 backbone、neck、detector 三层。

### 5.1 `tests/test_backbone.py`

**主要验证什么：**
- `LEM` 模块级 forward shape。
- `DPSStem` 模块级 forward shape。
- `DEB` 模块级 forward shape。
- `LGCB` 模块级 forward shape。
- `MBFM` 模块级 forward shape。
- `AMFEBackbone` 集成后输出 `F3 / F4 / F5` 的 shape 是否符合设计。

**为什么重要：**
- backbone 中分支较多，shape 最容易在改结构时出问题。
- 这是后续做 backbone 实验时的第一道保护。

### 5.2 `tests/test_neck.py`

**主要验证什么：**
- `AMFNeck` 是否能接受 `(F3, F4, F5)`。
- 是否能输出 `(N3, N4, N5)`。
- 输出通道数与尺度是否符合 neck 契约。

**为什么重要：**
- neck 是 backbone 与 detect head 之间的接口层。
- 一旦 neck 形状错了，detect head 接入就会马上出问题。

### 5.3 `tests/test_detector.py`

**主要验证什么：**
- 完整 detector 的 forward smoke test。
- `forward_features()` 返回的 neck 输出 shape。
- Detect stride 初始化是否符合 `(8, 16, 32)`。
- 使用 synthetic batch 是否能完成：
  - loss 计算
  - backward
  - optimizer step
- 是否可以从 YAML 正常构建模型。

**为什么重要：**
- 这是当前仓库里最接近真实训练执行链路的自动测试。
- 如果它失败，通常说明 backbone / neck / detect 三者之间至少有一处接口断裂。

### 5.4 synthetic smoke test 与 pytest 的关系

仓库里实际上存在两类“验证”：

1. **pytest 自动测试**
   - 通过 `tests/` 中的测试文件执行。
   - 更适合开发中频繁回归验证。

2. **命令行 synthetic smoke test**
   - 通过 `python tools/train.py --config ... --synthetic-smoke` 执行。
   - 更接近“模拟一次真正训练 step”的流程。

简单理解：
- `pytest` 偏向模块与接口正确性验证。
- synthetic smoke 偏向训练主链路闭环验证。

---

## 6. 后续开发时应该改哪些地方

这一部分非常重要，目的是告诉后来的研究者：如果你想改某一层，不要到处乱翻仓库，先看对的位置。

### 6.1 如果以后要改 backbone，主要看哪些文件

优先阅读和修改：

- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/backbone/lem.py`
- `amfe/models/backbone/dps_stem.py`
- `amfe/models/backbone/msb.py`
- `amfe/models/backbone/adb.py`
- `amfe/models/backbone/lgcb.py`
- `amfe/models/backbone/mbfm.py`
- `tests/test_backbone.py`

**建议理解顺序：**
- 先看 `amfe_backbone.py`，理解总装顺序。
- 再看各个子模块文件，理解每个 block 的角色。
- 改完以后先跑 `tests/test_backbone.py`。

### 6.2 如果以后要改 neck，主要看哪些文件

优先阅读和修改：

- `amfe/models/neck.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

**原因：**
- 当前 neck 的核心实现集中在一个文件里。
- 改 neck 后不仅要验证 neck 自己，还要验证 detect head 接口有没有被破坏。

### 6.3 如果以后要改模型注册或配置，主要看哪些文件

优先阅读和修改：

- `amfe/models/registry.py`
- `amfe/models/detector.py`
- `configs/model/amfe_amf_yolo.yaml`
- `configs/train/train_default.yaml`
- `amfe/models/__init__.py`
- `amfe/__init__.py`

**典型场景：**
- 新增模型配置字段。
- 修改默认类别数或 neck 通道数。
- 调整 build API。
- 改变外部导入方式。

### 6.4 如果以后要接入真实数据集和正式训练，主要看哪些文件

优先阅读和修改：

- `tools/train.py`
- `configs/data/dataset_example.yaml`
- `configs/train/train_default.yaml`
- `amfe/models/detector.py`
- `amfe/ultralytics_compat.py`
- `README.md`

**原因：**
- 当前真实数据训练尚未真正启动，`tools/train.py` 只是做输入校验后直接报错退出。
- 所以正式训练功能的扩展，大概率会以 `tools/train.py` 为第一修改点。
- 数据集 YAML 也需要从示例版升级为真实可用版本。

### 6.5 如果以后要增强测试体系，主要看哪些文件

优先阅读和修改：

- `tests/test_backbone.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

**建议后续补充方向：**
- 更细粒度的异常路径测试。
- 更完整的 synthetic batch 变体测试。
- 与真实 Ultralytics 数据流更接近的训练集成测试。

---

## 7. 新人阅读建议

下面给出一个面向“需要继续做实验的研究者”的推荐阅读顺序。目标不是最快看完，而是最快建立正确的系统心智模型。

### 7.1 第一阶段：先看总入口，建立全局图

建议先看：

1. `README.md`
2. `configs/model/amfe_amf_yolo.yaml`
3. `configs/train/train_default.yaml`
4. `tools/train.py`
5. `amfe/models/detector.py`

**为什么这样读：**
- `README.md` 能帮你先知道当前仓库已经做到什么程度。
- 两个配置文件可以帮助你理解系统是如何被参数化的。
- `tools/train.py` 告诉你一次运行是怎么启动的。
- `detector.py` 告诉你三段主线是如何被真正装配到一起的。

读完这一轮后，你应该先在脑中建立一个简单图：

**配置 -> 构建 detector -> backbone -> neck -> detect -> loss -> backward -> optimizer step**

### 7.2 第二阶段：再看核心模型实现

建议再看：

1. `amfe/models/backbone/amfe_backbone.py`
2. `amfe/models/backbone/lem.py`
3. `amfe/models/backbone/dps_stem.py`
4. `amfe/models/backbone/msb.py`
5. `amfe/models/backbone/adb.py`
6. `amfe/models/backbone/lgcb.py`
7. `amfe/models/backbone/mbfm.py`
8. `amfe/models/neck.py`
9. `amfe/models/common.py`

**为什么这样读：**
- 先看 backbone 总装，再下钻到每个子模块，能避免一开始陷入局部细节。
- 读完 backbone 再读 neck，会更容易理解 `F3 / F4 / F5 -> N3 / N4 / N5` 的接口关系。
- `common.py` 放在后面看更合适，因为此时你已经知道这些基础组件被哪些模块复用了。

### 7.3 第三阶段：最后看测试与兼容层

建议最后看：

1. `tests/test_backbone.py`
2. `tests/test_neck.py`
3. `tests/test_detector.py`
4. `amfe/models/registry.py`
5. `amfe/ultralytics_compat.py`
6. `configs/data/dataset_example.yaml`

**为什么这样读：**
- 测试文件可以帮助你反过来确认“系统认为正确的输入输出契约是什么”。
- `registry.py` 属于构建辅助层，理解它能帮助你改配置或扩展 build 流程。
- `ultralytics_compat.py` 属于兼容辅助层，通常不是先读的重点，但在排查环境问题时很重要。
- `dataset_example.yaml` 是未来真实训练的起点，但当前不是系统主逻辑核心。

### 7.4 一句话版推荐阅读顺序

如果只能记一个最短顺序，可以记成：

**README -> train config -> train.py -> detector.py -> amfe_backbone.py -> neck.py -> 各 backbone 子模块 -> tests**

这是因为：
- 先看“系统怎么跑”。
- 再看“模型怎么装”。
- 最后看“每个模块怎么实现”和“系统如何验证自己没坏”。

---

## 8. 核心文件与辅助文件的区分建议

为了避免新同学把精力平均分配到不重要的地方，最后再给一个简化判断。

### 8.1 核心模型文件

优先级最高：

- `amfe/models/backbone/amfe_backbone.py`
- `amfe/models/backbone/lem.py`
- `amfe/models/backbone/dps_stem.py`
- `amfe/models/backbone/msb.py`
- `amfe/models/backbone/adb.py`
- `amfe/models/backbone/lgcb.py`
- `amfe/models/backbone/mbfm.py`
- `amfe/models/neck.py`
- `amfe/models/detector.py`

这些文件直接定义了：

**AMFE-Backbone -> AMF-Neck -> Ultralytics Detect Head**

### 8.2 重要但偏辅助的文件

第二优先级：

- `amfe/models/registry.py`
- `amfe/models/common.py`
- `tools/train.py`
- `tests/test_backbone.py`
- `tests/test_neck.py`
- `tests/test_detector.py`

这些文件不一定直接定义模型结构，但决定了模型是否容易构建、运行、验证和迭代。

### 8.3 兼容性、占位性、说明性文件

阅读优先级可以稍后：

- `amfe/ultralytics_compat.py`：兼容性辅助文件
- `configs/data/dataset_example.yaml`：占位/示例配置文件
- `amfe/__init__.py`、`amfe/models/__init__.py`：导出辅助文件
- `pyproject.toml`、`requirements.txt`：环境与安装文件

这些文件也重要，但通常不是你理解模型结构本身的第一站。

---

## 9. 总结

如果用一句话概括当前仓库，可以写成：

> 这是一个已经把 **AMFE-Backbone -> AMF-Neck -> Ultralytics Detect Head** 跑通，并通过 synthetic smoke test 验证了最小训练闭环，但真实数据正式训练入口仍待后续扩展的研究型目标检测代码库。

对于后续实验开发者来说，最值得优先掌握的是三件事：

1. `amfe/models/detector.py` 如何把三段主线接起来。
2. `amfe/models/backbone/` 和 `amfe/models/neck.py` 如何定义核心特征流。
3. `tools/train.py` 与 `tests/` 如何验证当前修改没有破坏训练闭环。

只要先掌握这三层，你后续无论是改 backbone、改 neck、调配置，还是准备接真实数据集，都会更稳。
