# AMFE-Backbone + AMF-Neck 实验代码实现需求文档（给 Codex）

## 1. 任务目标

请基于 **Ultralytics YOLO 检测框架** 实现一个可训练、可验证、可推理的自定义目标检测模型，模型总结构为：

**Input → LEM → DPS Stem → AMFE-Backbone（MSB + ADB + LGCB + MBFM）→ AMF-Neck（CAF + TDSF + BURF）→ Ultralytics 原生 Detect Head**

本项目的核心要求：

1. **只改 backbone 和 neck，不改 Detect Head 的原理与结构。**
2. **训练、验证、推理、导出流程尽量复用 Ultralytics 现有能力。**
3. **代码必须工程化、可维护、可复现实验，不接受一次性脚本式实现。**
4. **实现优先级高于花哨设计，必须保证能稳定跑通。**
5. **不要把旧方案中的“去雾分支”重新加回来。当前方案不是图像复原网络，而是检测导向的特征增强网络。**

---

## 2. 最终固定网络方案

### 2.1 总体结构

```text
Input Image
→ LEM
→ DPS Stem
→ Shared Shallow Feature S2
→ Main Semantic Branch (MSB, ResNet50-based)
→ Auxiliary Detail Branch (ADB)
→ Lightweight Global Context Block (LGCB)
→ Multi-Branch Fusion Module (MBFM)
→ F3 / F4 / F5
→ AMF-Neck
→ N3 / N4 / N5
→ Ultralytics Detect Head
```

### 2.2 Backbone 模块命名（固定）

- `AMFEBackbone`: Asymmetric Multi-branch Feature Enhancement Backbone
- `LEM`: Lightweight Enhancement Module
- `DPSStem`: Detail-Preserving Stem
- `MSB`: Main Semantic Branch
- `ADB`: Auxiliary Detail Branch
- `DEB`: Detail Enhancement Block
- `LGCB`: Lightweight Global Context Block
- `MBFM`: Multi-Branch Fusion Module
- `CDG`: Context-Detail Gate

### 2.3 Neck 模块命名（固定）

- `AMFNeck`: Attention-guided Multi-scale Fusion Neck
- `CAF`: Channel Alignment Fusion
- `TDSF`: Top-Down Selective Fusion
- `BURF`: Bottom-Up Refinement Fusion
- `SPG`: Semantic Prior Gate
- `DPG`: Detail Prior Gate
- `MCA`: Multi-kernel Context Aggregator

### 2.4 Head（固定）

- 使用 **Ultralytics 原生 Detect Head**。
- 不改 Detect Head 的解耦/耦合方式、不改损失函数接口、不自定义新的检测头。
- 只做输入特征层级和通道适配。

---

## 3. 张量层级与尺度定义（以输入 640×640 为例）

### 3.1 前端与主干

- 输入：`[B, 3, 640, 640]`
- `LEM` 输出：`[B, 32, 640, 640]`
- `DPSStem` 输出 `S2`：`[B, 64, 160, 160]`（stride = 4）

### 3.2 MSB 主语义分支（ResNet50）

- `C2`: `[B, 256, 160, 160]`
- `C3`: `[B, 512, 80, 80]`
- `C4`: `[B, 1024, 40, 40]`
- `C5`: `[B, 2048, 20, 20]`

### 3.3 ADB 辅助细节分支

- `D3`: `[B, 256, 80, 80]`
- `D4`: `[B, 512, 40, 40]`

### 3.4 全局上下文分支

- `G5`: `[B, 512, 20, 20]`
- `G4`: `[B, 512, 40, 40]`
- `G3`: `[B, 512, 80, 80]`

### 3.5 Backbone 最终输出

- `F3`: `[B, 512, 80, 80]`
- `F4`: `[B, 1024, 40, 40]`
- `F5`: `[B, 2048, 20, 20]`

### 3.6 Neck 输出

CAF 统一通道到 `256`：

- `L3`: `[B, 256, 80, 80]`
- `L4`: `[B, 256, 40, 40]`
- `L5`: `[B, 256, 20, 20]`

AMF-Neck 输出：

- `N3`: `[B, 256, 80, 80]`
- `N4`: `[B, 256, 40, 40]`
- `N5`: `[B, 256, 20, 20]`

Detect Head 直接接收 `N3/N4/N5`。

---

## 4. Backbone 详细实现要求

## 4.1 LEM

### 固定结构

```text
Input
→ Conv3×3(s=1, c=32)
→ DWConv3×3(s=1, c=32)
→ PWConv1×1(s=1, c=32)
→ Residual Add
→ Conv3×3(s=1, c=32)
→ LEM Output
```

### 要求

- 不允许下采样。
- 这是检测导向浅层增强模块，不是去雾重建模块。
- 若输入通道与残差相加不一致，需要用 1×1 shortcut 对齐。

---

## 4.2 DPS Stem

### 固定结构

```text
LEM Output
→ Conv3×3(s=2, c=64)
→ Conv3×3(s=1, c=64)
→ Conv3×3(s=1, c=64)
→ Conv3×3(s=2, c=64)
→ S2
```

### 要求

- 明确替代原始 ResNet50 的 `7×7 / 2 + maxpool / 2` 前端。
- 不允许再使用原始大核 + maxpool stem。
- 保持 stride=4 的浅层共享特征输出。

---

## 4.3 MSB（主语义分支）

### 实现策略

- 使用 `torchvision.models.resnet50` 的 Bottleneck 结构实现主干。
- 允许加载 ImageNet 预训练权重，但只用于 **MSB 主分支**。
- 前端 stem 不使用 torchvision 原始 stem；请改为从 `S2` 接入后续 stage。

### 输出要求

- 返回 `C2/C3/C4/C5`。
- `C3/C4/C5` 必须用于后续融合。

### 注意

- 需要正确处理与原始 ResNet50 stem 不同带来的权重加载问题。
- 若无法直接加载完整预训练权重，至少要保证 layer1~layer4 能正确部分加载。

---

## 4.4 ADB（辅助细节分支）

### 设计原则

- 从 `S2` 分出。
- 只做到 `1/16`，不继续做到 `1/32`。
- 这是轻量细节补偿分支，不是第二个主干。

### DEB 固定定义

```text
Input
→ DWConv3×3(stride=s)
→ PWConv1×1(out_channels)
→ DWConv3×3(stride=1, dilation=2)
→ PWConv1×1(out_channels)
→ Residual/Shortcut
→ Output
```

### 分支结构

```text
S2
→ DEB-1(stride=2, out=256) → D3
→ DEB-2(stride=2, out=512) → D4
```

### 要求

- shortcut 分支必须自动适配 stride 与通道不一致情况。
- 不允许堆叠重型注意力模块（如非局部、Transformer block、大型 CBAM 串联等）。
- 保持轻量化。

---

## 4.5 LGCB（轻量全局上下文块）

### 固定结构

```text
C5
→ Conv1×1(2048→512)
→ DWConv1×7
→ DWConv7×1
→ GlobalAvgPool
→ Conv1×1
→ Sigmoid Gate
→ Channel Reweight
→ Conv1×1
→ G5
```

### 输出

- `G5` from `C5`
- `G4 = upsample(G5 → size(C4))`
- `G3 = upsample(G5 → size(C3))`

### 要求

- 这是深层上下文补充分支，不允许从输入或 S2 重新单独开一条全局分支。
- 必须保持轻量，使用 depthwise separable 风格。

---

## 4.6 MBFM（多分支融合模块）

### 融合原则

- 主分支特征是锚点。
- 细节分支提供局部补偿。
- 全局分支提供语义校正。
- 使用轻量门控 `CDG`，不使用重型通道+空间双注意力堆叠。

### CDG 定义

```text
Input: [c, d, g]
→ Concat
→ Conv1×1
→ Sigmoid
→ Gate M
```

### F3 融合

```text
D3 → AlignConv1×1 → D3'
G3 → AlignConv1×1 → G3'
M3 = CDG(C3, D3', G3')
F3 = Conv3×3([C3, M3⊙D3', (1-M3)⊙G3']) + C3
```

### F4 融合

```text
D4 → AlignConv1×1 → D4'
G4 → AlignConv1×1 → G4'
M4 = CDG(C4, D4', G4')
F4 = Conv3×3([C4, M4⊙D4', (1-M4)⊙G4']) + C4
```

### F5 融合

```text
G5 → AlignConv1×1 → G5'
F5 = Conv3×3([C5, G5']) + C5
```

### 要求

- `F3/F4/F5` 输出通道分别保持与 `C3/C4/C5` 一致：512/1024/2048。
- 必须写清楚 shape assert。
- 融合后残差保留主分支。

---

## 5. AMF-Neck 详细实现要求

## 5.1 总体结构

```text
F3/F4/F5
→ CAF
→ TDSF (top-down)
→ BURF (bottom-up)
→ N3/N4/N5
```

### 核心思想

- Top-down：高层语义先验引导低层融合。
- Bottom-up：低层细节先验反向校正高层融合。
- 不复制 backbone 里的“分支增强”逻辑，而专注跨尺度融合。

---

## 5.2 CAF

### 固定要求

```text
L3 = Conv1×1(F3 → 256)
L4 = Conv1×1(F4 → 256)
L5 = Conv1×1(F5 → 256)
```

### 要求

- 只做通道对齐，不附加复杂注意力。
- 默认输出通道统一为 256，但要求做成可配置参数。

---

## 5.3 SPG（Semantic Prior Gate）

### 定义

```text
Input: high-level feature
AvgPool + MaxPool
→ shared MLP or 1×1 conv bottleneck
→ Sigmoid
→ channel weight
```

### 要求

- 输出 shape 可广播到 `[B, C, H, W]`。
- 保持轻量，可采用 SE 风格实现。

---

## 5.4 DPG（Detail Prior Gate）

### 定义

```text
Input: low-level feature
DWConv3 + DWConv5 + DWConv7
→ sum or concat
→ Conv1×1
→ Sigmoid
→ spatial weight
```

### 要求

- 这是空间细节先验，不要实现成纯通道注意力。
- 输出 shape 需可广播到输入特征图。

---

## 5.5 TDSF（Top-Down Selective Fusion）

### 固定计算流程

```text
xc = x_low + upsample(x_high)
wc = SPG(x_high)
ws = DPG(x_low)
y  = Conv3×3(ws ⊙ (wc ⊙ xc)) + x_low
```

### 输出

- `T4 = TDSF(L4, Up(L5))`
- `T3 = TDSF(L3, Up(T4))`

### 要求

- 残差基底必须保留 `x_low`。
- 使用 nearest 或 bilinear 上采样均可，但整个项目需统一。
- 代码中写成独立模块，不要把逻辑散落在 forward 主函数里。

---

## 5.6 BURF（Bottom-Up Refinement Fusion）

### 固定计算流程

```text
xc = downsample(x_low_refined) + x_high
ws = DPG_B(x_low_refined)
wc = SPG_B(x_high)
y  = Conv3×3(wc ⊙ (ws ⊙ xc)) + x_high
```

### 输出

- `B4 = BURF(Down(T3), T4)`
- `B5 = BURF(Down(B4), L5)`

### 要求

- downsample 默认使用 `Conv3×3(stride=2)`，不要直接 `maxpool`。
- `DPG_B` 与 `SPG_B` 可以复用 `DPG`/`SPG` 的实现，但需写清楚是否复用同一模块类。
- 输出为：
  - `N3 = T3`
  - `N4 = B4`
  - `N5 = B5`

---

## 6. Head 与 Ultralytics 集成要求

## 6.1 Head 要求

- 直接使用 Ultralytics 的 Detect Head。
- 不改 head 结构。
- 不改损失函数定义。
- 不改正负样本分配逻辑。

## 6.2 集成要求

实现方式优先选择以下策略：

### 推荐方案（优先）

在 **Ultralytics 源码仓库/可编辑安装环境** 中新增自定义模块与自定义模型定义，直接复用其：

- model parser
- DetectionModel / trainer
- Detect head
- loss
- dataloader
- augmentations
- evaluation pipeline

### 不推荐方案

- 完全重写训练框架
- 自己重写 Detect Head 和 loss
- 自己写一套新的 dataloader/trainer 替代 Ultralytics

## 6.3 自定义模型要求

需要提供一个完整可实例化模型，例如：

- `AMFEYOLODetectionModel`

并保证支持：

- `train`
- `val`
- `predict`
- `export`（至少 ONNX；TensorRT 可选）

---

## 7. 建议代码目录结构（必须工程化）

```text
project_root/
├─ README.md
├─ requirements.txt
├─ pyproject.toml 或 setup.py
├─ configs/
│  ├─ model/
│  │  └─ amfe_amf_yolo.yaml
│  ├─ data/
│  │  └─ dataset_example.yaml
│  └─ train/
│     └─ train_default.yaml
├─ models/
│  ├─ backbone/
│  │  ├─ lem.py
│  │  ├─ dps_stem.py
│  │  ├─ deb.py
│  │  ├─ lgcb.py
│  │  ├─ mbfm.py
│  │  └─ amfe_backbone.py
│  ├─ neck/
│  │  ├─ gates.py
│  │  ├─ tdsf.py
│  │  ├─ burf.py
│  │  └─ amf_neck.py
│  ├─ model.py
│  └─ registry.py
├─ tools/
│  ├─ train.py
│  ├─ val.py
│  ├─ predict.py
│  ├─ export.py
│  └─ profile_model.py
├─ tests/
│  ├─ test_backbone_shapes.py
│  ├─ test_neck_shapes.py
│  ├─ test_model_forward.py
│  └─ test_ultralytics_integration.py
└─ docs/
   └─ architecture.md
```

### 强制要求

- 所有模块分文件实现。
- 不允许把所有代码堆在一个 `.py` 文件中。
- 必须包含最基本的测试。

---

## 8. 代码成熟度要求（这是硬要求）

Codex 必须产出 **成熟可维护代码**，至少满足以下标准：

### 8.1 代码规范

- Python 代码带类型注解。
- 每个模块和关键函数有 docstring。
- 关键 shape 处有断言或显式检查。
- 不允许神秘常数散落；全部通过配置或类参数传入。

### 8.2 可复现

- 提供随机种子设置。
- 训练参数写入配置文件。
- 支持恢复训练（resume）。
- 保存 best / last checkpoint。

### 8.3 鲁棒性

- 对输入尺寸、通道数不匹配给出明确报错。
- 对预训练权重加载失败给出清晰提示。
- 对缺失数据路径给出清晰提示。

### 8.4 测试

至少实现以下测试：

1. `AMFEBackbone` forward shape 测试
2. `AMFNeck` forward shape 测试
3. 整网 forward shape 测试
4. 与 Ultralytics trainer 的最小集成测试
5. 单张随机输入可正常完成 loss 前向图构建

### 8.5 可读性

- 模块命名与本需求文档保持一致。
- 不允许随意改名。
- 注释要解释“为什么这样做”，不是只解释“做了什么”。

---

## 9. 环境要求与安装要求

### 9.1 推荐基础环境

- OS：Ubuntu 22.04 LTS（首选） / Windows 11（可选）
- Python：**3.10**
- CUDA：按本机 GPU 驱动兼容的 **PyTorch 官方稳定版** 选择
- GPU：建议 NVIDIA 显卡，显存 >= 12GB；实验更稳建议 16GB+

PyTorch 官方安装页说明，当前稳定版要求 **Python 3.10 或更高**。citeturn405269search6turn405269search12

Ultralytics 官方文档说明，可通过 `pip install ultralytics` 安装稳定版，也支持直接克隆源码仓库；本项目为了便于加入自定义模块，建议使用 **固定版本源码 + 可编辑安装** 的方式，而不是漂移的未固定版本。citeturn405269search0turn405269search2

Ultralytics 官方文档也提供了 Python API 和训练能力，并支持高级自定义；本项目应最大限度复用其现有训练器，而不是重写训练引擎。citeturn405269search5turn405269search8

### 9.2 推荐库

#### 必装

- `torch`
- `torchvision`
- `ultralytics`
- `numpy`
- `opencv-python`
- `PyYAML`
- `tqdm`
- `matplotlib`
- `tensorboard`
- `pytest`
- `rich`

#### 可选

- `onnx`
- `onnxruntime-gpu` 或 `onnxruntime`
- `thop`（FLOPs/Params 粗略统计）
- `albumentations`（若与 Ultralytics 默认增强存在兼容方案）

### 9.3 推荐 requirements.txt（先按此生成）

```text
python==3.10
ultralytics
torch
torchvision
numpy
opencv-python
PyYAML
tqdm
matplotlib
tensorboard
pytest
rich
onnx
thop
```

### 9.4 推荐安装流程

```bash
conda create -n amfe_yolo python=3.10 -y
conda activate amfe_yolo

# 根据本机 CUDA/驱动，在 PyTorch 官网选择官方稳定版安装命令
# 然后再安装其余库

pip install -U pip
pip install ultralytics opencv-python PyYAML tqdm matplotlib tensorboard pytest rich onnx thop
```

### 9.5 许可证提醒

Ultralytics 官方文档提醒其框架采用 **AGPL-3.0** 许可证；若基于其源码进行修改、分发或公开部署，需要注意许可证合规。citeturn405269search10

---

## 10. 训练与实验要求

## 10.1 数据集接口

- 优先兼容 Ultralytics YOLO 检测数据集格式。
- 提供 `dataset_example.yaml` 示例。
- 路径、类别名、类别数不得硬编码。

## 10.2 默认训练超参数（首版可先给默认值）

建议默认值：

- `imgsz = 640`
- `epochs = 300`
- `batch = 8`（视显存可调）
- `workers = 8`
- `optimizer = SGD` 或 `AdamW`（默认保留 Ultralytics 方案即可）
- `lr0 = 0.01`（若沿用 SGD）
- `weight_decay = 5e-4`
- `device = 0`
- `amp = True`
- `cache = False`

## 10.3 训练脚本要求

需要提供：

```bash
python tools/train.py --config configs/train/train_default.yaml
python tools/val.py --weights path/to/best.pt --data configs/data/dataset_example.yaml
python tools/predict.py --weights path/to/best.pt --source path/to/images
python tools/export.py --weights path/to/best.pt --format onnx
```

## 10.4 实验记录要求

- 自动保存实验配置副本。
- 自动保存模型结构摘要。
- 记录 Params、FLOPs（若可行）、mAP50、mAP50-95、推理速度。

---

## 11. 实现策略要求（非常重要）

Codex 实现时必须遵守以下策略：

### 11.1 先跑通，再优化

第一阶段先完成：

1. 模型结构搭建成功
2. forward shape 全对
3. 可接入 Ultralytics Detect Head
4. 能在随机数据和小样本上完成 1 epoch 训练

第二阶段再做：

- 预训练权重加载细化
- 更完整的 profile
- 导出功能补强
- 性能优化

### 11.2 不要过度设计

- 不要引入 Transformer
- 不要引入额外检测头
- 不要再加 P2 分支
- 不要改 loss
- 不要再新增第三个创新模块

### 11.3 允许做的合理工程补充

- BN + SiLU/ReLU 的标准封装
- Conv / DWConv / PWConv 基础模块封装
- shape debug 工具
- model summary 工具
- 参数统计脚本

---

## 12. 交付物要求

Codex 最终必须交付：

1. 完整源码
2. `requirements.txt`
3. `README.md`
4. 可运行的训练/验证/推理命令示例
5. 最低限度测试文件
6. 模型配置文件
7. 一个最小可跑通 demo

---

## 13. 验收标准（必须满足）

### 必须全部通过

- [ ] `AMFEBackbone` 单独 forward 正常
- [ ] `AMFNeck` 单独 forward 正常
- [ ] 整网输出 `N3/N4/N5` 尺度正确
- [ ] Detect Head 能成功接入
- [ ] 在 Ultralytics 训练流程下能跑通至少 1 epoch
- [ ] 能保存 best/last 权重
- [ ] 能执行验证与推理
- [ ] 至少有 4 个基础测试通过
- [ ] README 中说明清楚环境、安装、训练、验证、推理步骤

### 加分项（可选）

- [ ] ONNX 导出可用
- [ ] TensorRT 导出尝试
- [ ] 支持 profile 与可视化
- [ ] 支持从 torchvision ResNet50 加载预训练权重

---

## 14. 给 Codex 的最后指令

请按以下原则实现：

1. **以“能稳定训练”为最高优先级。**
2. **所有模块命名、层级、接口必须严格遵守本需求文档。**
3. **不要擅自改动 backbone / neck 的核心设计。**
4. **如果某个实现细节存在工程不确定性，优先选择最稳、最常规、最容易维护的方案。**
5. **禁止为了追求所谓新颖性而引入本需求文档之外的大型模块。**
6. **代码必须像可以长期维护的研究代码，而不是一次性比赛脚本。**

