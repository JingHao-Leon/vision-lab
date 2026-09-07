# vision-lab · ViT 与 DDPM 从零实现

**不看 timm、不看 diffusers，亲手写出 Vision Transformer 和扩散模型的每一个模块——在 RTX 3090 上用真实数据集训练，给出可复现的实测指标与生成样本。**

[![tests](https://img.shields.io/badge/tests-9%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 实测结果（RTX 3090 24GB）

| 模型 | 数据集 | 参数量 | 训练配置 | 实测结果 | 耗时 |
|---|---|---|---|---|---|
| **ViT**（从零） | CIFAR-10 | 4.77M | 40 epochs · bs256 · RandAugment+RandomErasing · bf16 AMP | **测试精度 83.52%** | 18.5 min |
| **DDPM**（从零） | MNIST | 2.49M | 12 epochs · 400 步加噪 | ** ancestral 采样直接出可辨数字**（下图） | 4.8 min |

<div align="center">
  <img src="docs/ddpm_samples.png" width="384" alt="DDPM 采样的 32 个手写数字（4×8 网格）"/>
  <p><sub>DDPM  ancestral 采样 · 仅训练 12 epoch · 400 去噪步 · 无任何预训练权重</sub></p>
</div>

> 两个诚实标注：① ViT 83.5% 是 40 epoch 的成绩——从零训练的 ViT 要到 90%+ 通常需要 100-200 epochs 或更强蒸馏增广（DeiT 式），本仓库保留这一诚实数字并把延长训练列为 Roadmap；② DDPM 的样例是 12 epoch 的快速收敛版，偶尔仍有笔画噪声——ε-MSE 收敛到 0.030（完整 JSON 在 `docs/`）。

原始数据：`docs/vit_cifar10.json` · `docs/ddpm_mnist.json`（复现命令见下）。

## 两个模型，两种理解深度

| 模块 | 内容 | 关键实现细节 |
|---|---|---|
| `vision_lab/vit.py` | ViT 全模块 | Conv2d(patch,stride)=patch embedding；pre-norm 块；[class] token + learned position；分类只读 class 位 |
| `vision_lab/ddpm.py` | DDPM 全流程 | 线性 β schedule；闭式前向 q(x_t\|x_0)；ε-prediction 训练目标；FiLM 式时间注入的 Residual 块；8×8 分辨率自注意力；ancestral 采样 |
| `train_vit.py` | CIFAR-10 训练 | RandAugment + RandomErasing、AdamW+warmup cosine、bfloat16 AMP、每 epoch 出测试精度 |
| `train_ddpm.py` | MNIST 训练+采样 | 400 步加噪/去噪；32 数字 4×8 网格 png 直观验收 |

## 快速开始

```bash
uv sync
uv run pytest                       # 9 passed（CPU 即可，含闭式前向过程数学验证）

# GPU 训练（3090 实测时长见 README 顶部）
uv run python train_vit.py          # CIFAR-10 分类
uv run python train_ddpm.py         # MNIST 生成
```

## 测试为什么有意义

- `test_q_sample_closed_form_matches_definition`：零噪声时 x_t 必须精确等于 √ᾱ·x₀——扩散前向过程的闭式解不是"大约对"
- `test_vit_cls_token_is_used`：扰动 class token 必须改变输出——防止"摆设 token"式错误实现
- `test_unet_output_shape_any_input_size`：28×28（MNIST）与 32×32（CIFAR）双尺寸通过——UNet 的上下采样数学对奇数尺寸也成立
- `test_sampling_starts_from_noise_ends_bounded`：采样输出有界 [-1,1]—— ancestrsal 采样数值稳定性的最低保证

## 🛣 Roadmap

- [ ] ViT 延长训练（100+ epochs）冲击 90%+，附训练曲线
- [ ] Classifier-free guidance 引导采样
- [ ] Grad-CAM 可解释性模块

## License

MIT
