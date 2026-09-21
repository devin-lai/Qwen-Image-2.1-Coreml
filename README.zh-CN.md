# Qwen-Image-2.1 的 Core ML 转换：在 Apple 芯片 Mac 上本地生成图像

**在 Mac 上使用 Core ML 生成 1024 × 1024 图像。**

[English / 完整技术文档](README.md) · [下载模型](https://huggingface.co/devin-lai/Qwen-Image-2.1-Coreml) · [性能记录](benchmarks/README.md) · [提交硬件测试结果](https://github.com/devin-lai/Qwen-Image-2.1-Coreml/issues/new?template=benchmark_report.yml)

本项目提供 Qwen-Image-2.1 的 FP16 Core ML 模型、Python 推理代码，以及四组预计算提示词嵌入。
下载模型并安装依赖后，可以使用这些嵌入在本地生成图像，无需云端推理 API。
模型权重托管在 Hugging Face；GitHub 仓库包含代码、示例和原始测试记录。

在 **M5 MacBook Pro、32 GB 统一内存**的已记录测试中，Core ML 的单步去噪耗时中位数比
PyTorch bf16/MPS **快 2.4–2.6 倍**。这是去噪步骤的比较，不是完整生成流程的加速比。
40 步去噪耗时约 221–250 秒，不含文本编码、模型加载及提示词前缀计算。
[测试条件与计算依据](benchmarks/README.md#performance-summary)。

| 英文招牌 | 中文招牌 | 动物 | 插画 |
| :---: | :---: | :---: | :---: |
| ![雨夜霓虹招牌](assets/gallery/neon_sign.jpg) | ![清风茶舍木质招牌](assets/gallery/tea_house.jpg) | ![雪地中的狐狸](assets/gallery/fox.jpg) | ![龟背竹植物插画](assets/gallery/botanical.jpg) |

*实际 Core ML 输出：1024 × 1024，40 步，随机种子 42。*

## 快速开始

需要 Apple 芯片 Mac、macOS 15 或更新版本，以及 Python 3.11–3.13。
六个模型包共 14.74 GB，还需为依赖和 Core ML 编译预留空间。
已测试硬件为 M5 MacBook Pro、32 GB 统一内存；较小内存配置的需求尚未验证。

```bash
git clone https://github.com/devin-lai/Qwen-Image-2.1-Coreml.git
cd Qwen-Image-2.1-Coreml
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_models.py
python generate.py --out neon.png
```

模型下载到 `models/`。默认使用仓库内的霓虹招牌提示词嵌入。
首次运行还需编译、加载模型，并计算提示词 KV 缓存，因此比后续运行更慢。

生成中文招牌示例：

```bash
python generate.py --prompt-embeds assets/prompts/tea_house.npz --out tea.png
```

对应提示词：`一块木质招牌上写着「清风茶舍」，暖黄灯笼，雨后的青石板街`。
其他示例为 `assets/prompts/fox.npz` 和 `assets/prompts/botanical.npz`。

## 使用自己的提示词

自定义文本需要单独的上游文本编码器，涉及额外下载和内存开销。
先安装可选依赖并编码，再用 Core ML 生成图像：

```bash
python -m pip install '.[torch-reference]'
python encode_prompt.py "一座海边灯塔，暴风雨，长曝光摄影" --name lighthouse
python generate.py --prompt-embeds assets/prompts/lighthouse.npz --out lighthouse.png
```

`encode_prompt.py` 通过 Diffusers 加载 Qwen3-VL 文本编码器。
文本编码器未转换为 Core ML；可使用 `--checkpoint` 指向本地上游检查点。
提示词 KV 缓存保存在 `.cache/`，每组约 67 MB。

## 适用范围

- 固定形状：批量大小 1、1024 × 1024、最多 64 个编码后的提示词 token。
- 当前支持文生图；推理循环未实现图像编辑或 classifier-free guidance。
- 默认使用 `cpu_and_gpu`。性能数据来自这一设置，不代表 Neural Engine 的性能。
- 提供 Python 推理代码，没有附带 Swift 应用，尚未测试 iOS 运行。
- 其他 Apple 芯片和内存配置需要实测，欢迎提交成功或失败记录。

## 参与项目

请通过[硬件测试表单](https://github.com/devin-lai/Qwen-Image-2.1-Coreml/issues/new?template=benchmark_report.yml)
分享芯片、内存、macOS 版本和原始计时文件。
[贡献指南](CONTRIBUTING.md)说明了测试命令、问题反馈和 pull request 要求。
如果项目对你有帮助，欢迎 Star 收藏，并向正在探索 Mac 本地图像生成的朋友分享。

## 许可与致谢

Core ML 转换与推理代码由 Devin Lai 维护，基于 Qwen 模型。
推理代码采用 [Apache-2.0](LICENSE)；改编代码的归属见 [NOTICE](NOTICE)。
模型权重遵循 [Qwen Research License](LICENSE-QWEN)，用于非商业研究与评估；
商业使用需要另行获得上游许可。代码的 Apache 许可不改变模型权重的许可。

本文是中文快速上手指南；Python API、完整性能表格和验证方法请参阅[英文 README](README.md)。
