# TRIG-Bench (Text Relevance In Geo-localization)

**评估视觉语言模型 (VLMs) 在面对对抗性文本攻击时，其地理定位任务的鲁棒性表现。**

TRIG-Bench 提供了一套基于 **LLM (Qwen3-VL)** 和 **ComfyUI** 的全自动化对抗样本生成与评估流水线。它能生成具有高度欺骗性的“幻觉文本”，并将其逼真地融入街景图片中，从而精确测量模型在不同语义干扰下的定位偏差。

---

## 🚀 核心架构

本基准测试包含三个核心阶段：

1.  **攻击生成 (Attack Generation)**
    *   **引擎**: `Qwen/Qwen3-VL-30B-A3B-Thinking`
    *   **策略**: 根据原始图片内容，智能生成三种类型的干扰文本：
        *   🔤 **Similar**: 形似/音似词（如 McDonald's -> McDonnel's）。
        *   🎲 **Random**: 随机无关词。
        *   😈 **Adversarial**: 语义相反或误导性强的词（如将 "Stop" 改为 "Go"，或地名误导）。

2.  **图像合成 (Image Synthesis)**
    *   **引擎**: **ComfyUI** (Local API)
    *   **技术**: 利用 VLM 指令编辑能力 (`image_qwen_image_edit` 工作流)，通过 Prompt 引导将干扰文本自然地“生长”在图片中，保持光影和透视的一致性。

3.  **效能评估 (Evaluation with Paper Metrics)**
    *   **指标**:
        *   **WLA (Weighted Localization Accuracy)**: 分级加权定位精度 (1km - 2500km 多尺度)。
        *   **TBS (Text Bias Score)**: 文本偏差分数 ($Error_{Adv} - Error_{Clean}$)，量化干扰造成的额外误差。
        *   **TFR (Trap Fall Rate)**: 陷阱命中率（实验性）。

---

## 🛠️ 环境要求

*   **Python 3.10+**
*   **核心依赖**: `openai` (vLLM client), `torch`, `matplotlib`, `seaborn`, `folium` (见 `requirements.txt`)
*   **服务依赖**:
    *   **ComfyUI**: 需在本地 `127.0.0.1:8188` 启动，并安装 Qwen-Image-Edit 相关工作流节点。
    *   **vLLM**: 需部署 `Qwen/Qwen3-VL-30B-A3B-Thinking` 模型，默认端口 `8001`。

## 📦 安装

```bash
git clone https://github.com/inorganicwriter/TRIG-Bench.git
cd TRIG-Bench
pip install -r requirements.txt
```

## 🖥️ Server Deployment

For deploying the Qwen3-VL vLLM service, you can use the following command (auto TP=2 for 30B model):

```bash
vllm serve "/home/nas/lsr/huggingface/Qwen/Qwen3-VL-30B-A3B-Thinking" \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key qwen-local-key
```

---

## 📖 使用指南

### 第一步：生成攻击配置 (Generate Attacks)
使用 LLM 分析原图并生成攻击策略。

```bash
python data_collector/generate_attacks.py \
  --clean-meta ./data/clean_images/metadata.jsonl \
  --original-dir ./data/raw_images \
  --output ./data/attacks.jsonl \
  --model "Qwen/Qwen3-VL-30B-A3B-Thinking"
```

### 第二步：合成对抗样本 (Synthesize)
调用 ComfyUI 将文字注入图片。

```bash
python main_benchmark.py \
  --attack-file ./data/attacks.jsonl \
  --output-dir ./data/bench_dataset \
  --comfy-server 127.0.0.1:8188
```

### 第三步：模型评估 (Evaluate)
运行评估脚本，计算 MGD, WLA, TBS 等指标。

```bash
python evaluate.py \
  --img-dir ./data/bench_dataset \
  --metadata-file ./data/yfcc100m_dataset.txt \
  --bench-meta ./data/bench_dataset/benchmark_meta.jsonl \
  --output ./results_qwen.jsonl
```

### 第四步：可视化分析 (Visualize)
生成 CDF 曲线、误差柱状图和交互式地图。

```bash
python visualize_results.py \
  --results Qwen3-VL=./results_qwen.jsonl \
  --output-dir ./results_viz
```

---

## 📂 项目结构

```text
├── data_collector/         # [模块] 攻击生成与 ComfyUI 客户端
│   ├── generate_attacks.py # Step 1: LLM 攻击生成
│   ├── comfy_client.py     # ComfyUI 通信类
│   └── image_qwen_image_edit.json # ComfyUI 工作流模板
├── evaluation/             # [模块] 评估指标计算
│   ├── metric_calculator.py # WLA, TBS, TFR 核心公式
│   └── vllm_client.py      # 模型推理接口
├── main_benchmark.py       # [入口] Step 2: 图像合成脚本
├── evaluate.py             # [入口] Step 3: 评估脚本
├── visualize_results.py    # [入口] Step 4: 可视化脚本
└── requirements.txt
```

## 📄 许可证

[MIT License](LICENSE)
