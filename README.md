# TRIG-Bench (Text Relevance In Geo-localization)

**评估视觉语言模型 (VLMs) 在面对对抗性文本攻击时，其地理定位任务的鲁棒性表现。**

本工具包提供了一套端到端的流水线，包含以下核心功能：
1.  **清洗 (Clean)**：移除基准测试图片中的原始文本（基于 ComfyUI）。
2.  **生成 (Benchmark Generator)**：基于 **CLIP 语义相关性**，自动筛选出不同难度的干扰地名（Hard/Mid/Easy）。
3.  **合成 (Synthesize)**：将干扰文本自然地融合到图片中，生成量化的测试数据集。
4.  **评估 (Evaluate)**：测量添加文本后，SOTA 模型（如 Qwen, Llama, DeepSeek 等）地理定位精度的下降程度。

---

## 🚀 核心特性

本基准测试包含两个正交的评估维度：

### 维度一：语义难度 (Semantic Difficulty)
基于 **CLIP Score ($S$)** 衡量干扰文本与视觉场景的相关性：
1.  **空白对照组 (Control Group)**：$I_{clean}$，测定原生视觉理解能力。
2.  **语义正交模式 (Simple Mode, $S \le 0.20$)**：文本与场景差异显著（如雪山+"热带雨林"），检测OCR盲从。
3.  **语义平行模式 (Hard Mode, $S > 0.28$)**：文本与场景视觉风格相似（如东京+"大阪"），构成高似然性陷阱。

### 维度二：物理交互 (Physical Interaction)
基于干扰文本与物理环境的结合方式：
*   🏎️ **Level 1: 移动物体解离 (Moving Object Dissociation)**：利用 **YOLOv8** 将文本附着于汽车/行人，测试前景解耦。
*   🏯 **Level 2: 文化错位 (Cultural Displacement)**：将冲突文本植入固定环境（背景），构建逻辑悖论。
*   🗼 **Level 3: 实体幻觉 (Entity Hallucination)**：将高相关文本植入地标建筑，构建多模态协同幻觉。

*   **全自动流水线**：从原始图片清洗(LaMa) -> 语义分级生成(CLIP+YOLO) -> 自动化评测(vLLM)。
*   **多模型评估**：支持 Qwen-VL, Llama-Vision, DeepSeek 等所有 OpenAI 兼容接口模型。

---

## 🛠️ 环境要求

*   **Python 3.10+**
*   **依赖库**：`torch`, `transformers`, `ultralytics` (YOLOv8), `pillow`, `openai` 等（见 requirements.txt）。
*   **ComfyUI**（仅第一步清洗需要）：本地 `127.0.0.1:8188`。

## 📦 安装说明

1.  克隆代码仓库：
    ```bash
    git clone https://github.com/inorganicwriter/TRIG-Bench.git
    cd TRIG-Bench
    ```

2.  安装依赖：
    ```bash
    pip install -r requirements.txt
    ```

---

## 📖 使用指南

### 第一步：生成清洗样本 (Clean Sample Generation)
使用 ComfyUI 移除图片中的原有文字（路牌、广告等），建立“干净”基准。

```bash
# 需启动 ComfyUI
python pipeline.py \
  --input ./data/raw_images \
  --output ./data/clean_images \
  --mode remove \
  --prompt "Remove all UI text elements from the image."
```

### 第二步：生成基准数据集 (Benchmark Generation)
运行核心生成器。它会自动分析图片内容，从全球城市库中匹配干扰词，并生成带有文字干扰的图片。

```bash
python main_benchmark.py \
  --clean-dir ./data/clean_images \
  --output-dir ./data/bench_dataset \
  --clip-model "openai/clip-vit-base-patch32"
```
*输出：`bench_dataset/` 目录下将包含生成的图片以及 `benchmark_meta.jsonl`（记录了每张图的干扰类型和难度）。*

### 第三步：模型评估 (Evaluation)
评估目标模型（如 Qwen3-VL-30B）在对抗数据集上的表现。

```bash
# 需先启动 vLLM 服务 (例如 Qwen3-VL 或 GPT-4o 兼容接口)
python evaluate.py \
  --img-dir ./data/bench_dataset \
  --metadata-file ./data/yfcc100m_dataset.txt \
  --output ./results_qwen.jsonl \
  --model "Qwen/Qwen3-VL-30B-A3B-Thinking" \
  --api-base http://localhost:8001/v1
```

---

## 📂 项目结构

```text
├── benchmark_engine/       # [核心] 基准生成引擎
│   ├── relevance_scorer.py # CLIP 语义相似度计算
│   ├── text_injector.py    # PIL 视觉攻击合成
│   └── distractor_pool.py  # 干扰城市词库
├── data_collector/         # [模块] 数据采集与清洗
│   ├── clean_pipeline.py   # 清洗流水线 (Step 1)
│   ├── comfy_client.py     # ComfyUI 客户端
│   └── utils.py            # 工具函数
├── evaluation/             # [模块] 评测工具
│   ├── metric_calculator.py
│   └── vllm_client.py
├── main_benchmark.py       # [入口] 定量生成脚本 (Step 2)
├── evaluate.py             # [入口] 评估脚本 (Step 3)
├── LICENSE                 # MIT 许可证
└── requirements.txt
```

## 📄 许可证

[MIT License](LICENSE)
