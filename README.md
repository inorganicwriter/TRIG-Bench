# TRIG-Bench (Text Relevance In Geo-localization)

**评估视觉语言模型 (VLMs) 在面对对抗性文本攻击时，其地理定位任务的鲁棒性表现。**

TRIG-Bench 提供了一套全自动化的对抗样本生成与评估流水线，通过生成具有高度欺骗性的“幻觉文本”并将其融入街景图片，精确测量模型在不同语义干扰下的定位偏差。

---

## 📚 完整工作流程 (Complete Workflow)

### 1. 环境配置 (Environment Configuration)

建议使用 Conda 创建隔离环境：

```bash
# 1. 创建并激活环境
conda create -n trig python=3.10 -y
conda activate trig

# 2. 克隆项目
git clone https://github.com/inorganicwriter/TRIG-Bench.git
cd TRIG-Bench

# 3. 安装依赖
# 包含 openai, torch, matplotlib, folium 等
pip install -r requirements.txt
```

---

### 2. 启动基础服务 (Start Basic Programs)

本基准测试依赖两个核心服务：**大模型推理服务 (vLLM)** 和 **图像合成服务 (ComfyUI)**。

#### A. 启动 Qwen3-VL 服务 (vLLM)
在拥有 A100/A800/L20 等大显存 GPU 的服务器上运行：

```bash
# 自动检测 TP=2 (适用于 30B 模型)，监听 8001 端口
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

#### B. 启动 ComfyUI 服务
在本地或服务器上启动 ComfyUI（需安装 Qwen-Image-Edit 节点）：

```bash
# 进入 ComfyUI 目录
cd ComfyUI
# 启动并监听端口 (默认为 8188)
python main.py --listen 0.0.0.0 --port 8188
```

---

### 3. 执行基准测试 (Run Benchmark)

#### Step 1: 生成攻击配置 (Generate Attacks)
让 LLM (Qwen3-VL) 分析原始图片，生成 Similar/Random/Adversarial 三种攻击策略。

```bash
python data_collector/generate_attacks.py \
  --clean-meta ./data/clean_images/metadata.jsonl \
  --original-dir ./data/raw_images \
  --output ./data/attacks.jsonl \
  --model "Qwen/Qwen3-VL-30B-A3B-Thinking" \
  --api-base http://localhost:8001/v1
```

#### Step 2: 合成对抗样本 (Synthesize Images)
通过 ComfyUI 将生成的干扰文本逼真地注入到图片中。

```bash
python main_benchmark.py \
  --attack-file ./data/attacks.jsonl \
  --output-dir ./data/bench_dataset \
  --comfy-server 127.0.0.1:8188
```

---

### 4. 获取与分析结果 (Get Results)

#### Step 3: 模型表现评估 (Evaluate)
将合成的对抗图片喂给模型，计算**平均误差距离 (MGD)**、**加权定位精度 (WLA)** 和 **文本偏差分数 (TBS)**。

```bash
python evaluate.py \
  --img-dir ./data/bench_dataset \
  --metadata-file ./data/yfcc100m_dataset.txt \
  --bench-meta ./data/bench_dataset/benchmark_meta.jsonl \
  --output ./results_qwen.jsonl \
  --api-base http://localhost:8001/v1
```

#### Step 4: 可视化分析 (Visualize)
生成详细的图表报告：
*   📈 **CDF 曲线**: 误差累积分布
*   📊 **柱状图**: clean vs adversarial 性能对比
*   🗺️ **交互地图**: 幻觉连线可视化

```bash
python visualize_results.py \
  --results Qwen3-VL=./results_qwen.jsonl \
  --output-dir ./results_viz
```

生成的图表将保存在 `./results_viz` 目录下。

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
