# SIGNPOST-Bench

**Street Image Geo-localization with Noisy Perturbation on Observed Sign Text**

评估视觉语言模型 (VLMs) 在面对对抗性文本攻击时，其地理定位任务的鲁棒性表现。

SIGNPOST-Bench 提供了一套全自动化的对抗样本生成与评估流水线，通过生成具有高度欺骗性的"幻觉文本"并将其融入街景图片，精确测量模型在不同语义干扰下的定位偏差。

---

## 📚 完整工作流程 (Complete Workflow)

### 1. 环境配置 (Environment Configuration)

建议使用 Conda 创建隔离环境：

```bash
# 1. 创建并激活环境
conda create -n signpost python=3.10 -y
conda activate signpost

# 2. 克隆项目
git clone https://github.com/inorganicwriter/SIGNPOST-Bench.git
cd SIGNPOST-Bench

# 3. 安装依赖
pip install -r requirements.txt
```

---

### 2. 启动基础服务 (Start Basic Programs)

本基准测试依赖两个核心服务：**大模型推理服务 (vLLM)** 和 **图像合成服务 (ComfyUI)**。

#### A. 启动 Qwen3-VL 服务 (vLLM)
在拥有 A100/A800/L20 等大显存 GPU 的服务器上运行：

```bash
vllm serve "Qwen/Qwen3-VL-30B-A3B-Thinking" \
  --dtype auto \
  --trust-remote-code \
  --max-model-len 8192 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port 8001 \
  --api-key your-api-key
```

#### B. 启动 ComfyUI 服务
在本地或服务器上启动 ComfyUI（需安装 Qwen-Image-Edit 节点）：

```bash
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188
```

---

### 3. 一键执行流水线 (Run Pipeline)

使用 `run_pipeline.py` 统一管理所有阶段：

```bash
# 生成攻击方案 + 合成图片 + 评测（一键全流程）
python run_pipeline.py --dataset im2gps3k --stage all --model qwen3-30b --api-key your-api-key

# 或分阶段执行：
# Step 1: 生成攻击配置
python run_pipeline.py --dataset im2gps3k --stage attack_gen --model qwen3-30b --api-key your-api-key

# Step 2: 合成对抗样本
python run_pipeline.py --dataset im2gps3k --stage synthesize

# Step 3: 评测
python run_pipeline.py --dataset im2gps3k --stage evaluate --model qwen3-30b --api-key your-api-key
```

#### 支持的数据集
- `im2gps3k` — Im2GPS3k 测试集
- `yfcc4k` — YFCC4k 测试集
- `googlesv` — Google Street View 采样
- `baidusv` — 百度街景采样

#### 支持的模型

| 类别 | 模型 | 短名称 |
|------|------|--------|
| 本地 vLLM | Qwen3-VL-30B-A3B-Thinking | `qwen3-30b` |
| 本地 vLLM | Qwen3-VL-8B-Thinking | `qwen3-8b` |
| SiliconFlow | Qwen3-VL-235B | `qwen3-vl-235b-sf` |
| SiliconFlow | Qwen2.5-VL-72B | `qwen2.5-vl-72b-sf` |
| OpenRouter | GPT-5 | `gpt-5` |
| OpenRouter | GPT-4o | `gpt-4o` |
| OpenRouter | Claude Sonnet 4.6 | `claude-sonnet-4.6` |
| OpenRouter | Gemini 2.5 Pro | `gemini-2.5-pro` |

完整模型列表见 `evaluation/api_client.py`。

---

### 4. 获取与分析结果 (Get Results)

```bash
# 计算汇总结果
python compute_results.py --datasets im2gps3k yfcc4k googlesv baidusv

# 计算 Trap-Fit Rate (TFR)
python compute_tfr.py --datasets im2gps3k yfcc4k

# 场景文本分类
python classify_taxonomy.py --datasets im2gps3k yfcc4k

# 可视化
python visualize_results.py \
  --results Qwen3-VL=./results_qwen.jsonl \
  --output-dir ./results_viz
```

---

## 📂 项目结构

```text
SIGNPOST-Bench/
├── run_pipeline.py             # [入口] 统一流水线管理
├── data_collector/             # [模块] 攻击生成与图像合成
│   ├── generate_attacks.py     # LLM 攻击方案生成
│   ├── llm_provider.py         # LLM 接口封装 (OpenAI/vLLM)
│   ├── comfy_client.py         # ComfyUI 通信客户端
│   ├── filter_images.py        # OCR 图像筛选
│   └── image_qwen_image_edit.json  # ComfyUI 工作流模板
├── evaluation/                 # [模块] 评估与 API 客户端
│   ├── api_client.py           # 统一多平台 API 客户端
│   ├── metric_calculator.py    # WLA, TBS, TFR 指标计算
│   └── vllm_client.py          # vLLM 推理接口
├── main_benchmark.py           # 图像合成脚本
├── evaluate.py                 # 模型评测脚本
├── compute_results.py          # 结果汇总
├── compute_tfr.py              # TFR 计算
├── classify_taxonomy.py        # 场景文本分类
├── visualize_results.py        # 可视化
├── convert_metadata.py         # 元数据格式转换
├── sample_googlesv.py          # Google 街景采样
├── sample_baidusv.py           # 百度街景采样（多 GPU 并行 OCR）
├── analyze_invalid_samples.py  # 无效样本清理
├── count_dataset.py            # 数据集统计
├── requirements.txt            # Python 依赖
└── paper/
    └── main.tex                # 论文 LaTeX 源码
```

## 📊 评估指标

| 指标 | 全称 | 说明 |
|------|------|------|
| **MGD** | Mean Geo-Distance | 平均地理误差距离 (km) |
| **WLA** | Weighted Location Accuracy | 加权定位精度 (多阈值) |
| **TBS** | Text Bias Score | 文本偏差分数 (攻击前后差异) |
| **TFR** | Trap-Fit Rate | 陷阱命中率 (模型被误导到攻击目标的比例) |

## 📄 许可证

[MIT License](LICENSE)
