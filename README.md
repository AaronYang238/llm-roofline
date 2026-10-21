# roofline — 大模型推理算力强度 (FLOP/B) 分析工具

基于 [ModelScope](https://modelscope.cn/models) 上主流开源大模型的 `config.json`，
按**不同推理阶段**（QKV 投影、注意力核心、MLP/FFN、MoE 路由、LM Head、元素级/归一化等）
解析计算每阶段的 **FLOPs 与 DRAM 访存 Bytes**，输出各模型各阶段的**算力强度 (FLOP/B)** 交互式 HTML 报告。

## 快速开始

```bash
# 无任何第三方依赖 (纯 Python 标准库)
python main.py
```

首次运行会自动爬取 ModelScope 上 52 个主流大模型的 `config.json`（约 1–2 分钟），
缓存到 `configs_cache/`，之后秒级完成。输出为自包含单文件：

```
output/roofline_report.html   ← 浏览器直接打开（需启用 JavaScript）
```

```bash
python main.py --refresh      # 忽略缓存，重新爬取
python main.py --out r.html   # 指定输出路径
```

## 报告内容（5 个标签页）

| 标签页 | 内容 |
|---|---|
| **总览矩阵** | 52 个模型 × 8 个推理阶段的强度热力矩阵，可切换 预填充(Prefill)/解码(Decode)、提示词长度 S、批量 B、上下文长度 |
| **强度曲线** | 选中模型各阶段强度随解码批量 B / 提示词长度 S 的变化曲线（对数坐标，叠加 GPU 脊点参考线） |
| **Roofline 图** | 选中模型各阶段在所选 GPU（V100/A100/H100/H20/MI308X/B200）屋顶线上的位置：斜线段 = 访存受限，水平段 = 算力受限 |
| **架构总表** | 各模型架构参数（层数/hidden/头数/FFN/MoE）、总参数量、激活参数量、每 token 流量、理论 decode 上限 |
| **方法论** | 阶段划分、公式、关键假设与局限性 |

## 输入区（交互式参数）

页面顶部的**输入区**可实时调整推理场景，所有图表（矩阵/曲线/Roofline）即时重算：

- **推理阶段**：Prefill（预填充，GEMM 密集计算）或 Decode（解码，GEMV + 全量 KV 读），
  两者算力强度相差 2–3 个数量级；
- **Prefill 序列长度 S**：预填充一次前向处理的提示词长度；
- **Decode 批量 B**：并行解码的序列数（权重读一次由全批共享，GEMV 强度 ≈ B）；
- **KV 缓存长度**：解码时逐步累积的上下文长度，决定每步 KV 读量与注意力强度。

重算由页面内置的 **JS 公式引擎**完成（与 Python `stages.py` 逐公式一致，
已验证 Qwen2.5-7B / DeepSeek-V3 等模型在 prefill/decode 两阶段的数值与 Python 参考完全吻合），
修改参数后无需重新生成报告。

## 覆盖模型（52 个）

### 经典开源模型（33 个）

- **Qwen2.5**：0.5B / 1.5B / 3B / 7B / 14B / 32B / 72B（dense · GQA · tied embeddings）
- **Qwen3**：0.6B / 1.7B / 4B / 8B / 14B / 32B + **QwQ-32B**
- **Qwen3 MoE**：30B-A3B、235B-A22B
- **DeepSeek**：V3、R1（MoE · MLA）
- **Llama 3.1**：8B / 70B / 405B
- **Mistral / Mixtral**：Mistral-7B、Mixtral-8x7B、Mixtral-8x22B
- **Google Gemma 2**：9B / 27B；**Microsoft Phi-3-medium**
- **Zhipu GLM-4-9B**、**GLM-4.5**（MoE · MLA）
- **01.AI Yi-34B**、**Baichuan2-7B**、**InternLM2.5-7B**、**Moonshot Kimi-K2**

### 2025–2026 新一代模型（19 个）

- **Qwen3.5**：0.8B / 4B / 9B / 35B-A3B（dense 或 MoE · 混合线性注意力 GDN + 全注意力）
- **Qwen3.6-35B-A3B**、**Qwen3.8-27B**、**Qwen3.8-Flash-Next**（混合线性注意力）
- **DeepSeek-V4.1-Flash**、**DeepSeek-V4-Pro**（MoE · DSA 稀疏注意力，decode 仅读 topk 条 KV）
- **GLM-5.3**、**GLM-5.3-Flash**（MoE · MLA · DSA / 混合线性注意力）
- **Kimi-K3**（MoE · MLA · KDA 混合线性注意力）
- **Ling-3.1-flash**、**Ring-2.6-1T**（MoE · MLA）
- **Tencent Hunyuan**：Hy3、Hy-Memory-30B-A3B
- **Holo3-35B-A3B**（MoE · 混合线性注意力）
- **openPangu-2.0-Pro**（MoE · MLA）
- **Solar-Open2-250B**（MoE · KDA 混合线性注意力）

## 项目结构

```
roofline/
├── main.py               # 主入口：爬取 → 计算 → 渲染
├── roofline/
│   ├── crawler.py        # ModelScope config.json 爬取器（带缓存）
│   ├── arch.py           # 各家族 config 归一化（GQA/MLA/MoE/tied 等）
│   ├── stages.py         # 各推理阶段 FLOPs/Bytes 模型（Python 真值）
│   ├── compute.py        # 场景网格 → 强度数据立方
│   ├── report.py         # HTML 报告渲染
│   └── template.html     # 报告模板：输入区 + JS 公式引擎（computeStagesJS）
├── configs_cache/        # 爬取的 config.json 缓存（自动生成）
└── output/
    └── roofline_report.html
```

## 核心方法

- **算力强度** = 阶段 FLOPs ÷ 阶段 DRAM 访存量（单位 FLOP/B）。
  Roofline 模型中性能上限 = min(峰值算力, 带宽 × 强度)；GPU 脊点一般在 100–300 FLOP/B。
- **权重读一次**：每个 GEMM/GEMV pass 中权重只从 DRAM 读一次，整个 batch 共享（理想 batching）。
- **激活读/写一次**：每阶段输入激活读一次、输出写一次，阶段之间不重复计数。
- **MLA 近似**（DeepSeek-V3/R1、GLM-4.5、Kimi-K2）：按 (nope+rope) 与 v_head_dim 维度建模，KV 缓存按压缩后 (kv_lora_rank + rope) 计。
- **混合线性注意力**（Qwen3.5/3.6/3.8、Kimi-K3、GLM-5.3-Flash、Solar-Open2-250B）：
  线性注意力层（GDN/KDA）使用**固定大小循环状态**，不随序列长度累积 KV 缓存；
  仅全注意力层累积 KV 缓存。decode 时线性层每步读写状态（≈ state_elems 元素）。
- **DSA 稀疏注意力**（DeepSeek-V4.1-Flash、DeepSeek-V4-Pro、GLM-5.3、GLM-5.3-Flash）：
  indexer 每步只选 topk 个 token 的 KV 参与注意力，decode 注意力流量按 min(topk, S) 计。
- **MoE decode**：每步激活 n_active = min(E, B×K) 个专家，激活专家的权重各读一次。
- 所有公式按 BF16 权重 (2 B/参数)、BF16 激活、BF16 KV 缓存统一口径计算，便于跨模型公平比较。

## 局限性

- 解析模型未覆盖算子融合、kernel 调度、量化 (W8A8/FP8/Int4)、P/D 分离、投机解码等系统优化。
- 滑动窗口注意力 (Mistral/Gemma-2) 与因果掩码按简化方式处理；超长上下文下注意力核心的强度估计偏保守。
- 元素级/归一化阶段的 FLOPs 与访存为量级估算，非逐算子精确定义。
- 理论 decode 上限 (tok/s) = 带宽 ÷ 每 token 流量，为理想上界，实际通常低 30–60%。

## 数据来源

- ModelScope 模型页：<https://modelscope.cn/models>
- API：`GET /api/v1/models/{namespace}/{name}/repo?FilePath=config.json&Revision=master`
- 爬取的 config.json 缓存于 `configs_cache/`，二次运行不再联网。
