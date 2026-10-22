# roofline — 大模型推理算力强度 (FLOP/B) 分析工具

基于 [ModelScope](https://modelscope.cn/models) 上主流开源大模型的 `config.json`，
按**不同推理阶段**（QKV 投影、注意力核心、MLP/FFN、MoE 路由、LM Head、元素级/归一化等）
解析计算每阶段的 **FLOPs 与 DRAM 访存 Bytes**，输出各模型各阶段的**算力强度 (FLOP/B)** 交互式 HTML 报告。

报告不只“能算”，也尽量“好用、可信”：

- 52 个模型可**搜索、按家族筛选、按强度/参数量排序**，点击任意模型查看阶段级 FLOPs/访存/占比明细；
- 支持**快速场景**（单用户解码 / 批量解码 / 长上下文 / 预填充）、**URL 深链分享**与**矩阵 CSV 导出**；
- 架构总表会标注**单卡显存是否放得下**，放不下时不再给出误导性的单卡 tok/s 上界；
- 对 config 解析中的推断与异常（如 MoE 专家数缺失、MLA 字段不完整）给出**自动校验提示**。

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
python -m unittest discover -s tests   # 数据归一化 + Python/JS 公式一致性测试
```

## 报告内容（5 个标签页）

| 标签页 | 内容 |
|---|---|
| **总览矩阵** | 52 个模型 × 8 个推理阶段的强度热力矩阵；支持搜索/家族筛选/排序，点击行查看阶段明细；可切换 预填充(Prefill)/解码(Decode)、提示词长度 S、批量 B、上下文长度 |
| **强度曲线** | 选中模型各阶段强度随解码批量 B / 提示词长度 S 的变化曲线（对数坐标，叠加 GPU 脊点参考线） |
| **Roofline 图** | 选中模型各阶段在所选 GPU（V100/A100/H100/H20/MI308X/B200）屋顶线上的位置：斜线段 = 访存受限，水平段 = 算力受限 |
| **架构总表** | 各模型架构参数（层数/hidden/头数/FFN/MoE）、总参数量、激活参数量、每 token 流量、单卡显存适配、理论 decode 上限 |
| **方法论** | 阶段划分、公式、关键假设、数据校验规则与局限性 |

## 输入区（交互式参数）

页面顶部的**输入区**可实时调整推理场景，所有时序图（矩阵/曲线/Roofline）即时重算：

- **推理阶段**：Prefill（预填充，GEMM 密集计算）或 Decode（解码，GEMV + 全量 KV 读），
  两者算力强度相差 2–3 个数量级；
- **Prefill 序列长度 S**：预填充一次前向处理的提示词长度；
- **Decode 批量 B**：并行解码的序列数（权重读一次由全批共享，GEMV 强度 ≈ B）；
- **KV 缓存长度**：解码时逐步累积的上下文长度，决定每步 KV 读量与注意力强度；
- **快速场景**：一键切换 单用户解码 B=1 / 批量解码 B=32 / 长上下文 32K / 预填充 8K；
- **复制当前视图链接**：把 tab、阶段、B/S、模型、GPU、搜索/筛选/排序写入 URL，便于分享与复现。

重算由页面内置的 **JS 公式引擎**完成（与 Python `stages.py` 逐公式一致，
`tests/test_parity.py` 会用 Node 对全部模型、多组场景做交叉校验），
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
├── tests/
│   ├── test_arch.py      # 架构归一化 / 专家数 / 参数量 / 注意力公式回归
│   └── test_parity.py    # Python 与 JS 公式引擎交叉校验（需 Node.js）
└── output/
    └── roofline_report.html
```

## 核心方法

- **算力强度** = 阶段 FLOPs ÷ 阶段 DRAM 访存量（单位 FLOP/B）。
  Roofline 模型中性能上限 = min(峰值算力, 带宽 × 强度)；GPU 脊点一般在 100–300 FLOP/B。
- **权重读一次**：每个 GEMM/GEMV pass 中权重只从 DRAM 读一次，整个 batch 共享（理想 batching）。
- **激活读/写一次**：每阶段输入激活读一次、输出写一次，阶段之间不重复计数。
- **注意力 FLOPs**：QK<sup>T</sup> 与 AV 各计 2·d FLOPs/对；prefill 再乘 causal ×0.5，decode 对每步 min(topk, S) 个 KV 计算。
  旧版只计了 QK<sup>T</sup>，会把注意力 FLOPs 低估一半。
- **MLA 近似**（DeepSeek-V3/R1、GLM-4.5、Kimi-K2）：按 (nope+rope) 与 v_head_dim 维度建模，KV 缓存按压缩后 (kv_lora_rank + rope) 计。
- **混合线性注意力**（Qwen3.5/3.6/3.8、Kimi-K3、GLM-5.3-Flash、Solar-Open2-250B）：
  线性注意力层（GDN/KDA）使用**固定大小循环状态**，不随序列长度累积 KV 缓存；
  仅全注意力层累积 KV 缓存。decode 时线性层每步读写状态（≈ state_elems 元素）。
- **DSA 稀疏注意力**（DeepSeek-V4.1-Flash、DeepSeek-V4-Pro、GLM-5.3、GLM-5.3-Flash）：
  indexer 每步只选 topk 个 token 的 KV 参与注意力，decode 注意力流量按 min(topk, S) 计。
- **MoE decode**：每步激活 n_active = min(E, B×K) 个专家，激活专家的权重各读一次。
  每 token 专家数优先读 `num_experts_per_tok` / `num_experts_per_token`，**不会**误用采样参数 `top_k`；
  专家 FFN 维度优先读 `expert_hidden_dim` / `routed_expert_hidden_size`。
- **共享专家**：显式声明的 `n_shared_experts` / `num_shared_experts` 优先；只有
  `shared_expert_intermediate_size` 时按 1 个共享专家估算。
- **激活参数**：包含注意力、MLP/激活专家、embedding 与 LM head（untied 时）；tied embedding 不重复计 LM head。
- **单卡适配**：架构总表只比较 BF16 权重与所选 GPU 显存（按 90% 可用估算）；放不下时显示“需 N 卡”，
  不再给出单卡 tok/s 上界。
- 所有公式按 BF16 权重 (2 B/参数)、BF16 激活、BF16 KV 缓存统一口径计算，便于跨模型公平比较。

## 数据校验

`data_warnings()` 会对归一化结果做轻量检查，并在报告模型名旁以 ⚠ 提示：

- 解析参数量与模型名标注（如 30B）偏差超过 35%；
- MoE 的 K 大于 E，或 config 未声明每 token 专家数而按 K=2 估算；
- 存在 `q_lora_rank` / `o_lora_rank` 但缺少 `kv_lora_rank`，MLA 只能按 GQA 近似；
- 缺少 vocab_size / 层数 / 注意力头数等关键字段。

这些提示表示**解析口径风险**，不代表模型本身有问题；用于避免把“看起来精确”的数字当成事实。

## 局限性

- 解析模型未覆盖算子融合、kernel 调度、量化 (W8A8/FP8/Int4)、P/D 分离、投机解码等系统优化。
- 滑动窗口注意力 (Mistral/Gemma-2) 与因果掩码按简化方式处理；超长上下文下注意力核心的强度估计偏保守。
- 元素级/归一化阶段的 FLOPs 与访存为量级估算，非逐算子精确定义。
- 理论 decode 上限 (tok/s) = 带宽 ÷ 每 token 流量，为理想上界，实际通常低 30–60%；
  且仅在 BF16 权重可放入单卡显存时给出，多卡场景需按 TP/EP 与互联带宽另算。
- 混合线性注意力 (GDN/KDA) 与 DeepSeek-V4 的新式注意力仍按阶段做近似，DSA 的 indexer 开销未单独计入。

## 数据来源

- ModelScope 模型页：<https://modelscope.cn/models>
- API：`GET /api/v1/models/{namespace}/{name}/repo?FilePath=config.json&Revision=master`
- 爬取的 config.json 缓存于 `configs_cache/`，二次运行不再联网。
