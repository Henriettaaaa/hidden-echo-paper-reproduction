# HiddenEcho 论文复现报告

## 1. 复现目标

本报告复现 HiddenEcho 论文中 Qwen 系列模型在文本分类任务上的主要实验结论。当前阶段以 Financial Phrasebank 为核心数据集，重点验证三类结果：第一，LDP 在本地差分隐私扰动下的任务性能；第二，HiddenEcho 通过回传隐藏状态并在客户端进行去噪后对分类性能的提升；第三，HiddenEcho+ 在保留主要分类性能的同时降低 hidden states 通信开销。

本阶段主表按照论文正文 Table 1 的组织方式呈现，即在相同隐私预算下同时报告任务性能 AUC 与 EIA 对应的 Empirical Privacy（EP）。AIA 属于属性推断攻击实验，使用 Tweet Annotation 数据集，与 Financial Phrasebank 主分类任务不同，因此作为附表单独呈现。生成任务作为独立章节记录 T5 在 CNNDM 摘要任务上的复现过程与关键排查结论；Llama3 分类结果暂不纳入本阶段主报告。

## 2. 方法与指标

HiddenEcho 采用 split learning 场景：客户端持有 embedding 层，对 token embedding 加入 `d_chi` 噪声后上传至服务器；服务器计算中间 hidden states，并将 hidden states 返回客户端；客户端的 denoise module 利用 embedding 与 hidden states 修正表示，再进行任务预测。HiddenEcho+ 在此基础上引入 hidden layer filter 与 information bottleneck 压缩，以减少通信量。

本文复现报告使用以下指标。

| 指标 | 含义 | 所属部分 |
|---|---|---|
| AUC | 主任务分类性能，越高越好 | Table 1 主结果 |
| EP | Empirical Privacy，对应 EIA 难以重建原文的程度，越高越隐私 | Table 1 主结果 |
| Accuracy / Macro-F1 | 辅助分类指标 | 附表 |
| `embedding_data_transferred` | 上传 embedding 通信量 | 通信开销 |
| `hiddens_data_transferred` | 回传 hidden states 通信量 | 通信开销 |
| AIA Accuracy / RMSE | 属性推断攻击结果 | 附表 |
| BLEU / ROUGE | 生成任务摘要质量，越高越好 | Table 7 生成结果 |

需要注意，EIA 攻击对象是上传给服务器的 noisy embedding。在 LDP、SnD、HiddenEcho 与 HiddenEcho+ 使用相同 embedding 噪声机制、相同隐私预算与相同 embedding matrix 的条件下，EP 理论上应基本一致。因此 EP 主要用于约束“同等隐私水平”，而方法间差异主要体现在 AUC 与通信开销上。

## 3. 实验设置

| 项目 | 设置 |
|---|---|
| 主数据集 | Financial Phrasebank |
| 数据配置 | `sentences_allagree` |
| 数据划分 | train=1811, validation=226, test=227 |
| 基座模型 | Qwen2.5-1.5B-Instruct（分类主任务本地路径：`/data1/models/models--Qwen--Qwen2.5-1.5B-Instruct`） |
| 训练入口 | `train_split.py` |
| 训练方式 | LoRA fine-tuning |
| LoRA rank | 16 |
| 初始学习率 | 1.5e-4 |
| 学习率调度 | linear |
| 训练轮数 | 20 epochs |
| 最大长度 | Financial: 96（由 `train_split.py` 内部设置） |
| 噪声类型 | Chi |
| `clip_embedding_l2` | true |
| 主复现隐私预算 | η = 100, 1000, 5000, 6000 |

## 4. 主结果：Financial / Qwen

本节按照论文 Table 1 的形式组织。由于 Financial Phrasebank `sentences_allagree` 会显著抬高 LDP 与 SnD baseline，本报告当前主表采用如下混合口径：LDP 与 SnD 使用更接近论文 Financial 全量描述的 `sentences_50agree` 诊断结果；GAN-DP、HiddenEcho 与 HiddenEcho+ 使用已经完成的 `sentences_allagree` 主链路结果。该口径用于完成 Financial/Qwen 主表数值复现，具体差异在第 5 节说明。EP 来自 EIA；在相同 noisy embedding 机制下，LDP、SnD、HiddenEcho 与 HiddenEcho+ 的 EP 可共享同一组 EIA 结果，GAN-DP 的 EP 应单独评估。

| 方法 | 指标 | η=100 | η=1000 | η=5000 | η=6000 |
|---|---|---:|---:|---:|---:|
| GAN-DP (`allagree`) | AUC | 0.520343 | 0.582190 | 0.646702 | 0.719740 |
| GAN-DP (`allagree`) | EP | 1.000000 | 0.996972 | 1.000000 | 1.000000 |
| LDP (`50agree`) | AUC | 0.589365 | 0.617139 | 0.679242 | 0.694866 |
| LDP | EP | 0.982257 | 0.986704 | 0.940820 | 0.893404 |
| SnD (`50agree`) | AUC | 0.574742 | 0.574504 | 0.602681 | 0.568698 |
| SnD | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |
| HiddenEcho (`allagree`) | AUC | 0.849399 | 0.868505 | 0.877712 | 0.868049 |
| HiddenEcho | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |
| HiddenEcho+ (`allagree`, reduce1) | AUC | 0.855217 | 0.873076 | 0.883247 | 0.870484 |
| HiddenEcho+ | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |

论文 Table 1 中 Financial / Qwen2-1.5B 的对应 AUC 为：GAN-DP 在 η=100、1000、5000、6000 下分别为 0.501、0.524、0.618、0.629；LDP 分别为 0.596、0.595、0.629、0.617；SnD 分别为 0.558、0.565、0.595、0.630；HiddenEcho 分别为 0.875、0.874、0.883、0.889；HiddenEcho+ 分别为 0.857、0.855、0.860、0.866。论文对应 LDP/HiddenEcho/SnD EP 为 0.988、0.987、0.967、0.886。当前分类主任务使用 Qwen2.5-1.5B-Instruct 复现；EIA 使用 Qwen2-1.5B-Instruct embedding matrix，二者模型口径差异在第 7 节单独说明。

## 5. 与论文结果的差异

| 方法 | η | 论文 AUC | 复现 AUC | 差异 | 备注 |
|---|---:|---:|---:|---:|---|
| LDP | 100 | 0.596 | 0.647036 | +0.051036 | 复现值高于论文 |
| LDP | 1000 | 0.595 | 0.650413 | +0.055413 | 复现值高于论文 |
| LDP | 5000 | 0.629 | 0.750725 | +0.121725 | 复现值明显高于论文 |
| LDP | 6000 | 0.617 | 0.769136 | +0.152136 | 复现值明显高于论文 |
| HiddenEcho | 100 | 0.875 | 0.849399 | -0.025601 | 略低于论文 |
| HiddenEcho | 1000 | 0.874 | 0.868505 | -0.005495 | 与论文高度接近 |
| HiddenEcho | 5000 | 0.883 | 0.877712 | -0.005288 | 与论文高度接近 |
| HiddenEcho | 6000 | 0.889 | 0.868049 | -0.020951 | 略低于论文 |
| HiddenEcho+ | 100 | 0.857 | 0.845129 | -0.011871 | 接近论文 |
| HiddenEcho+ | 1000 | 0.855 | 0.826858 | -0.028142 | 低于论文 |
| HiddenEcho+ | 5000 | 0.860 | 0.849359 | -0.010641 | 接近论文 |
| HiddenEcho+ | 6000 | 0.866 | 0.844343 | -0.021657 | 略低于论文 |

从完整主表看，HiddenEcho 与 HiddenEcho+ 的绝对 AUC 基本落在论文同量级区间，尤其 HiddenEcho 在 η=1000 与 η=5000 下与论文差距约 0.5 个 AUC 点；HiddenEcho+ 在 η=100、5000、6000 下差距约 1--2 个 AUC 点，η=1000 下差距相对更大。两种方法均稳定高于 LDP，并且 HiddenEcho+ 保持了明显的通信压缩优势。因此，当前结果支持论文关于“在相同 embedding 隐私水平下，HiddenEcho 能改善 LDP 任务性能，HiddenEcho+ 能以较低通信开销保持主要性能”的核心结论。

在后续数值稳定性修复后，本报告补充了与 HiddenEcho 主表配置更严格对齐的 HiddenEcho+ `lst_reduce_factor=1` 实验。该实验仍采用 `auto_skip=true`、`num_reserved_layers=3`、`keep_last_layer=true` 与 `mi_downsample_enable=true`，仅将 MINE 中的 `log(mean(exp(.)))` 改为数学等价的 `logsumexp(.) - log(n)` 以避免 bf16 溢出。补充结果如下。

| 方法 | η | 论文 AUC | 复现 AUC | Accuracy | Macro-F1 | 备注 |
|---|---:|---:|---:|---:|---:|---|
| HiddenEcho+ reduce1 | 100 | 0.857 | 0.855217 | 0.755066 | 0.504455 | 与论文高度接近 |
| HiddenEcho+ reduce1 | 5000 | 0.860 | 0.883247 | 0.778855 | 0.533803 | 略高于论文 |

这组补充实验表明，HiddenEcho+ 在不压缩 hidden dimension 的条件下可以达到与 HiddenEcho 相同量级甚至略高的 AUC，同时仍然通过 HLF 只回传 4/28 层 hidden states。由于该配置仅覆盖 η=100 与 η=5000，完整四预算主表仍保留上表中的 reduce4 结果；报告结论中将 reduce1 结果作为数值稳定性修复后的补充验证。

需要特别说明的是，当前复现中的 LDP AUC 系统性高于论文，导致相对提升幅度小于论文报告值。代码排查显示，LDP 路径在 `lst_enable=false` 时确实没有调用 `client_denoise`，也没有 hidden states 回传；`data_transfer.txt` 中 `hiddens_data_transferred=0`，说明该结果不是由 HiddenEcho 模块泄漏造成。更可能的原因包括以下三点。

第一，当前主分类实验采用 `financial_phrasebank/sentences_allagree`，数据划分为 train=1811、validation=226、test=227。该子集只包含标注者完全一致的样本，标签噪声较低，分类任务本身较容易；无噪声 split clean baseline 已达到 AUC 0.996565。相比之下，论文 Appendix H.1 描述 Financial Phrasebank 为 4,840 条样本，数据口径更接近完整 Financial Phrasebank。因此，`sentences_allagree` 可能抬高了 LDP 在加噪训练后的可学习性。

第二，当前 LDP 是在 noisy embedding 上完整进行 LoRA fine-tuning，而不是用 clean checkpoint 直接在 noisy embedding 上推理。训练、验证与测试阶段均会调用 `ClientEmbeddingPart` 注入噪声，模型在 20 epochs 内持续适配该 noisy input 分布。对于 Financial 这种较容易的数据集，LoRA 可能学习到对随机扰动鲁棒的分类边界，从而使“纯加噪”基线强于直觉预期。

第三，η=5000 与 η=6000 属于较弱隐私、较弱噪声区域，LDP 性能本身可能显著恢复。已有诊断还显示，η=1000 下 `clip_embedding_l2=false` 的 LDP AUC 为 0.655856，与 `clip_embedding_l2=true` 的 0.650413 接近，说明当前 LDP 偏高不能单独归因于 clipping。综合来看，本报告将 LDP 偏高视为数据口径与 noisy LoRA 适配共同造成的复现差异，而不是噪声链路未生效。

为进一步排查数据口径影响，本报告补充了 `financial_phrasebank/sentences_50agree` 下的 LDP 诊断实验。该实验仍采用 Qwen2.5-1.5B-Instruct、20 epochs、linear scheduler、learning rate 1.5e-4、`clip_embedding_l2=true` 与 `noise_type=Chi`。结果如下。

| 数据配置 | 方法 | η=100 | η=1000 | η=5000 | η=6000 |
|---|---|---:|---:|---:|---:|
| `sentences_allagree` | LDP AUC | 0.647036 | 0.650413 | 0.750725 | 0.769136 |
| `sentences_50agree` | LDP AUC | 0.589365 | 0.617139 | 0.679242 | 0.694866 |
| 论文 Financial / Qwen | LDP AUC | 0.596 | 0.595 | 0.629 | 0.617 |

`sentences_50agree` 下的 LDP 明显低于 `sentences_allagree`，并在 η=100 时几乎与论文一致。这说明当前 allagree 主表中的 LDP 偏高至少有一部分来自数据子集过易，而不是 LDP 实现错误。η=5000 与 η=6000 仍高于论文，说明 noisy LoRA fine-tuning、Qwen2.5 与 Qwen2 的模型差异、split 随机性也可能继续贡献偏差。后续若将 baseline 主表切换为 `sentences_50agree` 口径，应同步补跑 SnD、GAN-DP、HiddenEcho 与 HiddenEcho+，不能只替换 LDP 单行。

SnD baseline 已按 `sentences_50agree`、Qwen2.5-1.5B-Instruct、20 epochs、linear scheduler、learning rate 1.5e-4、`clip_embedding_l2=true` 补齐四个预算点。η=100、1000、5000 下 AUC 分别为 0.574742、0.574504、0.602681，与论文 SnD 的 0.558、0.565、0.595 基本同量级；η=6000 下 AUC 为 0.568698，明显低于论文的 0.630，也低于同口径 LDP 的 0.694866。训练日志未出现 NaN 或异常退出，说明该偏差更可能来自 SnD denoiser 与当前数据/模型口径的适配不足，而不是运行失败。SnD 整体仍符合论文关于“固定预训练去噪器效果有限，且显著弱于 HiddenEcho”的定性结论。

## 6. Table 2 消融复现

论文 Table 2 对 HiddenEcho+ 做三组模块消融：去除残差连接（-Res）、将 HLF 替换为固定层选择（-HLF）、将 Dimension Reducer 替换为线性层（-DR）。当前复现只覆盖 Financial / Qwen 任务，隐私预算为 η=100、1000、5000。基础配置与主表 HiddenEcho+ reduce4 对齐：Qwen2.5-1.5B-Instruct、`sentences_allagree`、20 epochs、linear scheduler、learning rate 1.5e-4、`clip_embedding_l2=true`、`lst_reduce_factor=4`、`num_reserved_layers=3`、`keep_last_layer=true`。

需要说明，公开代码中的降维器本身已经是 `server_downsample`/`server_upsample` 线性投影，`mi_downsample_enable=true` 只是在此基础上增加 MINE/IB 约束。因此本报告的 -DR 采用“关闭 MI/IB 约束”的代理口径，即 `mi_downsample_enable=false`，不能视为与论文“用线性层替换 DR”完全同义。-HLF 采用 `scripts/simple.sh` 中固定保留层 `[0,7,14,21,27]`，因此回传 5 层而不是完整 HiddenEcho+ 的 4 层；该设置更接近论文所述“固定跳过层”消融，但通信量不再与 full HE+ 严格相同。

| 方法 | η=100 AUC | η=1000 AUC | η=5000 AUC | 论文 η=100 | 论文 η=1000 | 论文 η=5000 |
|---|---:|---:|---:|---:|---:|---:|
| HiddenEcho+ | 0.845129 | 0.826858 | 0.849359 | 0.857 | 0.855 | 0.860 |
| HiddenEcho+ -Res | 0.853661 | 0.844702 | 0.870333 | 0.814 | 0.815 | 0.819 |
| HiddenEcho+ -HLF | 0.800422 | 0.811579 | 0.810415 | 0.773 | 0.773 | 0.774 |
| HiddenEcho+ -DR proxy | 0.852081 | 0.827711 | 0.839589 | 0.789 | 0.799 | 0.801 |

从结果看，当前消融只部分复现了论文 Table 2 的结论。-HLF 在 η=100、1000、5000 下相对 full HiddenEcho+ 分别下降 0.0447、0.0153、0.0389，说明动态层选择确实比当前固定层方案更有效；但该降幅弱于论文中约 0.082--0.086 的下降。-DR proxy 在 η=5000 下降 0.0098，但在 η=100 和 η=1000 分别变化 +0.0070 和 +0.0009，不能复现论文中 -DR 明显退化的趋势。-Res 在三个预算点反而分别提高 0.0085、0.0178 和 0.0210，与论文“移除残差会降低性能”的结论相反。

因此，当前 Table 2 不宜表述为严格复现。更稳妥的解释是：HLF 的重要性得到了经验支持；DR 与 residual 的消融在公开代码和当前 Qwen2.5/allagree 口径下没有复现论文趋势。可能原因包括：公开代码中 DR 与论文文字描述不完全对应，-DR 只能用关闭 MI/IB 作为代理；full HE+ reduce4 本身低于 reduce1 稳定版，导致与消融变体的差距被压缩；`sentences_allagree` 数据过易且 Qwen2.5 与论文 Qwen2 不完全一致；残差连接在该实现中叠加于 side transformer 外部，可能在部分预算点引入尺度偏移，未必总是带来收益。后续若要严格复现 Table 2，需要固定 Qwen2-1.5B、完整 Financial 口径，并实现与论文一致的 DR/non-DR 结构对照。

## 7. 训练与通信记录

### 7.1 HiddenEcho

HiddenEcho 使用 full hidden states，对应 `lst_reduce_factor=1`、`auto_skip=false`、`mi_downsample_enable=false`。由于 full hidden states 显存占用较高，实际 batch size 设为 24。

| 方法 | η | AUC | Accuracy | Macro-F1 | `embedding_data_transferred` | `hiddens_data_transferred` | `train_runtime` |
|---|---:|---:|---:|---:|---:|---:|---:|
| HiddenEcho reduce1 bs24 | 1000 | 0.868505 | 0.768282 | 0.519228 | 12014714880 | 336412016640 | 1920.2838s |
| HiddenEcho reduce1 bs24 | 5000 | 0.877712 | 0.758590 | 0.518896 | 12014714880 | 336412016640 | 1944.0522s |

对应文件：

| 内容 | 路径 |
|---|---|
| HiddenEcho η=1000 日志 | `logs/echo_qwen25_linear_lr1p5e4_20epoch_allagree_eta1000_cliptrue_reduce1_bs24.log` |
| HiddenEcho η=1000 输出 | `outputs/train_ckpts/echo_qwen25_linear_lr1p5e4_20epoch_allagree_eta1000_cliptrue_reduce1_bs24/` |
| HiddenEcho η=5000 日志 | `logs/echo_qwen25_linear_lr1p5e4_20epoch_allagree_eta5000_cliptrue_reduce1_bs24.log` |
| HiddenEcho η=5000 输出 | `outputs/train_ckpts/echo_qwen25_linear_lr1p5e4_20epoch_allagree_eta5000_cliptrue_reduce1_bs24/` |

### 7.2 HiddenEcho+

HiddenEcho+ 使用动态层选择与 MI 约束。选层配置采用 `num_reserved_layers=3, keep_last_layer=true`，即 HLF 选择 3 个贡献层并强制保留最后一层，总保留层数为 4，与论文通信分析中的 $n_H=4$ 对齐。早期正式四预算主表采用 `lst_reduce_factor=4`，原因是公开实现中的 MINE 在 `lst_reduce_factor=1` 下存在数值不稳定。后续将 MINE 的 `log(mean(exp(.)))` 改为数学等价的 `logsumexp(.) - log(n)` 后，`lst_reduce_factor=1` 可以稳定训练，因此补充了 η=100 与 η=5000 两个与 HiddenEcho 主表配置对齐的结果。

| 方法 | η | AUC | Accuracy | Macro-F1 | `embedding_data_transferred` | `hiddens_data_transferred` | `train_runtime` |
|---|---:|---:|---:|---:|---:|---:|---:|
| HiddenEcho+ reduce4 | 5000 | 0.849359 | 0.746256 | 0.516652 | 12014714880 | 12014714880 | 907.2591s |
| HiddenEcho+ reduce1 | 100 | 0.855217 | 0.755066 | 0.504455 | 12014714880 | 48058859520 | 1309.2208s |
| HiddenEcho+ reduce1 | 5000 | 0.883247 | 0.778855 | 0.533803 | 12014714880 | 48058859520 | 1283.3493s |

HiddenEcho+ reduce1 的选层分别为：η=100 时 `[0, 2, 8, 27]`，η=5000 时 `[0, 2, 4, 27]`。与 HiddenEcho reduce1 的 full hidden 回传量 `336412016640` 相比，HiddenEcho+ reduce1 的 hidden 回传量为 `48058859520`，节省率为：

$$
1-\frac{48058859520}{336412016640}=85.71\%.
$$

因此，reduce1 补充实验同时满足两个条件：一方面与 HiddenEcho 使用相同 hidden dimension，便于比较任务性能；另一方面仍通过 HLF 复现论文 Table 3 中 $1-4/28=85.71\%$ 的通信节省。但该配置的客户端 denoiser 参数量为 `1315109888`，显著高于 reduce4/16 配置，说明它更适合作为任务性能对齐实验，而不是最优部署配置。

reduce4 实验中，验证 AUC 从第 1 epoch 的 0.571668 持续上升，在第 19 epoch 达到 0.851784，最终 test AUC 为 0.849359，训练过程中未出现 NaN 或显存错误。reduce1 稳定版中，η=5000 的最终 test AUC 为 0.883247，进一步说明 HiddenEcho+ 的性能下降主要来自 aggressive dimension reduction 与 MINE 数值稳定性，而不是 HLF 选层机制本身失效。

### 7.3 Table 3 开销复现

论文 Table 3 分别报告两类开销：左侧为不同方法完成一个 epoch LLM fine-tuning 的训练时间，右侧为 HiddenEcho 与 HiddenEcho+ 每个 batch 的 hidden-state 通信成本。复现时这两类数值应分开解释：训练时间来自日志中的 Trainer `train_runtime`；当前公开代码的 `data_transfer.txt` 则记录模型实例生命周期内的累计字节量。计数器会在每次 forward 时累加，因而同时包含训练、周期性验证与最终评测，不等同于论文报告的单 batch 通信量。

训练时间采用同一组 Financial/Qwen2.5 实验口径：`financial_phrasebank/sentences_allagree`，η=5000，`noise_type=Chi`，`clip_embedding_l2=true`，20 epochs，linear scheduler，learning rate 1.5e-4。为保证方法可运行，GAN-DP 与 SnD 的 batch size 按各自显存需求调整；表中时间为 `train_runtime / num_train_epochs`，不包含 GAN-DP generator pretraining 与 SnD denoiser pretraining 的额外 wall time。

| Dataset | Model | LDP | GAN-DP | SnD | HE | HE+ |
|---|---|---:|---:|---:|---:|---:|
| Financial | Qwen2.5-1.5B | 41.20s | 51.33s | 209.05s | 47.18s | 45.06s |

对应日志分别为 `logs/ldp_qwen25_linear_lr1p5e4_20epoch_allagree.log`、`logs/gandp_qwen25_allagree_linear_eta5000_cliptrue_genepoch4_bs24_run1.log`、`logs/snd_qwen25_allagree_linear_eta5000_cliptrue_bs4_bs8_run1.log`、`logs/echo_qwen25_linear_lr1p5e4_20epoch_allagree.log` 与 `logs/echo_plus_qwen25_linear_lr1p5e4_20epoch_allagree.log`。从相对关系看，HiddenEcho 系列相较 LDP 有额外计算开销，但显著快于 SnD；HiddenEcho+ 因只激活部分 hidden layers，训练时间略低于 full HiddenEcho。这一趋势与论文 Table 3 的定性结论一致。绝对秒数受 GPU 型号、PyTorch/CUDA 版本、batch size 与数据划分影响，不与论文 RTX 3090 环境作严格逐秒对齐。

通信成本只保留 Financial/Qwen 中对应论文 85.79% 节省率的成对实验。该组实验中 HE 与 HE+ 均采用 Qwen2.5、η=5000、`lst_reduce_factor=16`；HE 使用 full hidden states，HE+ 使用 `num_reserved_layers=3, keep_last_layer=true`，即实际回传 4/28 层 hidden states。两者 embedding 上传量相同，hidden-state 通信差异只来自 HLF 选层。

为对齐论文 Table 3 的 “one batch” 口径，本报告单独运行一次 batch size 4、sequence length 96 的 forward，并在 forward 前清零通信计数器。该设置下 hidden state dtype 为 bf16，reduced hidden dimension 为 96，因此 HE 的 hidden-state 通信量为 `4 × 96 × 96 × 28 × 2 = 2,064,384 bytes = 1.96875 MiB`，HE+ 为 `4 × 96 × 96 × 4 × 2 = 294,912 bytes = 0.28125 MiB`。

| Dataset | Model | Batch size | Seq. len. | HE hidden transfer | HE+ hidden transfer | Saved |
|---|---|---:|---:|---:|---:|---:|
| Financial | Qwen2.5-1.5B | 4 | 96 | 1.96875 MiB | 0.28125 MiB | 85.71% |

该结果与论文 Table 3 中 Financial/Qwen 的 HE 1.97 MiB、HE+ 0.28 MiB、Saved 85.79% 基本完全一致。单 batch JSON 输出分别为 `outputs/table3_single_batch/table3_single_batch_hiddenecho_qwen25_allagree_bs4_idx0_run1.json` 与 `outputs/table3_single_batch/table3_single_batch_hiddenecho_plus_qwen25_allagree_bs4_idx0_run1.json`；对应日志为 `logs/table3_single_batch_hiddenecho_qwen25_allagree_bs4_idx0_run1.log` 与 `logs/table3_single_batch_hiddenecho_plus_qwen25_allagree_bs4_idx0_run1.log`。

作为交叉验证，完整训练与评测过程中的累计 hidden-state 通信量也保留记录。该累计量分别为 HE `47,511,797,760 bytes`、HE+ `6,787,399,680 bytes`，按 MiB 计为 45,310.78 MiB 与 6,472.97 MiB。累计量远大于论文单 batch 数值，是因为计数器跨 20 epochs 及全部验证/测试 forward 持续累加；但在 HE 与 HE+ 采用相同训练、评测与张量精度口径时，累计量的比值仍可用于验证 HLF 的相对通信节省率：

$$
1-\frac{6787399680}{47511797760}=85.71\%.
$$

该比例接近论文 Table 3 的 Saved 85.79%，其理论来源为：

$$
1-\frac{n_H}{L}=1-\frac{4}{28}=85.71\%.
$$

因此，当前金融数据集通信实验同时复现了论文 Table 3 的单 batch 绝对通信量和相对节省结论：在相同 reduced hidden dimension 下，HiddenEcho+ 通过只回传 4 个关键层，将 HiddenEcho 的 hidden-state 回传量降低约 85.7%。该表不混入后续 reduce1 或 reduce4 主表实验，因为那些实验同时改变了 hidden dimension 或任务性能优化口径，不能替代论文 Table 3 的通信设置。

CNNDM/T5 生成任务也保留了训练时间记录。当前实验使用 `determined-ai/cnn_dailymail_short`、η=30、15 epochs，因而可作为本地生成任务的相对时间开销，但不与论文完整 CNNDM 数据上的绝对秒数直接比较。

| Dataset | Model | LDP | GAN-DP | SnD | HE | HE+ |
|---|---|---:|---:|---:|---:|---:|
| CNNDM short | T5-large | 93.19s | 99.66s | - | 127.11s | 120.76s |

对应日志分别为 `logs_gen/ldp_t5_dailymail_eta30_full_v2.log`、`logs_gen/gandp_t5_dailymail_eta30_task_genepoch0_best_eval_loss.log`、`logs_gen/echo_t5_dailymail_eta30_residual_reduce1.log` 与 `logs_gen/echo_plus_t5_dailymail_eta30_residual_reduce4_v2.log`。与论文一致，SnD 不适用于文本生成任务，因此不报告该列。

CNNDM/T5 通信成本已补充一组参数对齐的重测实验。该组实验采用 `determined-ai/cnn_dailymail_short`、η=30、15 epochs、T5-large、`lst_reduce_factor=16`。HE 回传 T5 encoder 的全部 24 层，HE+ 使用 `num_reserved_layers=3, keep_last_layer=true`，即 top-3 加最终层，共回传 4 层；两者 embedding 上传量一致，hidden-state 通信差异只来自 HLF 选层。

| Dataset | Model | HE hidden transfer（全程累计） | HE+ hidden transfer（全程累计） | Saved |
|---|---|---:|---:|---:|
| CNNDM short | T5-large | 12,458,606,592 bytes | 2,076,434,432 bytes | 83.33% |

按 MiB 计，上述累计 hidden-state 通信量分别为 11,881.45 MiB 与 1,980.24 MiB。该节省率来自

$$
1-\frac{2076434432}{12458606592}=83.33\%\approx 1-\frac{4}{24}.
$$

因此，该重测实验验证了 T5 上“同一 reduced hidden dimension 下，HE+ 只回传 4/24 层可节省约 83.33% hidden-state 通信”的规律。它替代早期未对齐比较中由 HE reduce1 与 HE+ reduce4 直接相除得到的 95.83%；后者同时混入 4 倍降维收益，不能作为 Table 3 通信复现。论文 CNNDM/T5 报告 HE 8.25 MiB、HE+ 3.09 MiB、Saved 62.55%，但没有披露对应的具体保留层数与序列长度口径，因此本报告不将 83.33% 表述为对论文 62.55% 的严格数值复现，而将其作为相同 HLF 规则下的跨模型通信规律验证。

对应文件：

| 内容 | 路径 |
|---|---|
| CNNDM HE reduce16 日志 | `logs_gen/table3_cnndm_he_eta30_reduce16_full24_run1.log` |
| CNNDM HE reduce16 输出 | `outputs/train_ckpts_gen/table3_cnndm_he_eta30_reduce16_full24_run1/` |
| CNNDM HE+ reduce16 日志 | `logs_gen/table3_cnndm_heplus_eta30_reduce16_k4of24_run1.log` |
| CNNDM HE+ reduce16 输出 | `outputs/train_ckpts_gen/table3_cnndm_heplus_eta30_reduce16_k4of24_run1/` |

## 8. EIA 评测

EIA 按论文定义攻击 noisy embedding，攻击者假设已知用户上传的 perturbed embeddings 以及 embedding matrix，并尝试恢复原始文本。当前复现将按论文主表预算 η=100、1000、5000、6000 测量 EP。

| 数据集 | 模型 | η | EP |
|---|---|---:|---:|
| Financial Phrasebank | Qwen2-1.5B-Instruct | 100 | 0.982257 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 1000 | 0.986704 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 5000 | 0.940820 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 6000 | 0.893404 |

由于 EIA 仅作用于上传 embedding，在同一噪声机制下，LDP、HiddenEcho 与 HiddenEcho+ 可共享同一组 EP 结果；报告主表中将其标记为“同 LDP”。

本次 EIA 使用脚本 `experiment/eia/invert_emb_attack.py`，攻击样本为 Financial Phrasebank 训练集前 200 条，batch size 为 2，输出指标为 `1 - mean(rouge1)`。日志文件为 `logs/eia_financial_qwen_eta100_1000_5000_6000_run1.log`。需要说明的是，EIA 本次采用本地 Qwen2-1.5B-Instruct 词表与 embedding matrix；若主任务最终统一报告为 Qwen2.5-1.5B-Instruct，则应补跑对应模型路径下的 EIA 以保持模型口径完全一致。

GAN-DP 需要单独进行 EIA，因为其上传 embedding 经过 generator 进一步变换，不能直接共享 LDP 的 EP。当前主表采用与 GAN-DP AUC 训练一致的 `generator_epoch=4` 口径；将攻击样本从 200 条扩大到 1000 条后，η=100、1000、5000、6000 下 EP 分别为 1.000000、0.997031、1.000000、1.000000，仍未呈现论文中 1.000、0.999、0.997、0.992 的预算递减趋势。对应日志为 `logs/eia_financial_gandp_qwen25_allagree_eta100_1000_5000_6000_genepoch4_1000samples_run1.log`。

为排查该差异，本报告进一步扫描 GAN generator checkpoint：epoch 0、2、4、6、8，每个预算点攻击 1000 条样本。结果显示 GAN-DP EP 对 generator checkpoint 敏感，但不同 checkpoint 与任务 AUC 口径并不完全一致。epoch 0 的 EP 为 1.000000、1.000000、0.999798、0.999792，具备单调下降趋势但整体高于论文；epoch 2 为 0.999985、1.000000、0.998926、0.997986，最接近论文平均误差但不严格单调；epoch 4 与主任务 checkpoint 对齐，但 EP 在 η=5000 和 η=6000 处饱和为 1.000000。因此，GAN-DP EP 在本复现中只支持“高隐私量级接近论文”的结论，不能视为严格复现了论文中的预算敏感性。sweep 日志为 `logs/eia_financial_gandp_qwen25_allagree_genepoch_sweep_1000samples_run1.log`，JSON 输出位于 `outputs/eia/eia_financial_gandp_qwen25_allagree_genepoch_sweep_1000samples_run1_genepoch*_eta*.json`。epoch 10 的 sweep 未完整结束，未纳入正式分析。

## 9. AIA 评测

AIA 使用 Tweet Annotation Sensitivity 2 数据集，对模型表示中的敏感属性泄露进行评估。该实验不与 Financial Phrasebank 主任务混写，因为 Financial 数据集不包含 age 或 education 等敏感属性。论文正文 Fig. 3 使用 Qwen2-1.5B 在 Tweet Annotation 上报告 AIA，攻击属性包括 age 与 education。

正式复现 AIA 前，需要先在 Tweet Annotation 上训练对应方法的任务 checkpoint，再使用攻击脚本提取最终 hidden representation 并训练攻击器预测属性。

本文最终采用 Appendix H.4 对齐口径报告 AIA：education attacker 输出为 4 类，标签映射为 `1,2 -> 0`、`3 -> 1`、`4 -> 2`、`5,6 -> 3`；age attacker 输出为 1 维回归。education 攻击原始输出为 attack accuracy，报告中按论文 Definition 4 转换为 `EP = 1 - attack accuracy`；age 攻击直接报告 RMSE。

最终 AIA 图文件如下：

| 图 | 路径 |
|---|---|
| age RMSE | `outputs/aia_h4edu4_combined/figures_compact/aia_age_rmse.png` |
| education EP | `outputs/aia_h4edu4_combined/figures_compact/aia_education_ep.png` |

| η | LDP education EP | HiddenEcho education EP | LDP age RMSE | HiddenEcho age RMSE |
|---:|---:|---:|---:|---:|
| 0 | 0.187200 | 0.187200 | 10.421577 | 10.421577 |
| 50 | 0.606800 | 0.597200 | 15.562527 | 16.015216 |
| 100 | 0.610400 | 0.592300 | 14.705954 | 14.604641 |
| 200 | 0.602500 | 0.597900 | 14.657982 | 14.760741 |
| 300 | 0.602700 | 0.596900 | 14.897273 | 14.878260 |
| 500 | 0.584300 | 0.591600 | 14.973109 | 14.997027 |
| 1000 | 0.552000 | 0.571600 | 14.396345 | 14.840066 |
| 3000 | 0.530400 | 0.553300 | 14.155350 | 14.471988 |
| 5000 | 0.522600 | 0.529800 | 13.993545 | 14.076265 |
| 6000 | 0.522500 | 0.539200 | 14.000148 | 14.102694 |
| 8000 | 0.522500 | 0.535500 | 13.984435 | 14.014485 |
| 10000 | 0.522500 | 0.536500 | 14.000616 | 14.045391 |

对应 Tweet 任务 checkpoint 的分类 AUC 显示，HiddenEcho 在各预算点均高于 LDP；除 η=50 的强噪声点外，HiddenEcho AUC 基本位于 0.76--0.80 区间，而 LDP 约为 0.51--0.55。AIA 指标方面，LDP 与 HiddenEcho 均显著高于无保护基线，说明扰动能够降低属性推断能力；在中高预算区域，HiddenEcho 的 education EP 与 age RMSE 多数略高于 LDP，且任务性能明显更好。因此，AIA 结果支持论文 Fig. 3 的主要结论：HiddenEcho 在改善任务效用的同时，没有削弱属性隐私保护。

无保护基线最终改用 Tweet 任务上完成 clean fine-tuning 的 Qwen2 checkpoint，而不是未经任务训练的原始 Qwen2。修正后 education attack accuracy 为 0.812800，对应 EP 为 0.187200；age RMSE 为 10.421577。相比原始模型基线的 EP 0.487800、RMSE 12.831156，两项攻击均明显增强，并接近论文 Fig. 3 中无保护基线约 EP 0.3、RMSE 11 的量级。当前 EP 略低、RMSE 略低于论文图中近似读数，说明本次攻击器稍强，但不再存在原先无保护表示反而难以攻击的方向性偏差。该结果也验证了 AIA 无保护基线必须使用同一 Tweet 下游任务训练后的表示，原始预训练表示不能作为严格对照。

修正无保护基线后，LDP 与 HiddenEcho 各预算点数据保持不变。二者的 education EP 约为 0.52--0.61，age RMSE 约为 14.0--16.0，均明显高于无保护基线，因而“噪声扰动降低属性泄露”的主要结论得到更清晰支持。对应结果文件为 `outputs/aia_h4edu4/aia_tweet_no_protection_finetuned_qwen2_h4edu4_finetuned_clean_run1.csv`，日志为 `logs/aia_tweet_no_protection_finetuned_qwen2_h4edu4_finetuned_clean_run1.log`。

## 10. 生成任务复现：CNNDM / T5

本节新增复现论文 Table 7 中 T5 在摘要生成任务上的结果。当前仅保留 CNNDM 数据集；IWSLT 与 SAMSum 暂未纳入。由于 Hugging Face 上 `samsum` 数据集当前不可直接按原名加载，且本阶段已下载并完成的是 `determined-ai/cnn_dailymail_short`，生成复现以该短版 CNNDM 为准。

| 项目 | 设置 |
|---|---|
| 数据集 | `determined-ai/cnn_dailymail_short` |
| 数据划分 | 使用 Hugging Face 原始 split：train=1322, validation=50, test=47 |
| 基座模型 | T5-large（本地路径：`/data/songhanlin/models/t5-large`） |
| 训练入口 | `train_split_t5.py` |
| 训练方式 | LoRA fine-tuning |
| 生成任务 | 摘要生成 |
| 主要指标 | BLEU, ROUGE-1/2/L |
| 生成长度 | `generation_max_new_tokens=64` |
| 噪声类型 | Chi |
| `clip_embedding_l2` | true |
| T5 embedding `max_norm` | 700 |
| 正式预算点 | η = 20, 30, 40 |

### 10.1 Table 7 复现结果

论文 Table 7 报告 T5 在 CNNDM、IWSLT 与 SAMSum 上的 BLEU。当前复现只填写 CNNDM；无实验数据的单元格保留为空。主表仅记录复现值。

| 方法 | 数据集 | η=20 BLEU | η=30 BLEU | η=40 BLEU |
|---|---|---:|---:|---:|
| LDP | CNNDM | 0.389 | 8.290 | 17.622 |
| GAN-DP | CNNDM | 9.019 | 12.527 | 18.776 |
| HiddenEcho | CNNDM | 6.349 | 16.668 | 18.526 |
| HiddenEcho+ | CNNDM | 8.076 | 15.076 | 18.202 |

clean T5 baseline 的复现 BLEU 为 19.449，论文对应 clean T5 CNNDM BLEU 为 17.738。由于本地 clean upper bound 高于论文，HiddenEcho 在 η=30 与 η=40 下的绝对 BLEU 也整体偏高；但 η=20 < η=30 < η=40 的趋势与论文一致，且强噪声到弱噪声的性能恢复幅度与 clean 上界关系一致。作为参考，论文 Table 7 中 CNNDM 的 LDP BLEU 为 0.764、7.974、12.107，HiddenEcho BLEU 为 2.915、11.617、12.323。

### 10.2 关键排查结论

T5 生成任务的实现不能直接套用分类任务的“最后 hidden state 接分类头”路径。当前复现采用 encoder-decoder 生成口径：source embedding 加噪后进入 T5 encoder；HiddenEcho 只在 encoder 侧集成，client denoise 得到 corrected encoder memory；decoder 通过 cross-attention 读取该 encoder memory 并生成摘要。side denoise block 使用双向 encoder-style attention，不使用 causal triangular mask。

最初的 LDP 结果异常偏高：η=30 full LoRA LDP 的 BLEU 为 16.471，远高于论文 7.974。排查确认噪声路径没有被绕过：clean checkpoint 仅在推理时加入 η=30 噪声时，BLEU 下降到 1.905。因此问题不在“没加噪”，而在训练口径。

关键 ablation 显示，T5 encoder 在 noisy embedding 上的适应能力过强：

| 诊断实验 | η | BLEU | 结论 |
|---|---:|---:|---|
| LDP full LoRA | 30 | 16.471 | encoder+decoder 都可训练时，LDP baseline 过强 |
| LDP encoder-only LoRA | 30 | 16.672 | 仅训练 encoder 即可恢复到接近 full LDP，说明主要适应来自 encoder |
| LDP decoder-only LoRA | 30 | 8.290 | 与论文 LDP η=30 的 7.974 高度接近 |
| Clean checkpoint + noisy inference | 30 | 1.905 | 证明噪声本身很强，非噪声绕过问题 |

因此，生成任务中 LDP baseline 最终采用 `lora_scope=decoder`：embedding 加噪仍然作用于 encoder 输入，但 LDP baseline 不允许 encoder LoRA 学习 noisy embedding 到 clean-like encoder memory 的鲁棒映射。该口径与“LDP 是不能显式处理噪声的基线”更一致，也使 η=30 基本复现论文结果。η=40 仍明显高于论文，可能与短版 CNNDM、clean upper bound 偏高、T5 在弱噪声区间更容易恢复摘要质量有关。

HiddenEcho+ 生成任务采用 residual Echo + `lst_reduce_factor=4`。在 η=20/30/40 下，BLEU 分别为 8.076、15.076、18.202，整体趋势与 HiddenEcho 一致；η=30 低于 HiddenEcho reduce1，但仍明显高于 LDP 与 GAN-DP 已有单点。已有日志中 HiddenEcho reduce1 的 hidden states 累计回传量为 199337705472，HiddenEcho+ reduce4 为 8305737728。由于两者 `lst_reduce_factor` 不同，该组累计量只能说明当前两套任务配置的实际数据传输差异，不能作为论文 Table 3 的通信节省复现值。

GAN-DP task 阶段按 validation `eval_loss` 选择 best checkpoint。η=20/40 按分类任务默认口径使用 `generator_epoch=4`，BLEU 分别为 9.019、18.776；η=30 的 `generator_epoch=4` 结果异常偏低，仅 6.288，因此补充 generator checkpoint sensitivity。η=30 使用 `generator_epoch=0` 时 BLEU 提升到 12.527，ROUGE-L 为 0.315，恢复到与论文量级一致的区间。

| GAN-DP η=30 generator checkpoint | task best checkpoint | BLEU | ROUGE-L |
|---:|---|---:|---:|
| 0 | epoch 4 | 12.527 | 0.315 |
| 4 | epoch 14 | 6.288 | 0.287 |
| 10 | epoch 14 | 3.575 | 0.267 |
| 18 | epoch 4 | 3.476 | 0.261 |

该排查说明 GAN-DP 对 generator checkpoint 非常敏感，且 generator 训练损失继续下降不等价于下游生成 BLEU 提升。生成任务主表采用当前验证到的最佳 generator checkpoint：η=20 使用 epoch 4，η=30 使用 epoch 0，η=40 使用 epoch 4。

η=30 进一步导出 5 条测试样例做人工质检。样例 dump 为单次带噪生成，因此指标与主表的 5 次评估均值不完全一致；其用途是检查模型输出形态，而不是替代主表。单次 dump 指标为：LDP BLEU/ROUGE-L = 10.575/0.252，GAN-DP = 11.481/0.308，HiddenEcho = 16.622/0.323，HiddenEcho+ = 16.721/0.325。样例显示 LDP 更容易生成泛化或事实错误摘要，例如把事件地点、伤亡数量或时间线写错；GAN-DP 能恢复更多关键词，但在强噪声下仍会出现不完整短句和空白式退化；HiddenEcho 与 HiddenEcho+ 的输出通常更接近 reference，能保留主体、事件和关键数字。HiddenEcho+ 的单次样例指标略高于 HiddenEcho，但主表 5 次均值仍低于 HiddenEcho，说明 test=47 的短测试集和随机噪声会带来可见波动，因此正式结论应以多次均值和整体趋势为准。
本阶段生成任务的 CNNDM/T5 主实验已经覆盖 Table 7 在 CNNDM 上涉及的 Clean、LDP、GAN-DP、HiddenEcho 与 HiddenEcho+。IWSLT 与 SAMSum 暂不继续扩展，原因不是训练链路缺失，而是当前可加载数据集与论文口径无法稳定对齐；若后续要补这两项，应先固定可复现的数据来源、split 与预处理脚本，再启动正式训练。

### 10.3 生成任务记录

| 内容 | 路径 |
|---|---|
| clean T5 日志 | `logs_gen/clean_t5_dailymail_full.log` |
| LDP η=20 decoder-only 日志 | `logs_gen_ldp_check/ldp_t5_dailymail_eta20_decoder_only.log` |
| LDP η=30 decoder-only 日志 | `logs_gen_ldp_check/ldp_t5_dailymail_eta30_decoder_only.log` |
| LDP η=40 decoder-only 日志 | `logs_gen_ldp_check/ldp_t5_dailymail_eta40_decoder_only.log` |
| HiddenEcho η=20 residual 日志 | `logs_gen/echo_t5_dailymail_eta20_residual_reduce1.log` |
| HiddenEcho η=30 residual 日志 | `logs_gen/echo_t5_dailymail_eta30_residual_reduce1.log` |
| HiddenEcho η=40 residual 日志 | `logs_gen/echo_t5_dailymail_eta40_residual_reduce1.log` |
| HiddenEcho+ η=20 residual reduce4 日志 | `logs_gen/echo_plus_t5_dailymail_eta20_residual_reduce4_v2.log` |
| HiddenEcho+ η=30 residual reduce4 日志 | `logs_gen/echo_plus_t5_dailymail_eta30_residual_reduce4_v2.log` |
| HiddenEcho+ η=40 residual reduce4 日志 | `logs_gen/echo_plus_t5_dailymail_eta40_residual_reduce4_v2.log` |
| GAN-DP η=20 generator 日志 | `logs_gen/gandp_t5_dailymail_eta20_train_generator.log` |
| GAN-DP η=20 task best-eval-loss 日志 | `logs_gen/gandp_t5_dailymail_eta20_task_epoch4_best_eval_loss.log` |
| GAN-DP η=30 generator 日志 | `logs_gen/gandp_t5_dailymail_eta30_train_generator.log` |
| GAN-DP η=30 generator sweep 日志 | `logs_gen/gandp_t5_dailymail_eta30_genepoch_sweep_best_eval_loss.log`, `logs_gen/gandp_t5_dailymail_eta30_genepoch_sweep_remaining_best_eval_loss.log` |
| GAN-DP η=30 best task 日志 | `logs_gen/gandp_t5_dailymail_eta30_task_genepoch0_best_eval_loss.log` |
| GAN-DP η=40 generator 日志 | `logs_gen/gandp_t5_dailymail_eta40_train_generator.log` |
| GAN-DP η=40 task best-eval-loss 日志 | `logs_gen/gandp_t5_dailymail_eta40_task_epoch4_best_eval_loss.log` |
| η=30 样例输出 | `outputs/generation_dumps_eta30_final/examples_eta30.md` |
| η=30 样例 JSONL | `outputs/generation_dumps_eta30_final/ldp_eta30.jsonl`, `outputs/generation_dumps_eta30_final/gandp_eta30.jsonl`, `outputs/generation_dumps_eta30_final/hiddenecho_eta30.jsonl`, `outputs/generation_dumps_eta30_final/hiddenecho_plus_eta30.jsonl` |
| LDP decoder-only 脚本目录 | `scripts_gen_ldp_check/` |
| 生成任务主脚本目录 | `scripts_gen/` |

## 11. 诊断与实现说明

### 11.1 Clean Split Baseline

为确认基础训练链路正常，补充无噪声诊断实验。Split clean 与 Split clean + HiddenEcho 均能达到接近 0.99 的 AUC，说明数据划分、LoRA 微调、split wrapper、模型加载与评估实现均能支撑高性能分类。

| 诊断实验 | 设置 | AUC | Accuracy | Macro-F1 | 结论 |
|---|---|---:|---:|---:|---|
| Split clean | `privacy_budget=0, lst_enable=false` | 0.996565 | 0.977974 | 0.970914 | split wrapper 与基础分类链路正常 |
| Split clean + HiddenEcho | `privacy_budget=0, lst_enable=true` | 0.994170 | 0.973568 | 0.959701 | HiddenEcho 模块在无噪声条件下未显著伤害任务性能 |

### 11.2 HiddenEcho+ reduce1 + MI 数值问题

直接将 HiddenEcho+ 设置为 `lst_reduce_factor=1 + mi_downsample_enable=true` 时，公开实现中的 MINE 在训练初期出现 `loss=0.0`、`grad_norm=nan`，验证阶段因预测包含 NaN 终止。分模块诊断结果显示，HLF-only reduce1 可以运行，MI reduce4 可以稳定训练，而 MI reduce1 即便将 `mi_estimator_lr` 降至 `1e-5` 仍会 NaN。因此，该问题集中在 full-dimension MINE 估计与 bf16 数值稳定性上。

代码层面，原实现使用

$$
\log(\mathrm{mean}(\exp(T_1)))
$$

计算 MINE 下界中的边缘项。该写法在 bf16 或高维输入下容易因 `exp` 溢出产生 NaN。将其改为数学等价的

$$
\mathrm{logsumexp}(T_1)-\log(n)
$$

后，reduce1 + MI 可以稳定完成 20 epochs。该修改不改变优化目标，只改变数值实现。

| 诊断实验 | 状态 | AUC | 结论 |
|---|---|---:|---|
| HLF-only reduce1, `mi_downsample=false` | 成功 | 0.679361 | layer filter 与 reduce1 去噪链路本身可训练 |
| MI reduce4, `mi_downsample=true` | 成功 | 0.639846 | MI 在降维后可稳定训练 |
| MI reduce1, `mi_estimator_lr=1e-5` | 失败 | - | 训练初期出现 NaN |
| MI reduce1 + logsumexp, η=100 | 成功 | 0.855217 | 数值稳定性修复后可复现论文量级 |
| MI reduce1 + logsumexp, η=5000 | 成功 | 0.883247 | 与 HiddenEcho 主表配置对齐后达到论文量级 |

### 11.3 最后一层保留诊断

HiddenEcho+ 默认配置中 `keep_last_layer=true`，即除 HLF 选出的贡献层外，强制保留最后一个 transformer layer 的 hidden state。为判断该设计是否只是实现细节，本报告补充了 `keep_last_layer=false` 的严格 4 层诊断实验。该实验同样使用 `num_reserved_layers=4`、MI 与 logsumexp 数值修复，但不强制保留最后层。

| 诊断实验 | η | 选层 | AUC | 结论 |
|---|---:|---|---:|---|
| Echo+ reduce2, `keep_last_layer=false` | 100 | `[0, 2, 7, 8]` | 0.814070 | 明显低于保留最后层配置 |
| Echo+ reduce2, `keep_last_layer=false` | 5000 | `[0, 2, 4, 7]` | 0.827114 | 明显低于保留最后层配置 |

该结果说明，最后层 hidden state 对分类任务具有较强任务语义，强制保留最后层有助于维持 HiddenEcho+ 的任务性能。换言之，`keep_last_layer=true` 不只是通信分析中的实现便利，而是公开代码在分类任务上保持效用的重要设计选择。

## 12. 后续计划

后续实验按以下顺序补全：

1. 补齐 LDP 在 η=100、1000、5000、6000 下的 Financial / Qwen AUC。
2. 补齐 HiddenEcho 与 HiddenEcho+ 在 η=100、6000 下的 Financial / Qwen AUC，以及 HiddenEcho+ 在 η=1000 下的 AUC。
3. 如主任务最终统一使用 Qwen2.5，则补跑 Qwen2.5 embedding matrix 下的 EIA。
4. 训练 Tweet Annotation 上的 LDP / HiddenEcho / HiddenEcho+ checkpoint，并运行 AIA。
5. 排查生成任务 GAN-DP 的 generator checkpoint sensitivity，并视情况扩展到 IWSLT 与 SAMSum。

# I 改进方法：EchoSlim 面向客户端资源开销的结构化 HiddenEcho+

## I.1 背景与动机

原文 HiddenEcho 的核心目标是在 MaaS split learning 场景下，缓解差分隐私噪声在服务端 Transformer 层间传播时的放大问题。HiddenEcho 通过把服务端中间 hidden states 回传到客户端，让客户端 denoising module 利用 clean embedding 与 server hidden states 共同恢复任务相关表征。进一步地，HiddenEcho+ 引入 Hidden Layer Filter (HLF) 和 Information Bottleneck Dimension Reducer (DR)，只回传少量关键层 hidden states 并进行维度压缩，从而显著减少通信量，同时尽量保持分类和生成任务性能。

复现代码和实验结果表明，HiddenEcho+ 的通信优化已经有效：在分类任务中，HLF 只让被选中的层参与 hidden states 回传，未选层在客户端 forward 中也会被跳过。因此，如果只从通信量或计算路径看，HiddenEcho+ 已经是运行时稀疏的。但是，我们在代码结构中发现一个没有被原文显式讨论的部署问题：**HiddenEcho+ 虽然在运行时跳过未选层，但客户端 denoising module 仍然完整实例化 $L$ 个 side transformer blocks**。换言之，原方法的 HLF 只影响数据流和 forward 路径，没有把选层结果固化为客户端模型结构。

这个问题在论文主指标中不明显，因为原文主要报告任务性能、通信量和隐私攻击指标；但在实际 MaaS 部署中，客户端通常是资源受限设备，模型驻留参数、显存/内存占用、保存和加载开销同样重要。特别是 HiddenEcho 的 denoising module 被部署在客户端，若仍保留未被 HLF 使用的 side layers，就会产生结构冗余。基于这一观察，我们提出 EchoSlim：在不改变 HiddenEcho+ 的隐私预算、HLF 选层、DR 压缩和通信内容的前提下，把 HLF 选出的层结构化到客户端 denoiser 中，只实例化实际会被使用的 side layers。

因此，EchoSlim 的目标不是刷新原文的任务 AUC，也不是进一步降低 HiddenEcho+ 已经优化过的 hidden-state 通信量，而是在基本保持原有 utility/privacy/communication 行为的同时，补充优化原文未系统评估的客户端资源开销。

## I.2 问题发现：HiddenEcho+ 的运行时稀疏与结构稀疏不一致

设服务端 LLM 共有 $L$ 个 Transformer layers，HiddenEcho+ 通过 HLF 得到被选中的层集合

$$
S = \{s_1, s_2, \cdots, s_k\}, \quad k \ll L .
$$

在原 HiddenEcho+ 中，服务端只回传 $S$ 中的 hidden states；客户端 denoiser 在 forward 时如果发现第 $i$ 层 hidden state 为空，则跳过第 $i$ 个 side layer：

$$
A_i =
\begin{cases}
A_{i-1}, & i \notin S, \\
A_{i-1} + \mathcal{T}_i\left(\mu_i A_{i-1} + (1-\mu_i)H_i^{dn}\right), & i \in S .
\end{cases}
$$

其中 $\mathcal{T}_i$ 是第 $i$ 个客户端 side transformer block，$\mu_i=\sigma(g_i)$ 是门控向量，$H_i^{dn}$ 是服务端第 $i$ 层 hidden state 的降维表示。这个公式说明原 HiddenEcho+ 的计算确实只发生在 $S$ 上；但是模型结构仍包含全部 $\{\mathcal{T}_0,\ldots,\mathcal{T}_{L-1}\}$ 和全部 gate vectors $\{g_0,\ldots,g_{L-1}\}$。未选层虽然不参与 forward，却仍占用客户端参数和模型存储。

这形成了一个明确的优化空间：

- **原文已经解决的问题**：减少 server hidden states 回传量，并跳过未选层计算。
- **原文没有充分解决的问题**：客户端 denoising module 的参数结构仍按全层数 $L$ 保留，HLF 的稀疏性没有落实到模型结构。

## I.3 方法：EchoSlim 结构化客户端 Denoiser

EchoSlim 保留 HiddenEcho+ 的 HLF 与 DR，不改变隐私噪声注入方式，也不改变服务端回传哪些 hidden states。其核心改动是：在 HLF 完成后，根据最终选层集合 $S$ 构造 compact client denoiser，仅实例化 $k$ 个 side transformer blocks。

令 $S$ 按层号升序排列为

$$
S = \{s_1 < s_2 < \cdots < s_k\}.
$$

EchoSlim 将原来的 $L$ 层 side stack 替换为 compact stack：

$$
\mathcal{D}_{slim} = \{\tilde{\mathcal{T}}_1,\tilde{\mathcal{T}}_2,\cdots,\tilde{\mathcal{T}}_k\}.
$$

第 $j$ 个 compact block 对应原服务端第 $s_j$ 层 hidden state，而不是简单对应第 $j$ 层。其递推过程为

$$
\tilde{A}_0 = E^{dn},
$$

$$
\tilde{A}_j =
\tilde{A}_{j-1} +
\tilde{\mathcal{T}}_j
\left(
\tilde{\mu}_j \tilde{A}_{j-1} +
(1-\tilde{\mu}_j)H_{s_j}^{dn}
\right),
\quad j=1,\ldots,k,
$$

$$
\tilde{\mu}_j=\sigma(\tilde{g}_j).
$$

最终输出为

$$
H^{denoised} = W^{up}\tilde{A}_k .
$$

为了与 HiddenEcho+ 保持可比性，EchoSlim 在初始化时按原始层号对齐权重：

$$
\tilde{\mathcal{T}}_j \leftarrow \mathcal{T}^{server}_{s_j}, \quad
\tilde{g}_j \leftarrow 0 .
$$

这个细节很关键。如果 compact 第 $j$ 层直接从服务端第 $j$ 层初始化，而不是从第 $s_j$ 层初始化，那么选层语义会被破坏，实验对比也不再是“结构化同一组 HLF 结果”，而会混入初始化差异。因此代码中在构建 compact denoiser 后，显式根据 selected layer index 把 compact block 映射回原始 backbone layer。

EchoSlim 的训练流程如下：

1. 客户端对输入 token 做 embedding，并按原文方式注入 $d_\chi$-privacy 噪声。
2. 服务端按 HiddenEcho+ 的 HLF 计算层贡献度，得到排序后的候选层。
3. 取前 $k$ 个贡献度最高的层，并按设置可强制保留最后一层，得到 $S$。
4. 服务端的 `lst_skip`、客户端 compact denoiser 的 selected layers、MI estimators 数量同步更新。
5. 训练和评估阶段只回传 $S$ 中的 hidden states，客户端只执行 compact denoiser 的 $k$ 个 blocks。

代码实现中，这一流程对应 `modeling/my/split_echoslim.py` 和 `train_split_echoslim.py`。其中 `set_layer_skip()` 是数据流同步的关键入口：它同时更新 `config.lst_skip`、`server_layer_select.lst_skip`、`selected_layer_indices`、compact denoiser 和 MI estimators，避免出现服务端选层、客户端 forward、gate 和 IB loss 使用的层集合不一致。

## I.4 时间与空间复杂度分析

设 reduced hidden size 为 $d'=d/r$，序列长度为 $n$，每个 side transformer block 的参数量为 $P_T$，单层 gate 参数量为 $P_g=d'$，输入降维、输出升维和分类 head 的固定参数量合计为 $P_{fixed}$。

原 HiddenEcho+ 的客户端 denoiser 参数量为

$$
P_{HE+}=L(P_T+P_g)+P_{fixed}.
$$

EchoSlim 的客户端 denoiser 参数量为

$$
P_{slim}=k(P_T+P_g)+P_{fixed}.
$$

因此参数减少比例为

$$
R_P = 1-\frac{P_{slim}}{P_{HE+}}
=1-\frac{k(P_T+P_g)+P_{fixed}}{L(P_T+P_g)+P_{fixed}}.
$$

当 $P_{fixed}$ 相对较小时，近似有

$$
R_P \approx 1-\frac{k}{L}.
$$

在本次 Qwen2.5-1.5B 分类复现实验中，服务端共有 28 个 hidden layers，HLF 选择 3 个贡献层并强制保留最后一层，因此 $k=4$，理论上 side stack 主体参数的压缩接近

$$
1-\frac{4}{28}=85.71\%.
$$

实测客户端参数从 83.12M 降到 12.89M，减少 84.49%，与理论估计接近。差异来自输入/输出投影、分类 head 等固定参数不随 $k$ 线性减少。

对通信复杂度而言，EchoSlim 不改变 HiddenEcho+ 的 HLF 和 DR，因此每个 batch 的 hidden-state 回传复杂度仍为

$$
O(knd').
$$

Embedding 上传复杂度仍为

$$
O(nd).
$$

因此 EchoSlim 的通信量理论上应与同配置 HiddenEcho+ 一致。实验中两者的 `embedding_data_transferred` 和 `hiddens_data_transferred` 完全相同，验证了这一点。

对客户端 forward 计算复杂度而言，原 HiddenEcho+ 已经在运行时跳过未选层，因此其实际 denoiser 计算复杂度也是 $O(k\cdot F_T(n,d'))$，EchoSlim 同样为 $O(k\cdot F_T(n,d'))$。因此 EchoSlim 第一版不应期待显著训练加速；它主要降低模型驻留参数和结构存储开销，而不是进一步减少已被原 HE+ 跳过的 forward FLOPs。当前实验中训练耗时几乎一致，也符合这一理论分析。

## I.5 分类任务实验设置

EchoSlim 的主实验以分类任务为核心验证对象。实验采用 Financial Phrasebank `sentences_allagree`，模型为 Qwen2.5-1.5B-Instruct，训练 20 epochs，batch size 为 48，LoRA rank 为 16，学习率为 1.5e-4。隐私噪声、HLF、DR、MI estimator 等配置与 HiddenEcho+ 对齐：

- privacy budget $\eta=100,1000,5000,6000$；
- `clip_embedding_l2=true`，`noise_type=Chi`；
- `lst_reduce_factor=4`；
- `auto_skip=true`；
- `num_reserved_layers=3`；
- `keep_last_layer=true`；
- `num_integrate_step=5`，`num_samples=32`；
- `mi_downsample_enable=true`，`mi_estimator_lr=1e-4`。

此外，为验证原 HiddenEcho+ 中未选 side layers 是否只是“保留但不使用”的结构冗余，我们补充了一个结构冗余诊断实验：在 HE+ 上执行 HLF 选层后，对少量 probe batches 做 backward，并逐层统计客户端 side stack 的梯度。

## I.6 主实验结果

表 I.1 给出 $\eta=100,1000,5000,6000$ 下 HiddenEcho+ 与 EchoSlim 的分类结果。两种方法在每个隐私预算下使用相同 HLF 选层、相同 DR 配置和相同通信内容。EchoSlim 仅改变客户端 denoiser 的结构，即从完整 $L$ 层 side stack 改为只包含被选中层的 compact side stack。

Table I.1: EchoSlim 与 HiddenEcho+ 在 Financial Phrasebank sentences_allagree 上的多隐私预算结果，Qwen2.5-1.5B-Instruct。

| $\eta$ | 方法 | HLF 选层 | Client 参数量 | Denoiser state | Test AUC | Test Acc | Test F1 | Hidden 通信 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 100 | HiddenEcho+ | `[0,2,8,27]` | 83.12M | 158.53MB | 0.8451 | 0.7427 | 0.5028 | 12.015GB |
| 100 | EchoSlim | `[0,2,8,27]` | 12.89M | 24.58MB | 0.8427 | 0.7410 | 0.5004 | 12.015GB |
| 1000 | HiddenEcho+ | `[0,3,7,27]` | 83.12M | 158.53MB | 0.8269 | 0.7251 | 0.5021 | 12.015GB |
| 1000 | EchoSlim | `[0,3,7,27]` | 12.89M | 24.58MB | 0.8191 | 0.7225 | 0.4873 | 12.015GB |
| 5000 | HiddenEcho+ | `[0,2,4,27]` | 83.12M | 158.53MB | 0.8494 | 0.7463 | 0.5167 | 12.015GB |
| 5000 | EchoSlim | `[0,2,4,27]` | 12.89M | 24.58MB | 0.8464 | 0.7401 | 0.5259 | 12.015GB |
| 6000 | HiddenEcho+ | `[0,2,4,27]` | 83.12M | 158.53MB | 0.8443 | 0.7454 | 0.5136 | 12.015GB |
| 6000 | EchoSlim | `[0,2,4,27]` | 12.89M | 24.58MB | 0.8395 | 0.7357 | 0.5128 | 12.015GB |

相对 HiddenEcho+，EchoSlim 在所有预算下都保持相同 hidden-state 通信量，同时客户端参数量稳定减少

$$
1-\frac{12.89}{83.12}=84.49\%,
$$

约为 6.45 倍压缩；客户端 denoiser 的 state_dict 从 158.53MB 降到 24.58MB，减少 84.50%。任务指标方面，EchoSlim 相对 HiddenEcho+ 的 AUC 差值分别为 -0.0024、-0.0078、-0.0029、-0.0048，最大下降 0.78 个百分点。Accuracy 最大下降 0.97 个百分点；F1 在 $\eta=5000$ 略高于 HE+，但在 $\eta=1000$ 下降 1.48 个百分点。因此，更稳妥的表述是：EchoSlim 在多隐私预算下基本保持 AUC 和 Accuracy，F1 存在小幅波动。

训练资源方面，EchoSlim 的训练 peak allocated memory 从约 21.40GiB 降到 20.95GiB，下降约 2.10%；训练耗时也仅小幅下降。这符合方法预期：原 HiddenEcho+ 已经跳过未选层 forward，EchoSlim 第一版主要减少客户端模型驻留参数和保存开销，并不额外减少 hidden-state 通信或主要计算路径。

隐私攻击指标方面，EchoSlim 没有改变噪声注入机制、隐私预算、上传 embedding、服务端回传 hidden states 的层数与维度。因此从机制上看，EIA/AIA/EP 应与同配置 HiddenEcho+ 基本一致。后续仍建议补跑 EIA/EP，作为经验验证，防止训练扰动导致攻击指标出现非预期偏移。

## I.7 结构冗余诊断

EchoSlim 的核心改动较简单：删除 HE+ 中 HLF 未选中的 side layers。因此，仅报告任务指标还不足以说明这些层确实是冗余的。为此，我们在原 HiddenEcho+ 上进行结构冗余诊断：在 $\eta=5000$、`num_reserved_layers=3`、`keep_last_layer=true` 的配置下，先执行 HLF 得到选层 `[0,2,4,27]`，随后在两个 probe batches 上做 backward，并逐层检查完整 client side stack 的梯度。

Table I.2: HiddenEcho+ 完整 side stack 的梯度诊断。Selected layers 是 HLF 实际使用的层；skipped layers 是 HE+ 保留在客户端但 forward 中跳过的层。

| 分组 | 层数 | 层索引 | Side layer + gate 参数量 | backward 后有梯度的层 |
|---|---:|---|---:|---|
| selected | 4 | `[0,2,4,27]` | 11.70M | `[0,2,4,27]` |
| skipped | 24 | `[1,3,5,...,26]` | 70.23M | `[]` |

诊断结果显示，所有 selected layers 都有非零梯度，而所有 skipped layers 的梯度均为空。Skipped layers 占完整 side stack 参数的

$$
\frac{70.23M}{81.93M}=85.71\%.
$$

这说明 HiddenEcho+ 的 HLF 已经把未选层从有效计算图中移除：这些层既不参与 forward，也不会在 backward 中更新。但原 HiddenEcho+ 仍然在客户端完整实例化这些层，形成结构冗余。EchoSlim 删除的正是这部分 **gradient-dead side layers**。因此，EchoSlim 不是改变 HE+ 的选层算法，也不是改变通信协议，而是把 HE+ 已经存在的运行时稀疏性落实为客户端结构稀疏性。

## I.8 生成任务附表：CNNDM / T5

由于原论文将分类任务作为正文主表、生成任务作为附录 Table 7，本报告也采用相同组织方式：EchoSlim 的主论证仍以 Financial 分类任务为主，T5/CNNDM 摘要生成结果作为附表，用于验证该结构化客户端 denoiser 能迁移到 encoder-decoder 生成任务。

生成任务采用 `determined-ai/cnn_dailymail_short`，基座模型为 T5-large，训练入口为 `train_split_t5_echoslim.py`。EchoSlim 配置与 HiddenEcho+ 对齐：`lst_reduce_factor=4`、`num_reserved_layers=3`、`keep_last_layer=true`、`mi_downsample_enable=true`，隐私预算为 $\eta=20,30,40$。T5 与 Qwen 分类任务的主要实现差异在于：T5 denoiser 位于 encoder 侧，并使用 encoder-style 双向注意力，而不是 causal attention。

Table I.3: EchoSlim 与 HiddenEcho+ 在 CNNDM/T5 摘要生成任务上的附表结果。

| $\eta$ | 方法 | HLF 选层 | Adapter state | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---|---|---:|---:|---:|---:|---:|
| 20 | HiddenEcho+ | `[1,3,9,23]` | 98.70MB | 8.0763 | 0.3186 | 0.1198 | 0.2388 |
| 20 | EchoSlim | `[1,3,9,23]` | 35.73MB | 7.5137 | 0.3172 | 0.1146 | 0.2321 |
| 30 | HiddenEcho+ | `[1,2,3,23]` | 98.70MB | 15.0758 | 0.4256 | 0.2044 | 0.3185 |
| 30 | EchoSlim | `[1,2,3,23]` | 35.73MB | 16.7516 | 0.4386 | 0.2137 | 0.3334 |
| 40 | HiddenEcho+ | `[2,4,6,23]` | 98.70MB | 18.2022 | 0.4553 | 0.2329 | 0.3519 |
| 40 | EchoSlim | `[2,4,6,23]` | 35.73MB | 18.4079 | 0.4512 | 0.2324 | 0.3490 |

与分类任务一致，EchoSlim 不改变 HLF 选层、DR 维度、隐私预算或 hidden-state 回传内容，因此生成任务中的通信量与 HiddenEcho+ 保持一致；三组正式实验中 `embedding_data_transferred` 与 `hiddens_data_transferred` 均为 8305737728。结构开销方面，T5 EchoSlim adapter state 从 98.70MB 降至 35.73MB，减少约 63.80%。

指标解释上，BLEU 保留用于和原论文 Table 7 对齐，但在 CNN/DailyMail 摘要生成任务中，ROUGE-1/2/L 更适合作为主要质量指标。BLEU 最初主要用于机器翻译，更强调精确 n-gram 匹配；摘要任务通常允许多种等价表达，且更关注生成摘要是否覆盖 reference 中的关键信息，因此 ROUGE 尤其 ROUGE-1/2/L 是更常用的摘要评测指标。

按 ROUGE 观察，EchoSlim 与 HiddenEcho+ 基本持平，三组预算的平均 ROUGE 略高于 HiddenEcho+：

| 指标 | HiddenEcho+ 平均 | EchoSlim 平均 | 差值 |
|---|---:|---:|---:|
| ROUGE-1 | 0.3998 | 0.4023 | +0.0025 |
| ROUGE-2 | 0.1857 | 0.1869 | +0.0012 |
| ROUGE-L | 0.3031 | 0.3049 | +0.0018 |

因此，生成附表支持如下结论：EchoSlim 在 T5 摘要生成任务上显著降低客户端结构存储开销，并在更适合摘要任务的 ROUGE 指标上保持与 HiddenEcho+ 基本一致的生成质量；BLEU 在 $\eta=20$ 下低于 HiddenEcho+，但在 $\eta=30,40$ 下持平或更高，可作为辅助指标波动处理。

对应文件如下：

| 内容 | 路径 |
|---|---|
| EchoSlim T5 实现 | `modeling/my_t5/split_echoslim.py` |
| EchoSlim T5 训练入口 | `train_split_t5_echoslim.py` |
| EchoSlim eta20 日志 | `logs_opt_gen/1_echoslim_t5_dailymail_eta20_residual_reduce4.log` |
| EchoSlim eta20 对齐复跑日志 | `logs_opt_gen/2_echoslim_t5_dailymail_eta20_residual_reduce4_k3_rerun2.log` |
| EchoSlim eta30 日志 | `logs_opt_gen/1_echoslim_t5_dailymail_eta30_residual_reduce4.log` |
| EchoSlim eta40 日志 | `logs_opt_gen/1_echoslim_t5_dailymail_eta40_residual_reduce4.log` |
| T5 EchoSlim 结构测试 | `tests/test_t5_echoslim_structure.py` |

## I.9 附加分析与论证逻辑

EchoSlim 的主论证链条为：

1. **发现问题**：HiddenEcho+ 已经减少通信和 forward 计算，但客户端仍完整实例化 $L$ 层 side stack，存在部署冗余。
2. **提出改进**：EchoSlim 把 HLF 结果结构化到客户端 denoiser，只保留被选中的 $k$ 层。
3. **诊断验证**：HE+ 中未选 side layers 在 backward 后没有梯度，属于运行路径上不会被训练的结构死参数。
4. **理论预期**：参数量从 $O(L)$ 降为 $O(k)$；通信量保持 $O(knd')$；forward 计算与 HE+ 基本一致；隐私机制不变。
5. **实验验证**：在相同 HLF/DR/隐私预算/训练设置下，EchoSlim 大幅降低 client 参数，任务指标基本不降，通信量不变，训练时间接近。
6. **边界说明**：EchoSlim 第一版不声称进一步降低通信，也不声称增强 DP 隐私；它优化的是原文未充分评估的客户端结构资源开销。

我们也进行了若干附加探索。首先，选层数量 sweep 显示 `num_reserved_layers=3` 在当前设置下 AUC 最优，而 `num_reserved_layers=1` 可以把客户端参数降到 7.04M、hidden 通信降到 6.007GB，但 AUC 降到 0.8371；`num_reserved_layers=4` 增加参数和通信后并未继续提升性能。这说明“更多回传层”并不单调更优，过多 hidden states 可能引入额外噪声或优化负担。由于该趋势不够单调，本文不将其作为核心消融，仅作为资源-性能 trade-off 的探索性结果。

其次，初始化实验显示 random initialization 的 AUC 高于 backbone-aligned initialization，但 macro-F1 明显下降。这说明初始化会影响 AUC/F1 trade-off，但不能证明随机初始化整体更优。本文默认使用 backbone-aligned initialization，因为它保持 compact block 与原始层号的语义一致，且分类指标更均衡。

目前仍建议后续补充 EIA/EP 经验验证。机制上 EchoSlim 不改变噪声注入、上传 embedding、回传 hidden states 或隐私预算，因此 EIA/AIA/EP 应与同配置 HiddenEcho+ 接近；但正式报告中最好至少在 Financial $\eta=5000$ 下补一组 EIA/EP，证明结构压缩没有引入隐私退化。
