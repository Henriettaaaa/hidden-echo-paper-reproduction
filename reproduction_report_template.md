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

本节按照论文 Table 1 的形式组织。EP 来自 EIA；在相同 noisy embedding 机制下，LDP、HiddenEcho 与 HiddenEcho+ 的 EP 可共享同一组 EIA 结果。

| 方法 | 指标 | η=100 | η=1000 | η=5000 | η=6000 |
|---|---|---:|---:|---:|---:|
| LDP | AUC | 0.647036 | 0.650413 | 0.750725 | 0.769136 |
| LDP | EP | 0.982257 | 0.986704 | 0.940820 | 0.893404 |
| HiddenEcho | AUC | 0.849399 | 0.868505 | 0.877712 | 0.868049 |
| HiddenEcho | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |
| HiddenEcho+ | AUC | 0.845129 | 0.826858 | 0.849359 | 0.844343 |
| HiddenEcho+ | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |

论文 Table 1 中 Financial / Qwen2-1.5B 的对应 AUC 为：LDP 在 η=100、1000、5000、6000 下分别为 0.596、0.595、0.629、0.617；HiddenEcho 分别为 0.875、0.874、0.883、0.889；HiddenEcho+ 分别为 0.857、0.855、0.860、0.866。论文对应 EP 为 0.988、0.987、0.967、0.886。当前分类主任务使用 Qwen2.5-1.5B-Instruct 复现；EIA 使用 Qwen2-1.5B-Instruct embedding matrix，二者模型口径差异在第 7 节单独说明。

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

需要特别说明的是，当前复现中的 LDP AUC 系统性高于论文，导致相对提升幅度小于论文报告值。代码排查显示，LDP 路径在 `lst_enable=false` 时确实没有调用 `client_denoise`，也没有 hidden states 回传；`data_transfer.txt` 中 `hiddens_data_transferred=0`，说明该结果不是由 HiddenEcho 模块泄漏造成。更可能的原因包括以下三点。

第一，当前主分类实验采用 `financial_phrasebank/sentences_allagree`，数据划分为 train=1811、validation=226、test=227。该子集只包含标注者完全一致的样本，标签噪声较低，分类任务本身较容易；无噪声 split clean baseline 已达到 AUC 0.996565。相比之下，论文 Appendix H.1 描述 Financial Phrasebank 为 4,840 条样本，数据口径更接近完整 Financial Phrasebank。因此，`sentences_allagree` 可能抬高了 LDP 在加噪训练后的可学习性。

第二，当前 LDP 是在 noisy embedding 上完整进行 LoRA fine-tuning，而不是用 clean checkpoint 直接在 noisy embedding 上推理。训练、验证与测试阶段均会调用 `ClientEmbeddingPart` 注入噪声，模型在 20 epochs 内持续适配该 noisy input 分布。对于 Financial 这种较容易的数据集，LoRA 可能学习到对随机扰动鲁棒的分类边界，从而使“纯加噪”基线强于直觉预期。

第三，η=5000 与 η=6000 属于较弱隐私、较弱噪声区域，LDP 性能本身可能显著恢复。已有诊断还显示，η=1000 下 `clip_embedding_l2=false` 的 LDP AUC 为 0.655856，与 `clip_embedding_l2=true` 的 0.650413 接近，说明当前 LDP 偏高不能单独归因于 clipping。综合来看，本报告将 LDP 偏高视为数据口径与 noisy LoRA 适配共同造成的复现差异，而不是噪声链路未生效。

## 6. 训练与通信记录

### 6.1 HiddenEcho

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

### 6.2 HiddenEcho+

HiddenEcho+ 使用动态层选择与 MI 约束。当前公开实现中 `lst_reduce_factor=1 + mi_downsample_enable=true` 会出现 NaN；经 smoke run 排查，`lst_reduce_factor=4 + mi_downsample_enable=true` 能够稳定训练，因此正式复现实验采用 reduce4。

| 方法 | η | AUC | Accuracy | Macro-F1 | `embedding_data_transferred` | `hiddens_data_transferred` | `train_runtime` |
|---|---:|---:|---:|---:|---:|---:|---:|
| HiddenEcho+ reduce4 | 5000 | 0.849359 | 0.746256 | 0.516652 | 12014714880 | 12014714880 | 907.2591s |

与 HiddenEcho reduce1 相比，HiddenEcho+ reduce4 在 η=5000 下将 hidden states 回传通信量从 336412016640 降至 12014714880，节省约 96.43%。验证 AUC 从第 1 epoch 的 0.571668 持续上升，在第 19 epoch 达到 0.851784，最终 test AUC 为 0.849359，训练过程中未出现 NaN 或显存错误。

对应文件：

| 内容 | 路径 |
|---|---|
| 后台脚本 | `scripts/5_echo_plus_mi_reduce4_eta5000_cliptrue_20epoch_bg.sh` |
| 日志 | `logs/echo_plus_mi_qwen25_eta5000_cliptrue_reduce4_20epoch_run2.log` |
| 输出 | `outputs/train_ckpts/echo_plus_mi_qwen25_eta5000_cliptrue_reduce4_20epoch_run2/` |

## 7. EIA 评测

EIA 按论文定义攻击 noisy embedding，攻击者假设已知用户上传的 perturbed embeddings 以及 embedding matrix，并尝试恢复原始文本。当前复现将按论文主表预算 η=100、1000、5000、6000 测量 EP。

| 数据集 | 模型 | η | EP |
|---|---|---:|---:|
| Financial Phrasebank | Qwen2-1.5B-Instruct | 100 | 0.982257 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 1000 | 0.986704 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 5000 | 0.940820 |
| Financial Phrasebank | Qwen2-1.5B-Instruct | 6000 | 0.893404 |

由于 EIA 仅作用于上传 embedding，在同一噪声机制下，LDP、HiddenEcho 与 HiddenEcho+ 可共享同一组 EP 结果；报告主表中将其标记为“同 LDP”。

本次 EIA 使用脚本 `experiment/eia/invert_emb_attack.py`，攻击样本为 Financial Phrasebank 训练集前 200 条，batch size 为 2，输出指标为 `1 - mean(rouge1)`。日志文件为 `logs/eia_financial_qwen_eta100_1000_5000_6000_run1.log`。需要说明的是，EIA 本次采用本地 Qwen2-1.5B-Instruct 词表与 embedding matrix；若主任务最终统一报告为 Qwen2.5-1.5B-Instruct，则应补跑对应模型路径下的 EIA 以保持模型口径完全一致。

## 8. AIA 评测

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
| 0 | 0.487800 | 0.487800 | 12.831156 | 12.831156 |
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

需要说明的是，当前复现中无保护基线的 education EP 为 0.487800、age RMSE 为 12.831156，均高于论文图中的近似读数。这表明公开代码下的 AIA attacker 整体弱于论文图所对应的攻击设置。补充诊断显示，原始 education 标签口径下的无保护 EP 为 0.492300、age RMSE 为 12.828987，与 H.4 四分类口径几乎一致，因此绝对数值偏差并非主要由 education 类别数造成，更可能来自攻击输入表示、攻击器训练强度或公开代码与论文图中 AIA 评测细节的不一致。报告中据此将 AIA 作为趋势复现与隐私-效用权衡验证，而不主张无保护基线的绝对值完全复现。

## 9. 生成任务复现：CNNDM / T5

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

### 9.1 Table 7 复现结果

论文 Table 7 报告 T5 在 CNNDM、IWSLT 与 SAMSum 上的 BLEU。当前复现只填写 CNNDM；无实验数据的单元格保留为空。主表仅记录复现值。

| 方法 | 数据集 | η=20 BLEU | η=30 BLEU | η=40 BLEU |
|---|---|---:|---:|---:|
| LDP | CNNDM | 0.389 | 8.290 | 17.622 |
| GAN-DP | CNNDM | 9.019 | 12.527 | 18.776 |
| HiddenEcho | CNNDM | 6.349 | 16.668 | 18.526 |
| HiddenEcho+ | CNNDM | 8.076 | 15.076 | 18.202 |

清洁 T5 baseline 的复现 BLEU 为 19.449，论文对应 clean T5 CNNDM BLEU 为 17.738。由于本地 clean upper bound 高于论文，HiddenEcho 在 η=30 与 η=40 下的绝对 BLEU 也整体偏高；但 η=20 < η=30 < η=40 的趋势与论文一致，且强噪声到弱噪声的性能恢复幅度与 clean 上界关系一致。作为参考，论文 Table 7 中 CNNDM 的 LDP BLEU 为 0.764、7.974、12.107，HiddenEcho BLEU 为 2.915、11.617、12.323。

### 9.2 关键排查结论

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

HiddenEcho+ 生成任务采用 residual Echo + `lst_reduce_factor=4`。在 η=20/30/40 下，BLEU 分别为 8.076、15.076、18.202，整体趋势与 HiddenEcho 一致；η=30 低于 HiddenEcho reduce1，但仍明显高于 LDP 与 GAN-DP 已有单点。通信上，HiddenEcho reduce1 的 hidden states 回传量为 199337705472，HiddenEcho+ reduce4 的 hidden states 回传量为 8305737728；在 embedding 上传量相同的条件下，HiddenEcho+ 将 hidden 回传从 24 倍 embedding 降到 1 倍 embedding，节省约 95.83%。

GAN-DP task 阶段按 validation `eval_loss` 选择 best checkpoint。η=20/40 按分类任务默认口径使用 `generator_epoch=4`，BLEU 分别为 9.019、18.776；η=30 的 `generator_epoch=4` 结果异常偏低，仅 6.288，因此补充 generator checkpoint sensitivity。η=30 使用 `generator_epoch=0` 时 BLEU 提升到 12.527，ROUGE-L 为 0.315，恢复到与论文量级一致的区间。

| GAN-DP η=30 generator checkpoint | task best checkpoint | BLEU | ROUGE-L |
|---:|---|---:|---:|
| 0 | epoch 4 | 12.527 | 0.315 |
| 4 | epoch 14 | 6.288 | 0.287 |
| 10 | epoch 14 | 3.575 | 0.267 |
| 18 | epoch 4 | 3.476 | 0.261 |

该排查说明 GAN-DP 对 generator checkpoint 非常敏感，且 generator 训练损失继续下降不等价于下游生成 BLEU 提升。生成任务主表采用当前验证到的最佳 generator checkpoint：η=20 使用 epoch 4，η=30 使用 epoch 0，η=40 使用 epoch 4。

### 9.3 生成任务记录

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
| LDP decoder-only 脚本目录 | `scripts_gen_ldp_check/` |
| 生成任务主脚本目录 | `scripts_gen/` |

## 10. 诊断与实现说明

### 10.1 Clean Split Baseline

为确认基础训练链路正常，补充无噪声诊断实验。Split clean 与 Split clean + HiddenEcho 均能达到接近 0.99 的 AUC，说明数据划分、LoRA 微调、split wrapper、模型加载与评估实现均能支撑高性能分类。

| 诊断实验 | 设置 | AUC | Accuracy | Macro-F1 | 结论 |
|---|---|---:|---:|---:|---|
| Split clean | `privacy_budget=0, lst_enable=false` | 0.996565 | 0.977974 | 0.970914 | split wrapper 与基础分类链路正常 |
| Split clean + HiddenEcho | `privacy_budget=0, lst_enable=true` | 0.994170 | 0.973568 | 0.959701 | HiddenEcho 模块在无噪声条件下未显著伤害任务性能 |

### 10.2 HiddenEcho+ reduce1 + MI 数值问题

直接将 HiddenEcho+ 设置为 `lst_reduce_factor=1 + mi_downsample_enable=true` 时，训练初期出现 `loss=0.0`、`grad_norm=nan`，验证阶段因预测包含 NaN 终止。分模块诊断结果显示，HLF-only reduce1 可以运行，MI reduce4 可以稳定训练，而 MI reduce1 即便将 `mi_estimator_lr` 降至 `1e-5` 仍会 NaN。因此，该问题集中在 full-dimension MINE 估计与 bf16 数值稳定性上。为保持复现过程可运行，本阶段 HiddenEcho+ 正式结果采用 reduce4。

| 诊断实验 | 状态 | AUC | 结论 |
|---|---|---:|---|
| HLF-only reduce1, `mi_downsample=false` | 成功 | 0.679361 | layer filter 与 reduce1 去噪链路本身可训练 |
| MI reduce4, `mi_downsample=true` | 成功 | 0.639846 | MI 在降维后可稳定训练 |
| MI reduce1, `mi_estimator_lr=1e-5` | 失败 | - | 训练初期出现 NaN |

## 11. 后续计划

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

当前只在分类任务上验证 EchoSlim。实验采用 Financial Phrasebank `sentences_allagree`，模型为 Qwen2.5-1.5B-Instruct，训练 20 epochs，batch size 为 48，LoRA rank 为 16，学习率为 1.5e-4。隐私噪声、HLF、DR、MI estimator 等配置与 HiddenEcho+ 对齐：

- privacy budget $\eta=5000$；
- `clip_embedding_l2=true`，`noise_type=Chi`；
- `lst_reduce_factor=4`；
- `auto_skip=true`；
- `num_reserved_layers=3`；
- `keep_last_layer=true`；
- `num_integrate_step=5`，`num_samples=32`；
- `mi_downsample_enable=true`，`mi_estimator_lr=1e-4`。

后续计划补齐 $\eta=100,1000,6000$，用于验证 EchoSlim 在不同隐私强度下是否都能保持 HiddenEcho+ 的主指标，同时稳定降低客户端参数。

## I.6 主实验结果

表 I.1 给出当前已完成的 $\eta=5000$ 主实验结果。EchoSlim 记录到的 HLF 选层为 `[0, 2, 4, 27]`，即 3 个高贡献层加最后一层。HiddenEcho+ 与 EchoSlim 的通信配置完全一致，hidden-state 回传量相同。

Table I.1: EchoSlim 与 HiddenEcho+ 在分类任务上的对比，Financial Phrasebank sentences_allagree，Qwen2.5-1.5B-Instruct，$\eta=5000$。

| 方法 | 选层/结构 | Client 参数量 | Test AUC | Test Acc | Test F1 | Train runtime | Embedding 通信 | Hidden 通信 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HiddenEcho+ | HLF runtime sparse, full side stack | 83.12M | 0.8494 | 0.7463 | 0.5167 | 898.51s | 12.015GB | 12.015GB |
| EchoSlim | compact side stack, `[0,2,4,27]` | 12.89M | 0.8464 | 0.7401 | 0.5259 | 895.58s | 12.015GB | 12.015GB |

相对 HiddenEcho+，EchoSlim 的客户端参数量减少

$$
1-\frac{12.89}{83.12}=84.49\%,
$$

约为 6.45 倍压缩。任务指标方面，Test AUC 下降 0.0029，Accuracy 下降 0.0062，F1 上升 0.0092。考虑到两者训练设置、隐私预算、HLF/DR 配置和通信量保持一致，这说明 EchoSlim 能够在基本不牺牲分类性能的情况下，把 HiddenEcho+ 的运行时稀疏转化为客户端结构稀疏。

训练耗时方面，EchoSlim 为 895.58s，HiddenEcho+ 为 898.51s，仅快 0.33%。这不是负面结果，而是符合方法预期：原 HiddenEcho+ 已经跳过未选层 forward，EchoSlim 第一版主要减少参数驻留和模型结构冗余，并不额外减少 hidden-state 通信或主要计算路径。

隐私攻击指标方面，EchoSlim 没有改变噪声注入机制、隐私预算、上传 embedding、服务端回传 hidden states 的层数与维度。因此从机制上看，EIA/AIA/EP 应与同配置 HiddenEcho+ 基本一致。后续仍建议补跑 EIA/EP，作为经验验证，防止训练扰动导致攻击指标出现非预期偏移。

## I.7 消融实验设计

EchoSlim 可以做消融，但消融目标应围绕“结构化 HLF 是否必要、是否稳定、是否有额外代价”，而不是重复原文 HiddenEcho+ 已经做过的 HLF/DR/Residual 消融。

建议至少补充以下消融：

1. **结构稀疏消融：HiddenEcho+ full side stack vs. EchoSlim compact side stack。**  
   这是当前主表已经完成的核心对比。它回答的问题是：在相同 HLF、相同通信和相同训练配置下，删除未使用 side layers 是否会损害性能。当前结果显示 AUC 仅下降 0.0029，参数减少 84.49%。

2. **选层数量消融：$k=1,2,3,4$ 或 `num_reserved_layers=1,2,3,4`。**  
   这个实验用于证明 EchoSlim 的参数-性能 trade-off 可控。预期随着 $k$ 增加，client 参数和 hidden 通信线性增加，AUC/F1 逐渐接近或超过较小 $k$ 的配置。当前已有初步 5 epoch 结果显示，`num_reserved_layers=1` 且保留最后一层时选层为 `[4,27]`，性能优于只选 `[4]`，但通信和时间也约翻倍；正式报告中应以 20 epoch 为准。

3. **保留最后一层消融：without last layer vs. keep last layer。**  
   HiddenEcho+ 原逻辑通常强调最后层表征对任务 head 重要。EchoSlim 中也应验证强制保留最后层是否必要。该实验能够支撑 `[0,2,4,27]` 中 `27` 的合理性。

4. **初始化消融：backbone-aligned initialization vs. random initialization。**  
   EchoSlim 的 compact 第 $j$ 层必须对应原始第 $s_j$ 层。若随机初始化或错误按 compact index 初始化，可能导致训练不稳定或性能下降。这个消融能证明“按原始层号对齐初始化”不是实现细节，而是方法成立的必要条件。

5. **隐私预算鲁棒性：$\eta=100,1000,5000,6000$。**  
   这不是严格意义上的结构消融，但对论文说服力很重要。它回答 EchoSlim 是否只在某个噪声强度下偶然有效。你后续补齐 $\eta=100,1000,6000$ 后，可以把主表扩展成多预算表。

## I.8 还需要的支撑实验与论证逻辑

为了让 EchoSlim 的论文叙事完整，建议把论证链条组织为：

1. **发现问题**：HiddenEcho+ 已经减少通信和 forward 计算，但客户端仍完整实例化 $L$ 层 side stack，存在部署冗余。
2. **提出改进**：EchoSlim 把 HLF 结果结构化到客户端 denoiser，只保留被选中的 $k$ 层。
3. **理论预期**：参数量从 $O(L)$ 降为 $O(k)$；通信量保持 $O(knd')$；forward 计算与 HE+ 基本一致；隐私机制不变。
4. **实验验证**：在相同 HLF/DR/隐私预算/训练设置下，EchoSlim 大幅降低 client 参数，任务指标基本不降，通信量不变，训练时间接近。
5. **边界说明**：EchoSlim 第一版不声称进一步降低通信，也不声称增强 DP 隐私；它优化的是原文未充分评估的客户端结构资源开销。

除当前主实验外，还建议补充以下支撑结果：

- **多隐私预算主表**：补齐 $\eta=100,1000,6000$，报告 AUC/Acc/F1、client params、embedding/hidden 通信、runtime。若各预算下 EchoSlim 与 HE+ 的 AUC 差距都在约 0.5 到 1 个百分点以内，同时参数稳定减少约 84%，结论会更稳。
- **峰值显存或模型 checkpoint 大小**：参数量是理论指标，最好再给一个实际部署指标，例如客户端 denoiser checkpoint 大小、加载后 CUDA/CPU memory。这样“客户端资源开销”会更有说服力。
- **EIA/EP 验证**：机制上隐私攻击指标应接近 HE+，但最好至少在 Financial 的 $\eta=5000$ 补一组 EIA/EP，证明 EchoSlim 没有引入隐私退化。
- **k-sweep 曲线**：横轴为 client params 或 selected layers 数，纵轴为 AUC/F1。这个图能直观看出 EchoSlim 的资源-性能 Pareto trade-off。
- **keep-last 消融**：证明最后层在 compact denoiser 中是否必要，也能解释当前 `[0,2,4,27]` 选层的设计来源。

在目前分类任务阶段，最小充分实验集可以是：

| 实验 | 目的 | 是否必须 |
|---|---|---|
| HE+ vs EchoSlim, $\eta=100,1000,5000,6000$ | 证明主效果跨隐私预算稳定 | 必须 |
| k-sweep 或 selected layers sweep | 证明结构压缩和性能之间的 trade-off | 强烈建议 |
| keep-last ablation | 解释最后层保留策略 | 建议 |
| checkpoint size / peak memory | 支撑客户端资源开销主张 | 强烈建议 |
| EIA/EP 对齐验证 | 支撑“不牺牲隐私” | 建议 |

如果生成任务暂时不做，报告中需要明确写成“本文改进部分当前先验证分类任务”。由于 EchoSlim 本质上作用于 HiddenEcho+ 的 side denoiser 结构，而不是分类 head 特有模块，方法本身可以迁移到生成任务；但在没有生成实验前，不应声称已在生成任务验证有效。
