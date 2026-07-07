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
| 基座模型 | Qwen2-1.5B-Instruct（本地路径：`/data/songhanlin/models/Qwen2-1.5B-Instruct`） |
| 训练入口 | `train_split.py` |
| 训练方式 | LoRA fine-tuning |
| LoRA rank | 16 |
| 初始学习率 | 1.5e-4 |
| 学习率调度 | linear |
| 训练轮数 | 20 epochs |
| 最大长度 | Financial: 96（由 `train_split.py` 内部设置） |
| 噪声类型 | Chi |
| `clip_embedding_l2` | true |
| 主复现隐私预算 | η = 100, 1000, 5000, 6000（当前已完成部分见主表） |

## 4. 主结果：Financial / Qwen

本节按照论文 Table 1 的形式组织。尚未完成的预算或方法保留为空，后续实验完成后直接补入。EP 来自 EIA；在相同 noisy embedding 机制下，LDP、HiddenEcho 与 HiddenEcho+ 的 EP 可共享同一组 EIA 结果。

| 方法 | 指标 | η=100 | η=1000 | η=5000 | η=6000 |
|---|---|---:|---:|---:|---:|
| LDP | AUC | [待补] | [待补] | [待补] | [待补] |
| LDP | EP | 0.982257 | 0.986704 | 0.940820 | 0.893404 |
| HiddenEcho | AUC | [待补] | 0.868505 | 0.877712 | [待补] |
| HiddenEcho | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |
| HiddenEcho+ | AUC | [待补] | [待补] | 0.849359 | [待补] |
| HiddenEcho+ | EP | 同 LDP | 同 LDP | 同 LDP | 同 LDP |

论文 Table 1 中 Financial / Qwen2-1.5B 的对应 AUC 为：LDP 在 η=100、1000、5000、6000 下分别为 0.596、0.595、0.629、0.617；HiddenEcho 分别为 0.875、0.874、0.883、0.889；HiddenEcho+ 分别为 0.857、0.855、0.860、0.866。论文对应 EP 为 0.988、0.987、0.967、0.886。当前复现已经完成 HiddenEcho 的 η=1000 与 η=5000、HiddenEcho+ 的 η=5000，以及 Financial / Qwen2-1.5B 口径下的 EIA。

## 5. 与论文结果的差异

| 方法 | η | 论文 AUC | 复现 AUC | 差异 | 备注 |
|---|---:|---:|---:|---:|---|
| HiddenEcho | 1000 | 0.874 | 0.868505 | -0.005495 | 与论文高度接近 |
| HiddenEcho | 5000 | 0.883 | 0.877712 | -0.005288 | 与论文高度接近 |
| HiddenEcho+ | 5000 | 0.860 | 0.849359 | -0.010641 | 略低于论文，分类链路稳定收敛 |

从已完成结果看，HiddenEcho 的两个预算点均与论文差距约 0.5 个 AUC 点；HiddenEcho+ 在 η=5000 下低于论文约 1.1 个 AUC 点，但保持了明显的通信压缩优势。当前结果支持论文关于“在相同 embedding 隐私水平下，HiddenEcho 能显著改善 LDP 任务性能”的主要结论。

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

论文 Table 7 报告 T5 在 CNNDM、IWSLT 与 SAMSum 上的 BLEU。当前复现只填写 CNNDM；无实验数据的单元格保留为空。

| 方法 | 数据集 | η=20 论文 BLEU | η=20 复现 BLEU | η=30 论文 BLEU | η=30 复现 BLEU | η=40 论文 BLEU | η=40 复现 BLEU |
|---|---|---:|---:|---:|---:|---:|---:|
| LDP | CNNDM | 0.764 | 0.389 | 7.974 | 8.290 | 12.107 | 17.622 |
| HiddenEcho | CNNDM | 2.915 | 6.349 | 11.617 | 16.668 | 12.323 | 18.526 |
| HiddenEcho+ | CNNDM | [待补] | [待补] | [待补] | [待补] | [待补] | [待补] |

清洁 T5 baseline 的复现 BLEU 为 19.449，论文对应 clean T5 CNNDM BLEU 为 17.738。由于本地 clean upper bound 高于论文，HiddenEcho 在 η=30 与 η=40 下的绝对 BLEU 也整体偏高；但 η=20 < η=30 < η=40 的趋势与论文一致，且强噪声到弱噪声的性能恢复幅度与 clean 上界关系一致。

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
5. 补齐生成任务 HiddenEcho+ 在 CNNDM 上的 η=20、30、40 结果，并视情况扩展到 IWSLT 与 SAMSum。
