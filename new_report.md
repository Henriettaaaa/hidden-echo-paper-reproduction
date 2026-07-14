# HiddenEcho 论文复现与 EchoSlim 改进方法验证

## 摘要

本文围绕 HiddenEcho 论文进行复现和改进。复现部分以 Model-as-a-Service（MaaS）场景下的 split learning 为背景，重点验证在差分隐私（DP）扰动输入的条件下， HiddenEcho 框架对于缓解层间噪声放大、保留任务信号的效果，以及 HiddenEcho+ 通过基于梯度的隐藏层选择（HLF）和信息瓶颈压缩（DR）降低通信开销的结论。复现结果表明，HiddenEcho 与 HiddenEcho+ 在 Financial Phrasebank / Qwen 分类任务以及 CNNDM / T5 摘要生成任务上达到原论文同量级性能，HiddenEcho+ 的 hidden-state 通信节省率约为 85.71%，与论文一致；少部分 baseline 结果与原论文存在偏差，主要来自未公开的实现细节以及随机性差异。

在复现过程中，我们发现 HiddenEcho+ 虽然已经在 forward 过程中跳过未选 hidden layers，但客户端 denoising module 仍完整实例化所有 side transformer blocks，存在结构冗余。基于此，本文提出 EchoSlim：在不改变隐私噪声、HLF 选层、DR 压缩和通信内容的前提下，仅实例化被 HLF 选中的客户端 side layers。 我们在相同的 MaaS场
景中评估 EchoSlim， 结果显示，在文本分类任务上，相比于 HiddenEcho+，EchoSlim 将客户端参数量从 83.12M 降至 12.89M，减少 84.49%；denoiser state 从 158.53MB 降至 24.58MB，减少 84.50%；AUC 最大下降仅为 0.78%，hidden-state 通信量保持不变。结果说明 EchoSlim 能把 HiddenEcho+ 的运行时稀疏性转化为客户端结构稀疏性，优化 MaaS场景下的端侧部署开销。

## 1. 引言

MaaS 范式的兴起为无法获取高性能计算资源的用户提供了平台，使其能够利用 LLMs 进行推理、微调及定制化智能体开发等多种用途（David 等人 2014）；split learning 将主要 Transformer 层放在服务端，客户端只保留embedding 层，极大地减少客户端计算负担。然而 MaaS 也存在隐私风险：用户若直接上传原始文本，服务端可能接触到姓名、联系方式、财务信息等敏感内容。

在基于扰动的隐私保护方法中，差分隐私（DP）因其较低的计算开销而受到广泛关注， 它在客户端对用户输入施加指定强度的噪声，然后再上传至服务器，以牺牲精度为代价实现隐私保护。

实验表明，embedding 层加入噪声后，噪声会随着深层 Transformer 的非线性变换逐层传播并放大，导致下游任务性能下降。现有去噪方法依赖预训练且与大语言模型动态脱节，无法有效缓解层间噪声。HiddenEcho 的核心思想是让服务端将中间 hidden states 回传给客户端，由客户端 denoising module 结合 clean embedding 和服务端中间表示进行校正，从而缓解层间噪声放大。HiddenEcho+ 进一步通过 HLF 只选择关键 hidden layers，并通过 DR 降低 hidden-state 维度，以减少通信量。

本文完成两个任务。第一，复现 HiddenEcho 论文的主要实验结论，包括 Financial Phrasebank / Qwen 文本分类、通信开销、EIA/AIA 隐私评测和 CNNDM / T5 摘要生成任务。第二，在复现基础上，本文提出优化方案 EchoSlim。EchoSlim 针对 HiddenEcho+ 客户端 denoiser 的结构冗余：原方法在客户端模型结构仍加载完整 side stack。EchoSlim 将 HLF 的选层结果固化为客户端 compact denoiser，从而降低客户端参数量和存储开销。

总而言之，本文的贡献如下：

1. 复现 HiddenEcho 与 HiddenEcho+ 的主要性能和通信结论，并分析与原论文数值不一致的原因。
2. 通过梯度诊断确认 HiddenEcho+ 中未选 side layers 不参与 forward/backward，是结构上可删除的 gradient-dead parameters。
3. 提出 EchoSlim，将客户端 denoiser 从完整 $L$ 层结构压缩为只包含 $k$ 个选中层的 compact 结构。
4. 在分类和生成任务上验证 EchoSlim 能显著降低客户端结构开销，同时基本保持 HiddenEcho+ 的性能。

## 2. 国内外相关工作

MaaS 将大模型部署在服务端，客户端只需调用模型能力，但用户数据需要经过网络传输，隐私风险集中在客户端到服务端的数据上传环节。已有隐私保护方法大致包括 cryptographic methods（密码学方法）和 perturbation-based methods（扰动方法）。前者如安全多方计算、同态加密，安全性强但计算开销较高；后者通过对输入或表示加入扰动换取隐私与效用之间的折中，更适合资源受限客户端。

split learning 将模型切分到客户端和服务端。对于文本模型，一种常见设置是客户端保留 embedding 层，服务端保留后续 Transformer 层。客户端上传的不是原始 token，而是经过扰动的 embedding。这一设置降低了原文直接暴露风险，但仍可能受到 embedding inversion attack（EIA，嵌入反演攻击）和 attribute inference attack（AIA，属性推断攻击）。

SnD 使用预训练 denoising module 对 noisy embedding 进行去噪，但该模块与后续 LLM fine-tuning 动态脱节，难以充分适应深层 hidden distribution 的变化。HiddenEcho 改为端到端训练客户端 denoising module，并利用服务端中间 hidden states 进行校正。HiddenEcho+ 在此基础上增加 HLF 和 DR，减少需要回传的 hidden states 数量与维度。

本文的改进关注另一个部署问题：HiddenEcho+ 的运行时稀疏性尚未完全转化为客户端模型结构稀疏性。EchoSlim 针对这一点压缩客户端 denoiser 的参数和 state_dict 大小。

## 3. 原论文方法概述

### 3.1 问题定义

在 split MaaS 场景中，客户端持有输入文本 $x$ 和 embedding 层 $\mathcal{E}$，服务端持有后续 LLM backbone。客户端先得到 clean embedding：

$$
E = \mathcal{E}(x),
$$

然后加入差分隐私噪声：

$$
E' = E + \delta.
$$

服务端以 $E'$ 为输入计算 Transformer hidden states。普通 LDP baseline 直接使用 noisy embedding 训练或推理，无法显式修正噪声传播。HiddenEcho 的目标是学习一个客户端 denoising module $\mathcal{D}$，利用 clean embedding 与服务端 hidden states 生成校正后的表示：

$$
H^{denoised} = \mathcal{D}(E, \mathbf{H}).
$$

### 3.2 HiddenEcho

HiddenEcho 将服务端所有 $L$ 层 hidden states 回传给客户端。客户端 denoising module 是一个降维 side network，hidden size 为 $d'=d/r$。第 $i$ 层将上一层输出 $A_{i-1}$ 与服务端第 $i$ 层 hidden state 的降维表示 $H_i^{dn}$ 混合：

$$
Z_i = \mu_i A_{i-1} + (1-\mu_i)H_i^{dn},
$$

其中 $\mu_i=\sigma(g_i)$ 是可学习 gate。随后通过 side transformer block 与 residual connection 得到：

$$
A_i = A_{i-1} + \mathcal{T}_i(Z_i).
$$

最终输出经 upsample 回到原始 hidden dimension，并接入任务头计算 loss。HiddenEcho 的主要优势是不需要像 SnD 一样单独预训练 denoiser，而是在下游任务 fine-tuning 中端到端优化。

### 3.3 HiddenEcho+

HiddenEcho 回传全部 hidden states，通信成本较高。HiddenEcho+ 引入两个模块降低通信量。

第一，Hidden Layer Filter 根据梯度贡献选择关键层。设服务端共有 $L$ 层，HLF 计算每层 hidden state 对最终输出的贡献度，并选出 $k$ 个关键层。训练和推理时只回传这些层的 hidden states。

第二，Dimension Reducer 将回传 hidden states 从 $d$ 维压缩到 $d'=d/r$ 维，并通过 information bottleneck 约束保留任务相关信息。

因此，HiddenEcho 的 hidden-state 通信量近似为：

$$
V_{HE}=Lnd',
$$

HiddenEcho+ 的通信量近似为：

$$
V_{HE+}=knd',
$$

理论节省率为：

$$
1-\frac{k}{L}.
$$


### 3.4 Baseline 与评估指标

本文复现中，LDP、GAN-DP 和 SnD 是对比基线：LDP 直接在 embedding 上加入 $d_\chi$ 噪声；GAN-DP 使用生成式扰动方法；SnD 使用预训练 denoiser。

评估指标包括原文指标以及本文新增指标：

| 指标 | 含义 |
|---|---|
| AUC | 分类任务性能，越高越好 |
| Accuracy / Macro-F1 | 辅助分类指标 |
| EP | Empirical Privacy，越高表示攻击越困难 |
| BLEU / ROUGE | 生成任务质量指标 |
| hidden-state transfer | 服务端回传 hidden states 的通信量 |
| client parameters / state_dict size | 客户端 denoiser 结构开销 |

## 4. 复现实验设置

在文本分类和生成任务上评估扰动方法，分类任务使用Qwen2-1.5B，生成任务使用T5-Large（参数0.75B）。数据集包括用于分类的 Financial Phrasebank 和Tweet Annotation，以及用于生成的 CNN/DailyMail 。我们通过Transformers（Wolf等，2020）和PEFT
（Mangrulkar等，2022）进行LoRA微调，使用AdamW和线性调度器（初始学习率=1.5e-4）
。所有实验在NVIDIA RTX 5090 GPU上运行。

## 5. 复现实验结果

### 5.1 Financial / Qwen 主表结果

表 1 给出 Financial / Qwen 上的主要复现结果。为避免把不同数据配置混成严格公平比较，表中将 `sentences_allagree` 主链路和 `sentences_50agree` 诊断结果显式标出。

表 1  Financial Phrasebank 上的 AUC / EP 复现结果。

| 方法 | 指标 | $\eta=100$ | $\eta=1000$ | $\eta=5000$ | $\eta=6000$ |
|---|---|---:|---:|---:|---:|
| GAN-DP (`allagree`) | AUC | 0.5203 | 0.5822 | 0.6467 | 0.7197 |
| GAN-DP (`allagree`) | EP | 1.0000 | 0.9970 | 1.0000 | 1.0000 |
| LDP (`allagree`) | AUC | 0.6470 | 0.6504 | 0.7507 | 0.7691 |
| LDP (`50agree`, 诊断) | AUC | 0.5894 | 0.6171 | 0.6792 | 0.6949 |
| LDP | EP | 0.9823 | 0.9867 | 0.9408 | 0.8934 |
| SnD (`50agree`, 诊断) | AUC | 0.5747 | 0.5745 | 0.6027 | 0.5687 |
| HiddenEcho (`allagree`) | AUC | 0.8494 | 0.8685 | 0.8777 | 0.8680 |
| HiddenEcho+ (`allagree`, reduce1) | AUC | 0.8552 | 0.8731 | 0.8832 | 0.8705 |

原论文 Table 1 中 Financial / Qwen2-1.5B 的 AUC 为：GAN-DP 0.501、0.524、0.618、0.629；LDP 0.596、0.595、0.629、0.617；SnD 0.558、0.565、0.595、0.630；HiddenEcho 0.875、0.874、0.883、0.889；HiddenEcho+ 0.857、0.855、0.860、0.866。对比可见，HiddenEcho 与 HiddenEcho+ 的绝对 AUC 基本处于原论文同量级；尤其 HiddenEcho 在 $\eta=1000,5000$ 下与论文差距约 0.5 个 AUC 点。HiddenEcho+ reduce1 在 $\eta=100,5000$ 下分别达到 0.8552 和 0.8832，与论文 0.857 和 0.860 接近或略高。

### 5.2 通信与训练开销复现



结果趋势与原论文一致：HiddenEcho 相比 LDP 有额外 denoising 开销，但显著快于 SnD；HiddenEcho+ 因只使用部分 hidden layers，训练时间略低于 full HiddenEcho。

通信复现结果与原论文 Financial / Qwen 的 HE 1.97 MiB、HE+ 0.28 MiB 基本一致，且符合理论计算值。

### 5.3 EIA 与 AIA 复现

EIA 结果 EP 见实验主表。论文对应 LDP/HiddenEcho/SnD EP 为 0.988、0.987、0.967、0.886。本文结果处于同一量级，且预算增大后 EP 整体下降，符合更弱噪声带来更低隐私保护的趋势。

AIA 使用 Tweet Annotation Sensitivity 2 数据集，攻击属性包括 education 和 age。最终结果显示，LDP 与 HiddenEcho 均显著高于无保护基线，说明扰动能够降低属性推断能力；在中高预算区域，HiddenEcho 的 education EP 和 age RMSE 多数略高于 LDP，同时任务 AUC 明显更好。因此 AIA 结果支持原论文 Fig. 3 的主要结论：HiddenEcho 在改善任务效用的同时没有削弱属性隐私保护。

### 5.4 CNNDM / T5 生成任务复现

clean T5 baseline 的复现 BLEU 为 19.449，原论文 clean T5 CNNDM BLEU 为 17.738。由于本地 clean upper bound 更高，HiddenEcho 在 $\eta=30,40$ 下的绝对 BLEU 也整体偏高。总体上，$\eta=20<30<40$ 的性能恢复趋势与论文一致。

## 6. 改进方法：EchoSlim

### 6.1 问题发现

复现实验结果的过程中，我们发现 HiddenEcho+  在客户端 denoising module 仍完整加载 $L$ 个 side transformer blocks 和对应 gate vectors。未选层虽然不参与 forward，也不会在 backward 中更新，但仍占用客户端参数、state_dict 存储、加载时间和模型驻留内存。

设 HLF 得到选层集合：

$$
S=\{s_1,s_2,\cdots,s_k\}, \quad k\ll L.
$$

原 HiddenEcho+ 的 forward 对未选层近似为：

$$
A_i=A_{i-1},\quad i\notin S,
$$

对选中层执行：

$$
A_i=A_{i-1}+\mathcal{T}_i(\mu_i A_{i-1}+(1-\mu_i)H_i^{dn}),\quad i\in S.
$$

这说明未选层在计算图上是空路径，但原结构仍保留全部 $\mathcal{T}_0,\ldots,\mathcal{T}_{L-1}$。EchoSlim 基于此进行优化。

### 6.2 EchoSlim 结构

EchoSlim 保留 HiddenEcho+ 的 HLF、DR、隐私预算和通信内容，仅改变客户端 denoiser 的实例化方式。将 $S$ 按层号升序排列：

$$
S=\{s_1<s_2<\cdots<s_k\}.
$$

EchoSlim 构造 compact denoiser：

$$
\mathcal{D}_{slim}=\{\tilde{\mathcal{T}}_1,\tilde{\mathcal{T}}_2,\cdots,\tilde{\mathcal{T}}_k\}.
$$

第 $j$ 个 compact block 对应原服务端第 $s_j$ 层 hidden state，而不是简单对应第 $j$ 层。递推过程为：

$$
\tilde{A}_0=E^{dn},
$$

$$
\tilde{A}_j=\tilde{A}_{j-1}+\tilde{\mathcal{T}}_j(\tilde{\mu}_j\tilde{A}_{j-1}+(1-\tilde{\mu}_j)H_{s_j}^{dn}),
$$

$$
\tilde{\mu}_j=\sigma(\tilde{g}_j).
$$

最终输出为：

$$
H^{denoised}=W^{up}\tilde{A}_k.
$$

为了保持与 HiddenEcho+ 的可比性，EchoSlim 按原始层号初始化 compact block：

$$
\tilde{\mathcal{T}}_j \leftarrow \mathcal{T}^{server}_{s_j}.
$$

### 6.3 复杂度分析

设每个 side transformer block 参数量为 $P_T$，gate 参数量为 $P_g$，固定参数量为 $P_{fixed}$。原 HiddenEcho+ 客户端 denoiser 参数量为：

$$
P_{HE+}=L(P_T+P_g)+P_{fixed}.
$$

EchoSlim 参数量为：

$$
P_{slim}=k(P_T+P_g)+P_{fixed}.
$$

参数减少比例为：

$$
R_P=1-\frac{k(P_T+P_g)+P_{fixed}}{L(P_T+P_g)+P_{fixed}}.
$$

当固定参数较小时，近似为：

$$
R_P\approx 1-\frac{k}{L}.
$$

通信复杂度和客户端 forward 计算复杂度理论上应该基本不变，因为 EchoSlim 与 HiddenEcho+ 回传同一组选中 hidden states：

$$
O(knd').
$$

## 7. EchoSlim 实验与结果

### 7.1 实验设置

分类任务使用Qwen2-1.5B，生成任务使用T5-Large（参数0.75B）。数据集包括用于分类的 Financial Phrasebank 和Tweet Annotation，以及用于生成的 CNN/DailyMail 。我们通过Transformers（Wolf等，2020）和PEFT
（Mangrulkar等，2022）进行LoRA微调，使用AdamW和线性调度器（初始学习率=1.5e-4）
。所有实验在NVIDIA RTX 5090 GPU上运行。(与 HiddenEcho+ 完全对齐)

### 7.2 分类任务上的实验结果

表 4 给出 EchoSlim 与 HiddenEcho+ 在相同 HLF 选层、相同 DR、相同隐私预算和相同通信内容下的结果。

表 4  EchoSlim 与 HiddenEcho+ 分类结果。

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

EchoSlim 的客户端参数量减少：

$$
1-\frac{12.89}{83.12}=84.49\%.
$$

denoiser state_dict 从 158.53MB 降至 24.58MB，减少 84.50%。AUC 相对 HiddenEcho+ 的下降分别为 0.0024、0.0078、0.0029、0.0048，最大下降 0.78 个百分点。Accuracy 最大下降 0.97 个百分点；Macro-F1 在 $\eta=5000$ 略高于 HiddenEcho+，在 $\eta=1000$ 下降较明显。总体结论是：EchoSlim 显著降低客户端结构开销，并基本保持 AUC 与 Accuracy，F1 有小幅波动。

训练资源方面，EchoSlim 的 peak allocated memory 从约 21.40GiB 降至 20.95GiB，下降约 2.10%；训练耗时仅小幅下降。这符合复杂度分析：原 HiddenEcho+ 已经跳过未选层计算，EchoSlim 主要减少参数驻留和保存开销。

### 7.3 改进策略的实验验证

为验证未选 side layers 是否确实冗余，本文在原 HiddenEcho+ 上执行 HLF 选层后，对两个 probe batches 做 backward，并统计完整 client side stack 中各层梯度。实验配置为 $\eta=5000$，选层为 `[0,2,4,27]`。

表 5  HiddenEcho+ side stack 梯度诊断。

| 分组 | 层数 | 层索引 | Side layer + gate 参数量 | backward 后有梯度的层 |
|---|---:|---|---:|---|
| selected | 4 | `[0,2,4,27]` | 11.70M | `[0,2,4,27]` |
| skipped | 24 | `[1,3,5,...,26]` | 70.23M | `[]` |

结果显示，所有 selected layers 均有非零梯度，所有 skipped layers 梯度为空。Skipped layers 占完整 side stack 参数的：

$$
\frac{70.23M}{81.93M}=85.71\%.
$$

这说明 HiddenEcho+ 的 HLF 已经将未选层从有效计算图中移除，但结构上仍实例化这些层。EchoSlim 删除的是这部分 gradient-dead side layers。

### 7.4 T5 / CNNDM 生成任务实验结果

表 6  EchoSlim 与 HiddenEcho+ 在 CNNDM / T5 上的结果。

| $\eta$ | 方法 | HLF 选层 | Adapter state | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---:|---|---|---:|---:|---:|---:|---:|
| 20 | HiddenEcho+ | `[1,3,9,23]` | 98.70MB | 8.0763 | 0.3186 | 0.1198 | 0.2388 |
| 20 | EchoSlim | `[1,3,9,23]` | 35.73MB | 7.5137 | 0.3172 | 0.1146 | 0.2321 |
| 30 | HiddenEcho+ | `[1,2,3,23]` | 98.70MB | 15.0758 | 0.4256 | 0.2044 | 0.3185 |
| 30 | EchoSlim | `[1,2,3,23]` | 35.73MB | 16.7516 | 0.4386 | 0.2137 | 0.3334 |
| 40 | HiddenEcho+ | `[2,4,6,23]` | 98.70MB | 18.2022 | 0.4553 | 0.2329 | 0.3519 |
| 40 | EchoSlim | `[2,4,6,23]` | 35.73MB | 18.4079 | 0.4512 | 0.2324 | 0.3490 |

T5 EchoSlim adapter state 从 98.70MB 降至 35.73MB，减少约 63.80%。三组预算下，EchoSlim 与 HiddenEcho+ 的 `embedding_data_transferred` 和 `hiddens_data_transferred` 均为 8305737728，说明通信量保持一致。按 ROUGE 观察，EchoSlim 与 HiddenEcho+ 基本持平，三组预算平均 ROUGE 略高于 HiddenEcho+：

| 指标 | HiddenEcho+ 平均 | EchoSlim 平均 | 差值 |
|---|---:|---:|---:|
| ROUGE-1 | 0.3998 | 0.4023 | +0.0025 |
| ROUGE-2 | 0.1857 | 0.1869 | +0.0012 |
| ROUGE-L | 0.3031 | 0.3049 | +0.0018 |

BLEU 在 $\eta=20$ 下低于 HiddenEcho+，在 $\eta=30,40$ 下持平或更高。由于摘要任务通常更重视内容覆盖，ROUGE 比 BLEU 更适合作为主要参考。

## 8. 结论

本文完成了 HiddenEcho 论文的主要复现与 EchoSlim 改进验证。复现结果支持原论文的核心判断：在 embedding-level DP 场景下，HiddenEcho 能通过服务端 hidden states 引导客户端 denoising，缓解噪声层间放大带来的性能下降；HiddenEcho+ 能通过 HLF 和 DR 显著降低 hidden-state 通信量。在 Financial / Qwen 通信复现中，HiddenEcho+ 只回传 4/28 层 hidden states，使单 batch hidden-state 通信量从 1.96875 MiB 降至 0.28125 MiB，节省 85.71%。

在复现的基础上，本文进一步提出改进方案 EchoSlim，将 HiddenEcho+ 的 HLF 运行时稀疏性转化为客户端 denoiser 的结构稀疏性。分类任务中，EchoSlim 将客户端参数减少 84.49%，denoiser state 减少 84.50%，同时 AUC 最大下降仅 0.78%，hidden-state 通信量保持不变。生成任务中，EchoSlim 将 T5 adapter state 减少 63.80%，ROUGE 指标基本保持。结果说明 EchoSlim 是对 HiddenEcho+ 的部署侧补充优化，适合资源受限客户端场景。

## 参考文献

[1] Anonymous authors. HiddenEcho: Hidden-State Correction to Mitigate Inter-Layer Noise Amplification in LLMs under Differential Privacy.

[2] Qu et al. $d_\chi$-privacy based perturbation for text embeddings.

[3] Mai et al. SnD: Server-side noise denoising for differentially private language model inference.

[4] Song C., Raghunathan A. Information Leakage in Embedding Models. CCS 2020.

[5] Gupta O., Raskar R. Distributed learning of deep neural network over multiple agents. 2018.

[6] Zhang et al. Privacy and efficiency of communications in federated split learning. IEEE Transactions on Big Data, 2023.

[7] Alemi et al. Deep Variational Information Bottleneck. ICLR 2017.

[8] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.

[9] Malo et al. Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts. Journal of the Association for Information Science and Technology, 2014.

[10] Papineni et al. BLEU: a Method for Automatic Evaluation of Machine Translation. ACL 2002.

[11] Lin C.-Y. ROUGE: A Package for Automatic Evaluation of Summaries. 2004.
