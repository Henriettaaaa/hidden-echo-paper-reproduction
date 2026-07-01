# HiddenEcho 论文复现报告

## 1. 复现目标

本报告旨在复现 HiddenEcho 论文中关于 Qwen 系列模型的文本分类实验，重点验证三项结论：其一，LDP 在差分隐私扰动下的性能退化；其二，HiddenEcho 通过回传隐藏状态并进行客户端去噪后，对分类性能的提升效果；其三，HiddenEcho+ 在保持性能的同时降低通信开销的有效性。

本阶段复现范围聚焦于 Financial Phrasebank 数据集上的分类任务。生成任务与 Llama3 结果不纳入本阶段主报告，相关内容仅作为后续扩展计划保留。

## 2. 方法概述

HiddenEcho 是一种面向 split learning 场景的隐私增强框架。与仅对 embedding 加噪的 LDP 不同，HiddenEcho 将服务器侧中间隐藏状态回传至客户端，由轻量去噪模块结合初始 embedding 与隐藏状态共同完成修正；HiddenEcho+ 则进一步引入 hidden layer selection 与 information bottleneck 压缩，以降低通信量。

从论文实验设计看，分类任务的核心比较对象包括 LDP、HiddenEcho 和 HiddenEcho+；攻击评测主要采用 EIA，AIA 则对应 Tweet Annotation 数据集上的属性推断实验；消融实验用于验证 residual connection、hidden layer filter 与 dimension reducer 的作用；优化与通信开销分析分别对应论文中的 Fig. 4 与 Table 3。

## 3. 复现范围

| 模块 | 是否纳入本阶段 | 说明 |
|---|---:|---|
| Financial Phrasebank 分类 | 是 | 当前主复现对象 |
| LDP 基线 | 是 | 作为隐私扰动下的对照基线 |
| HiddenEcho | 是 | 验证隐藏状态回传 + 客户端去噪 |
| HiddenEcho+ | 是 | 验证通信压缩与性能折中 |
| EIA | 是 | 与分类任务配套的主要攻击评测 |
| AIA | 可选 | 对应 Tweet Annotation，建议后续补充 |
| 消融实验 | 建议补充 | 用于解释 HiddenEcho+ 的结构贡献 |
| MRPC / BBC News | 可选 | 用于增强论文表 1 的完整性 |
| Llama3 分类 | 不纳入 | 本阶段范围外 |
| 生成任务 | 不纳入 | 本阶段范围外 |

## 4. 实验设置

| 项目 | 设置 |
|---|---|
| 数据集 | Financial Phrasebank |
| 数据划分 | [待填：与论文一致 / 当前实现口径] |
| 基座模型 | [待填：Qwen2-1.5B 或当前使用模型] |
| 训练方式 | LoRA fine-tuning |
| 优化器 | AdamW |
| 学习率调度 | linear |
| 初始学习率 | 1.5e-4 |
| 训练轮数 | 20 |
| 批大小 | 48 / 48 |
| 最大长度 | 128 |
| LoRA rank | 16 |
| 隐私预算 | η = [待填] |
| 噪声类型 | Chi |
| HiddenEcho | enabled / disabled |
| HiddenEcho+ | enabled / disabled |

## 5. 主结果

### 5.1 分类性能

| 方法 | η | AUC | Accuracy | F1 |
|---|---:|---:|---:|---:|
| LDP | [待填] | [待填] | [待填] | [待填] |
| HiddenEcho | [待填] | [待填] | [待填] | [待填] |
| HiddenEcho+ | [待填] | [待填] | [待填] | [待填] |

### 5.2 与论文对齐情况

| 方法 | 论文结果 | 复现结果 | 差异 | 结论 |
|---|---:|---:|---:|---|
| LDP | [待填] | [待填] | [待填] | [待填] |
| HiddenEcho | [待填] | [待填] | [待填] | [待填] |
| HiddenEcho+ | [待填] | [待填] | [待填] | [待填] |

### 5.3 通信开销

| 方法 | embedding_data_transferred | hiddens_data_transferred | 相对 HiddenEcho 节省比例 |
|---|---:|---:|---:|
| HiddenEcho | [待填] | [待填] | - |
| HiddenEcho+ | [待填] | [待填] | [待填] |

### 5.4 训练时间

| 方法 | train_runtime | 平均每 epoch 时间 | 备注 |
|---|---:|---:|---|
| LDP | [待填] | [待填] | [待填] |
| HiddenEcho | [待填] | [待填] | [待填] |
| HiddenEcho+ | [待填] | [待填] | [待填] |

## 6. 训练过程分析

### 6.1 HiddenEcho

本部分对应论文 Fig. 4(a)。请给出训练 loss、验证 loss 与验证 AUC 随 epoch 变化的总体趋势，并说明最优 epoch、是否出现过拟合，以及测试结果与验证最优点之间的关系。

[待填：曲线分析]

### 6.2 HiddenEcho+

本部分对应论文 Fig. 4(b)-(d)。请说明不同保留层数设置下的收敛速度、性能上限与通信代价，并给出最终采用配置的原因。

[待填：曲线分析]

## 7. 攻击评测

### 7.1 EIA

EIA 用于衡量在已知 embedding 与 embedding matrix 条件下的重建风险。该部分应与分类任务共享相同的隐私预算设置，并优先报告与主实验一致的 η 取值。

| 方法 | η | 指标 | 结果 |
|---|---:|---|---:|
| LDP | [待填] | AUC / EP | [待填] |
| HiddenEcho | [待填] | AUC / EP | [待填] |
| HiddenEcho+ | [待填] | AUC / EP | [待填] |

### 7.2 AIA

AIA 对应 Tweet Annotation 数据集上的属性推断实验，建议作为独立扩展实验呈现，不与 Financial Phrasebank 混写。

| 方法 | 数据集 | 指标 | 结果 |
|---|---|---|---:|
| LDP | Tweet Annotation | EP / RMSE | [待填] |
| HiddenEcho | Tweet Annotation | EP / RMSE | [待填] |
| HiddenEcho+ | Tweet Annotation | EP / RMSE | [待填] |

## 8. 消融实验

消融实验用于解释 HiddenEcho+ 的结构贡献，建议作为主报告的补充部分或附录保留。

| 变体 | 含义 | AUC | 结论 |
|---|---|---:|---|
| HiddenEcho+ | 完整模型 | [待填] | [待填] |
| -Res | 去除 residual connection | [待填] | [待填] |
| -HLF | 改为固定选层 | [待填] | [待填] |
| -DR | 以线性层替代 dimension reducer | [待填] | [待填] |

## 9. 讨论

### 9.1 复现一致性

请概述哪些结果与论文保持一致，哪些结果存在偏差，并说明偏差主要来自模型版本、数据划分、实现细节还是训练超参。

[待填：一致性讨论]

### 9.2 结论

请用一段话总结本阶段复现是否支持论文的核心结论：LDP 作为基线的性能水平、HiddenEcho 的有效性、HiddenEcho+ 的通信压缩效果，以及当前复现中仍未完全对齐的部分。

[待填：总结]

## 10. 附录

| 内容 | 路径 / 说明 |
|---|---|
| 训练脚本 | [待填] |
| 日志文件 | [待填] |
| 输出目录 | [待填] |
| 环境配置 | [待填] |
| 备注 | [待填] |

