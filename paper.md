# HIDDENECHO: HIDDEN-STATE CORRECTION TO MIT-IGATE INTER-LAYER NOISE AMPLIFICATION IN LLMS UNDER DIFFERENTIAL PRIVACY

Anonymous authors Paper under double-blind review

## ABSTRACT

The rise of large language models (LLMs) has driven the adoption of Modelas-a-Service (MaaS), but transmitting raw text to servers raises critical privacy concerns. Existing approaches employ deep neural networks (DNNs) or differential privacy (DP) to perturb inputs, while they face key limitations: DNNbased methods often require task-specific pre-training, and conventional DP techniques, though privacy-preserving, suffer from noise amplification as perturbed inputs propagate through the deep transformer layer, leading to significant degradation in downstream task performance. To alleviate this, we propose HiddenEcho, an end-to-end framework with client noise correction, where hidden states are sent from the server to the client and refined by a lightweight module using both embeddings and intermediate representations. HiddenEcho suppresses inter-layer noise amplification without pretraining, effectively preserving task-relevant signals under DP constraints. To further reduce communication, HiddenEcho + incorporates gradient-based hidden layer selection and information bottleneck compression, reducing communication cost while preserving essential task information. Experiments across text classification and generation tasks demonstrate that HiddenEcho achieves up to 46.89% performance improvement over DP baselines, over 85% communication reduction, and up to 72.52% faster training compared to existing denoising approaches, establishing a new privacy-utility trade-off for privatized LLMs. Codes are available at https://anonymous.4open.science/r/hidden-echo.

## 1 INTRODUCTION

The advancement of large language models (LLMs) has profoundly transformed scientific research Kulmanov et al. (2024); VM et al. (2024); Li et al. (2023b); Yang et al. (2024b). The substantial computational costs associated with the growing number of parameters in LLMs have driven the emergence of the Model-as-a-Service (MaaS) paradigm. MaaS offers a platform for users without access to high-performance computing resources, enabling them to leverage LLMs for various purposes, including inference, fine-tuning, and the development of customized agents (David et al., 2014). Nevertheless, MaaS also raises significant security concerns. Specifically, sensitive information, such as personally identifiable information (PII), including names, phone numbers, email addresses, and financial details, may be exposed when users upload data to LLM vendors.

Privacy protection for LLMs in the MaaS framework mainly relies on cryptography-based and perturbation-based methods. While cryptographic techniques like secure multiparty computation (Hou et al., 2024) and homomorphic encryption (Liu & Liu, 2023) provide strong security, their high computational overhead makes them impractical for resource-constrained clients.

In contrast, perturbation-based methods have gained attention because of their flexibility to add perturbations to the data as a privacy-preserving mechanism. For instance, deep neural network (DNN)-based perturbation methods leverage learned data distributions to generate perturbed data that can deceive adversaries. However, these approaches typically require pretraining phases for the whole training, limiting their practicality. Differential privacy (DP) as a perturbation-based method with lower computational overhead, has emerged as an alternative. It introduces noise of a specified intensity to the input on the client side before transmitting it to the server. For example, Qu et al. (Qu et al., 2021) proposed adding $d _ { \chi }$ -based noise to text embeddings, achieving enhanced privacy protection at the cost of reduced accuracy. However, when such noise is left unprocessed, it leads to significant performance degradation in downstream tasks when applied to LLM. Mai et al. (Mai et al., 2024) improve this issue with their SnD framework, which involves pretraining a denoising module on the server and deploying it on the client. This approach filters out a part of noises and enhances model performance.

<!-- image-->  
Figure 1: Mean squared error (MSE) between clean hidden states and noisy hidden states under different privacy budgets based on Qwen2-1.5B (Yang et al., 2024a) with 27 hidden layers retaining on the server side on the MRPC dataset (Wang et al., 2018).

Nevertheless, Experiments show that differential privacy noise in text embeddings is progressively amplified through LLM transformer blocks, leading to increasing MSE and significant performance degradation, as seen in the ${ } ^ { " } \mathrm { L D P } ( d _ { \chi } ) ( \eta { = } 1 0 0 ) ^ { 3 }$ curve in Fig 1. Existing denoising methods, relying on pretraining and disconnected from LLM dynamics, fail to mitigate inter-layer noise effectively.

Based on this, we propose an end-to-end framework HiddenEcho that integrates noise correction in the MaaS to protect data privacy in LLMs. Unlike existing denoising approaches: (1) it eliminates the need for pretraining, enabling effective denoising of inter-layer noise from the server; (2) it fully leverages the internal hidden layer information of LLMs, optimizing their performance; and (3) Considering the communication overhead between the client and server, we introduce HiddenEcho +, which incorporates a gradient-based hidden layer filter to identify and select critical hidden layers, alongside an information bottleneck-based dimension reducer to retain essential information from the hidden states. This design enables near-complete noise correction with minimal data transmission, striking an effective balance between communication efficiency and model performance. As illustrated by the ”HiddenEcho ” curves in Fig 1, In the final hidden layer, HiddenEcho (η=100) reduces noise (14.69 → 8.31) by 43.43% compared to $\mathrm { L D P } ( d _ { \chi } ) ( \eta = 1 0 0 )$

In summary, our contributions are: ❶ We identify and analyze the critical issue of noise amplification in LLMs under differential privacy, where injected noise grows progressively through hidden layers, severely degrading model performance. ❷ We propose HiddenEcho, an end-to-end framework that enables pretraining-free, progressive noise correction via client-side denoising guided by hidden states from server, which is applicable to both inference and fine-tuning with balanced privacy, utility, and communication cost. ❸ We evaluate HiddenEcho in MaaS scenarios, showing up to 46.89% performance gain in text classification over baselines, over 85% communication reduction with HiddenEcho +, and 72.52% faster denoising compared to existing methods.

## 2 RELATED WORKS

Privacy Preservation for LLMs Privacy preservation in LLMs has become critical with widespread deployment (Miranda et al., 2024). Existing approaches fall into cryptographic and perturbation-based methods. Cryptographic techniques, such as secure multi-party computation (Hou et al., 2024) and homomorphic encryption (Hao et al., 2022; Liu & Liu, 2023), offer strong privacy guarantees but incur high computational costs and are limited to defending against external adversaries, making them impractical for resource-constrained clients. Perturbation-based methods provide a more flexible trade-off between privacy and utility. While some approaches perturb model outputs (Liu et al., 2019) or use adversarial training (Coavoux et al., 2018a), differential privacy has emerged as a popular choice in the MaaS paradigm due to its lightweight noise injection into embeddings (Lyu et al., 2020; Qu et al., 2021; Shen et al., 2023; Li et al., 2023a). However, DP noise is amplified through transformer layers, degrading model performance. SnD (Mai et al., 2024) introduces a client-side denoising module to mitigate this effect, but fails to fully address noise propagation across deep transformer blocks—a challenge our work aims to resolve.

## 3 PRELIMINARIES

## 3.1 THREAT MODELS

For language models, attackers typically aim to extract sensitive information from the original user data. We consider a split MaaS deployment in which the client hosts the embedding layer and the server hosts the remaining model Shen et al. (2023). They follow the protocol but may attempt to infer additional information from observed artifacts. An attacker may be either (i) a malicious service provider), or (ii) an eavesdropper possessing any subset of the following: ❶ Perturbed embeddings: the attacker observes perturbed token embeddings $\Psi ( x ) = \mathcal { E } ( x ) ) + \delta$ submitted by the client. ❷ Embedding layer parameters: the attacker observes the embedding matrix $\bar { W _ { e m b } }$ used to map tokens to vectors. As highlighted in (Song & Raghunathan, 2020; Shen et al., 2023), Embedding Inversion Attacks (EIA) and Attribute Inference Attacks (AIA) represent significant privacy threats in machine learning:

Definition 1 (Embedding Inversion Attack (EIA)) Given perturbed embeddings $\Psi ( \boldsymbol { x } ) \in \mathbb { R } ^ { l \times d }$ and the embedding matrix $W _ { e m b } ,$ , the goal is to reconstruct each token t is recovered by

$$
\hat { v } _ { t } = \arg \operatorname* { m i n } _ { v \in \mathcal { V } } \| W _ { e m b } [ v ] - \Psi ( x ) \| _ { 2 } .
$$

Definition 2 (Attribute Inference Attack (AIA)) Let $a \in { \mathcal { A } }$ be a sensitive attribute. Given auxiliary labeled samples $\boldsymbol { S } = \{ ( \tilde { x } _ { i } , \tilde { a } _ { i } ) \}$ }, the attacker trains

$$
f _ { w } : \mathbb { R } ^ { l \times d }  \mathcal { A }
$$

on $( \Psi ( \tilde { x } _ { i } ) , \tilde { a } _ { i } )$ and predicts $\hat { a } = f _ { w } ( \Psi ( x ) )$ for target x.

## 3.2 PROBLEM DEFINITION

Based on threat models, we focus on the privacy concerns associated with data transfer between the client and server when utilizing LLMs in the MaaS. In this scenario, the client holds a private dataset $X ~ = ~ \{ x _ { 1 } , x _ { 2 } , \cdot \cdot \cdot ~ , x _ { n } \}$ . Following a split learning framework Gupta & Raskar (2018); Zhang et al. (2023), we mitigate the client’s resource constraints by deploying the word embedding layer E of the LLM on the client, while the remaining layers are hosted on the server. To ensure privacy, perturbations based on differential privacy, denoted as $\delta ,$ are applied to the embeddings on the client. The optimization of the global LLM after incorporating these perturbations can be formalized as follows:

$$
\theta ^ { * } = \underset { \theta } { \arg \operatorname* { m i n } } \frac { 1 } { \vert X \vert } \sum _ { x _ { i } \in X } \mathcal { L } ( \theta , \Psi ( \mathcal { E } ( x _ { i } ) + \delta ) ) .\tag{1}
$$

Here, θ are the model parameters to be optimized, and Ψ denotes a denoising module. To enhance the feedback received by the client from the server, the design of an effective Ψ for mitigating the impact of added noise on the model’s outputs is crucial.

## 4 METHODOLOGY

HiddenEcho leverages a full hidden layer correction algorithm to address noise amplification in LLMs. Due to the communication constraints inherent to the split learning framework, we propose

<!-- image-->  
Figure 2: Framework of HiddenEcho. The denoise module is deployed on the client side, and the operations related to LLM’s hidden layer of HiddenEcho and HiddenEcho + are deployed on the server side.

HiddenEcho +, which reduces transmission significantly with only a minor performance tradeoff. Fig. 2 shows the framework, Algorithm 1 details the workflow, and complexity analysis and theoretical justification are in Appendices D and F, respectively.

## 4.1 HIDDENECHO

In HiddenEcho, server-side hidden layer states are transmitted back to the client for correction.   
This process is designed to be integrated with the fine-tuning of the LLM.

Perturbation Tokenized texts are converted to embeddings $E = \mathcal { E } ( x _ { i } ) \in \mathbb { R } ^ { n \times d }$ on the client, where n is the sequence length and d is the hidden size of the server-side LLM. To ensure privacy, noise is added to embeddings, yeilding $E ^ { \prime } = E + \delta _ { \mathrm { { \scriptsize { i } } } }$ , which are then transmitted to the server.

Server-side Forward Propagation The server inputs the noisy embeddings $E ^ { \prime }$ into the LLM B. During forward propagation, intermediate hidden states $\pmb { H } = \bar { \pmb { \mathrm { B } } ( \ o E ^ { \prime } ) } = \{ \bar { H _ { 0 } } , \cdots , \bar { H _ { L - 1 } } \}$ are collected from all $L$ layers. However, injected noise progressively distorts the hidden states’ feature space, which prevents LLM from effectively learning the task information. Consequently, a denoising mechanism is crucial to correct these hidden states for effective task learning.

Denoising The client-side denoising module refines the hidden states received from the server. Drawing inspiration from the LST method (Sung et al., 2022), which uses a dimension-reduced LLM as a side network for downstream task learning, the denoising module takes the initial noise-free embedding and the hidden states of the LLM on the server side as input. By utilizing the information contained in the initial embedding, it generates optimized hidden states: $H ^ { \mathrm { d e n o i s e d } } = { \cal D } ( E , \pmb { H } )$ where D is the denoise module.

The denoise module has a hidden size of $d ^ { \prime } = d / r$ , where r is the reduction factor, and has L layers. Each layer i contains a transformer $\mathcal { T } _ { i }$ and a gate vector $\mathbf { \pmb { g } } _ { i }$ . To integrate the server-side hidden states, the input to layer i is a combination of $H _ { i }$ and the previous layer’s output $A _ { i - 1 }$ , with the gate vector $\mathbf { \vec { \mathbf { g } } } _ { i }$ controlling the proportion of this mixture. The proportion is computed by $\pmb { \mu } _ { i } = \mathrm { s i g m o i d } ( \pmb { g } _ { i } )$ Thus, the input to the transformer Ti is

$$
Z _ { i } = \pmb { \mu _ { i } } A _ { i - 1 } + ( 1 - \pmb { \mu _ { i } } ) H _ { i } ^ { \mathrm { d n } } ,\tag{2}
$$

where $H _ { i } ^ { \mathrm { d n } } \in \mathbb { R } ^ { n \times d ^ { \prime } }$ is the downsampled $H _ { i }$ . Specifically, for the first layer $A _ { i - 1 } = E ^ { \mathrm { d n } }$ , Edn also represents the downsampled E.

The mechanism adjusts the influence of the server-side hidden states on the denoising process, ensuring that the refined hidden state optimally balances the client-side and server-side information.

To further enhance the learning ability of the denoise module, residual connections are introduced, which propagate the information of the initial embeddings to the deeper layers, preserving the integrity of the original signals during denoising. The output of layer i is recursively defined as:

$$
A _ { i } = A _ { i - 1 } + { \mathcal { T } } _ { i } ( Z _ { i } ) .\tag{3}
$$

The downsampling process, along with the subsequent upsampling, is learned by linear layers on the server side to reduce communication cost:

$$
H _ { i } ^ { \mathrm { d n } } = \mathcal { W } _ { i } ^ { \mathrm { d n } } ( H _ { i } ) ,\tag{4}
$$

$$
E ^ { \mathrm { d n } } = { \mathcal { W } } _ { \mathrm { E m b } } ^ { \mathrm { d n } } ( E ) .\tag{5}
$$

The final output $A _ { L - 1 }$ of the denoising module is then upsampled back to the original dimension d to create the denoised hidden state:

$$
H ^ { \mathrm { d e n o i s e d } } = \mathcal { W } ^ { \mathrm { u p } } ( A _ { L - 1 } ) .\tag{6}
$$

Optimization The denoised hidden state is fed into a task-specific head to generate predictions, and the corresponding loss is computed for model optimization. For classification tasks, the head outputs logits, and cross-entropy loss is applied:

$$
\hat { y } = W ^ { \mathrm { t a s k } } ( H ^ { \mathrm { d e n o i s e d } } ) ,\tag{7}
$$

$$
\mathcal { L } ( \hat { y } , y ) = - \sum _ { i } y _ { i } \log ( \hat { y } _ { i } ) ,\tag{8}
$$

where y represents the vector of ground-truth labels. Both the denoising module and the task-specific parameters are optimized to minimize this loss, improving classification accuracy and denoising effectiveness. This ensures denoised hidden states effectively contribute to the task performance.

## 4.2 HIDDENECHO+

While leveraging all intermediate hidden states achieves relatively optimal denoising performance, the associated communication overhead reduces practicality. To optimize our framework, HiddenEcho + is designed to strike a balance between LLM performance and communication efficiency. Specifically, HiddenEcho + integrates a hidden layer filter and a dimension reducer to mitigate communication costs without substantial performance degradation.

Hidden Layer Filter Transmitting all intermediate hidden states between server and client incurs prohibitive communication costs. We observe that not all layers contribute equally to the final output, suggesting that selectively transmitting only the most informative layers could maintain performance while reducing overhead.

To quantify the contribution of each hidden layer to the final output, a gradient-based filter is designed. For a given layer $i ( i < L - 1 )$ ), we gradually vary the value of its hidden state from 0 to $H _ { i }$ and observe the corresponding changes in the output of the last layer. Denoting $\mathcal { T } _ { i } ^ { S }$ as layer i of the server-side LLM, we have:

$$
\hat { H } _ { L - 1 } = \mathcal { T } _ { L - 1 } ^ { S } \circ . . . \circ \mathcal { T } _ { i } ^ { S } ( \hat { H } _ { i } ) ,\tag{9}
$$

where $\hat { H } _ { i }$ is the current value of layer $i ,$ and $\hat { H } _ { L - 1 }$ is the output of the last layer corresponding to the hidden state $\hat { H } _ { i }$ . ◦ signifies the sequential application of layers, with each layer’s output feeding into the next layer in the sequence.

The layer’s contribution $C _ { i }$ is defined by the cumulative gradient of these output changes:

$$
C _ { i } = H _ { i } \int _ { 0 } ^ { H _ { i } } \frac { \partial \hat { H } _ { L - 1 } } { \partial \hat { H } _ { i } } \mathrm { d } \hat { H } _ { i } .\tag{10}
$$

However, in practice, calculating the continuous integral is computationally challenging. Following (Dai et al., 2022), we use Riemann approximation to compute the integral:

$$
C _ { i } = \frac { H _ { i } } { m } \sum _ { j = 1 } ^ { m } \frac { \partial \hat { H } _ { L - 1 } } { \partial \hat { H } _ { i } } \Bigg | _ { \hat { H } _ { i } = ( j / m ) H _ { i } } ,\tag{11}
$$

where m denotes the number of approximation steps.

This calculation is performed before fine-tuning. We sample a small subset from the training dataset. Each sample undergoes standard preprocessing: tokenization, embedding, and perturbation, but not denoising. The server computes the layer contributions for each sample using Eq. (11) and averages these contributions across all samples.

Layers with the highest k contributions are selected to minimize communication overhead while maintaining performance, where k is a small hyperparameter. During each forward pass, only these layers’ hidden states are transmitted, significantly reducing communication costs. Upon receiving these hidden states, the client’s denoising module correspondingly skips unselected layers, accelerating computation and lowering resource requirements.

Dimension Reducer While layer selection reduces the number of transmitted states, each hidden state remains high-dimensional. Projecting the hidden states of the server-side LLM using linear layers is often effective, but it may fail to learn optimal representations due to the lack of explicit optimization objectives. We address this by applying the information bottleneck technique (Alemi et al., 2017) to compress hidden states while preserving task-relevant information.

In HiddenEcho +, we formulate dimension reduction as an information bottleneck problem: minimize the mutual information (MI) between the noisy embedding $E ^ { \prime }$ and the downsampled hidden states $H _ { i } ^ { \mathrm { { d n } } }$ , while maximizing the MI between the denoised output Hdenoised and the downsampled hidden states $H _ { i } ^ { \mathrm { { d n } } }$ . The corresponding loss function is:

$$
\mathcal { L } ^ { \mathrm { I B } } = \frac { 1 } { n } \sum _ { i = 0 } ^ { n - 1 } I ( E ^ { \prime } ; H _ { i } ^ { \mathrm { d n } } ) - \beta I ( H ^ { \mathrm { d e n o i s e d } } ; H _ { i } ^ { \mathrm { d n } } ) .\tag{12}
$$

Consequently, the overall model optimization loss is a combination of the task loss and the information bottleneck loss, weighted by α, β:

$$
\mathcal { L } = \mathcal { L } ( \hat { y } , y ) + \alpha \mathcal { L } ^ { \mathrm { I B } } .\tag{13}
$$

Although exact MI computation for high-dimensional variables is inherently challenging (Belghazi et al., 2018), an exact value is often unnecessary for optimization. Based on this, MINE (Belghazi et al., 2018), a neural network-based approach, is employed to estimate MI effectively.

MINE uses a statistics network to learn a function $f _ { \theta }$ that maximizes the difference between its expectation over the joint distribution $P ( X , Y )$ , and the exponential expectation over the product of the marginal distributions $P ( X ) P ( Y )$ . Mathematically, this can be expressed as

$$
\operatorname* { m a x } _ { \theta } \left( \mathbb { E } _ { P ( X , Y ) } [ f _ { \theta } ( X , Y ) ] - \exp ( \mathbb { E } _ { P ( X ) } [ \mathbb { E } _ { P ( Y ) } [ f _ { \theta } ( X , Y ) ] ] ) \right) .\tag{14}
$$

The estimated MI is then approximated by the supremum of this difference:

$$
I ( X ; Y ) \approx \operatorname* { s u p } _ { \theta } \left( \mathbb { E } _ { P ( X , Y ) } [ f _ { \theta } ( X , Y ) ] - \exp ( \mathbb { E } _ { P ( X ) } [ \mathbb { E } _ { P ( Y ) } [ f _ { \theta } ( X , Y ) ] ] ) \right) .\tag{15}
$$

This neural network-based estimator allows for an efficient computation of MI in scenarios where traditional methods are computationally prohibitive.

Specially, we prepare two statistics networks for each hidden state $H _ { i } ^ { \mathrm { { d n } . } }$ one to estimate the MI $I ( E ^ { \prime } ; H _ { i } ^ { \mathrm { d n } } )$ , and the other to estimate $I ( H ^ { \mathrm { d e n o i s e d } } ; H _ { i } ^ { \mathrm { d n } } )$ . After calculating the task loss at each step, these statistics networks are optimized for several steps according to Eq. equation 14. Once the optimization process is finished, the networks are used to compute the MI estimates. The information bottleneck loss is computed based on these estimates, as described in Eq. equation 12.

## 5 EXPERIMENTS

We evaluate perturbation methods on text classification and generation tasks using Qwen2-1.5B and Llama3-1B (1.54B and 1.23B parameters) for classification, and T5-Large (0.75B parameters) for generation. Datasets include Financial Phrasebank, MRPC, BBC News, and Tweet Annotation for classification; IWSLT2014, CNN/DailyMail, and Samsum for generation. Details are provided in Appendix H.1. We employ LoRA fine-tuning via Transformers (Wolf et al., 2020) and PEFT (Mangrulkar et al., 2022), with AdamW and a linear scheduler (initial lr = 1.5e-4). Performance is measured using AUC and Empirical Privacy (Definition 4) for classification (Li et al., 2023a), and BLEU for generation (Papineni et al., 2002). All experiments run on an NVIDIA RTX 3090 GPU.

Attacks Following prior studies (Song & Raghunathan, 2020), we evaluate the privacy protection effectiveness of HiddenEcho and baseline methods under simulated attacks within the split federated learning framework (Shen et al., 2023). In our experiments, a white-box attack setting is assumed, where attackers have access to user-submitted text embeddings and the parameters of the embedding model. As described in 3.1, the Embedding Inversion Attack (EIA) and Attribute Inference Attack (AIA) models are used to evaluate the effectiveness of privacy preservation methods.

Table 1: Performance of different perturbation methods on text classification tasks based on Qwen2-1.5B.
<table><tr><td colspan="2">Dataset</td><td colspan="4">MRPC</td><td colspan="4">Financial</td><td colspan="4">BBC News</td></tr><tr><td colspan="2">Privacy Budget η</td><td>100</td><td>1000</td><td>5000</td><td>6000</td><td>100</td><td>1000</td><td>5000</td><td>6000</td><td>100</td><td>1000</td><td>5000</td><td>6000</td></tr><tr><td rowspan="2">GAN-DP</td><td rowspan="2">AUC EP</td><td>0.497</td><td>0.532</td><td>0.597</td><td>0.612</td><td>0.501</td><td>0.524</td><td>0.618</td><td>0.629</td><td>0.606</td><td>0.620</td><td>0.684</td><td>0.720</td></tr><tr><td>1.000</td><td>0.999</td><td>0.999</td><td>0.998</td><td>1.000</td><td>0.999</td><td>0.997</td><td>0.992</td><td>0.995</td><td>0.991</td><td>0.971</td><td>0.962</td></tr><tr><td rowspan="2">LDP</td><td>AUC</td><td>0.551</td><td>0.557</td><td>0.553</td><td>0.599</td><td>0.596</td><td>0.595</td><td>0.629</td><td>0.617</td><td>0.648</td><td>0.646</td><td>0.736</td><td>0.803</td></tr><tr><td>EP</td><td>0.988</td><td>0.987</td><td>0.956</td><td>0.867</td><td>0.988</td><td>0.987</td><td>0.967</td><td>0.886</td><td>0.973</td><td>0.972</td><td>0.914</td><td>0.820</td></tr><tr><td rowspan="2">SnD</td><td>AUC</td><td>0.513</td><td>0.513</td><td>0.526</td><td>0.533</td><td>0.558</td><td>0.565</td><td>0.595</td><td>0.630</td><td>0.627</td><td>0.628</td><td>0.629</td><td>0.637</td></tr><tr><td>AUC</td><td>0.646</td><td>0.657</td><td>0.661</td><td>0.667</td><td>0.875</td><td>0.874</td><td>0.883</td><td>0.889</td><td>0.685</td><td>0.803</td><td>0.839</td><td>0.960</td></tr><tr><td rowspan="2">HiddenEcho HiddenEcho+</td><td>AUC</td><td>0.660</td><td>0.655</td><td>0.666</td><td>0.668</td><td>0.857</td><td>0.855</td><td>0.860</td><td>0.866</td><td>0.732</td><td>0.747</td><td>0.805</td><td>0.951</td></tr><tr><td>AUC Improve %</td><td>19.78</td><td>15.22</td><td>11.56</td><td>9.15</td><td>46.81</td><td>46.89</td><td>40.38</td><td>41.11</td><td>12.96</td><td>24.30</td><td>13.99</td><td>19.55</td></tr></table>

1 The EP of SnD and HiddenEcho is consistent with that of LDP, while GAN-DP differs from the other methods. Subsequent tables follow this format in reporting EP.

## 5.1 RESULTS OF EMBEDDING INVERSION ATTACK

We evaluate various methods against embedding inversion attacks in text classification using Qwen2-1.5B under $d _ { \chi }$ -privacy budgets η = 100, 1000, 5000 (definition in Appendix B); results on Llama3-1B are in Appendix H.2. As shown in Table 1, HiddenEcho achieves consistently higher AUC scores, confirming its effectiveness in mitigating noise amplification. HiddenEcho, using full hidden states, delivers the best performance, which improves AUC by up to 46.89% (Financial Phrasebank) and 24.30% (BBC News). HiddenEcho +, which selectively transmits high-impact layers via gradient-based filtering, achieves competitive results with significantly reduced communication, even outperforming HiddenEcho on MRPC (+19.78%) and BBC News (+12.96%), suggesting that not all layers contribute positively to denoising. SnD underperforms due to its reliance on a fixed pre-trained denoising model, which fails to adapt to the shifting hidden distributions during fine-tuning, leading to ineffective noise removal. Additional EIA evaluation on text generation is provided in Appendix H.3.

## 5.2 ABLATION STUDY

We conduct ablation studies on HiddenEcho +, which subsumes all components of HiddenEcho. We evaluate three variants: removing residual connections (−Res), replacing the Hidden Layer Filter with fixed skip layers (−HLF ), and substituting the Dimension Reducer with a linear layer (−DR). As shown in Fig 2, the full HiddenEcho + consistently achieves the highest AUC across datasets and privacy budgets. Removing residual connections degrades performance by 1.1%–11.51%, with the largest drop on BBC News (9.4%–11.51%). Replacing the HLF causes the most significant decline—up to 14.1% (e.g., 0.732→0.629 on BBC News at η=100)—demonstrating the importance of dynamic layer selection in noise suppression. The −DR variant reduces AUC by 0.9%–13.9%, with greater impact on complex tasks (e.g., 6.5%–7.9% drop on Financial).

Table 2: Ablation study of HiddenEcho on text classification tasks based on Qwen2-1.5B.
<table><tr><td>Dataset</td><td colspan="3">MRPC</td><td colspan="3">Financial</td><td colspan="3">BBC News</td></tr><tr><td>Privacy Budget η</td><td>100</td><td>1000</td><td>5000</td><td>100</td><td>1000</td><td>5000</td><td>100</td><td>1000</td><td>5000</td></tr><tr><td>HiddenEcho+</td><td>0.660</td><td>0.655</td><td>0.666</td><td>0.857</td><td>0.855</td><td>0.860</td><td>0.732</td><td>0.747</td><td>0.805</td></tr><tr><td>HiddenEcho + -Res</td><td>0.646</td><td>0.648</td><td>0.658</td><td>0.814</td><td>0.815</td><td>0.819</td><td>0.659</td><td>0.661</td><td>0.729</td></tr><tr><td>HiddenEcho +-HLF</td><td>0.637</td><td>0.640</td><td>0.641</td><td>0.773</td><td>0.773</td><td>0.774</td><td>0.629</td><td>0.630</td><td>0.719</td></tr><tr><td>HiddenEcho + -DR</td><td>0.632</td><td>0.649</td><td>0.644</td><td>0.789</td><td>0.799</td><td>0.801</td><td>0.630</td><td>0.663</td><td>0.789</td></tr></table>

These results confirm that residual connections stabilize training, the HLF enhances communication and noise control, and the dimension reducer improves feature robustness, collectively ensuring architectural efficacy under DP perturbations.

## 5.3 RESULTS OF ATTRIBUTE INFERENCE ATTACK

Compared to other text classification datasets, the Tweet Annotation dataset includes critical attributes such as the author’s age and education, making it well-suited for attribute inference attacks. Following the approach in (Song & Raghunathan, 2020), we train an MLP model to predict related information for each tweet. For detailed architecture, refer to Appendix H.4. Specifically, we evaluate the model’s robustness using RMSE for age prediction and Empirical Privacy (EP) for education inference, where higher values indicate stronger resistance to attacks. As depicted in Fig 3, the red dashed line represents the privacy

<!-- image-->

<!-- image-->  
Figure 3: AIA performance on Tweet Annotation Sensitivity 2 (Kern et al., 2023) with Qwen2-1.5B.

protection capability without perturbation. Both HiddenEcho and standard LDP exhibit performance degradation as privacy protection increases. However, except in scenarios with high privacy budgets $( \mathbf { e . g . } , \eta = 1 0 0 )$ , where both methods show nearly comparable, HiddenEcho consistently outperforms LDP in terms of privacy protection under other conditions.

## 5.4 OPTIMIZATION

<!-- image-->  
(a) HiddenEcho

<!-- image-->  
(b) HiddenEcho + (4)

<!-- image-->  
(c) HiddenEcho + (8)

<!-- image-->  
(d) HiddenEcho + (16)

Figure 4: Optmization performance of HiddenEcho and HiddenEcho + with different hidden layers on BBC News based on Qwen2-1.5B.

We document the optimization process of the HiddenEcho framework, with results on the BBC News dataset using Qwen2-1.5B visualized in Fig 4. Specifically, we compare the optimization trajectories of two configurations: HiddenEcho, which utilizes full hidden layer states, and HiddenEcho +, which employs filtered hidden layers. The evaluation metrics encompass training loss, evaluation loss, and evaluation AUC, providing a comprehensive view of model convergence and classification performance. During optimization, HiddenEcho shows stable decline in evaluate loss in the early period, while overfitting starting at the 14th epoch, with increased evaluation loss and performance degradation, likely due to the use of full hidden layers for correction. In contrast, we observe the optimization trajectories of HiddenEcho + with 4, 8, and 16 hidden layers. The 4-layer configuration achieves an AUC above 75% by the 12th epoch. The hidden layer filter in HiddenEcho + enables more focused corrections, reducing overfitting. These findings suggest that using fewer hidden layers in HiddenEcho can lead to faster convergence and lower communication overhead without sacrificing performance.

Table 3: Training time cost overhead of different methods for one epoch of llm fine-tuning (left) and communication cost of HiddenEcho (HE) and HiddenEcho + (HE+) for one batch (right).
<table><tr><td colspan="6">Training time cost (Second)</td><td colspan="6">Communication cost (MiB)</td></tr><tr><td>Approaches</td><td></td><td>LDP</td><td>GAN-DP</td><td>SnD</td><td>HE</td><td>HE+</td><td colspan="2">Approaches</td><td>HE</td><td>HE+</td><td>Saved</td></tr><tr><td rowspan="3">MRPC Financial BBC News</td><td rowspan="3">Q</td><td>125 74</td><td>118</td><td>248</td><td>196</td><td>166</td><td rowspan="3">MRPC Financial</td><td rowspan="3">Q</td><td>2.63 1.97</td><td>0.38 0.28</td><td>85.55% 85.79%</td></tr><tr><td></td><td>76 97</td><td>184</td><td>115</td><td>92 108</td><td></td><td></td><td></td></tr><tr><td>95</td><td></td><td>393</td><td>118</td><td>BBC News</td><td>10.50</td><td>1.50</td><td>85.71%</td></tr><tr><td>IWSLT</td><td rowspan="3">T</td><td>25 35</td><td>26</td><td>-</td><td>51</td><td>37</td><td>IWSLT</td><td rowspan="3">T</td><td>6.00</td><td>2.25</td><td>62.50%</td></tr><tr><td>CNNDM</td><td></td><td>37</td><td>-</td><td>64</td><td>46</td><td>CNNDM</td><td>8.25</td><td>3.09</td><td>62.55%</td></tr><tr><td>Samsum</td><td>32</td><td>33</td><td></td><td>62</td><td>40</td><td>Samsum</td><td>3.05</td><td>1.14</td><td>62.62%</td></tr></table>

## 5.5 TIME COST

We compare the time overhead of different methods for perturbing embeddings by recording the training time for one epoch for each method. Statistics are shown in the left side of Table 3, where Q and T denotes Qwen2-1.5B and T5-Large, respectively. Since SnD is not applicable to text generation, we do not report statistics for it in this context. The HiddenEcho framework, which builds upon LDP, incurs higher computational overhead compared to LDP alone. However, when compared to SnD, which also includes a denoising module, HiddenEcho demonstrates faster training speeds, with time costs reduced by up to 72.52% on the BBC News dataset. Although HiddenEcho + incorporates additional steps such as a hidden layer filter and dimension reduction, it still achieves faster training speeds due to the use of fewer hidden layers. Notably, while the GAN-DP method based on DNN shows advantages in a single training epoch, it requires a pre-training process for the GAN, which adds to its overall time cost.

## 5.6 COMMUNICATION COST

This section analyzes the communication overhead of HiddenEcho. HiddenEcho requires transmitting hidden layer states between the server and client to enable correction. The full hidden states are transmitted in HiddenEcho, resulting in large data volumes and high real-time transmission demands during LLM fine-tuning. In contrast, HiddenEcho + compresses communication by selecting key hidden layers for transmission. The communication costs per data batch for both HiddenEcho variants are shown in the right side of Table 3. The results indicate that HiddenEcho + reduces communication overhead by over 60% compared to HiddenEcho. Specifically, for text classification tasks, it achieves a remarkable space saving of over 85%. For text generation tasks, which require HiddenEcho + to filter more hidden layers to achieve optimal performance, the space saving is approximately 62%. Under typical network bandwidth, client-server communication using HiddenEcho + remains unaffected.

## 6 CONCLUSION

Large language models (LLMs) in the Model-as-a-Service paradigm enable convenient customization but raise privacy concerns. While differential privacy (DP) mitigates these risks, it degrades model performance, especially as injected noise is amplified through multi-layer transformer blocks. To address this, we propose HiddenEcho, a split learning-based framework that integrates with hidden layers and supports both fine-tuning and inference. Experiments show that HiddenEcho achieves a superior privacy-utility trade-off and significantly improves downstream task performance under DP constraints, offering a novel solution to noise mitigation in privatized LLMs.

## REFERENCES

Iwslt2014, international workshop on spoken language translation. https://workshop2014. iwslt.org/, 2014.

Alexander A. Alemi, Ian Fischer, Joshua V. Dillon, and Kevin Murphy. Deep variational information bottleneck. In 5th International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26, 2017, Conference Track Proceedings. OpenReview.net, 2017.

Rouzbeh Behnia, Mohammadreza Reza Ebrahimi, Jason Pacheco, and Balaji Padmanabhan. Ewtune: A framework for privately fine-tuning large language models with differential privacy. In 2022 IEEE International Conference on Data Mining Workshops (ICDMW), pp. 560–566. IEEE, 2022.

Mohamed Ishmael Belghazi, Aristide Baratin, Sai Rajeswar, Sherjil Ozair, Yoshua Bengio, R. Devon Hjelm, and Aaron C. Courville. Mutual information neural estimation. In Jennifer G. Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, ICML 2018, Stockholmsmassan, Stockholm, Sweden, July 10-15, 2018 ¨ , volume 80 of Proceedings of Machine Learning Research, pp. 530–539. PMLR, 2018.

Maximin Coavoux, Shashi Narayan, and Shay Cohen. Privacy-preserving neural representations of text. In 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1–10. Association for Computational Linguistics, 2018a.

Maximin Coavoux, Shashi Narayan, and Shay B Cohen. Privacy-preserving neural representations of text. In 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1–10. Association for Computational Linguistics, 2018b.

Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8493–8502, 2022.

Olaf David, Wes Lloyd, Ken Rojas, Mazdak Arabi, Frank Geter, James C Ascough II, Tim Green, George Leavesley, and Jack Carlson. Model-as-a-service (maas) using the cloud services innovation platform (csip). 2014.

Oluwaseyi Feyisetan, Borja Balle, Thomas Drake, and Tom Diethe. Privacy-and utility-preserving textual analysis via calibrated multivariate perturbations. In Proceedings of the 13th international conference on web search and data mining, pp. 178–186, 2020.

Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. SAMSum corpus: A humanannotated dialogue dataset for abstractive summarization. In Proceedings of the 2nd Workshop on New Frontiers in Summarization, pp. 70–79, Hong Kong, China, November 2019. Association for Computational Linguistics.

Derek Greene and Padraig Cunningham. Practical solutions to the problem of diagonal dominance ´ in kernel document clustering. In Proceedings of the 23rd international conference on Machine learning, pp. 377–384, 2006.

Otkrist Gupta and Ramesh Raskar. Distributed learning of deep neural network over multiple agents. Journal of Network and Computer Applications, 116:1–8, 2018.

Meng Hao, Hongwei Li, Hanxiao Chen, Pengzhi Xing, Guowen Xu, and Tianwei Zhang. Iron: private inference on transformers. In Proceedings of the 36th International Conference on Neural Information Processing Systems, pp. 15718–15731, 2022.

Xiaoyang Hou, Jian Liu, Jingyu Li, Jiawen Zhang, and Kui Ren. Faster lookup table evaluation with application to secure llm inference. Cryptology ePrint Archive, 2024.

Christoph Kern, Stephanie Eckman, Jacob Beck, Rob Chew, Bolei Ma, and Frauke Kreuter. Annotation sensitivity: Training data collection methods affect model performance. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14874–14886, Singapore, December 2023. Association for Computational Linguistics.

Maxat Kulmanov, Francisco J Guzman-Vega, Paula Duek Roggli, Lydie Lane, Stefan T Arold, and ´ Robert Hoehndorf. Protein function prediction as approximate semantic entailment. Nature Machine Intelligence, 6(2):220–228, 2024.

Yansong Li, Zhixing Tan, and Yang Liu. Privacy-preserving prompt tuning for large language model services. arXiv preprint arXiv:2305.06212, 2023a.

Yinheng Li, Shaofei Wang, Han Ding, and Hang Chen. Large language models in finance: A survey. In Proceedings of the fourth ACM international conference on AI in finance, pp. 374–382, 2023b.

Xuanqi Liu and Zhuotao Liu. Llms can understand encrypted prompt: Towards privacy-computing friendly transformers. arXiv preprint arXiv:2305.18396, 2023.

Yi Liu, Jialiang Peng, JQ James, and Yi Wu. Ppgan: Privacy-preserving generative adversarial network. In 2019 IEEE 25Th international conference on parallel and distributed systems (ICPADS), pp. 985–989. IEEE, 2019.

Lingjuan Lyu, Xuanli He, and Yitong Li. Differentially private representation for nlp: Formal guarantee and an empirical study on privacy and fairness. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 2355–2365, 2020.

Peihua Mai, Ran Yan, Zhe Huang, Youjia Yang, and Yan Pang. Split-and-denoise: Protect large language model inference with local differential privacy. In International Conference on Machine Learning, pp. 34281–34302. PMLR, 2024.

Pekka Malo, Ankur Sinha, Pekka Korhonen, Jyrki Wallenius, and Pyry Takala. Good debt or bad debt: Detecting semantic orientations in economic texts. Journal of the Association for Information Science and Technology, 65(4):782–796, 2014.

Sourab Mangrulkar, Sylvain Gugger, Lysandre Debut, Younes Belkada, Sayak Paul, and Benjamin Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods, 2022.

Michele Miranda, Elena Sofia Ruzzetti, Andrea Santilli, Fabio Massimo Zanzotto, Sebastien ´ Bratieres, and Emanuele Rodol \` a. Preserving privacy in large language models: A survey on \` current threats and solutions. arXiv preprint arXiv:2408.05212, 2024.

Ramesh Nallapati, Bowen Zhou, Caglar Gulcehre, Bing Xiang, et al. Abstractive text summarization using sequence-to-sequence rnns and beyond. arXiv preprint arXiv:1602.06023, 2016.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Chen Qu, Weize Kong, Liu Yang, Mingyang Zhang, Michael Bendersky, and Marc Najork. Natural language understanding with privacy-preserving bert. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management, pp. 1488–1497, 2021.

Xicong Shen, Yang Liu, Huiqi Liu, Jue Hong, Bing Duan, Zirui Huang, Yunlong Mao, Ye Wu, and Di Wu. A split-and-privatize framework for large language model fine-tuning. arXiv preprint arXiv:2312.15603, 2023.

Congzheng Song and Ananth Raghunathan. Information leakage in embedding models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pp. 377–390, 2020.

Yi-Lin Sung, Jaemin Cho, and Mohit Bansal. Lst: Ladder side-tuning for parameter and memory efficient transfer learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 12991–13005. Curran Associates, Inc., 2022.

Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.

Kushala VM, Harikrishna Warrier, Yogesh Gupta, et al. Fine tuning llm for enterprise: Practical guidelines and recommendations. arXiv preprint arXiv:2404.10779, 2024.

Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pp. 353–355. Association for Computational Linguistics, November 2018.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick ´ von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 38–45, Online, October 2020. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024a.

Qimin Yang, CHEN JIEXIN, Runqi Su, Tao Tan, et al. Fine-tuning medical language models for enhanced long-contextual understanding and domain expertise. In First Workshop on Long-Context Foundation Models@ ICML 2024, 2024b.

Zongshun Zhang, Andrea Pinto, Valeria Turina, Flavio Esposito, and Ibrahim Matta. Privacy and efficiency of communications in federated split learning. IEEE Transactions on $B i g$ Data, 9(5): 1380–1391, 2023.

## A THE USE OF LARGE LANGUAGE MODELS

The language of this paper was edited using a large language model (LLM) to enhance clarity and readability. The final content and academic integrity remain the responsibility of the authors.

## B $d _ { \chi }$ PRIVACY

Differential Privacy (DP) is a perturbation-based privacy-preserving mechanism that provides a rigorous framework for safeguarding data confidentiality. By introducing carefully calibrated noise during the training or fine-tuning of LLMs, DP makes it significantly harder to extract sensitive information from the perturbed data (Behnia et al., 2022).

In particular, the $d _ { \chi }$ -based DP method is more suitable for text structural embeddings (Feyisetan et al., 2020). Based on the differential privacy, we define the $d _ { \chi }$ -Privacy.

Definition 3 (dχ-Privacy) Let X be the input domain, Y be the output domain, and $d _ { \chi }$ be a distance metric over X. A randomized mechanism $M : X  Y$ satisfies ηdχ-privacy if for any two inputs $x , x ^ { \prime } \in X$ and any subset $S \subseteq Y$ , the following inequality holds:

$$
\frac { \operatorname* { P r } [ M ( x ) \in S ] } { \operatorname* { P r } [ M ( x ^ { \prime } ) \in S ] } \leq e ^ { \eta d _ { \chi } ( x , x ^ { \prime } ) } ,\tag{16}
$$

where $\eta \geq 0$ represents the privacy budget, controlling the trade-off between privacy and utility.

HiddenEcho offers a novel solution to mitigate LLM performance degradation caused by noisebased differential privacy mechanisms.

## C PRIVACY DEFINITION

Building on prior research (Coavoux et al., 2018b), which defines privacy as the adversary’s inability to infer information about the input from its latent representations, we adopt a similar perspective in our work.

Definition 4 (Empirical Privacy) Empirical Privacy $( E P )$ quantifies the adversary’s inability to reconstruct the original input or infer sensitive attributes from perturbed text. The degree of privacy protection increases as it becomes more challenging for an attacker to recover the original text or extract sensitive information.

$$
E P = 1 - \frac { \sum _ { x _ { i } \in X } \mathbb { I } ( f ( \Phi ( x _ { i } ) ) , x _ { i } ) } { | X | } ,\tag{17}
$$

where $\Phi ( x _ { i } )$ represents the embedding layer of the LLM, f denotes a general inversion process, and I indicates the correct predictions.

## D TIME AND SPACE COMPLEXITY

## D.1 HIDDENECHO

The computational cost of HiddenEcho is primarily driven by its denoising module. For the time complexity:

1. Transformer Layers: Each Transformer layer processes hidden states with a complexity of $O ( n ^ { 2 } d ^ { \prime } + n d ^ { \prime 2 } ) $ , where $d ^ { \prime } = d / r$ (reduced hidden size), n is the sequence length, and L is the number of layers. The total complexity for all layers is:

$$
O ( L ( n ^ { 2 } d / r + n d ^ { 2 } / r ^ { 2 } ) ) .
$$

2. Down/Upsampling: The linear transformations for downsampling and upsampling the embeddings have a complexity of $O ( L n d d ^ { \prime } )$ .

3. Computing gate vectors and performing mixing operations incurs a complexity of $O ( L n d ^ { \prime } )$

Combining these, the total time complexity is:

$$
O ( L ( n ^ { 2 } d / r + n d ^ { 2 } / r ^ { 2 } + n d ^ { 2 } / r ) ) .
$$

For the space complexity:

1. Parameter storage: The Transformer layers and linear transformations require $O ( L d ^ { \prime 2 } +$ $L d d ^ { \prime } )$ for storing parameters.

2. Intermediate Representations: The hidden states and gate vectors contribute $O ( L n d ^ { \prime } { + } L d ^ { \prime } )$ to memory usage.

Thus, the total space complexity is:

$$
O ( L ( d ^ { 2 } / r ^ { 2 } + n d / r + d ^ { 2 } / r ) ) .
$$

## D.2 HIDDENECHO+

To address the high communication overhead, HiddenEcho + compresses the hidden layer states using selective filtering and dimensionality reduction. For the time complexity:

1. Hidden Layer Filter: Estimating the gradient $\frac { \partial \hat { H } _ { L - 1 } } { \partial \hat { H } _ { i } }$ for each approximation step involves backpropagation through the layers following $H _ { i }$ . This incurs a complexity of $O ( m n ^ { 2 } d )$ per layer, where m denotes the number of approximation steps. Summing across L layers, the total cost is:

$$
O ( m L n ^ { 2 } d ) .
$$

2. Dimension Reducer: Downsampling and upsampling hidden states incur $O ( n d d ^ { \prime } )$ , where $d ^ { \prime } = d / r$ is the reduced dimension, and r is the reduction factor. MINE operations over $n _ { H }$ selected layers require $O ( k n n _ { H } d ^ { \prime } )$ ), where k is the optimization steps for MINE.

The total time complexity is:

$$
O ( m L n ^ { 2 } d + k n n _ { H } d / r + n d ^ { 2 } / r ) .
$$

For the space complexity:

1. Hidden Layer Filter: Requires $O ( L n d )$ for storing gradients and contributions.

2. Dimension Reducer: MINE statistics networks require $O ( n _ { H } d ^ { \prime 2 } )$ Downsampled/upsampled states add $O ( n n _ { H } d ^ { \prime } )$ .

The total space complexity is:

$$
O ( L n d + n _ { H } d ^ { 2 } / r ^ { 2 } + n n _ { H } d / r ) .
$$

## E COMMUNICATION ANALYSIS

In the HiddenEcho, all L hidden states of the server-side LLM are transmitted. Each hidden state has dimensions of $n \cdot d ,$ where n represents the sequence length and $d ^ { \prime } = d / r$ denotes the reduced hidden dimension achieved via dimensionality reduction by a factor r. The total communication volume can be expressed as:

$$
V _ { \mathrm { \scriptsize { H i d d e n E c h o } } } = L \cdot n \cdot d ^ { \prime } .
$$

In contrast, the HiddenEcho + configuration transmits only $n _ { H }$ selected hidden layers, resulting in a total communication volume of:

$$
V _ { \mathrm { { H i d d e n E c h o } } + } = n _ { H } \cdot n \cdot d ^ { \prime } .
$$

To quantify the reduction in transmission, the ratio of communication volumes between the two configurations is given by:

$$
\frac { V _ { \mathrm { H i d d e n E c h o ~ + } } } { V _ { \mathrm { H i d d e n E c h o } } } = \frac { n _ { H } \cdot n \cdot d ^ { \prime } } { L \cdot n \cdot d ^ { \prime } } = \frac { n _ { H } } { L } .
$$

The percentage of transmission volume saved is therefore:

$$
\mathrm { S a v i n g s } \ : ( \% ) = \left( 1 - { \frac { n _ { H } } { L } } \right) \cdot 1 0 0 .
$$

Example Case: When $n _ { H } \ll L ,$ , significant communication savings can be achieved. For instance, consider $n _ { H } = 4$ and $L = 2 8$ . The percentage savings in transmission volume is calculated as:

$$
{ \mathrm { S a v i n g s ~ } } ( \% ) = \left( 1 - { \frac { 4 } { 2 8 } } \right) \cdot 1 0 0 \approx 8 7 . 5 0 \% .
$$

## F PROOF OF NOISE MITIGATION IN HIDDENECHO

We provide proof demonstrating how the HiddenEcho framework mitigates interlayer noise amplification by analyzing noise propagation through transformer layers and the corrective effects of the denoising module.

## F.1 NOISE AMPLIFICATION IN TRANSFORMER LAYERS

Let the hidden state at the i-th layer be $H _ { i } ,$ and the corresponding noise be $\delta _ { i }$ . The hidden state at the (i + 1)-th layer can be expressed as:

$$
H _ { i + 1 } = T _ { i + 1 } ( H _ { i } + \delta _ { i } ) ,
$$

where $\tau _ { i + 1 }$ represents the transformer operation. Due to the nonlinear nature of $\tau _ { i + 1 }$ , noise $\delta _ { i }$ propagates and is amplified. The noise at the $( i + 1 )$ -th layer can be approximated as:

$$
\delta _ { i + 1 } = f ( \delta _ { i } ) ,
$$

where $f ( \cdot )$ denotes the transformation applied by the layer. The magnitude of $\delta _ { i + 1 }$ is bounded by the Jacobian norm of the transformation:

$$
\lVert \boldsymbol { \delta } _ { i + 1 } \rVert \leq \lVert \boldsymbol { J } _ { f } ( \boldsymbol { H } _ { i } ) \rVert \cdot \lVert \boldsymbol { \delta } _ { i } \rVert ,
$$

where $\| J _ { f } ( H _ { i } ) \|$ is the Jacobian norm. Defining the noise amplification factor as $\alpha _ { i } = \mathbb { E } [ \| J _ { f } ( H _ { i } ) \| ]$ we obtain:

$$
\| \delta _ { i + 1 } \| \leq \alpha _ { i } \| \delta _ { i } \| , \quad { \mathrm { w h e r e ~ } } \alpha _ { i } > 1 .
$$

Over L layers, the noise at the final layer is amplified as:

$$
\| \delta _ { L } \| \leq \prod _ { i = 1 } ^ { L } \alpha _ { i } \| \delta _ { 0 } \| ,
$$

where $\delta _ { 0 }$ denotes the initial noise introduced by the privacy-preserving mechanism.

## F.2 NOISE DECOMPOSITION AND DENOISING

The hidden state $H _ { i }$ can be decomposed into two components:

$$
H _ { i } = { \cal S } _ { i } + \delta _ { i } ,
$$

where:

$S _ { i } { : }$ Signal component containing task-relevant information.

$\delta _ { i } \colon$ Noise component introduced for privacy preservation.

The HiddenEcho module D utilizes the noise-free initial embedding E and the set of server-side hidden states $H = \left\{ H _ { 0 } , H _ { 1 } , \dots , H _ { L - 1 } \right\}$ to produce a denoised hidden state:

$$
H _ { i } ^ { \mathrm { d e n o i s e d } } = { \cal D } ( E , { \cal H } ) .
$$

The denoised hidden state can be expressed as:

$$
H _ { i } ^ { \mathrm { d e n o i s e d } } = S _ { i } + \delta _ { i } ^ { \mathrm { d e n o i s e d } } ,
$$

where $\delta _ { i } ^ { \mathrm { d e n o i s e d } }$ represents the residual noise after applying the denoising module.

## F.3 DYNAMIC MIXING AND RESIDUAL CONNECTIONS

The HiddenEcho module incorporates dynamic mixing and residual connections to enhance signal retention and suppress noise. The input to the i-th layer of the module is given by:

$$
Z _ { i } = \mu _ { i } A _ { i - 1 } + ( 1 - \mu _ { i } ) H _ { i } ^ { \mathrm { d n } } ,
$$

where:

$A _ { i - 1 } { \mathrm { : } }$ Output from the previous layer with reduced noise.

$H _ { i } ^ { \mathrm { d n } } = \mathcal { W } ^ { \mathrm { d n } } ( H _ { i } )$ : Compressed version of the hidden state, containing both signal and noise.

The gate parameter $\mu _ { i } \in ( 0 , 1 )$ dynamically adjusts the contributions of $A _ { i - 1 }$ and $H _ { i } ^ { \mathrm { d n } }$ . Expanding $Z _ { i }$ in terms of its components:

$$
Z _ { i } = \mu _ { i } ( S _ { A _ { i - 1 } } + \delta _ { A _ { i - 1 } } ) + ( 1 - \mu _ { i } ) ( S _ { H _ { i } } + \delta _ { H _ { i } } ) .
$$

The contributions of signal and noise can be written as

$$
S _ { Z _ { i } } = \mu _ { i } S _ { A _ { i - 1 } } + ( 1 - \mu _ { i } ) S _ { H _ { i } } , \quad \delta _ { Z _ { i } } = \mu _ { i } \delta _ { A _ { i - 1 } } + ( 1 - \mu _ { i } ) \delta _ { H _ { i } } .
$$

Using the triangle inequality, the noise magnitude satisfies:

$$
\lVert \delta _ { Z _ { i } } \rVert \leq \mu _ { i } \lVert \delta _ { A _ { i - 1 } } \rVert + ( 1 - \mu _ { i } ) \lVert \delta _ { H _ { i } } \rVert .
$$

This demonstrates the effectiveness of dynamic mixing and residual connections in amplifying the signal while suppressing sparse noise. Generally, it ensures that $\| \mathcal { D } ( \delta , E , H ) \| > 0 .$

## F.4 NOISE REDUCTION AT THE FINAL LAYER

The residual noise after denoising is given by:

$$
\lVert \delta ^ { \mathrm { d e n o i s e d } } \rVert = \lVert \delta \rVert \cdot \left( 1 - \frac { \lVert \mathcal { D } ( \delta , E , H ) \rVert } { \lVert \delta \rVert } \right) .
$$

We have $\| \mathcal { D } ( \delta , E , H ) \| > 0$ , ensuring:

$$
0 < 1 - \frac { \| \mathcal { D } ( \delta , E , H ) \| } { \| \delta \| } < 1 ,
$$

which implies:

$$
\lVert \delta ^ { \mathrm { d e n o i s e d } } \rVert < \lVert \delta \rVert .
$$

Let $\begin{array} { r } { 0 < \beta = \frac { \| \delta ^ { \mathrm { d e n o i s e d } } \| } { \| \delta \| } < 1 } \end{array}$ . The corrected noise at the i-th layer satisfies:

$$
\| \delta _ { i } ^ { \mathrm { d e n o i s e d } } \| \leq \beta _ { i } \| \delta _ { i } \| .
$$

At the (i + 1)-th layer, the noise satisfies:

$$
\lVert \delta _ { i + 1 } ^ { \mathrm { d e n o i s e d } } \rVert \leq \beta _ { i + 1 } \alpha _ { i } \lVert \delta _ { i } ^ { \mathrm { d e n o i s e d } } \rVert .
$$

By recursively applying this relationship across L layers, the noise at the final layer satisfies:

$$
\| \delta _ { L } ^ { \mathrm { d e n o i s e d } } \| \le \left( \prod _ { i = 1 } ^ { L } \beta _ { i } \alpha _ { i } \right) \| \delta _ { 0 } \| < \prod _ { i = 1 } ^ { L } \alpha _ { i } \| \delta _ { 0 } \| = \| \delta _ { L } \| .
$$

## G WORKFLOW OF HIDDENECHO

Algorithm 1 outlines the training process for HiddenEcho. In particular, the lines highlighted in blue distinguish the specific workflow of HiddenEcho +.

Algorithm 1 Workflow of a Training Step of HiddenEcho   
Require: Input tokens x, grouth truth y   
Ensure: LossClient Phase   
1: Embed tokens: $E  \mathcal { E } ( x ) ;$   
2: Inject sampled noise to $\begin{array} { r } { \boldsymbol { E } \colon \boldsymbol { E } ^ { \prime }  \boldsymbol { E } + \boldsymbol { \delta } ; } \end{array}$   
3: Send E′ to server;   
Server Phase   
4: Compute hidden states: $H \gets B ( E ^ { \prime } ) ;$   
5: Filter the hidden states according to the precomputed layer contributions to create a subset S;   
▷ HiddenEcho +   
6: Downsample the hidden states in S by Eq. equation 4;   
7: Return the downsampled S to client;   
Client Phase   
8: Compute d9: Denoising: $E ^ { \mathrm { d n } }$ by Eq. equation 5;   
$H _ { \mathrm { d e n o i s e d } }  { \mathcal { D } } ( E ^ { \mathrm { d n } } , { \bar { S } } ) ;$   
10: Compute task loss $\mathcal { L } _ { \mathrm { t a s k } }$ by Eq. equation 7 and Eq. equation 8;   
11: Optimize the MI estimators by Eq. equation 14; ▷ HiddenEcho +   
12: Compute information bottleneck loss LIB by Eq. equation 12;   
13: Compute total loss L by Eq. equation 13;   
14: return Loss L;

Table 4: Performance of different perturbation methods on text classification tasks based on Llama3- 1B.
<table><tr><td colspan="2">Dataset</td><td colspan="3">MRPC</td><td colspan="3">Financial</td><td colspan="3">BBC News</td></tr><tr><td colspan="2">Privacy Budget η</td><td>1000</td><td>4000</td><td>5000</td><td>1000</td><td>4000</td><td>5000</td><td>1000</td><td>4000</td><td>5000</td></tr><tr><td rowspan="2">GAN-DP</td><td>AUC</td><td>0.506</td><td>0.502</td><td>0.513</td><td>0.540</td><td>0.550</td><td>0.576</td><td>0.619</td><td>0.647</td><td>0.664</td></tr><tr><td>EP</td><td>0.999</td><td>0.998</td><td>0.998</td><td>0.999</td><td>0.999</td><td>0.997</td><td>0.999</td><td>0.989</td><td>0.986</td></tr><tr><td>LDP</td><td>AUC EP</td><td>0.489 0.951</td><td>0.529 0.889</td><td>0.494 0.809</td><td>0.561 0.952</td><td>0.567 0.897</td><td>0.559 0.848</td><td>0.619 0.903</td><td>0.627 0.803</td><td>0.641 0.700</td></tr><tr><td rowspan="2">SnD</td><td>AUC</td><td>0.509</td><td>0.504</td><td>0.507</td><td>0.558</td><td>0.553</td><td>0.572</td><td>0.632</td><td>0.633</td><td>0.633</td></tr><tr><td></td><td></td><td></td><td>0.663</td><td>0.894</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HiddenEcho HiddenEcho +</td><td>AUC AUC</td><td>0.654 0.645</td><td>0.659 0.653</td><td>0.655</td><td>0.828</td><td>0.906 0.824</td><td>0.905 0.829</td><td>0.978 00.971</td><td>0.978 0.972</td><td>0.978 0.974</td></tr><tr><td colspan="2">AUC Improve %</td><td>28.48</td><td>24.57</td><td>29.24</td><td>59.36</td><td>59.79</td><td>57.12</td><td>54.75</td><td>51.16</td><td>47.29</td></tr></table>

Table 5: Statistics of datasets.
<table><tr><td>Dataset</td><td>Task</td><td>#Train</td><td>#Dev</td><td>#Test</td></tr><tr><td rowspan="3">FP MRPC BBC News Tweet</td><td>sentiment analysis</td><td>1,811</td><td>226</td><td>227</td></tr><tr><td>semantic equivalence judgment</td><td>3,301</td><td>1,725</td><td>1,725</td></tr><tr><td>news topic classification offensive speech detection</td><td>1225 1500</td><td>500 500</td><td>500 500</td></tr><tr><td>IWSLT</td><td>machine translation</td><td>1,044</td><td>130</td><td>131</td></tr><tr><td>CNNDM</td><td>summarization</td><td>1,322</td><td>50</td><td>47</td></tr><tr><td>Samsum</td><td>summarization</td><td>2,916</td><td>171</td><td>150</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## H EXPERIMENTAL SUPPLEMENTS

## H.0.1 BASELINES

We evaluate HiddenEcho against several strong baselines within the segmented framework, encompassing standard DP algorithms, DP-based denoising methods, and DNN-based perturbation approaches. The baselines include:

• Local Differential Privacy (LDP): Embeddings fed into the LLM’s word embedding layer are perturbed with $d _ { \chi }$ -noise (Qu et al., 2021), then transmitted to the server.

• GAN-DP: A GAN-based noise addition method designed to perturb embeddings by introducing $d _ { \chi }$ -based noise of varying magnitudes to generate perturbed vectors.

• SnD (Mai et al., 2024): A DP-based denoising approach where the denoising module is pre-trained on the server and then downloaded to the client for noise correction.

• HiddenEcho: Our end-to-end client-side denoising method transmits the full LLM hidden states for processing.

• HiddenEcho +: An improved version of HiddenEcho, featuring gradient-based hidden layer filtering and dimensionality reduction via information bottleneck theory to lower communication overhead while preserving performance.

## H.1 DATASET DETAILS AND BASE PERFORMANCE

For the text classification task, we utilize:

• Financial Phrasebank (Malo et al., 2014): A sentiment classification dataset with 4,840 financial news sentences, categorized by annotator agreement rates.

Table 6: Performances of centralized fine-tuning on six datasets for each LLMs.
<table><tr><td colspan="5">Text Classification</td></tr><tr><td>Base Model</td><td>Metric</td><td>MRPC</td><td>Financial</td><td>BBC News</td></tr><tr><td>Qwen2-1.5B</td><td>AUC</td><td>0.920</td><td>0.976</td><td>0.998</td></tr><tr><td>Llama3-1B</td><td>AUC</td><td>0.928</td><td>0.980</td><td>0.999</td></tr><tr><td colspan="5">Text Generation</td></tr><tr><td>Base Model</td><td>Metric</td><td>IWSLT</td><td>CNNDM</td><td>Samsum</td></tr><tr><td>T5-large</td><td>BLEU</td><td>34.047</td><td>17.738</td><td>24.371</td></tr></table>

Table 7: Performance of different perturbation methods on text generation tasks based on T5-Large.
<table><tr><td colspan="2">Dataset</td><td colspan="3">IWSLT</td><td colspan="3">CNNDM</td><td colspan="3">Samsum</td></tr><tr><td colspan="2">Privacy Budget η</td><td>20</td><td>30</td><td>40</td><td>20</td><td>30</td><td>40</td><td>20</td><td>30</td><td>40</td></tr><tr><td rowspan="2">GAN-DP</td><td>BLEU EP</td><td>| 0.109</td><td></td><td>10.30929.816</td><td>| 5.461</td><td>13.572</td><td>12.697</td><td>| 4.120</td><td>4.964</td><td>5.509</td></tr><tr><td></td><td>0.883</td><td>0.821</td><td>0.799</td><td>0.460</td><td>0.372</td><td>0.348</td><td>0.503</td><td>0.461</td><td>0.449</td></tr><tr><td rowspan="2">LDP</td><td>|BLEU|</td><td>| 0.035</td><td>15.553</td><td>24.576|</td><td>5 | 0.764</td><td>7.974</td><td>12.107</td><td>| 2.403</td><td>14.602</td><td>20.235</td></tr><tr><td>E</td><td>0.994</td><td>0.970</td><td>0.914</td><td>00.987</td><td>0.916</td><td>0.764</td><td>0.989</td><td>0.931</td><td>0.806</td></tr><tr><td>HiddenEcho</td><td>BLEU</td><td>| 1.092</td><td>20.080</td><td>26.366</td><td>| 2.915</td><td>11.617</td><td>12.323</td><td>| 4.618</td><td>20.636</td><td>21.851</td></tr><tr><td>HiddenEcho +</td><td>BLEU</td><td>0.824</td><td>22.403</td><td>25.654</td><td>| 0.971</td><td>10.925</td><td>12.718</td><td>4.323</td><td>18.192</td><td>20.867</td></tr></table>

• Microsoft Research Paraphrase Corpus (Wang et al., 2018): A sentence pairs dataset collected from news articles, each labeled by human annotators to indicate whether the pairs are paraphrases.

• BBC News (Greene & Cunningham, 2006): Consists of articles published on the BBC News between 2004 and 2005, with each article categorized into one of five topics: business, entertainment, politics, sports, or technology.

• Tweet Annotation (Kern et al., 2023): A dataset comprises annotated tweet data for hate speech and offensive language under five experimental conditions, which are utilized for attribute inference attacks.

For the text generation task, we utilize:

• IWSLT2014 (IWSLT) (201, 2014): A dataset for English-to-French machine translation, focusing on spoken language.

• CNN DailyMail Short (CNNDM) (Nallapati et al., 2016): A concise version of CNN DailyMail news summaries, paired with fill-in-the-blank questions.

• Samsum Short (Samsum): A shortened version from (Gliwa et al., 2019), comprising messenger-style dialogues with corresponding summaries.

More dataset statistics are reported in Table 5. For reference, the ground truth performance of each large model across various datasets is provided in Table 6.

## H.2 EIA AGAINST FOR TEXT CLASSIFICATION BASED ON LLAMA3-1B

Furthermore, we extend to evaluate the performance of baselines against EIA in text classification tasks using Llama3-1B. Given the significant differences in embedding layer parameter scales across different LLMs, privacy budgets of 1000, 4000, and 5000 are selected for this experiment. All other experimental settings are consistent with those outlined in 5.1. The detailed results are presented in Table 4.

In contrast to Qwen2-1.5B, HiddenEcho exhibits clear superiority when applied to Llama3, achieving significantly higher improvements over baselines, with a maximum performance gain of 59.79%. Although HiddenEcho + typically performs slightly below HiddenEcho, it remains a more advantageous choice in bandwidth-constrained scenarios.

## H.3 EIA AGAINST FOR TEXT GENERATION BASED ON T5-LARGE

We evaluate machine translation on the IWSLT dataset and text summarization on the CNN DailyMail Short and Samsum Short datasets, using T5-Large as the base model. The BLEU scores of HiddenEcho and other baseline methods are assessed against EIA at varying η. Note that SnD’s noise reduction model, which processes classification vectors, is unsuitable for text generation tasks.

As shown in Table 7, HiddenEcho consistently demonstrates near-optimal performance. On the IWSLT dataset, HiddenEcho achieves the highest BLEU scores at η = 20 (1.092) and η = 40 (26.366), while HiddenEcho + outperforms at η = 30 (22.403). A similar trend is observed on the CNNDM dataset, although HiddenEcho performs suboptimally at lower privacy budgets.

The Samsum dataset further confirms HiddenEcho ’s effectiveness, with HiddenEcho consistently delivering the highest BLEU scores across all privacy budgets (4.618 at η = 20, 20.636 at η = 30, and 21.851 at η = 40). HiddenEcho significantly outperforms GAN-DP and LDP, particularly at lower privacy budgets.

HiddenEcho strikes a better balance between privacy and utility in text generation, maintaining competitive EP values while achieving significantly higher generation quality, particularly in summarization tasks.

## H.4 AIA MODEL ARCHITECTURE

The architecture of the attacker model for attribute inference attacks is detailed in Table 8. The model’s output size is set to 4 for education inference and 1 for age prediction.

Table 8: Attacker Model Architecture
<table><tr><td>Layer</td><td>Shape</td></tr><tr><td>Input</td><td>Batch size × 1536</td></tr><tr><td>FC</td><td>1536 × 768</td></tr><tr><td>ReLU</td><td>-</td></tr><tr><td>FC</td><td>768 × Output size</td></tr></table>

<!-- image-->  
(a) LDP

<!-- image-->  
(b) SnD

<!-- image-->  
(c) GAN-DP

<!-- image-->  
(d) HiddenEcho  
Figure 5: Comparison of visualization of t-SNE between baselines and HiddenEcho on the Financial Phrasebank with Qwen2-1.5B.

## H.5 VISUALIZATION

Additionally, we extract the output of the final layer of the server-side LLM after training convergence and employ t-SNE (Van der Maaten & Hinton, 2008) to project the embeddings into a 2D space, maintaining consistent settings across all methods. This visualization enables a comparative analysis of the effects of different perturbation techniques on the feature space. Each perturbation algorithm is evaluated under the same privacy budget ϵ.

We conduct experiments using four perturbation baselines on the Financial Phrasebank dataset with the Qwen2-1.5B model and ϵ of 5000. The results are visualized in Fig 5.

The visualization of HiddenEcho reveals a triangular spatial distribution of clusters, with points from the same category forming compact groups. This clustering pattern is especially evident in the orange and green categories, highlighting effective feature separation. In contrast, other methods fail to form distinct clusters, with nodes exhibiting dispersed and overlapping distributions. The lack of clear intra-class cohesion and inter-class separation in the embedding space leads to their suboptimal performance.