# 复现 ICLR 2026

HIDDENECHO:HIDDEN-STATECORRECTIONTOMIT
IGATEINTER-LAYERNOISEAMPLIFICATIONINLLMS
UNDERDIFFERENTIALPRIVACY

> The rise of large language models (LLMs) has driven the adoption of Model-as-a-Service (MaaS). However, transmitting raw text to servers raises critical privacy concerns. Existing approaches employ deep neural networks (DNNs) or differential privacy (DP) to perturb inputs. Yet, these approaches suffer notable limitations: DNN-based methods often require task-specific pre-training, and conventional DP techniques, though privacy-preserving, suffer from noise amplification as perturbed inputs propagate through the deep transformer layer, leading to significant degradation in downstream task performance. To alleviate this, we propose HIDDENECHO, an end-to-end framework with client noise correction, where hidden states are sent from the server to the client and refined by a lightweight module using both embeddings and intermediate representations. HIDDENECHO suppresses inter-layer noise amplification without pretraining, effectively preserving task-relevant signals under DP constraints. To further reduce communication, HIDDENECHO incorporates gradient-based hidden layer selection and information bottleneck compression, reducing communication cost while preserving essential task information. Experiments across text classification and generation tasks demonstrate that HIDDENECHO achieves up to 46.89% performance improvement over DP baselines, over 85% communication reduction, and up to 72.52% faster training compared to existing denoising approaches, establishing a new privacy-utility trade-off for privatized LLMs.

## Quick Start

1. Install dependencies:

   `environment.yml` contains all the dependencies. You can create a new conda environment and install them.
2. Run the script:

   ```bash
   bash scripts/simple.sh
   ```

   The script trains a model on the financial phrasebank dataset with a privacy budget of 5000. You can modify the parameters in the script as needed.

## Baselines

### LDP

```bash
python train_split.py \
    --experiment_name "ldp" \
    --model_path "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --num_train_epochs 20 \
    --lr_scheduler_type "constant" \
    --learning_rate 4e-4 \
    --max_len 128 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --lora_rank 16 \
    --privacy_budget 5000 \
    --lst_enable false
```

### SnD

```bash
python -m baselines.snd.data
python -m baselines.snd.train_denoise
python -m baselines.snd.train_task \
    --model_name "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --privacy_budget 5000 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16
```

### GAN-DP

```bash
python -m baselines.gan.train_gan \
    --llm_path "/path/to/Qwen2-1.5B-Instruct" \
    --dataset_name "financial_phrasebank" \
    --privacy_budget 5000 \
    --train_epochs 20

python -m baselines.gan.train_task \
    --model_name "/path/to/Qwen2-1.5B-Instruct" \
    --dataset "financial_phrasebank" \
    --generator_epoch 20 \
    --privacy_budget 5000 \
    --num_train_epochs 15 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16
```
