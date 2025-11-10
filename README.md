# Modern AI with Unsloth — CMPE255 Colabs

A compact set of Colab notebooks showcasing multiple adaptation strategies for large language models (LLMs) and vision-language models with **Unsloth**, **Hugging Face Transformers**, **Datasets**, and **TRL (Reinforcement Learning for Transformers)**. Each notebook is self-contained and designed to run on Google Colab GPUs.

> 

---

## 📚 Project Index (Colab ↔️ Video)

### 1) CMPE255_Mistral_Finetuning_v1 — **Full-Parameter Fine-Tuning (SFT)**
- Colab: [Open in Colab](ADD_COLAB_LINK_HERE)
- Video: [YouTube](https://youtu.be/YkelSl8ZRFw)

### 2) CMPE255_Mistral_Lora_Finetunning_colab_2_Yashaswini_Dinesh — **PEFT with LoRA (Low-Rank Adaptation) + SFT**
- Colab: [Open in Colab](ADD_COLAB_LINK_HERE)
- Video: [YouTube](https://youtu.be/v2Hld-1gN2Y)

### 3) CMPE255_Mistral_Lora_Finetuning_Reinforcement_Sentimental_colab_3 — **LoRA + RL (PPO) for Positive Sentiment**
- Colab: [Open in Colab](ADD_COLAB_LINK_HERE)
- Video: [YouTube](https://youtu.be/aDRBJRug2M0)

### 4) CMPE255_Colab_4_Gemma3_Vision_GRPO — **Gemma 3 Vision + GRPO (Generalized Reinforcement Policy Optimization)**
- Colab: [Open in Colab](ADD_COLAB_LINK_HERE)
- Video: [YouTube](https://youtu.be/dRzSiNLbgLU)

### 5) CMPE255_unsloth_cpt_colab5 — **CPT/DAP (Continued/Domain-Adaptive Pretraining)**
- Colab: [Open in Colab](ADD_COLAB_LINK_HERE)
- Video: [YouTube](https://youtu.be/j5LzfcMUAl8)

---

## 🧪 What Each Notebook Demonstrates

| Notebook | Model | Method | Objective | Alignment | Best For |
|---|---|---|---|---|---|
| v1 Full FT | Mistral‑7B | **Full-parameter SFT** | Instruction following | Optional DPO/PPO later | Maximum capacity changes |
| LoRA SFT | Mistral‑7B | **LoRA (PEFT) + SFT** | Instruction following | Optional DPO/PPO later | Low‑VRAM, fast iteration |
| LoRA + RL | Mistral‑7B | **LoRA + PPO** | Task + **positive sentiment** reward | PPO with KL control | Controllable tone/style |
| Gemma3 Vision + GRPO | Gemma 3 Vision | **GRPO** | Vision‑language tasks | GRPO | Multimodal alignment |
| CPT/DAP | Mistral‑7B | **Continued pretraining** | CLM on raw domain text | Optional SFT + DPO/PPO | Domain knowledge/style |

> **SFT = Supervised Fine-Tuning** · **PEFT = Parameter‑Efficient Fine‑Tuning** · **LoRA = Low‑Rank Adaptation** · **PPO = Proximal Policy Optimization** · **DPO = Direct Preference Optimization** · **GRPO = Generalized Reinforcement Policy Optimization** · **CPT/DAP = Continued/Domain‑Adaptive Pretraining** · **CLM = Causal Language Modeling**

---

## ⚙️ Environment (Colab-friendly)

- Recommended GPU: T4 / L4 / A100 (BF16 preferred if available).
- Python packages (typical): `unsloth`, `transformers`, `datasets`, `trl`, `accelerate`, `bitsandbytes`, `peft`.

```bash
pip install -U unsloth transformers datasets trl accelerate bitsandbytes peft
```

> On first run, Colab may prompt you to restart the runtime after installing packages.

---

## 🗂️ Data & Formatting

- **Instruction datasets** as `(prompt, response)` pairs.
- Convert to **two-turn chat** (user → assistant) using `tokenizer.apply_chat_template(...)`.
- Set `eos_token_id` and `pad_token_id` explicitly for Mistral/Gemma families.

**Tip:** Use the **same chat template** for training and inference (`add_generation_prompt=True`) to improve stability.

---

## 🏃 Quick Recipes

### Full-Parameter SFT (v1)
- `FastLanguageModel.from_pretrained(..., full_finetune=True)`  
- **BF16**, gradient accumulation, gradient checkpointing (if needed)  
- `SFTTrainer` with linear LR schedule, warmup, AdamW (8‑bit optional)

### LoRA SFT (colab_2)
- `get_peft_model(...)` to inject LoRA into `q_proj/k_proj/v_proj/o_proj`  
- Tune **rank (r)**, **alpha**, **dropout** for VRAM/quality trade‑off  
- `SFTTrainer` with short `max_steps` for quick demos

### LoRA + RL (colab_3)
- Start from the SFT LoRA checkpoint  
- Define a **sentiment reward** (classifier or heuristic)  
- Use **TRL PPOTrainer** with a **KL penalty** to avoid drift

### CPT / DAP (colab_5)
- CLM on **raw domain text** with **packing** for long, efficient sequences  
- AdamW, warmup, cosine/linear scheduler; optional **LoRA‑CPT** for low VRAM

### Gemma 3 Vision + GRPO (colab_4)
- Vision‑language dataset (image + text)  
- Optimize with **GRPO** for stable alignment on multimodal prompts

---

## 🔎 Inference

1. Reuse the **same chat template** and set `add_generation_prompt=True`.  
2. Tokenize with **attention mask**.  
3. `model.generate(max_new_tokens=..., temperature=..., top_p=...)`.  
4. Decode and post‑process (strip special tokens).

---

## ✅ Evaluation (suggested)

- Hold out a small **validation set** that mirrors your target tasks.  
- Track **loss/perplexity** (CPT) and task metrics (SFT/RL).  
- Do **A/B** comparisons: base vs. adapted model for the same prompts.

---

## 🧯 Troubleshooting

- **CUDA OOM:** lower batch size, enable 8‑bit optimizer, reduce context length, or use LoRA.  
- **No BF16:** fall back to FP16; consider gradient checkpointing.  
- **Weird formatting:** ensure consistent chat templates train ↔️ infer.  
- **Unstable PPO:** adjust reward scaling and **KL coefficient**.

---

## 📎 Add Your Colab Links Here

Paste your Colab URLs above where it says `ADD_COLAB_LINK_HERE`. A common pattern is:
```
https://colab.research.google.com/github/<user-or-org>/<repo>/blob/main/<notebook>.ipynb
```

---

## 📝 License & Attribution

- Models and weights follow their original licenses (Mistral, Gemma, etc.).  
- This repo uses **Unsloth** and **Hugging Face** libraries; please cite/acknowledge accordingly.
