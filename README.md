# preghealthcarebot

Here's a README file for your project:

---

# Fine-Tuning Llama 2 with QLoRA

This project demonstrates how to fine-tune the Llama 2 model using parameter-efficient fine-tuning (PEFT) techniques like QLoRA to optimize the usage of limited resources.

## Step-by-Step Instructions

### Step 1: Install Required Packages

Install the necessary Python packages by running the following command:

```bash
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
```

### Step 2: Import Required Libraries

Import the necessary libraries in your Python environment:

```python
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
```

### Step 3: Prepare Dataset

The original dataset for training can be found [here](https://huggingface.co/datasets/timdettmers/openassistant-guanaco). This dataset needs to be reformatted to follow the Llama 2 template. You can use the following datasets:

- Reformat Dataset (1k samples): [guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
- Complete Reformat Dataset: [guanaco-llama2](https://huggingface.co/datasets/mlabonne/guanaco-llama2)

To understand how the dataset was created, you can refer to this [notebook](https://colab.research.google.com/drive/1Ad7a9zMmkxuXTOh1Z7-rNSICA4dybpM2?usp=sharing).

### Step 4: Fine-Tune Llama 2

Fine-tune the Llama 2 model using QLoRA. Due to resource limitations, the fine-tuning is done in 4-bit precision to reduce VRAM usage.

1. Load the `llama-2-7b-chat-hf` model.
2. Train it on the `mlabonne/guanaco-llama2-1k` dataset (1,000 samples) for one epoch.
3. Use QLoRA with the following configuration:
   - **Rank:** 64
   - **Scaling Parameter:** 16
   - **Precision:** 4-bit (NF4 type)

### Step 5: Start the Fine-Tuning Process

Load the dataset and configure the necessary settings for training.

```python
os.environ["HF_TOKEN"] = "your_hf_token"
dataset = load_dataset("Jhpiego/testhealth", split="train")

# Configure bitsandbytes for 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    quantization_config=bnb_config,
    device_map={"": 0}
)
```

### Step 6: Train the Model

Use the `SFTTrainer` from `trl` to train the model.

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=LoraConfig(lora_alpha=16, lora_dropout=0.1, r=64, task_type="CAUSAL_LM"),
    dataset_text_field="text",
    max_seq_length=None,
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )
)
trainer.train()
```

### Step 7: Monitor Training with TensorBoard

Use TensorBoard to monitor the training process:

```bash
%load_ext tensorboard
%tensorboard --logdir results/runs
```

### Step 8: Test the Model

Use the text generation pipeline to test the fine-tuned model:

```python
prompt = "What is a large language model?"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(f"<s>[INST] {prompt} [/INST]")
print(result[0]['generated_text'])
```

### Step 9: Save the Fine-Tuned Model

Merge the LoRA weights with the base model and push it to the Hugging Face Hub:

```python
model.push_to_hub("your-username/Llama-2-7b-chat-finetune", check_pr=True)
tokenizer.push_to_hub("your-username/Llama-2-7b-chat-finetune", check_pr=True)
```

## Notes

- This example uses free Google Colab resources, which have limitations. Consider using GPUs with higher memory capacity for full fine-tuning.
- You can use your trained model like any other Llama 2 model from the Hugging Face Hub.
