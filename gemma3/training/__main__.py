from datasets import load_dataset

from huggingface_hub import whoami

from unsloth import FastModel

from unsloth.chat_templates import (
    get_chat_template, 
    standardize_data_formats,
    train_on_responses_only
)

from trl import SFTTrainer, SFTConfig

import torch

# Variables to tweak
dataset_name = "mlabonne/FineTome-100k"
model_name = "unsloth/gemma-3-4b-it"
optimizer = "adamw_8bit"
seed = 3407
output_lora_model_name = "gemma-3"
output_merged_model_name = "gemma-3-finetune"
output_merged_model_gguf_name = "gemma-3-finetune-gguf"

# you may need to pass in your token here
try:
    hf_account_name = whoami()['name']
except:
    print("Unable to log into Huggingface, skipping push to HF Hub")
    hf_account_name = None
hf_token = None # if you're not logged in from the command line you'll need to provide this

# wandb Report To
report_to = "none"

# Use max_steps or num_training_steps
max_steps = 200
# num_training_epochs = 1

# Available models for training
fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

print(f"Using Huggingface User: {hf_account_name}")

# Retrieve pretrained model and associated tokenizer
model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

# Configure PEFT model for training on top of the pretrained model
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(tokenizer, chat_template = "gemma-3")

print(f"Loading and standardizing dataset: {dataset_name}")
dataset = load_dataset(dataset_name, split = "train")
dataset = standardize_data_formats(dataset)

print(f"Displaying entry 100 from dataset post standardization:\n{dataset[100]}")

print("Applying chat template to dataset")
def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return {"text": texts}

dataset = dataset.map(apply_chat_template, batched=True)

print(f"Displaying element 100 after applying chat template: {dataset[100]['text']}")

print(f"Wiring SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        # num_train_epochs = num_training_epochs, # Set this for 1 full training run.
        max_steps = max_steps,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = optimizer,
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = seed,
        report_to = report_to, # Use this for WandB etc
    ),
)

# Further trainer configuration for instruction/response parts
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

# Display current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

print("Starting training...")

trainer_stats = trainer.train()

print("Displaying post training memory stats.")

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

print("Saving fine-tuned LORA adapters")

model.save_pretrained(output_lora_model_name)  # Local saving
tokenizer.save_pretrained(output_lora_model_name)
    
model.save_pretrained_merged(output_merged_model_name, tokenizer)

if hf_account_name:
    model.push_to_hub_merged(
        f"{hf_account_name}/{output_merged_model_name}", 
        tokenizer,
        token=hf_token,
    )

model.save_pretrained_gguf(
    output_merged_model_gguf_name,
    quantization_type = "Q8_0", # For now only Q8_0, BF16, F16 supported
)

if hf_account_name:
    model.push_to_hub_gguf(
        output_merged_model_name,
        quantization_type = "Q8_0", # Only Q8_0, BF16, F16 supported
        repo_id = f"{hf_account_name}/{output_merged_model_gguf_name}",
        token=hf_token,
    )
