from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

INPUT_FILE = r"Datasets\Train\final_training.jsonl"
OUTPUT_DIR = r"Models\Dolphin_Adapter"

MODEL_NAME = "cognitivecomputations/dolphin-2.9.2-qwen2-7b"

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []

    for instruction, input, output in zip(instructions, inputs, outputs):

        text = f"<|im_start|>system\n{instruction}<|im_end|>\n" \
               f"<|im_start|>user\n{input}<|im_end|>\n" \
               f"<|im_start|>assistant\n{output}<|im_end|>"

        texts.append(text)
    return {"text": texts, }

if __name__ == "__main__":
    print(f"⏳ Loading {MODEL_NAME} in 4-bit...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        dtype=None,  # Auto-detect (Float16)
        load_in_4bit=True,
        device_map='cuda:0'
    )

    # LoRA (Low Rank Adaptation)
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
    )

    print("📚 Loading Dataset...")
    dataset = load_dataset("json", data_files=INPUT_FILE, split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print("🚀 Starting Training...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,

        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            warmup_steps=100,

            # --- RUN SETTINGS ---
            max_steps=8000,

            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            save_strategy="steps",
            save_steps=500,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train(resume_from_checkpoint=True)

    print(f"🎉 Training Complete! Saving to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)