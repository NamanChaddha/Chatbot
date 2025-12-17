from unsloth import FastLanguageModel
import torch
import os

CHECKPOINT_PATH = r"..\Models\Dolphin_Adapter\checkpoint-7000"
OUTPUT_DIR = r"..\Models\Dominator_Dolphin"

print(f"⏳ Loading Checkpoint from {CHECKPOINT_PATH}...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = CHECKPOINT_PATH,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
    device_map='cuda:0'
)

print(f"💾 Saving Final Model to {OUTPUT_DIR}...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save as GGUF (for Ollama / Jan.ai)
# print("📦 Converting to GGUF (This might take a while)...")
# model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")

print(f"✅ Done! Saved Chatbot to {CHECKPOINT_PATH}.")