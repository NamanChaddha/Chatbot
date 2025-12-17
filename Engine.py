from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread

MODEL_PATH = r"Models\Dominator_Dolphin"
SYSTEM_PROMPT = """You are a user in r/AskReddit.Reply to the post contextually."""

print("⏳ Loading Dominator Dolphin...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="cuda:0"
)
FastLanguageModel.for_inference(model)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|im_end|>")
]

print("\n🗑️ DOMINATOR DOLPHIN ONLINE. (Type 'exit' to quit)")
print("--------------------------------------------------")

while True:
    try:
        user_input = input("\nYou: ")
    except EOFError:
        break

    if user_input.lower() in ["exit", "quit"]:
        print("TrashBot: Bye.")
        break

    #Set Temperature
    is_technical = any(
        word in user_input.lower() for word in ['code', 'python', 'script', 'solve', 'calculate', 'math', 'function'])
    current_temp = 0.1 if is_technical else 0.8

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=current_temp,
        min_p=0.1,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Bot:", end=" ")

    for new_text in streamer:
        if "<|im_end|>" not in new_text:
            print(new_text, end="", flush=True)

    print()