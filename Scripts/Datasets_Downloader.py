from datasets import load_dataset
import json
import os

output_directory = r"..\Datasets"

def save_to_json(data, filename):
    path = os.path.join(output_directory,filename)
    print(f"Saving {len(data):,} rows to {filename}...")
    with open(path, 'a',encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

print("\n📦 Processing Math Instruct (PoT + CoT)...")
try:
    ds = load_dataset("TIGER-Lab/MathInstruct", split="train")
    count = 0
    excess=0
    PoT=[]
    CoT=[]
    target_count = 250000
    CoT_count = 0
    PoT_count = 0
    instruction=""
    for row in ds:
        if count >= target_count: break
        inp = row['instruction']
        out = row['output']
        source = row.get('source', '').lower()
        isCot = False
        if source.strip() == 'data/cot/aqua_rat.json':
            #Multiple choice question and answers, no code only explanation, no LaTeX
            instruction = "You are a Math tutor, solve the following multiple choice question giving the correct option and explanation without Latex formatting"
            CoT_count+=1
            isCot = True
        elif source.strip() == 'data/cot/math50k_camel.json':
            #Question-Answer type data, no code only explanation, LaTeX
            instruction = "You are a Math tutor, solve the following question giving a detailed explanation with Latex formatting"
            CoT_count+=1
            isCot = True
        elif source.strip() == 'data/cot/gsm_rft.json':
            #Question-Answer type data, no code only explanation, no LaTeX
            instruction = "You are a Math tutor, solve the following question giving an explanation without Latex formatting"
            CoT_count+=1
            isCot = True
        elif source.strip() == 'data/pot/mathqa.json':
            #Question-Code type data, only code without explanation, no LaTeX
            instruction = "You are a math tutor, solve the following question in python without comments giving the final answer"
            out = f"<TOOL_CALL>\n{out}\n</TOOL_CALL>"
            PoT_count+=1
        elif source.strip() == 'data/pot/gsm_gpt4.json':
            #Question-Code type data, code with comments without explanation, no LaTeX
            instruction = "You are a math tutor, solve the following question in python with comments giving the final answer"
            out = f"<TOOL_CALL>\n{out}\n</TOOL_CALL>"
            PoT_count+=1
        elif source.strip() == 'data/pot/numglue.json':
            #Question-Code type data, code with minimum comments without explanation, no LaTeX
            instruction = "You are a math tutor, solve the following question in python with minimum comments giving the final answer"
            out = f"<TOOL_CALL>\n{out}\n</TOOL_CALL>"
            PoT_count+=1
        elif source.strip() == 'data/cot/math_train.json':
            #Question-LaTeX type data, explanation with Asymptote diagrams, LaTeX
            instruction = "You are a Math tutor, solve the following question giving a detailed explanation and asymptote diagram and Latex formatting"
            CoT_count+=1
            isCot = True
        elif source.strip() == 'data/pot/math_train.json':
            #Question-Code type data, code with comments without explanation, no LaTeX
            instruction = "You are a math tutor, solve the following question in python with comments giving the final answer"
            out = f"<TOOL_CALL>\n{out}\n</TOOL_CALL>"
            PoT_count+=1
        elif source.strip() == 'data/pot/aqua_rat_filtered.json':
            #Question-Code type data, code with comments without explanation, no LaTeX
            instruction = "You are a math tutor, solve the following question in python with comments giving the final answer"
            out = f"<TOOL_CALL>\n{out}\n</TOOL_CALL>"
            PoT_count+=1
        elif source.strip() == 'data/cot/gsm_train.json':
            #Question-Answer type data, no code only explanaton, no LaTeX
            instruction = "You are a Math tutor, solve the following question giving a detailed explanation without Latex formatting"
            CoT_count+=1
            isCot = True
        else:
            excess+=1
            continue
        count+=1
        entry = {
                "instruction": instruction,
                "input": inp,
                "output": out
        }
        if isCot:
            CoT.append(entry)
        else:
            PoT.append(entry)
        if len(CoT)%10000 == 0:
            save_to_json(CoT, "CoT.jsonl")
            CoT.clear()
        if len(PoT)%10000 == 0:
            save_to_json(PoT, "PoT.jsonl")
            PoT.clear()
        if count%50000 == 0:
            print(f"Processed {count} rows of Math Instruct!")
    if CoT:
        save_to_json(CoT, "CoT.jsonl")
        CoT.clear()
    if PoT:
        save_to_json(PoT, "PoT.jsonl")
        PoT.clear()

    print(f"Saved {count}, Discarded {excess} rows.")
    print(f"Completed processing Math Instruct")
except Exception as e:
    print(f"Error saving Evol Instruct dataset.: {e}")


print(f"\n📦 Processing Evol-Instruct...")
try:
    ds = load_dataset('nickrosh/Evol-Instruct-Code-80k-v1',split='train')
    formatted_data =[]
    count = 0
    limit = 180000
    instruction = "You are a Senior Python Developer. Provide a comprehensive solution with code and explanation."
    for row in ds:
        if count >= limit:
            break
        formatted_data.append({
            "instruction": instruction,
            "input": row['instruction'],
            "output": row['output']
        })
        count+=1
    save_to_json(formatted_data, "Python_Code.jsonl")
    print(f"Completed processing Evol Instruct")
except Exception as e:
    print(f"Error saving Evol Instruct dataset.: {e}")


print("\n📦 Processing Python Syntax...")
try:
    ds = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split='train')
    formatted_data = []

    for row in ds:
        dataset_inst = row.get('instruction', '')
        dataset_inp = row.get('input', '')

        full_input = f"{dataset_inst}\n{dataset_inp}".strip()

        formatted_data.append({
            "instruction": "You are a Senior Python Developer. Write a Python program to solve this.",
            "input": full_input,
            "output": row['output']
        })

    save_to_json(formatted_data, "Python_Syntax.jsonl")
    print("✅ Saved Python_Syntax.jsonl!")
except Exception as e:
    print(f"❌ Error saving Python Syntax: {e}")


print("\n📦 Processing Math WordProblems...")
try:
    ds = load_dataset("microsoft/orca-math-word-problems-200k", split="train")
    formatted_data = []
    target_count = 100000
    count = 0

    for row in ds:
        if count >= target_count: break
        formatted_data.append({
            "instruction": "You are a Math Tutor. Solve the given question explaining every step in detail with Latex formatting.",
            "input": row['question'],
            "output": row['answer']
        })
        count += 1
    save_to_json(formatted_data, "Math_WordProblems.jsonl")
    print("✅ Saved Math_WordProblems.jsonl!")
except Exception as e:
    print(f"❌ Error saving Maths Words Problems: {e}")


print("Done processing all files!")