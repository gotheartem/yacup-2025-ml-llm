# Imports
import json
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm


# Fix seeds
import os
import torch
import random
import numpy as np

def seed_everything(seed):
    global SEED
    SEED = seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)


# Device & dtype
device = torch.device("cuda") # torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 # torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
torch.set_default_dtype(dtype)

print(f"device={device}; dtype={dtype}")


# Load model & tokenizer
model_path = "Qwen/Qwen3-1.7B"

tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
if tok.pad_token is None and tok.eos_token is not None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=dtype,
    device_map="cuda",
    trust_remote_code=True,
).to(device)


# Pipeline
def answer(question: str,
           system: str | None = "You are a helpful assistant.",
           max_new_tokens: int = 256,
           temperature: float = 0.7,
           top_p: float = 0.9,
           top_k: int = 50):
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    new_tokens = output_ids[0, input_ids.size(1):]
    text = tok.decode(new_tokens, skip_special_tokens=True)

    return text.strip()

def check(question, true, pred):
    request = f"Вопрос: \"{question}\"\nОтвет студента: \"{pred}\"\nПравильный ответ: \"{true}\"\n\n" + \
               "Опираясь на предоставленный правильный ответ, проверь ответ студента. Не придирайся к точной формулировке ответа. Если на вопрос нет правильного ответа, то считай правильными ответы студента типа \"Я не знаю\", \"Не знаю ответа на этот вопрос\", \"Не могу дать ответ на этот вопрос\" и т.д. Если на вопрос есть правильный ответ, то ответ студента должен ему соответствовать. Опирайся только на правильный ответ. Сначала проверь ответ пользователя, только потом выноси вердикт. В конце проверки напиши \"YES\" или \"NO\"."
    ans = answer(
        request,
        system="Ты — профессиональный жюри с огромным опытом проверки ответов студентов.",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.8,
        top_k=1,
    )

    yi = ans.rfind("YES")
    ni = ans.rfind("NO")

    if yi >= 0:
        if ni >= 0:
            return yi > ni
        else:
            return True
    else:
        if ni >= 0:
            return False
        else:
            return False


# Scoring
test = pd.read_csv('validation/test.csv')

with open('validation/output.json', 'r') as file:
    test['answer'] = json.load(file)

print("=" * 16, "START SCORING", "=" * 16)
test["verdict"] = True
for idx in tqdm(test.index, desc="Scoring..."):
    question = test.loc[idx, "correct"]
    true = test.loc[idx, "correct"] if test.loc[idx, "correct"] != "NONE" else "На этот вопрос нет правильного ответа"
    pred = test.loc[idx, "answer"]
    test.loc[idx, "verdict"] = check(question, true, pred)

by_question_fact = []
by_question_prov = []
for question_id in test["question_id"].unique():
    queries = test[test["question_id"] == question_id]
    if queries["correct"].iloc[0] == "NONE":
        # if queries["answer"].iloc[0] != "Ответ: помидор.":
        #     print(queries["answer"].iloc[0])
        # if queries["verdict"].all():
        #     print(queries["correct"].iloc[0], queries["answer"].iloc[0])
        by_question_prov.append(queries["verdict"].all())
    else:
        by_question_fact.append(queries["verdict"].all())

print(test[~test['verdict']])

acc_fact = np.mean(by_question_fact)
acc_prov = np.mean(by_question_prov)
print(f"Accuracy (by question, fact): {np.sum(by_question_fact)} / {len(by_question_fact)} = {100 * acc_fact:.4f}%")
print(f"Accuracy (by question, prov): {np.sum(by_question_prov)} / {len(by_question_prov)} = {100 * acc_prov:.4f}%")
print(f"Score:  {1000 * acc_fact:.6f} * 0.8 + {1000 * acc_prov:.6f} * 0.2 = {1000 * (acc_fact * 0.8 + acc_prov * 0.2):.6f}")
print("=" * 16, "SCORING FINISHED", "=" * 16)