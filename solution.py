# Imports
import os
import sys
import json
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from llama_cpp import Llama

devnull = open(os.devnull, 'w')


# Load models
embedder = Llama(
    model_path="models/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
    embedding=True,
    n_gpu_layers=-1,
    verbose=False,
)

llm = Llama(
    model_path="models/T-lite-it-1.0-Q8_0-GGUF/t-lite-it-1.0-q8_0.gguf",
    n_ctx=8192,
    n_gpu_layers=-1,
    verbose=False,
)


# Pipeline
with open('data/articles.json', 'r') as file:
    facts = json.load(file)
with open('data/embeds.npy', 'rb') as file:
    embeds = np.load(file, allow_pickle=True)

fact_lens = pd.Series([len(text) for text in facts])
fact_len_max_allowed = fact_lens.quantile(0.99)

embeds = np.asarray(embeds, dtype=np.float32)
embeds = (embeds.T / (np.linalg.norm(embeds, axis=-1) + 1e-12)).T

dim = embeds.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(embeds)

with open('prompts/validator-system-prompt.txt', 'r') as file:
    validator_system_prompt = file.read()

def validate(question, retrieved_text):
    request = f'\n\nВопрос: {question}\nОтрывок:\n{retrieved_text}'
    llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": validator_system_prompt + request},
            {"role": "user", "content": question},
        ],
        temperature=0.001,
        top_p=0.001,
        max_tokens=512,
    )
    result = out["choices"][0]["message"]["content"].replace('<think>', '').replace('</think>', '').strip()
    if 'no' in result.lower():
        return False
    else:
        if 'yes' in result.lower():
            return True
        else:
            return False

def retrieve(question):
    stderr = sys.stderr
    sys.stderr = devnull
    e = np.array(
        embedder.create_embedding(input=question)["data"][0]["embedding"],
        dtype=np.float32
    )
    sys.stderr = stderr

    e = e / (np.linalg.norm(e) + 1e-12)

    k = 6
    sims, idxs = faiss_index.search(e.reshape(1, -1), k)
    sims = sims[0]
    idxs = idxs[0]

    res = []
    for i, idx in enumerate(idxs):
        if (sims[i] >= 0.45) and (fact_lens[idx] < fact_len_max_allowed) and validate(question, facts[idx]):
            res.append(facts[idx])
    return res[::-1]

with open('prompts/confidence-system-prompt.txt', 'r') as file:
    confidence_system_prompt = file.read()
with open('prompts/system-prompt.txt', 'r') as file:
    system_prompt = file.read()
with open('prompts/rag-system-prompt.txt', 'r') as file:
    rag_system_prompt = file.read()

def postprocess_answer(result):
    if 'ответ:' in result.lower():
        result = result[result.lower().rfind('ответ:') + len('ответ:'):]
    if ('не знаю' in result.lower()) and ('ошибка' in result.lower()):
        result = result[result.lower().rfind('ошибка') + len('ошибка'):]
    while True:
        if result.startswith(','):
            result = result[1:]
        elif result.endswith(')'):
            result = result[:-1]
        else:
            break
    return result.strip()

def inference_rag(question, texts):
    texts_prepared = '\n\nФакты:\n\n' + '\n\n'.join(texts)
    llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": rag_system_prompt + texts_prepared},
            {"role": "user", "content": question},
        ],
        temperature=0.001,
        top_p=0.001,
        max_tokens=512,
    )
    result = out["choices"][0]["message"]["content"]
    return postprocess_answer(result)

def check_confidence(question):
    llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": confidence_system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.001,
        top_p=0.001,
        max_tokens=512,
    )
    result = out["choices"][0]["message"]["content"]
    if 'no' in result.lower():
        return False
    else:
        if 'yes' in result.lower():
            return True
        else:
            return False

def inference_standard(question):
    llm.reset()
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.001,
        top_p=0.001,
        max_tokens=512,
    )
    result = out["choices"][0]["message"]["content"]
    return postprocess_answer(result)

def inference(question):
    # print(f'Inferencing question "{question}"')
    if not check_confidence(question):
        relevant = retrieve(question)
        # print(f'Retrieved {len(relevant)} texts')
        if len(relevant) > 0:
            answer = inference_rag(question, relevant)
            if not ('none' in answer.lower()):
                # print('Answered via RAG')
                return answer
            else:
                # print('Answering via RAG failed')
                pass
    else:
        # print('Retrieval skipped due to model\'s high confidence')
        pass
    answer = inference_standard(question)
    # print('Answered via solo-mode')
    return answer


# Inference
with open('input.json') as input_file:
    model_input = json.load(input_file)

model_output = []
for question in tqdm(model_input):
    model_output.append(inference(question))

with open('output.json', 'w') as output_file:
    json.dump(model_output, output_file, ensure_ascii=False)


devnull.close()