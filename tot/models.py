import os
import openai

import torch
torch.cuda.empty_cache()

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#chatgpt
completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
hf_access_token = os.getenv("HF_ACCESS_TOKEN", "")

if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")

tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def completions_gpt(**kwargs):
    return openai.chat.completions.create(**kwargs)

def completions_Llama(model, tokenizer, messages, temperature, max_tokens=1024, n=1):
    
    outputs = []
    prompt = messages[0]['content']

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = len(input_ids)
    print("completions_Llama generating")
    generated_ids = model.generate(
        input_ids=input_ids,
        do_sample=True,
        num_return_sequences=n,
        max_new_tokens=max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    outputs = []
    for response in responses:
        response = response.strip()
        outputs.append(response)
    
    return outputs

def completions_qwen(model, tokenizer, messages, temperature, max_tokens=5000, n=1):
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    input_ids = model_inputs.input_ids[0]
    input_length = len(input_ids)
    
    generated_ids = model.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        do_sample=True,
        num_return_sequences=n,
        max_new_tokens=max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [output_ids[input_length:] for output_ids in generated_ids]
    
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    outputs = []
    for response in responses:
        response = response.strip()
        outputs.append(response)
    
    return outputs


def get_output(model, tokenizer, name, prompt, temperature, max_tokens=5000, n=1) -> list:
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        if name == "Llama-3.1-8B-Instruct" or name == 'Llama-3.2-3B-Instruct':
            res = completions_Llama(model, tokenizer, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
            outputs.extend(res)
        elif name == "Qwen2.5-1.5B-Instruct":
            res = completions_qwen(model, tokenizer, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
            outputs.extend(res)            
        else:
            raise ValueError(f"Unimplemented model: {name}")
    return outputs
    
def usage(backend):
    global completion_tokens, prompt_tokens
    if backend == "o1-mini":
        cost = completion_tokens / 1000 * 0.012 + prompt_tokens / 1000 * 0.003
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.01 + prompt_tokens / 1000 * 0.0025
    elif backend == "gpt-4o-mini":
        cost = completion_tokens / 1000 * 0.0006 + prompt_tokens / 1000 * 0.00015
    else:
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}