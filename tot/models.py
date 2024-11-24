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

def completions_gpt(**kwargs):
    return openai.chat.completions.create(**kwargs)

#llama
tokenizer = None
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def completions_Llama(messages, temperature=0.7, max_tokens=1000, n=1, stop=None):
    global tokenizer, model
    if tokenizer is None or model is None:

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Set to True for 4-bit quantization
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16  # You can also try torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', token=hf_access_token)
        model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct', quantization_config=bnb_config, token=hf_access_token)
        
    prompt = ''
    for message in messages:
        if message['role'] == 'system':
            prompt += f"{message['content']}\n"
        elif message['role'] == 'user':
            prompt += f"User: {message['content']}\n"
        elif message['role'] == 'assistant':
            prompt += f"Assistant: {message['content']}\n"

    outputs = []
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    for _ in range(n):
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_text = output_text[len(prompt):].strip()

        if stop:
            for stop_seq in stop:
                idx = generated_text.find(stop_seq)
                if idx != -1:
                    generated_text = generated_text[:idx]
                    break

        outputs.append(generated_text)

    return outputs

def get_output(prompt, model, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    if model == "gpt-4o" or model == "o1-mini":
        messages = [{"role": "user", "content": prompt}]
        global completion_tokens, prompt_tokens
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = completions_gpt(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
            outputs.extend([choice.message.content for choice in res.choices])
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
        return outputs
    
    elif model == "Llama3.1-8B-Instruct":
        messages = [{"role": "user", "content": prompt}]
        outputs = []
        while n > 0:
            cnt = min(n, 20)
            n -= cnt
            res = completions_Llama(messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
            outputs.extend(res)
        return outputs
    
def usage(backend):
    global completion_tokens, prompt_tokens
    if backend == "o1-mini":
        cost = completion_tokens / 1000 * 0.012 + prompt_tokens / 1000 * 0.003
    elif backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.01 + prompt_tokens / 1000 * 0.0025
    elif backend == "Llama3.1-8B-Instruct":
        cost = 0
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}