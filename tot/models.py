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

def completions_Llama(model_name, messages, temperature, max_tokens=5000, n=1):
    global tokenizer, model
    if tokenizer is None or model is None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Set to True for 4-bit quantization
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.float16  # You can also try torch.bfloat16
        )

        if model_name == "Llama-3.1-8B-Instruct":
            tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/{model_name}', token=hf_access_token)
            model = AutoModelForCausalLM.from_pretrained(f'meta-llama/{model_name}', quantization_config=bnb_config, token=hf_access_token)
        elif model_name == "Llama-3.2-3B-Instruct":
            tokenizer = AutoTokenizer.from_pretrained('/home/zs7353/Llama-3.2-3B-Instruct_tokenizer', local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained('/home/zs7353/Llama-3.2-3B-Instruct_model', local_files_only=True).to(device)

    outputs = []
    prompt = messages[0]['content']

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    input_length = len(input_ids)

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

def completions_qwen(messages, temperature, max_tokens=5000, n=1):
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained("/home/zs7353/Qwen2.5-1.5B-Instruct_tokenizer", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("/home/zs7353/Qwen2.5-1.5B-Instruct_model", local_files_only=True).to(device)
    
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


def get_output(prompt, model, temperature, max_tokens=5000, n=1) -> list:
    messages = [{"role": "user", "content": prompt}]
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        if model == "gpt-4o" or model == "o1-mini" or model == "gpt-4o-mini":
            global completion_tokens, prompt_tokens
            if model == "gpt-4o" or model == "gpt-4o-mini":
                res = completions_gpt(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
            elif model == "o1-mini":
                res = completions_gpt(model=model, messages=messages, n=cnt)
            outputs.extend([choice.message.content for choice in res.choices])
            completion_tokens += res.usage.completion_tokens
            prompt_tokens += res.usage.prompt_tokens
        elif model == "Llama-3.1-8B-Instruct" or model == 'Llama-3.2-3B-Instruct':
            res = completions_Llama(model_name=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
            outputs.extend(res)
        elif model == "Qwen2.5-1.5B-Instruct":
            res = completions_qwen(messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt)
            outputs.extend(res)            
        else:
            raise ValueError(f"Unimplemented model: {model}")
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