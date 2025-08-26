import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
from tqdm import tqdm
import argparse

def calculate_perplexity(sentence: str, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer,
                         device: str = "cuda:3") -> float:
    model.eval()
    encodings = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


def evaluate_perplexity(input_file: str, output_file: str, device: str = "cuda:3"):
    tokenizer = AutoTokenizer.from_pretrained(path_to_your_model)
    model = AutoModelForCausalLM.from_pretrained(path_to_your_model,torch_dtype=torch.bfloat16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sentences = []
    ppl = []
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.count('\n') > 0 and content.count('{') > 1:
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "generated_sentence" in data:
                    sentences.append(data["generated_sentence"])
            except json.JSONDecodeError as e:
                print(f"fail")
    else:
        try:
            data = json.loads(content)
            if "prompt" in data:
                sentences.append(data["prompt"])
        except json.JSONDecodeError as e:
            print(f"fail")

        return

    results = []
    total_ppl = 0.0
    evaluated_count = 0

    for i, sentence in enumerate(tqdm(sentences, desc="ppl")):
        sentence = str(sentence).strip()
        if not sentence:
            continue

        try:
            perplexity = calculate_perplexity(sentence, model, tokenizer, device)
            print(perplexity)
            total_ppl += perplexity
            evaluated_count += 1
            ppl.append(perplexity)
            print(f'ppl: {ppl}')

            result = {
                "sentence_idx": i,
                "sentence": sentence,
                "perplexity": perplexity
            }

            results.append(result)

        except Exception as e:
            continue

    avg_ppl = total_ppl / evaluated_count if evaluated_count > 0 else 0.0



if __name__ == "__main__":
    parser = argparse.ArgumentParser)
    parser.add_argument('--i', type=str)
    args = parser.parse_args()
    input_file = args.i
    output_file = "perplexity_results.jsonl"
    evaluate_perplexity(input_file, output_file, device="cuda")