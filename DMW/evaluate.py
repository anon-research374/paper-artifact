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
    """
    专门处理包含 generated_sentence 字段的 JSON 文件
    """
    # 加载模型
    print("加载模型...")
    tokenizer = AutoTokenizer.from_pretrained("/home/zlm/xuezhou/my_model/llama-3-8b")
    model = AutoModelForCausalLM.from_pretrained("/home/zlm/xuezhou/my_model/llama-3-8b",torch_dtype=torch.bfloat16).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 读取文件并提取句子
    sentences = []
    ppl = []
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # 检查是否是 JSON Lines 格式
    if content.count('\n') > 0 and content.count('{') > 1:
        # JSON Lines 格式
        print("检测到 JSON Lines 格式")
        for line_num, line in enumerate(content.split('\n'), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "generated_sentence" in data:
                    sentences.append(data["generated_sentence"])
                else:
                    print(f"第 {line_num} 行缺少 generated_sentence 字段")
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行解析失败: {e}")
    else:
        # 单个 JSON 对象
        print("检测到单个 JSON 对象")
        try:
            data = json.loads(content)
            if "prompt" in data:
                sentences.append(data["prompt"])
            else:
                print("缺少 generated_sentence 字段")
                print(f"可用字段: {list(data.keys())}")
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")

    if not sentences:
        print("没有找到任何句子，程序退出")
        return

    print(f"找到 {len(sentences)} 个句子")

    # 计算困惑度
    results = []
    total_ppl = 0.0
    evaluated_count = 0

    for i, sentence in enumerate(tqdm(sentences, desc="计算困惑度")):
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
            print(f"句子 {i + 1} 评估失败: {e}")
            continue

    # 计算平均困惑度
    avg_ppl = total_ppl / evaluated_count if evaluated_count > 0 else 0.0
    print(f"\n平均困惑度 (PPL): {avg_ppl:.2f}")

    # # 保存结果
    # with open(output_file, "w", encoding="utf-8") as f:
    #     for result in results:
    #         f.write(json.dumps(result, ensure_ascii=False) + "\n")
    #
    #     # 添加汇总信息
    #     summary = {
    #         "summary": {
    #             "average_perplexity": avg_ppl,
    #             "evaluated_count": evaluated_count,
    #             "total_sentences": len(sentences)
    #         }
    #     }
    #     f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(f"结果已保存到: {output_file}")
    print(f"总句子数: {len(sentences)}")
    print(f"成功评估: {evaluated_count}")
    print(f'ppl: {ppl}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用自定义参数运行 LLM 数据隐藏。")
    parser.add_argument('--i', type=str)
    args = parser.parse_args()
    input_file = args.i
    output_file = "perplexity_results.jsonl"
    evaluate_perplexity(input_file, output_file, device="cuda")