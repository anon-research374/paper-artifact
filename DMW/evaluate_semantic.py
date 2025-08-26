import json
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
import torch
import argparse
import math

# --- 模块导入 ---
# 假设这些模块来自您提供的代码，并且可以被正确导入
try:
    from sent_to_code.sent_to_code import initialize_resources, sent_to_code
except ImportError:
    print("警告：无法导入 'sent_to_code' 模块。将使用占位函数进行测试。")


    # 定义占位函数，以便在没有sent_to_code模块时也能运行基本逻辑
    def initialize_resources(**kwargs):
        pass


    def sent_to_code(sentence, device, threshold):
        # 返回一个固定长度的随机码字用于测试
        k = kwargs.get('bit_length', 4)
        return f"{np.random.randint(0, 2 ** k):0{k}b}"


# --- 核心功能函数 ---

def recover_codewords(text: str, device: str) -> list:
    """
    从单段文本中恢复一个码字列表。每个码字对应一个句子。
    """
    codewords = []
    # 使用nltk进行句子分割
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        # sent_to_code 将每个句子转换为一个码字 (bitstring)
        # 注意：这里的 0.01 是一个示例阈值，您可能需要根据 sent_to_code 的实现来调整
        bitstring = sent_to_code(sentence, device, 0.01)

        if bitstring is not None:
            codewords.append(bitstring)

    return codewords


def calculate_shannon_entropy(data_list: list) -> float:
    """
    计算一个列表中数据的香农熵 (Shannon Entropy)。
    熵值越高，代表数据分布越随机、越不可预测。
    """
    if not data_list:
        return 0.0

    # 1. 统计每个码字出现的频率
    counts = Counter(data_list)
    total_count = len(data_list)

    # 2. 计算每个码字的概率 p(x)
    probabilities = [count / total_count for count in counts.values()]

    # 3. 根据公式 H(X) = - sum(p(x) * log2(p(x))) 计算熵
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    return entropy


def main():
    """
    主执行函数：读取文件，提取码字，计算并输出香农熵。
    """
    parser = argparse.ArgumentParser(description="计算JSONL文件中所有文本的码字分布香农熵。")
    parser.add_argument('--i', type=str, required=True, help="包含生成文本的JSONL文件路径。")
    parser.add_argument('--bit-num', type=int, default=4, help="用于sent_to_code初始化的比特长度 (即 k 值)。")
    args = parser.parse_args()

    FILE_PATH = args.i
    BIT_LENGTH_k = args.bit_num  # 每个码字的比特长度 k
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 初始化解码所需的资源
    print(f"使用设备: {DEVICE}")
    try:
        # 路径应根据您的项目结构进行配置
        initialize_resources(
            cc_path="./sent_to_code/data/4_kmeans/cc.pt",
            embedder_path="./sent_to_code/SemStamp-c4-sbert",
            bit_length=BIT_LENGTH_k
        )
        print("资源初始化完成。")
    except Exception as e:
        print(f"错误：资源初始化失败。请检查路径。详细信息: {e}")
        return

    all_extracted_codewords = []
    line_count = 0
    error_count = 0

    print(f"\n开始处理文件: '{FILE_PATH}'")
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                if not line.strip():
                    continue
                try:
                    json_obj = json.loads(line.strip())
                    # 我们只关心 "generated_sentence" 字段
                    generated_text = json_obj.get("result")

                    if generated_text:
                        codewords_from_text = recover_codewords(generated_text, DEVICE)
                        all_extracted_codewords.extend(codewords_from_text)
                    else:
                        print(f"警告：第 {line_count} 行缺少 'generated_sentence' 字段。")
                        error_count += 1
                except json.JSONDecodeError:
                    print(f"警告：第 {line_count} 行 JSON 格式无效。")
                    error_count += 1

        # 计算并打印最终的熵值
        entropy_value = calculate_shannon_entropy(all_extracted_codewords)

        # 理论最大熵是一个均匀分布的熵，等于 log2(码本大小K) = log2(2^k) = k
        max_entropy = BIT_LENGTH_k

        print("\n========== 熵值分析结果 ==========")
        print(f"已处理的总行数: {line_count}")
        print(f"处理失败或跳过的行数: {error_count}")
        print(f"提取的总码字(句子)数量: {len(all_extracted_codewords)}")

        if all_extracted_codewords:
            unique_codewords = len(set(all_extracted_codewords))
            print(f"唯一码字数量: {unique_codewords}")
            print(f"码字分布的香农熵 (Shannon Entropy): {entropy_value:.4f} bits")
            if max_entropy > 0:
                normalized_entropy = entropy_value / max_entropy
                print(f"理论最大熵 (基于 k={BIT_LENGTH_k}): {max_entropy:.4f} bits")
                print(f"归一化熵 (Normalized Entropy): {normalized_entropy:.4f}")

        print("================================")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{FILE_PATH}'。")
    except Exception as e:
        print(f"处理过程中发生意外错误: {e}")


if __name__ == '__main__':
    main()

