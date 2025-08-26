# File: evaluate_attacks.py

import json
import numpy as np
import math
import time
import os
from nltk.tokenize import sent_tokenize
import torch
import argparse
from typing import List, Dict, Tuple, Optional

# 假设这些模块来自您提供的代码
# 请确保 sent_to_code_module.py 文件在同一目录下或在Python路径中
from sent_to_code.sent_to_code import initialize_resources, sent_to_code

# ============================================================================
# STC 消息提取函数 (来自您的代码)
# ============================================================================

STC_MATRIX_PATH = './STC_code/stc_matrix.npy'


def get_matrix(width, height):
    if 2 <= width <= 20 and 7 <= height <= 12:
        matrices = np.load(STC_MATRIX_PATH)
        start = (height - 7) * 400 + (width - 1) * 20
        return matrices[start:start + width]
    else:
        if (1 << (height - 2)) < width:
            raise ValueError("Cannot generate matrix for this payload.")
        np.random.seed(1)
        mask = (1 << (height - 2)) - 1
        bop = (1 << (height - 1)) + 1
        cols = []
        for i in range(width):
            while True:
                r = ((np.random.randint(1, mask + 1) & mask) << 1) + bop
                if r not in cols:
                    cols.append(r)
                    break
        return np.array(cols, dtype=np.uint32)


def arrange_matrices(shorter, longer, msg_length, inv_alpha):
    mat_type = np.zeros(msg_length, dtype=np.uint8)
    mat_width = np.full(msg_length, shorter, dtype=np.uint32)
    for i in range(msg_length):
        if np.sum(mat_width[:i]) + longer <= (i + 1) * inv_alpha + 0.5:
            mat_type[i] = 1
            mat_width[i] = longer
    return mat_type, mat_width


def stc_extract(vector: List[int], alpha: float, msg_length: int, mat_height: int) -> np.ndarray:
    """从二进制向量中提取 STC 编码的消息"""
    inv_alpha = 1 / alpha
    shorter, longer = math.floor(inv_alpha), math.ceil(inv_alpha)
    columns = [get_matrix(shorter, mat_height), get_matrix(longer, mat_height)]
    mat_type, mat_width = arrange_matrices(shorter, longer, msg_length, inv_alpha)

    binmat = [
        np.unpackbits(columns[0][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1],
        np.unpackbits(columns[1][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1]
    ]

    msg = np.zeros(msg_length, dtype=np.uint8)
    height = mat_height
    vec_idx = 0
    for msg_idx in range(msg_length):
        for k in range(mat_width[msg_idx]):
            if vec_idx < len(vector) and vector[vec_idx]:
                msg[msg_idx:msg_idx + height] ^= binmat[mat_type[msg_idx]][k][:height]
            vec_idx += 1
        if msg_length - msg_idx <= mat_height:
            height -= 1
    return msg


def recover_bit(text: str, bit_num: int, device: str) -> Optional[List[int]]:
    """从文本中恢复比特流 (vector)"""
    stego_bit = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence: continue

        bitstring = sent_to_code(sentence, device, 0.0001)
        if bitstring is None:
            # 如果任何一个句子解码失败，则认为整个文本解码失败
            return None
        stego_bit.extend(int(b) for b in bitstring)
    return stego_bit


# 在您的 evaluate_attack.py 脚本中，替换这个函数

def evaluate_attacks_on_record(json_obj: Dict, bit_length: int, mat_height: int, device: str) -> Dict[
    str, Tuple[int, int]]:
    """
    (已修正) 对单个JSON记录中的所有四种攻击进行评测。
    """
    results = {}
    attack_fields = {
        "Deletion": "delete_sen",
        "Substitution": "sub_sen",
        "Insertion": "insert_sen",
        "Paraphrase": "para_sen"
    }

    try:
        # 1. 提取通用信息
        original_message_str = json_obj.get("message")  # 得到的是一个字符串
        alpha = json_obj.get("alpha")
        msg_length = json_obj.get("msg_size")

        if not all([original_message_str, alpha is not None, msg_length is not None]):
            print(f"  - 警告: 跳过此记录，因缺少 'message', 'alpha', 或 'msg_size'。")
            return {name: (0, 0) for name in attack_fields}

        # --- 核心修正：在这里解析 message 字符串 ---
        try:
            # 使用 json.loads() 将字符串 "[1,0,1]" 解析为列表 [1, 0, 1]
            original_message_list = json.loads(original_message_str)
            original_message = np.array(original_message_list)
        except (json.JSONDecodeError, TypeError):
            print(f"  - 警告: 跳过此记录，因 'message' 字段格式不正确: {original_message_str}")
            return {name: (0, 0) for name in attack_fields}
        # --- 修改结束 ---

        total_bits = len(original_message)

        # 2. 遍历每种攻击
        for attack_name, field_name in attack_fields.items():
            attacked_text = json_obj.get(field_name)

            if not attacked_text:
                print(f"  - {attack_name}: 警告 - 未找到攻击后的文本字段 '{field_name}'。")
                results[attack_name] = (0, total_bits)
                continue

            # 3. 从攻击后的文本恢复消息
            vector = recover_bit(attacked_text, bit_length, device)

            if vector is None:
                print(f"  - {attack_name}: ❌ 恢复失败 (sent_to_code 返回 None)。")
                results[attack_name] = (0, total_bits)
                continue

            extracted_message = stc_extract(vector, alpha, msg_length, mat_height)

            # 4. 计算比特准确度
            compare_len = min(len(original_message), len(extracted_message))
            matching_bits = np.sum(original_message[:compare_len] == extracted_message[:compare_len])

            results[attack_name] = (matching_bits, total_bits)

            if matching_bits == total_bits and len(original_message) == len(extracted_message):
                print(f"  - {attack_name}: ✅ 消息完全一致。")
            else:
                print(f"  - {attack_name}: ❌ 消息不一致 ({matching_bits}/{total_bits} 比特匹配)。")

    except Exception as e:
        print(f"  - 错误: 处理此记录时发生意外错误: {e}")
        return {name: (0, 0) for name in attack_fields}

    return results


# ============================================================================
# 主执行函数 (Main Function)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="评测不同攻击下的水印鲁棒性。")
    parser.add_argument('--i', type=str, required=True, help="包含评测数据的JSON文件路径。")
    parser.add_argument('--bit-num', type=int, default=4, help="每个句子的比特长度。")
    parser.add_argument('--h', type=int, default=6, help="STC矩阵高度。")
    args = parser.parse_args()

    # --- 初始化 ---
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {DEVICE}")
    print("正在初始化资源 (sent_to_code)...")
    try:
        initialize_resources(
            cc_path="./sent_to_code/data/4_kmeans/cc.pt",
            embedder_path="./sent_to_code/SemStamp-c4-sbert",
            bit_length=args.bit_num
        )
        print("初始化完成。")
    except Exception as e:
        print(f"错误：资源初始化失败: {e}")
        return

    # --- 聚合结果的计数器 ---
    total_stats = {
        "Deletion": {"matching": 0, "total": 0},
        "Substitution": {"matching": 0, "total": 0},
        "Insertion": {"matching": 0, "total": 0},
        "Paraphrase": {"matching": 0, "total": 0}
    }
    processed_records = 0

    # --- 读取和处理数据 ---
    print(f"\n开始处理文件: '{args.i}'")
    try:
        with open(args.i, 'r', encoding='utf-8') as f:
            data = json.load(f)  # 直接加载整个JSON文件

        for i, record in enumerate(data):
            print(f"\n--- 正在处理第 {i + 1}/{len(data)} 条记录 ---")

            # 对当前记录进行评测
            record_results = evaluate_attacks_on_record(record, args.bit_num, args.h, DEVICE)

            # 聚合结果
            for attack_name, (matching, total) in record_results.items():
                if total > 0:  # 只有在有效处理后才聚合
                    total_stats[attack_name]["matching"] += matching
                    total_stats[attack_name]["total"] += total

            # 只有在至少有一种攻击有效处理后，才算作一条已处理记录
            if any(t > 0 for _, t in record_results.values()):
                processed_records += 1

    except FileNotFoundError:
        print(f"错误：找不到文件 '{args.i}'。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{args.i}' 不是有效的JSON格式。")
        return
    except Exception as e:
        print(f"处理过程中发生意外错误: {e}")

    # --- 打印最终摘要 ---
    print("\n========== 评测摘要 ==========")
    print(f"总共分析的记录数: {processed_records}")
    print("-" * 50)
    print(f"{'攻击类型':<15} | {'比特准确率':<15} | {'匹配/总比特数'}")
    print("-" * 50)

    for attack_name, stats in total_stats.items():
        matching = stats["matching"]
        total = stats["total"]
        accuracy = (matching / total * 100) if total > 0 else 0

        print(f"{attack_name:<15} | {accuracy:>14.2f}% | ({matching}/{total})")

    print("===============================")


if __name__ == '__main__':
    main()