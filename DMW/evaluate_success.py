# import json
# import numpy as np
# import math
# import time
# import os
# from transformers import AutoTokenizer, GPT2Tokenizer
# from sentence_transformers import SentenceTransformer
# from nltk.tokenize import sent_tokenize
# import torch
# import argparse
#
# # 假设这些模块来自您提供的代码
# from sent_to_code.sent_to_code import initialize_resources, sent_to_code
#
# parser = argparse.ArgumentParser(description="使用自定义参数运行 LLM 数据隐藏。")
# parser.add_argument('--i', type=str)
# parser.add_argument('--bit-num', type=int,default=4)
# parser.add_argument('--h', type=int,default=6)
# args = parser.parse_args()
# FILE_PATH = args.i
# BIT_LENGTH = args.bit_num
# MAT_HEIGHT = args.h
# # 模型和数据文件路径
# CC_PATH = "./sent_to_code/data/4_kmeans/cc.pt"
# EMBEDDER_PATH = "./sent_to_code/SemStamp-c4-sbert"
# STC_MATRIX_PATH = './STC_code/stc_matrix.npy'  # STC 矩阵文件路径
#
# # 提取参数 (这些参数应与生成数据时使用的参数匹配)
#
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动检测 GPU
#
#
# # --- STC 消息提取函数 (来自您的代码) ---
#
# def get_matrix(width, height):
#     if 2 <= width <= 20 and 7 <= height <= 12:
#         # Get matrix from the pre-defined array
#         matrices = np.load(STC_MATRIX_PATH)
#         start = (height - 7) * 400 + (width - 1) * 20
#         return matrices[start:start + width]
#     else:
#         # Generate a random matrix
#         if (1 << (height - 2)) < width:
#             raise ValueError("Cannot generate matrix for this payload. Choose a higher constraint height.")
#
#         np.random.seed(1)
#         mask = (1 << (height - 2)) - 1
#         bop = (1 << (height - 1)) + 1
#         cols = []
#
#         for i in range(width):
#             while True:
#                 r = ((np.random.randint(1, mask + 1) & mask) << 1) + bop
#                 if r not in cols:
#                     cols.append(r)
#                     break
#
#         return np.array(cols, dtype=np.uint32)
#
#
# def arrange_matrices(shorter, longer, msg_length, inv_alpha):
#     """安排子矩阵的顺序和数量"""
#     mat_type = np.zeros(msg_length, dtype=np.uint8)
#     mat_width = np.full(msg_length, shorter, dtype=np.uint32)
#     for i in range(msg_length):
#         if np.sum(mat_width[:i]) + longer <= (i + 1) * inv_alpha + 0.5:
#             mat_type[i] = 1
#             mat_width[i] = longer
#     return mat_type, mat_width
#
#
# def stc_extract(vector, alpha, msg_length, mat_height):
#     """从二进制向量中提取 STC 编码的消息"""
#     inv_alpha = 1 / alpha
#     assert inv_alpha >= 1, '消息长度不能超过向量长度!'
#     assert 4 <= mat_height <= 31, '子矩阵高度应在 [4, 31] 范围内!'
#
#     shorter = math.floor(inv_alpha)
#     longer = math.ceil(inv_alpha)
#     columns = [get_matrix(shorter, mat_height), get_matrix(longer, mat_height)]
#
#     binmat = [np.unpackbits(columns[0][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1],
#               np.unpackbits(columns[1][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1]]
#
#     mat_type, mat_width = arrange_matrices(shorter, longer, msg_length, inv_alpha)
#
#     msg = np.zeros(msg_length, dtype=np.uint8)
#     height = mat_height
#     vec_idx = 0
#
#     for msg_idx in range(msg_length):
#         for k in range(mat_width[msg_idx]):
#             if vec_idx < len(vector) and vector[vec_idx]:
#                 msg[msg_idx:msg_idx + height] ^= binmat[mat_type[msg_idx]][k][:height]
#             vec_idx += 1
#         if msg_length - msg_idx <= mat_height:
#             height -= 1
#
#     return msg
#
#
# def recover_bit(text: str, bit_num: int, device):
#     """从文本中恢复比特流 (vector)"""
#     stego_bit = []
#     # 使用 nltk 进行句子分割
#     for sentence in sent_tokenize(text):
#         sentence = sentence.strip()
#         if not sentence:
#             continue
#
#         # 调用 sent_to_code 将句子转换为比特串
#         bitstring = sent_to_code(sentence, device, 0.01)
#
#         if bitstring is None:
#             continue
#
#         stego_bit.extend(int(b) for b in bitstring)
#
#     return stego_bit
#
#
# # --- 主逻辑 ---
#
# def compare_message_accuracy(json_obj, bit_length, mat_height, device):
#     """
#     核心函数：提取并比较消息，并计算比特准确度。
#     返回一个元组: (是否完全一致, 是否成功处理, 匹配的比特数, 总比特数)。
#     """
#     try:
#         # 1. 从 JSON 对象中获取所需数据
#         idx = json_obj.get('idx', 'unknown')
#         generated_text = json_obj.get("generated_sentence")
#         original_message_str = json_obj.get("message")
#         alpha = json_obj.get("alpha")
#         msg_length = json_obj.get("msg_size")
#
#         if not all([generated_text, original_message_str, alpha is not None, msg_length is not None]):
#             print(f"对象 idx {idx}: 跳过，因缺少 'generated_sentence', 'message', 'alpha', 或 'msg_size'。")
#             # --- 修改：返回符合新格式的元组 ---
#             return False, False, 0, 0
#
#         # 2. 将 JSON 中的原始消息（字符串格式）解析为 numpy 数组
#         original_message = np.array(json.loads(original_message_str))
#
#         # 3. 从生成的文本中恢复隐藏的比特向量 (vector)
#         time1 = time.time()
#         vector = recover_bit(generated_text, bit_length, device)
#
#         # 4. 从恢复的向量中提取消息
#         extracted_message = stc_extract(np.array(vector), alpha, msg_length, mat_height)
#         time2 = time.time()
#         print(f'time: {time2 - time1}')
#         # --- 新增: 计算比特准确度 ---
#         original_len = len(original_message)
#         extracted_len = len(extracted_message)
#
#         # 基准总比特数永远是原始消息的长度
#         total_bits = original_len
#         matching_bits = 0  # 默认为0
#
#         # 只有在总比特数大于0时才进行有意义的计算
#         if total_bits > 0:
#             # 找出需要比较的共同部分长度
#             compare_len = min(original_len, extracted_len)
#             # 计算在共同部分的匹配比特数
#             matching_bits = np.sum(original_message[:compare_len] == extracted_message[:compare_len])
#
#         # 5. 比较整个消息是否完全相等 (需要长度和内容都相等)
#         are_equal = (original_len == extracted_len) and (matching_bits == total_bits)
#
#         if are_equal:
#             print(f"✅ 对象 idx {idx}: 消息一致。")
#         else:
#             print(f"❌ 对象 idx {idx}: 消息不一致。")
#             print(f"  - 原始消息: {original_message}")
#             print(f"  - 提取消息: {extracted_message}")
#             # --- 新增: 为不一致的消息打印其比特准确度 ---
#             if total_bits > 0:
#                 bit_acc_percent = (matching_bits / total_bits) * 100
#                 print(f"  - 比特准确度: {matching_bits}/{total_bits} ({bit_acc_percent:.2f}%)")
#
#         # --- 修改：返回包含比特计数的新元组 ---
#         return are_equal, True, matching_bits, total_bits
#
#     except Exception as e:
#         print(f"处理对象 idx {json_obj.get('idx', 'unknown')} 时发生错误: {e}")
#         # --- 修改：返回符合新格式的元组 ---
#         return False, False, 0, 0
#
#
# def main():
#     """
#     主执行函数
#     """
#     # 初始化解码所需的资源
#     print(f"使用设备: {DEVICE}")
#     print("正在初始化资源 (sent_to_code)...")
#     try:
#         initialize_resources(
#             cc_path=CC_PATH,
#             embedder_path=EMBEDDER_PATH,
#             bit_length=BIT_LENGTH
#         )
#         print("初始化完成。")
#     except Exception as e:
#         print(f"错误：资源初始化失败。请检查路径 '{CC_PATH}' 和 '{EMBEDDER_PATH}'。")
#         print(f"详细信息: {e}")
#         return
#
#     # 准备计数器
#     identical_count = 0
#     different_count = 0
#     processed_count = 0
#     error_count = 0
#     total_bits_processed = 0
#     total_matching_bits = 0
#
#     print(f"\n开始处理文件: '{FILE_PATH}'")
#
#     try:
#         with open(FILE_PATH, 'r', encoding='utf-8') as file:
#             for line_number, line in enumerate(file, 1):
#                 if not line.strip():
#                     continue
#
#                 try:
#                     json_obj = json.loads(line.strip())
#                     print(f"\n--- 正在处理第 {line_number} 行 (idx: {json_obj.get('idx', 'unknown')}) ---")
#
#                     # 对当前对象进行消息准确度评测
#                     are_equal, processed, matching, total= compare_message_accuracy(json_obj, BIT_LENGTH, MAT_HEIGHT, DEVICE)
#
#                     if processed:
#                         processed_count += 1
#                         if are_equal:
#                             identical_count += 1
#                         else:
#                             different_count += 1
#
#                         total_matching_bits += matching
#                         total_bits_processed += total
#                     else:
#                         error_count += 1
#
#                 except json.JSONDecodeError:
#                     print(f"第 {line_number} 行错误：JSON 格式无效。")
#                     error_count += 1
#
#         # 打印最终的评测摘要
#         print("\n========== 评测摘要 ==========")
#         print(f"总共分析的对象数: {processed_count}")
#         print(f"处理失败或跳过的对象数: {error_count}")
#         print(f"✅ 消息一致的对象数: {identical_count}")
#         print(f"❌ 消息不一致的对象数: {different_count}")
#
#         if processed_count > 0:
#             accuracy = (identical_count / processed_count) * 100
#             print(f"🎯 准确率 (Message Match Accuracy): {accuracy:.2f}%")
#             print(f'bit accuracy:{(total_matching_bits / total_bits_processed) * 100}')
#         print("===============================")
#
#     except FileNotFoundError:
#         print(f"错误：找不到文件 '{FILE_PATH}'。请检查文件路径。")
#     except Exception as e:
#         print(f"处理过程中发生意外错误: {e}")
#
#
# if __name__ == '__main__':
#     main()
#
#


import json
import numpy as np
import math
import time
import os
from transformers import AutoTokenizer, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import torch
import argparse
import traceback

# 假设这些模块来自您提供的代码
from sent_to_code.sent_to_code import initialize_resources, sent_to_code

# --- 参数解析 ---
parser = argparse.ArgumentParser(description="使用数据截断的近似方法，解码由非整数倍参数生成的数据。")
parser.add_argument('--i', type=str, required=True, help="要处理的JSONL数据文件路径。")
parser.add_argument('--bit-num', type=int, default=4, help="每个句子代表的比特数 (必须与生成时一致)。")
parser.add_argument('--h', type=int, default=6, help="STC矩阵高度 (必须与生成时一致)。")
parser.add_argument('--seg', type=int, required=True, help="生成数据时使用的段长度 (seg)。")
args = parser.parse_args()

# --- 全局常量 ---
FILE_PATH = args.i
BIT_LENGTH = args.bit_num
MAT_HEIGHT = args.h
SEG_LENGTH = args.seg
CC_PATH = "./sent_to_code/data/4_kmeans/cc.pt"
EMBEDDER_PATH = "./sent_to_code/SemStamp-c4-sbert"
STC_MATRIX_PATH = './STC_code/stc_matrix.npy'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# --- 未经修改的 STC 提取与辅助函数 ---

def get_matrix(width, height):
    if 2 <= width <= 20 and 7 <= height <= 12 and os.path.exists(STC_MATRIX_PATH):
        matrices = np.load(STC_MATRIX_PATH)
        start = (height - 7) * 400 + (width - 1) * 20
        return matrices[start:start + width]
    else:
        if (1 << (height - 2)) < width:
            raise ValueError("Cannot generate matrix for this payload. Choose a higher constraint height.")
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


def stc_extract(vector, alpha, msg_length, mat_height):
    inv_alpha = 1 / alpha
    assert inv_alpha >= 1, '消息长度不能超过向量长度!'
    assert 4 <= mat_height <= 31, '子矩阵高度应在 [4, 31] 范围内!'
    shorter = math.floor(inv_alpha)
    longer = math.ceil(inv_alpha)
    columns = [get_matrix(shorter, mat_height), get_matrix(longer, mat_height)]
    binmat = [np.unpackbits(columns[0][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1],
              np.unpackbits(columns[1][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1]]
    mat_type, mat_width = arrange_matrices(shorter, longer, msg_length, inv_alpha)
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


def recover_bit(text: str, bit_num: int, device: str) -> list:
    """从文本中恢复比特流 (vector)，保持原样"""
    stego_bit = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence: continue
        bitstring = sent_to_code(sentence, device, 0.01)
        if bitstring:
            stego_bit.extend(int(b) for b in bitstring)
    return stego_bit


# --- 修改后的核心函数 ---
def compare_message_accuracy(json_obj, bit_length, mat_height, seg_length, device):
    """
    核心函数：使用数据截断的近似方法，提取并比较消息。
    本方法会丢弃不完整段落的信息，只验证完整段落的部分。
    """
    try:
        # 1. 从 JSON 对象中获取所需数据
        idx = json_obj.get('idx', 'N/A')
        generated_text = json_obj.get("generated_sentence")
        original_message_raw = json_obj.get("message")
        alpha = json_obj.get("alpha")
        msg_length = json_obj.get("msg_size")

        if not all([generated_text, original_message_raw, alpha is not None, msg_length is not None]):
            print(f"对象 idx {idx}: 跳过，缺少必要字段。")
            return False, False, 0, 0

        # ==================== FIX STARTS HERE ====================
        # 修正: 稳健地将 "message" 字段加载为 numpy 数组
        # 检查它是否为字符串，如果是，则用 json.loads() 解析
        if isinstance(original_message_raw, str):
            original_message_list = json.loads(original_message_raw)
        else:
            # 如果它已经是列表（或其它可迭代对象），直接使用
            original_message_list = original_message_raw

        original_message = np.array(original_message_list)
        # ===================== FIX ENDS HERE =====================

        # 2. 计算截断点
        num_full_segments = msg_length // seg_length
        truncated_msg_length = num_full_segments * seg_length

        if num_full_segments == 0:
            print(f"  [信息] 对象 idx {idx}: 消息总长 ({msg_length}) 小于一个段长 ({seg_length})，本方法不适用，跳过。")
            return False, False, 0, msg_length

        print(f"  [分析] 原始消息 {msg_length} 比特，段长 {seg_length}。包含 {num_full_segments} 个完整段落。")
        print(f"  [计划] 将只验证前 {truncated_msg_length} 比特的信息。")

        seg_num = int(seg_length / alpha / bit_length)
        num_sentences_to_keep = num_full_segments * seg_num

        # 3. 对数据进行截断
        original_message_truncated = original_message[:truncated_msg_length]

        all_sentences = sent_tokenize(generated_text)
        if len(all_sentences) < num_sentences_to_keep:
            print(
                f"  [警告] 句子数量不足！需要约 {num_sentences_to_keep} 句，实际只有 {len(all_sentences)} 句。将使用所有句子。")
            num_sentences_to_keep = len(all_sentences)

        truncated_sentences = all_sentences[:num_sentences_to_keep]
        truncated_text = " ".join(truncated_sentences)

        # 4. 使用原始的、连续的解码逻辑处理截断后的数据
        time1 = time.time()
        vector = recover_bit(truncated_text, bit_length, device)

        extracted_message = stc_extract(np.array(vector), alpha, msg_length=truncated_msg_length, mat_height=mat_height)
        time2 = time.time()
        print(f'  [性能] 提取耗时: {time2 - time1:.4f} 秒')

        # 5. 比较截断后的结果
        total_bits_to_check = len(original_message_truncated)
        matching_bits = np.sum(
            original_message_truncated[:len(extracted_message)] == extracted_message[:len(original_message_truncated)])
        are_equal = (len(original_message_truncated) == len(extracted_message)) and (
                    matching_bits == total_bits_to_check)

        if are_equal:
            print(f"✅ 对象 idx {idx}: 消息的完整部分一致 (Partial Match)。")
            print(f"   已验证 {matching_bits}/{msg_length} 比特。")
        else:
            print(f"❌ 对象 idx {idx}: 消息的完整部分不一致。")
            print(f"  - 原始消息 (截断后 {len(original_message_truncated)} bits): {original_message_truncated}")
            print(f"  - 提取消息 ({len(extracted_message)} bits): {extracted_message}")
            if total_bits_to_check > 0:
                bit_acc_percent = (matching_bits / total_bits_to_check) * 100
                print(f"  - 部分比特准确度: {matching_bits}/{total_bits_to_check} ({bit_acc_percent:.2f}%)")

        return are_equal, True, matching_bits, total_bits_to_check

    except Exception as e:
        print(f"处理对象 idx {json_obj.get('idx', 'N/A')} 时发生严重错误: {e}")
        traceback.print_exc()
        return False, False, 0, 0


# --- 主逻辑 (与之前相同) ---
def main():
    print(f"使用设备: {DEVICE}")
    print("正在初始化资源 (sent_to_code)...")
    try:
        initialize_resources(
            cc_path=CC_PATH,
            embedder_path=EMBEDDER_PATH,
            bit_length=BIT_LENGTH
        )
        print("初始化完成。")
    except Exception as e:
        print(f"错误：资源初始化失败。请检查路径 '{CC_PATH}' 和 '{EMBEDDER_PATH}'。详细信息: {e}")
        return

    identical_count, different_count, processed_count, error_count = 0, 0, 0, 0
    total_bits_processed, total_matching_bits = 0, 0

    print(f"\n开始处理文件: '{FILE_PATH}'")
    print(f"使用参数: bit-num={BIT_LENGTH}, mat-height={MAT_HEIGHT}, seg={SEG_LENGTH}")
    print("注意：本脚本使用数据截断法，只会验证消息的完整段落部分。")

    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip(): continue
                try:
                    json_obj = json.loads(line.strip())
                    print(f"\n--- 正在处理第 {line_number} 行 (idx: {json_obj.get('idx', 'N/A')}) ---")

                    are_equal, processed, matching, total = compare_message_accuracy(
                        json_obj, BIT_LENGTH, MAT_HEIGHT, SEG_LENGTH, DEVICE
                    )

                    if processed:
                        processed_count += 1
                        if are_equal:
                            identical_count += 1
                        else:
                            different_count += 1
                        total_matching_bits += matching
                        total_bits_processed += total
                    else:
                        error_count += 1

                except json.JSONDecodeError:
                    print(f"第 {line_number} 行错误：JSON 格式无效。")
                    error_count += 1

        print("\n========== 评测摘要 (基于截断数据) ==========")
        print(f"总共分析的对象数: {processed_count}")
        print(f"处理失败或跳过的对象数: {error_count}")
        print(f"✅ 消息部分一致的对象数: {identical_count}")
        print(f"❌ 消息部分不一致的对象数: {different_count}")

        if processed_count > 0:
            accuracy = (identical_count / processed_count) * 100
            print(f"🎯 部分匹配准确率 (Partial Match Accuracy): {accuracy:.2f}%")
        if total_bits_processed > 0:
            bit_accuracy = (total_matching_bits / total_bits_processed) * 100
            print(
                f"部分比特准确度 (Partial Bit Accuracy): {bit_accuracy:.2f}% ({total_matching_bits}/{total_bits_processed})")
        print("==============================================")

    except FileNotFoundError:
        print(f"错误：找不到文件 '{FILE_PATH}'。请检查文件路径。")
    except Exception as e:
        print(f"处理过程中发生意外错误: {e}")


if __name__ == '__main__':
    main()