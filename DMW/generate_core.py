from cProfile import label
import numpy as np
import math
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForMaskedLM, GPT2Tokenizer, GPT2LMHeadModel
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from sentence_transformers import SentenceTransformer
from transformers.tokenization_utils import PreTrainedTokenizer
from sentence_transformers.util import cos_sim
from transformers import StoppingCriteria, StoppingCriteriaList
from sent_to_code.sent_to_code import initialize_resources, sent_to_code, compute_cost_cosine,compute_cost_norm
from nltk.tokenize import sent_tokenize
from string import punctuation
from itertools import groupby
import torch
from math import exp
from tqdm import tqdm
import traceback
import random
import json
import jsonlines
import os
import difflib
import ssl
import requests
from collections import Counter
import nltk
import argparse

try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

MAX_PPL_COST = 0  # 负无穷大，任何数都比它大
MIN_PPL_COST = 10   # 正无穷大，任何数都比它小
MAX_robust_COST = 0
MIN_robust_COST = 10
MAX_entro_COST = 0
MIN_entro_COST = 10

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
PUNCTS = '!.?'

# =========================================================================
# 核心 STC 编码与未修改的工具函数
# =========================================================================

def get_matrix(width, height):
    if 2 <= width <= 20 and 7 <= height <= 12:
        # Get matrix from the pre-defined array
        matrices = np.load('/home/zlm/xuezhou/DTMM/STC_code/stc_matrix.npy')
        start = (height - 7) * 400 + (width - 1) * 20
        return matrices[start:start + width]
    else:
        # Generate a random matrix
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


class STCEmbed():
    def __init__(self, alpha, msg, mat_height, seg):
        mat_height = np.clip(mat_height, 5, 31)
        assert len(msg) >= mat_height, 'The message length must be larger than the submatrix height!'
        assert 0 < seg <= len(msg), 'The segment length cannot be larger than the message length!'

        state_num = 1 << mat_height
        self.col_mask = state_num - 1

        inv_alpha = 1 / alpha
        assert inv_alpha >= 1, 'The message cannot be longer than the vector!'
        if inv_alpha < 2:
            print("The relative payload is greater than 1/2. This may result in poor embedding efficiency.")

        shorter = math.floor(inv_alpha)
        longer = math.ceil(inv_alpha)
        self.columns = [get_matrix(shorter, mat_height), get_matrix(longer, mat_height)]
        msg_length = len(msg)
        mat_type, mat_width = arrange_matrices(shorter, longer, msg_length, inv_alpha)
        self.mat_type = mat_type
        self.mat_width = mat_width

        vector_length = np.round(msg_length / alpha).astype(int)
        self.stego = np.zeros(vector_length, dtype=np.uint8)
        self.stego_seg = np.array([], dtype=np.uint8)
        self.state = np.arange(state_num, dtype=np.uint32)
        self.prices = np.full(state_num, np.inf, dtype=np.float32)
        self.prices[0] = 0.0
        self.path = np.zeros((vector_length, state_num), dtype=np.uint8)

        self.msg = msg
        self.seg = seg
        self.mat_height = mat_height
        self.msg_length = msg_length
        self.state_num = state_num
        self.vec_idx = 0
        self.msg_idx = 0
        self.col_idx = 0

    def embed_bit(self, vector_bit, cost):
        column = self.columns[self.mat_type[self.msg_idx]][self.col_idx] & self.col_mask
        c1, c2 = vector_bit * cost, (1 - vector_bit) * cost

        alt_state = self.state ^ column
        v1 = self.prices[self.state] + c1
        v2 = self.prices[alt_state] + c2
        self.prices = np.minimum(v1, v2)
        self.path[self.vec_idx, :] = (self.prices == v2).astype(np.uint8)

        sel_idx = self.path[self.vec_idx, :].astype(bool)
        self.path[:self.vec_idx, sel_idx] = self.path[:self.vec_idx, alt_state[sel_idx]]
        self.vec_idx += 1
        self.col_idx += 1

        if self.col_idx >= self.mat_width[self.msg_idx]:
            self.prices[:self.state_num // 2] = self.prices[
                                                self.msg[self.msg_idx]::2]  # self.state_num is an even number
            self.prices[self.state_num // 2:] = np.inf
            self.path[:, :self.state_num // 2] = self.path[:, self.msg[self.msg_idx]::2]

            if (self.msg_idx + 1) % self.seg == 0:  # or self.msg_idx + 1 == self.msg_length:
                start_idx = self.vec_idx - sum(self.mat_width[self.msg_idx - self.seg + 1:self.msg_idx + 1])
                min_idx = np.argmin(self.prices[:self.state_num // 2])
                self.stego[start_idx:self.vec_idx] = self.path[start_idx:self.vec_idx, min_idx]
                self.stego_seg = self.stego[start_idx:self.vec_idx]
                self.prices[self.state != min_idx] = np.inf

            self.msg_idx += 1
            self.col_idx = 0

    # ... (STCEmbed 其他方法保持不变) ...
    def has_final_bits(self):
        return self.msg_idx % self.seg > 0

    def roll_back_vec_idx(self):
        self.vec_idx -= self.col_idx

    def embed_final_bits(self):
        shift = self.msg_idx % self.seg
        start_idx = self.vec_idx - sum(self.mat_width[self.msg_idx - shift:self.msg_idx])
        min_idx = np.argmin(self.prices[:self.state_num // 2])
        self.stego[start_idx:self.vec_idx] = self.path[start_idx:self.vec_idx, min_idx]
        self.stego_seg = self.stego[start_idx:self.vec_idx]

    def get_hidden_msg_length(self):
        return self.msg_idx

    def get_stego(self):
        return self.stego[:self.vec_idx]

    def get_stego_seg(self):
        return self.stego_seg

    def is_seg_end(self):
        return self.col_idx == 0 and self.msg_idx > 0 and self.msg_idx % self.seg == 0

    def is_finished(self):
        return self.msg_idx >= self.msg_length


def stc_extract(vector, alpha, msg_length, mat_height):
    inv_alpha = 1 / alpha
    assert inv_alpha >= 1, 'The message cannot be longer than the vector!'
    assert 5 <= mat_height <= 31, 'Submatrix height should be in the range [5, 31]!'

    shorter = math.floor(inv_alpha)
    longer = math.ceil(inv_alpha)
    columns = [get_matrix(shorter, mat_height), get_matrix(longer, mat_height)]

    mat_type, mat_width = arrange_matrices(shorter, longer, msg_length, inv_alpha)

    binmat = [np.unpackbits(columns[0][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1],
              np.unpackbits(columns[1][..., np.newaxis].astype('>u4').view(np.uint8), axis=1)[:, -mat_height:][:, ::-1]]
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


class SentenceEndCriteria(StoppingCriteria):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.current_num_sentences = 0

    def update(self, current_text):
        """根据当前文本更新初始句子数量"""
        self.current_num_sentences = len(sent_tokenize(current_text))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 修正：移除过于严格的断言，因为它无法处理内部的批处理
        # assert input_ids.size(0) == 1

        # 创建一个列表，用于存放批处理中每个序列的“完成状态”
        is_done_list = []

        # 遍历批处理中的每一个序列
        for sequence_ids in input_ids:
            # 单独解码这个序列
            text = self.tokenizer.decode(sequence_ids, skip_special_tokens=True)

            # 对这个解码后的文本进行判断
            # 逻辑不变：当新生成的句子数超过1个时，就认为该序列完成了
            has_new_sentence = len(sent_tokenize(text)) > self.current_num_sentences + 1
            is_done_list.append(has_new_sentence)

        # 将布尔值列表转换为一个与输入在同一设备上的张量
        # StoppingCriteriaList 期望得到一个形状为 (batch_size,) 的布尔张量
        return torch.tensor(is_done_list, device=input_ids.device)


def discard_final_token_in_outputs(outputs):
    return outputs[:, :-1]


def first_upper(s):
    if len(s) == 0:
        return s
    else:
        return s[0].upper() + s[1:]


def clean_text(s):
    punc = set(punctuation) - set('.')
    punc.add("\n")
    newtext = []
    for k, g in groupby(s):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    return ''.join(newtext)


def well_formed_sentence(sent, end_sent=False):
    sent = first_upper(sent)
    sent = sent.replace('  ', ' ')
    sent = sent.replace(' i ', " I ")
    # 移除最后的清理，交由 normalize_sentence 和最终的 clean_text
    return sent


# =========================================================================
# 修复 1：创建统一的文本规范化函数
# =========================================================================
def normalize_sentence(s: str) -> str:
    """
    一个统一的、标准的函数，用于清理和规范化句子。
    所有送入 sent_to_code 的字符串都必须先经过此函数处理。
    """
    if not isinstance(s, str):
        return ""

    # 移除首尾空白
    s = s.strip()

    # 使用您已有的清理函数
    s = well_formed_sentence(s)

    # 确保句子以合适的标点结尾，这对于分句器至关重要
    if s and not any(s.endswith(p) for p in ".?!"):
        s += "."

    return s


# =========================================================================
# 保持原始结构的核心类与函数
# =========================================================================


class LLMSimulator:
    def __init__(self, tokenizer, model, device, debug_mode=False):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.context = []
        self.repetition_penalty = 1.2
        self.debug_mode = debug_mode
        if self.debug_mode:
            print("LLMSimulator is running in DEBUG mode. Batch generations will be printed.")

    # 更正和重构后的 predict 函数
    def predict(self, prompt, bit_num, device, num_candidates=1):
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be a string, got {type(prompt)}: {prompt}")

        torch.cuda.empty_cache()

        # --- 修改点 1: 使用 tokenizer() 调用替代 .encode() ---
        # 这会返回一个字典，包含 input_ids 和 attention_mask
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(self.model.device)

        # --- 修改点 2: 从新的 inputs 对象中获取 prompt 长度 ---
        num_prompt_tokens = inputs.input_ids.shape[1]

        with torch.no_grad():
            sentence_end_criteria = SentenceEndCriteria(self.tokenizer)
            sentence_end_criteria.update(prompt)
            stopping_criteria = StoppingCriteriaList([sentence_end_criteria])

            # --- 修改点 3: 使用 **inputs 将所有参数（包括 attention_mask）传入 ---
            outputs = self.model.generate(
                **inputs,  # 使用 **inputs 替代 input_ids=input_ids
                do_sample=True,
                stopping_criteria=stopping_criteria,
                max_new_tokens=128,
                top_p=0.8,
                temperature=0.9,
                repetition_penalty=1.2,
                min_new_tokens=6,
                num_return_sequences=num_candidates
            )

        # --- 后续处理代码无需修改 ---
        results = []
        for i in range(outputs.shape[0]):
            newly_generated_ids = outputs[i, num_prompt_tokens:]
            newly_generated_text = self.tokenizer.decode(newly_generated_ids, skip_special_tokens=True).strip()

            if len(newly_generated_ids) < 3:
                continue

            processed_text = normalize_sentence(newly_generated_text)
            sentences = sent_tokenize(processed_text)
            first_sentence = sentences[0] if sentences else ""

            if not first_sentence:
                continue

            bitstring = sent_to_code(first_sentence, device, 0.03)
            if bitstring:
                results.append((" " + first_sentence, bitstring))

        # ... (调试和返回逻辑保持不变)
        if self.debug_mode:
            print("\n" + "#" * 80)
            print(f"DEBUG | {time.strftime('%Y-%m-%d %H:%M:%S')} | predict call")
            print(f"PROMPT: '{prompt}'")
            print(f"REQUESTED: {num_candidates} candidates | GENERATED: {len(results)} valid candidates")
            print("-" * 80)
            for i, (sentence, bitstring) in enumerate(results):
                print(f"  [{i + 1:02d}/{len(results):02d}] Sentence: '{sentence.strip()}' | Bitstring: {bitstring}")
            print("#" * 80 + "\n")

        if num_candidates == 1:
            if results:
                return results[0]
            else:
                return (" ", None)
        else:
            return results


class LLMDataProcessing():
    def __init__(self, llm, bit_num, seg, seg_num, Wquality, Went, Wrobust, window_size,device):
        self.llm = llm
        self.bit_num = bit_num
        self.sen_list = []
        self.seg = seg
        self.seg_num = seg_num
        self.stego_sen = ''
        self.first_sen = ''
        self.first_bitstring = ''
        self.Wquality = Wquality
        self.Went = Went
        self.Wrobust = Wrobust
        self.bitstring_list = []
        self.sen_cost = []
        self.previous_id = []
        ssl._create_default_https_context = ssl._create_unverified_context
        self.window_size = window_size
        self.device = device
        self.bitstring_cache = {}
        self.debug_mode = True
        self.ppl_model = AutoModelForCausalLM.from_pretrained('/home/zlm/xuezhou/my_model/opt-2.7b',torch_dtype=torch.bfloat16).to(self.device)
        self.ppl_tokenizer = AutoTokenizer.from_pretrained('/home/zlm/xuezhou/my_model/opt-2.7b')

    def _add_elem(self, elem_list, first_elem, second_elem, idx):
        elem_list.insert(idx + 1, elem_list[idx].copy())
        elem_list[idx].append(first_elem)
        elem_list[idx + 1].append(second_elem)

    def entropy_cost(self, counter, k_categories=4):
        total = sum(counter.values())
        if total == 0: return 1.0
        H = -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)
        # Avoid division by zero if k_categories is 1
        if k_categories <= 1: return 0.0
        H_norm = H / math.log2(k_categories)
        return 1.0 - H_norm

    # MODIFIED FUNCTION
    def evaluate_entropy(self, input, reps=16):
        """
        Calculates entropy cost by generating 'reps' candidates in a single batch call.
        """
        bit_counter = Counter()

        # Key change: Single, efficient call to get all candidates instead of a loop
        candidates = self.llm.predict(input, self.bit_num, self.device, num_candidates=reps)

        for sen, bitstring in candidates:
            if bitstring:  # Ensure bitstring is not None
                if bitstring not in self.bitstring_cache:
                    self.bitstring_cache[bitstring] = sen.strip()
                bit_counter[bitstring] += 1

        entropy_cost = self.entropy_cost(bit_counter, 2 ** self.bit_num)
        return entropy_cost

    def sentence_perplexity(self, text: str, window_size: int = 5) -> float:
        # 1. 使用nltk将完整文本分割成句子
        sentences = sent_tokenize(text)

        # 2. 根据窗口大小，决定用于计算的句子范围
        if len(sentences) > window_size:
            # 如果句子总数大于窗口，只取最后 window_size 个
            target_sentences = sentences[-window_size:]
        else:
            # 否则，使用所有句子
            target_sentences = sentences

        # 3. 将选定的句子重新拼接成用于计算的文本
        text_for_ppl = " ".join(target_sentences)

        if not text_for_ppl.strip():
            return 0.0

        # 4. 计算这个窗口内文本的困惑度 (原有逻辑)
        enc = self.ppl_tokenizer(text_for_ppl, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)

        # 避免输入为空token的错误
        if input_ids.size(1) == 0:
            return 0.0

        with torch.no_grad():
            outputs = self.ppl_model(input_ids, labels=input_ids)
            # 增加对loss为nan或inf的健壮性处理
            if torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                ppl = float('inf')
            else:
                ppl = torch.exp(outputs.loss).item()

        # 返回 1/PPL，并处理 PPL 极大或为0的边界情况
        if ppl == float('inf') or ppl == 0:
            return 0.0
        else:
            return 1 / ppl

    def compute_cost(self, input, sentence):
        global MAX_PPL_COST, MIN_PPL_COST, MAX_entro_COST, MIN_entro_COST
        entropy_cost = self.evaluate_entropy(input)
        full_text = input + sentence
        ppl_cost = self.sentence_perplexity(full_text,self.window_size)
        print(f'entropy cost:{entropy_cost},ppl cost :{ppl_cost}')

        # # 更新最大最小值
        # if entropy_cost > MAX_entro_COST: MAX_entro_COST = entropy_cost
        # if entropy_cost < MIN_entro_COST: MIN_entro_COST = entropy_cost  # 修正了原代码的elif，确保min也能在第一次更新
        if ppl_cost > MAX_PPL_COST: MAX_PPL_COST = ppl_cost
        if ppl_cost < MIN_PPL_COST: MIN_PPL_COST = ppl_cost

        epsilon = 1e-9
        norm_entropy_cost = self.Went * (entropy_cost - MIN_entro_COST) / (MAX_entro_COST - MIN_entro_COST + epsilon)
        norm_ppl_cost = self.Wquality * (ppl_cost - MIN_PPL_COST) / (MAX_PPL_COST - MIN_PPL_COST + epsilon)

        cost = norm_entropy_cost + norm_ppl_cost
        return cost

    def store_data(self, prompt, stego_sens, cover_sens, max_retries=5):
        """
        生成并存储覆盖句子数据。如果生成失败，会重试几次。
        如果所有重试都失败，则抛出异常。
        """
        global MAX_robust_COST,MIN_robust_COST
        current_prompt = prompt + " ".join(stego_sens) + " ".join(cover_sens)

        for attempt in range(max_retries):
            opt_sen, opt_bitstring = self.llm.predict(current_prompt, self.bit_num, self.device, num_candidates=1)

            # 如果成功生成了比特串，就处理并成功返回
            if opt_bitstring:
                self.sen_list.append(opt_sen)
                int_list = list(map(int, opt_bitstring))
                self.bitstring_list.append(int_list)

                cost = self.compute_cost(current_prompt, opt_sen)
                epsilon = 1e-9
                cost_cluster = compute_cost_cosine(opt_sen)
                print(f'cost_cluster:{cost_cluster}')
                if cost_cluster > MAX_robust_COST:
                    MAX_robust_COST = cost_cluster
                elif cost_cluster < MIN_robust_COST:
                    MIN_robust_COST = cost_cluster

                cost_cluster = self.Wrobust * (cost_cluster - MIN_robust_COST) / (
                            MAX_robust_COST - MIN_robust_COST + epsilon)
                cost += cost_cluster
                self.sen_cost.append([cost] * self.bit_num)

                return  # 成功完成，退出函数

            # 如果执行到这里，说明生成失败，打印警告并准备重试
            print(
                f"Warning: Failed to generate a valid sentence/bitstring (Attempt {attempt + 1}/{max_retries}). Retrying...")
            time.sleep(1)  # 等待1秒再重试

        # 如果 for 循环结束都没有成功返回，说明所有重试都失败了
        # 抛出一个明确的异常，中断当前样本的处理
        raise RuntimeError(
            f"FATAL: Failed to generate a valid cover sentence for prompt '{current_prompt[:100]}...' after {max_retries} attempts.")

    def generate_special_sen(self, input_prompt, obj_bitstring, max_attempts=5):
        search_batch_size = 15
        for attempt in range(max_attempts):
            candidates = self.llm.predict(input_prompt, self.bit_num, self.device, num_candidates=search_batch_size)
            for sen, bitstring in candidates:
                if bitstring:
                    # 无论如何，都将新见到的句子存入缓存
                    if bitstring not in self.bitstring_cache:
                        self.bitstring_cache[bitstring] = sen.strip()
                    # 如果生成了目标句子，任务成功结束
                    if bitstring == obj_bitstring:
                        if self.debug_mode:
                            print(
                                f"[DEBUG|generate_special_sen]     > SUCCESS: Generated a new sentence for '{obj_bitstring}'.")
                        return " "+sen

        # --- 阶段二：缓存备胎查找 ---
        if self.debug_mode:
            print(f"[DEBUG|generate_special_sen]   ! Phase 1 FAILED. No new sentence generated.")
            print(f"[DEBUG|generate_special_sen]   Phase 2: Searching for '{obj_bitstring}' in the global cache...")


        print(self.bitstring_cache)
        if obj_bitstring in self.bitstring_cache:
            fallback_sentence = " " + self.bitstring_cache[obj_bitstring]
            if self.debug_mode:
                print(
                    f"[DEBUG|generate_special_sen]     > FALLBACK SUCCESS: Found a pre-existing sentence for '{obj_bitstring}' in cache.")
            return fallback_sentence

        # --- 阶段三：最终失败 ---
        # 如果实时生成和缓存查找都失败了，这是一个无法挽回的错误。
        if self.debug_mode:
            print(f"[DEBUG|generate_special_sen]   ! Phase 2 FAILED. Bitstring '{obj_bitstring}' not found in cache.")

        raise ValueError(
            f"FATAL: Failed to GENERATE a new sentence AND could not FIND any existing sentence for bitstring '{obj_bitstring}'."
        )

    def split_bits(self, bits):
        assert len(bits) % self.bit_num == 0, "bits length must be divisible by bit_num"
        return [bits[i:i + self.bit_num] for i in range(0, len(bits), self.bit_num)]

    def generate_seg_sen(self, input, obj_bitstrings):
        if self.debug_mode:
            print("\n" + "=" * 40 + " [STEP: Embed Stego Segment] " + "=" * 40)
            print(f"[DEBUG|generate_seg_sen] Target Stego Bits: {obj_bitstrings}")

        sen_bit = self.split_bits(obj_bitstrings)
        generated_sentences = []
        regeneration_chain_started = False

        i = 0
        for bit_list in sen_bit:
            current_context_prompt = input + " ".join(generated_sentences)
            bit_str_target = ''.join(map(str, bit_list))
            bit_arr_target = np.asarray(bit_list)
            original_cover_sentence = self.sen_list[-self.seg_num + i]
            original_cover_bits_arr = np.asarray(self.bitstring_list[-self.seg_num + i])
            bits_match = np.array_equal(bit_arr_target, original_cover_bits_arr)

            # --- 核心修改：更新决策逻辑 ---
            if bits_match and not regeneration_chain_started:
                sen = original_cover_sentence
                if self.debug_mode:
                    print(
                        f"[DEBUG|generate_seg_sen]   - Bits '{bit_str_target}': PREFIX CACHE HIT. Using existing sentence.")
            else:
                if not regeneration_chain_started:
                    regeneration_chain_started = True
                # 执行生成
                sen = self.generate_special_sen(current_context_prompt, bit_str_target)
                if sen is None:  # 假设 generate_special_sen 失败时返回 None
                    sen = original_cover_sentence
                    if self.debug_mode:
                        original_bits_str = ''.join(map(str, original_cover_bits_arr))
                        print(
                            f"[DEBUG|generate_seg_sen]     ! FALLBACK: Generation failed. Using original sentence with bits '{original_bits_str}'.")

            generated_sentences.append(sen.strip())
            i += 1

        final_segment = " ".join(generated_sentences)

        if self.debug_mode:
            print(f"[DEBUG|generate_seg_sen] Resulting Stego Segment: '{final_segment}'")
            print("=" * 109)

        return final_segment

    # 在 LLMDataProcessing 类中...

    # def generate_seg_sen(self, input, obj_bitstrings):
    #     if self.debug_mode:
    #         print("\n" + "=" * 40 + " [STEP: Embed Stego Segment] " + "=" * 40)
    #         print(f"[DEBUG|generate_seg_sen] Target Stego Bits: {obj_bitstrings}")
    #
    #     sen_bit = self.split_bits(obj_bitstrings)
    #
    #     # --- 核心修改：使用列表来收集句子，而不是直接拼接字符串 ---
    #     generated_sentences = []
    #
    #     i = 0
    #     for bit_list in sen_bit:
    #         # 每次循环都构建一个独立的 prompt，避免 prompt 过长
    #         current_context_prompt = input + " ".join(generated_sentences)
    #
    #         bit_str_target = ''.join(map(str, bit_list))
    #         bit_arr_target = np.asarray(bit_list)
    #
    #         original_cover_sentence = self.sen_list[-self.seg_num + i]
    #         original_cover_bits_arr = np.asarray(self.bitstring_list[-self.seg_num + i])
    #
    #         if np.array_equal(bit_arr_target, original_cover_bits_arr):
    #             sen = original_cover_sentence
    #             if self.debug_mode:
    #                 print(f"[DEBUG|generate_seg_sen]   - Bits '{bit_str_target}': CACHE HIT. Using existing sentence.")
    #         else:
    #             if self.debug_mode:
    #                 print(
    #                     f"[DEBUG|generate_seg_sen]   - Bits '{bit_str_target}': CACHE MISS. Attempting to generate new sentence...")
    #
    #             # 使用修正后的 prompt
    #             sen = self.generate_special_sen(current_context_prompt, bit_str_target)
    #
    #             if sen is None:
    #                 sen = original_cover_sentence
    #                 if self.debug_mode:
    #                     original_bits_str = ''.join(map(str, original_cover_bits_arr))
    #                     print(
    #                         f"[DEBUG|generate_seg_sen]     ! FALLBACK: Generation failed. Using original sentence with bits '{original_bits_str}'.")
    #
    #         # 将处理好的句子（确保去除首尾多余空格）添加到列表中
    #         generated_sentences.append(sen.strip())
    #         i += 1
    #
    #     # --- 使用 ' '.join() 进行一次性、安全的拼接 ---
    #     final_segment = " ".join(generated_sentences)
    #
    #     if self.debug_mode:
    #         print(f"[DEBUG|generate_seg_sen] Resulting Stego Segment: '{final_segment}'")
    #         print("=" * 109)
    #
    #     return final_segment

    def get_stego_sen(self):
        return self.stego_sen

    def get_sen(self):
        return self.sen_list[-1]

    def get_sen_bitstring(self):
        return self.bitstring_list[-1]

    def get_sen_cost(self):
        return self.sen_cost[-1]


# !!! 关键改动：应用了两个修复的主函数
# def llm_data_hiding(alpha, mat_height, msg, seg, bit_num, prompt, model, tokenizer, seg_num, Wq, We, Wr,device):
#     llm = LLMSimulator(tokenizer, model, device)
#     data_proc = LLMDataProcessing(llm, bit_num, seg, seg_num, Wq, We, Wr,device)
#     stc_embed = STCEmbed(alpha, msg, mat_height, seg)
#
#     print("正在生成一个不含水印的前缀句子...")
#     prefix_sentence, _ = llm.predict(prompt, bit_num, device, num_candidates=1)
#     watermarking_prompt = prompt + prefix_sentence
#     print(f'prefix_sentence:{prefix_sentence}')
#     final_sentence_list = [prefix_sentence.strip()]
#
#     # 这些列表用于在水印循环中存储生成的句子
#     stego_sentences = []
#     cover_sentences = []
#
#     # 其他变量保持原样
#     cover_bits, cover_costs, stego_bits = [], [], []
#
#     while True:
#         data_proc.store_data(watermarking_prompt, stego_sentences, cover_sentences)
#
#         cover_sen = data_proc.get_sen()
#         # 将生成的句子添加到 cover 列表中
#         cover_sentences.append(cover_sen)
#
#         cover_bit = data_proc.get_sen_bitstring()
#         cover_cost = data_proc.get_sen_cost()
#
#         for i in range(len(cover_bit)):
#             stc_embed.embed_bit(cover_bit[i], cover_cost[i])
#             cover_bits.append(cover_bit[i])
#             cover_costs.append(cover_cost[i])
#
#         if stc_embed.is_seg_end():
#             stego_seg = stc_embed.get_stego_seg()
#             # 构造 prompt
#             seg_prompt = prompt + " ".join(stego_sentences)
#             stego_sen_segment = data_proc.generate_seg_sen(seg_prompt, stego_seg)
#             stego_sentences.append(stego_sen_segment)
#             # 清空 cover 列表
#             cover_sentences = []
#
#         if stc_embed.is_finished():
#             break
#
#     stc_embed.roll_back_vec_idx()
#     if stc_embed.has_final_bits():
#         stego_seg = stc_embed.get_stego_seg()
#         seg_prompt = prompt + " ".join(stego_sentences)
#         stego_sen_segment = data_proc.generate_seg_sen(seg_prompt, stego_seg)
#         stego_sentences.append(stego_sen_segment)
#
#     # 修复 2：在所有过程结束后，用安全的方式拼接最终文本
#     final_sentences = prefix_sentence + " ".join(stego_sentences)
#
#     recover_stego = recover_bit(final_sentences, bit_num, device)
#     stego = stc_embed.get_stego()
#
#     return final_sentences, stego, recover_stego

def llm_data_hiding(alpha, mat_height, msg, seg, bit_num, prompt, model, tokenizer, seg_num, Wq, We, Wr, window_size,device):
    llm = LLMSimulator(tokenizer, model, device)
    data_proc = LLMDataProcessing(llm, bit_num, seg, seg_num, Wq, We, Wr,window_size, device)
    stc_embed = STCEmbed(alpha, msg, mat_height, seg)
    final_sentence_list = []


    # cover_sentences 列表现在只用于在单个段落内临时存储
    cover_sentences = []

    while True:
        # --- 修改：seg_prompt 现在基于最新的、已确定的隐写文本构建 ---
        # 注意：这里我们只传递 stego_sentences 部分
        current_embedding_prompt = prompt + " ".join(final_sentence_list)
        data_proc.store_data(current_embedding_prompt, [], cover_sentences)

        cover_sen = data_proc.get_sen()
        cover_sentences.append(cover_sen)

        cover_bit = data_proc.get_sen_bitstring()
        cover_cost = data_proc.get_sen_cost()

        for i in range(len(cover_bit)):
            stc_embed.embed_bit(cover_bit[i], cover_cost[i])

        if stc_embed.is_seg_end():
            stego_seg = stc_embed.get_stego_seg()

            # 构造用于生成隐写段落的 prompt
            seg_prompt = prompt + " ".join(final_sentence_list)
            stego_sen_segment = data_proc.generate_seg_sen(seg_prompt, stego_seg)

            # --- 核心修改：将生成的段落直接添加到主列表中 ---
            final_sentence_list.append(stego_sen_segment.strip())

            # 清空临时 cover 列表，为下一个段落做准备
            cover_sentences = []

        if stc_embed.is_finished():
            break

    # 处理可能剩余的未形成完整段落的比特
    stc_embed.roll_back_vec_idx()
    if stc_embed.has_final_bits():
        stego_seg = stc_embed.get_stego_seg()
        seg_prompt = prompt + " ".join(final_sentence_list)
        stego_sen_segment = data_proc.generate_seg_sen(seg_prompt, stego_seg)
        final_sentence_list.append(stego_sen_segment.strip())

    # --- 核心修改：使用 join 进行一次性、安全的最终拼接 ---
    final_text = " ".join(final_sentence_list)
    print(f'final_text:{final_text}')

    recover_stego = recover_bit(final_text, bit_num, device)
    stego = stc_embed.get_stego()

    return final_text, stego, recover_stego




def recover_bit(text: str, bit_num: int, device):
    print(f'final_text:{text}')
    stego_bit = []
    sentences = sent_tokenize(text)
    print(f'sentences:{sentences}')
    if len(sentences) <= 1:
        # print("警告: 文本中只找到一个或零个句子，无法提取水印。")
        return []
    for sentence in sentences:
        normalized = normalize_sentence(sentence)

        if not normalized:
            continue

        bitstring = sent_to_code(normalized, device, 0.0001)
        print(f'recover_sentence: {normalized}\nrecover_bitstring: {bitstring}')

        if bitstring is not None:
            stego_bit.extend(int(b) for b in bitstring)
        else:
            print(f"Warning: sent_to_code returned None for sentence: '{normalized}'")

    return stego_bit


def load_and_sample_data(file_path, num_samples):
    with open(file_path, "r") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        data = [r for r in obj if "prompt" in r and "natural_text" in r]
    elif isinstance(obj, dict):
        data = [obj] if "prompt" in obj and "natural_text" in obj else []
    else:
        data = []
    if len(data) <= num_samples:
        return data
    return random.sample(data, num_samples)

# def load_and_sample_data(file_path, sample_size):
#     import random
#
#     data = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         for line_num, line in enumerate(f, 1):
#             line = line.strip()
#             if line:
#                 try:
#                     obj = json.loads(line)
#                     data.append(obj)
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing line {line_num}: {e}")
#                     continue
#
#     print(f"Loaded {len(data)} total samples from {file_path}")
#
#     # 随机抽样
#     if sample_size and sample_size < len(data):
#         # 设置随机种子以确保可重现性（可选）
#         random.seed(42)
#         data = random.sample(data, sample_size)
#         print(f"Randomly sampled {sample_size} samples from dataset")
#     elif sample_size and sample_size >= len(data):
#         print(f"Requested {sample_size} samples, but dataset only has {len(data)} samples. Using all data.")
#     else:
#         print("No sampling specified, using all data")
#
#     return data

def save_results_to_file(results_list, output_file):
    if not results_list: return
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_results = json.load(f)
            if not isinstance(existing_results, list):
                existing_results = []
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = []
    else:
        existing_results = []
    existing_results.extend(results_list)
    try:
        with open(output_file, 'w') as f:
            json.dump(existing_results, f, indent=4)
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")


def generate_output_filename(msg_size, alpha, bit_number, seg, wq, we, wr,mat_height,window_size):
    alpha_str = str(alpha).replace('.', '')
    wq_str = str(wq).replace('.', '')
    we_str = str(we).replace('.', '')
    wr_str = str(wr).replace('.', '')
    filename = f"core_cosine_Falcon_C4_{alpha_str}_msg{msg_size}_bit{bit_number}_seg{seg}_wq{wq_str}_we{we_str}_wr{wr_str}_{mat_height}_{window_size}.json"
    return filename


CONFIG = {
    "input_file": "/home/zlm/xuezhou/DTMM/dataset/c4/processed_c4_sen.json",
    "sample_size": 30,
    "model_name": "/home/zlm/xuezhou/my_model/opt-1.3b",
    "max_length": 512,
    "batch_size": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def main():
    # 2. 创建解析器
    parser = argparse.ArgumentParser(description="使用自定义参数运行 LLM 数据隐藏。")

    # 3. 为每个参数添加命令行参数定义
    parser.add_argument('--msg', type=int, default=8, help='要隐藏的随机消息的大小。')
    parser.add_argument('--alpha', type=float, default=0.5, help='相对载荷（嵌入率）。')
    parser.add_argument('--bit_num', type=int, default=4, help='每个句子所代表的比特数。')
    parser.add_argument('--seg', type=int, default=8, help='用于 STC 嵌入的段长度。')
    parser.add_argument('--wq', type=float, default='1', help="Wquality")
    parser.add_argument('--we', type=float, default='1', help="Went")
    parser.add_argument('--wr', type=float, default='1', help="Wrobust")
    parser.add_argument("--h", type=int, default='6', help="mat_height")
    parser.add_argument('--window-size', type=int, default=8, help='用于计算PPL的滑动窗口大小。')

    # 4. 从命令行解析参数
    args = parser.parse_args()

    # 5. 使用解析后的参数，而不是硬编码的值
    device = CONFIG["device"]
    model = AutoModelForCausalLM.from_pretrained(CONFIG["model_name"],torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # 也要更新模型的配置
        model.config.pad_token_id = model.config.eos_token_id
    initialize_resources(
        cc_path="./sent_to_code/data/4_kmeans/cc.pt",
        embedder_path="./sent_to_code/SemStamp-c4-sbert",
        bit_length=args.bit_num,
    )
    np.random.seed(42)

    # 使用从命令行传入的 args
    msg_size = args.msg
    alpha = args.alpha
    bit_number = args.bit_num
    seg = args.seg
    seg_num = int(seg / alpha / bit_number)
    wq = args.wq
    we = args.we
    wr = args.wr
    mat_height = args.h
    window_size = args.window_size
    output_file = generate_output_filename(msg_size, alpha, bit_number, seg, wq, we, wr,mat_height,window_size)


    try:
        samples = load_and_sample_data(CONFIG["input_file"], CONFIG["sample_size"])
    except FileNotFoundError:
        print(f"Error: File {CONFIG['input_file']} not found")
        return
    except PermissionError:
        print(f"Error: Permission denied for file {CONFIG['input_file']}")
        return

    results = []
    batch_size = 5
    processed_cnt = 0

    for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
        torch.cuda.empty_cache()
        prompt = sample["prompt"]
        print(f'prompt: {prompt}')
        msg = np.random.randint(0, 2, size=msg_size, dtype='uint8')
        print(f'msg: {msg}')

        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                if retry_count > 0:
                    msg = np.random.randint(0, 2, size=msg_size, dtype='uint8')
                    print(f"[RETRY {retry_count}/{max_retries}] Regenerating with new message: {msg}")

                start_time = time.time()
                sen, stego, recover_stego = llm_data_hiding(
                    alpha, mat_height, msg, seg, bit_number, prompt, model, tokenizer, seg_num, wq, we, wr, window_size,device
                )
                end_time = time.time()
                success = True

            except ValueError as e:
                error_msg = str(e)
                if "FATAL: Failed to GENERATE" in error_msg:
                    retry_count += 1
                    print(f"[RETRY {retry_count}/{max_retries}] Generation failed: {e}")
                    if retry_count >= max_retries:
                        print(f"[ERROR] Sample {i} failed after {max_retries} retries, skipping...")
                        break
                    torch.cuda.empty_cache()
                    time.sleep(1)
                else:
                    print(f"[ERROR] Sample {i} skipped due to ValueError: {e}")
                    break
            except Exception as e:
                print(f"\n[ERROR] Sample {i} crashed with an unexpected error.")
                # 打印完整的错误堆栈信息
                traceback.print_exc()

        if not success:
            continue

        processed_cnt += 1
        emb_time = end_time - start_time

        # 生成参考句子的逻辑保持不变
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(model.device)
            gen_ids = model.generate(
                **inputs, do_sample=True, top_p=0.8, temperature=0.9,
                repetition_penalty=1.2, min_new_tokens=50, max_new_tokens=128,
            )
        ref_sen = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        results.append({
            "prompt": prompt,
            "generated_sentence": sen,
            "reference_sen": ref_sen,
            "embedding_time": emb_time,
            "seg": seg,
            "alpha": alpha,
            "message": msg.tolist(),
            "stego": stego.tolist(),
            "recover_stego": recover_stego,
            "msg_size": msg_size
        })


        if processed_cnt % batch_size == 0:
            save_results_to_file(results, output_file)
            print(f"Saved {processed_cnt} samples to {output_file}")
            results.clear()

    if results:
        save_results_to_file(results, output_file)
        print(f"Saved remaining {len(results)} samples to {output_file}")




if __name__ == '__main__':
    main()
