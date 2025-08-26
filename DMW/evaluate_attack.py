import json
import numpy as np
import math
import time
import os
from nltk.tokenize import sent_tokenize
import torch
import argparse
from typing import List, Dict, Tuple, Optional

from sent_to_code.sent_to_code import initialize_resources, sent_to_code


def get_matrix(width: int, height: int, stc_matrix_path: str) -> np.ndarray:
    if 2 <= width <= 20 and 7 <= height <= 12:
        matrices = np.load(stc_matrix_path)
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


def stc_extract(vector: List[int], alpha: float, msg_length: int, mat_height: int, stc_matrix_path: str) -> np.ndarray:
    """Extracts the STC-encoded message from a binary vector."""
    inv_alpha = 1 / alpha
    shorter, longer = math.floor(inv_alpha), math.ceil(inv_alpha)
    columns = [get_matrix(shorter, mat_height, stc_matrix_path), get_matrix(longer, mat_height, stc_matrix_path)]
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
    """Recovers the bitstream (vector) from text."""
    stego_bit = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence: continue

        bitstring = sent_to_code(sentence, device, 0.0001)
        if bitstring is None:
            return None
        stego_bit.extend(int(b) for b in bitstring)
    return stego_bit


def evaluate_attacks_on_record(json_obj: Dict, bit_length: int, mat_height: int, device: str, stc_matrix_path: str) -> Dict[str, Tuple[int, int]]:
    """
    (Corrected) Evaluates all four attacks on a single JSON record.
    """
    results = {}
    attack_fields = {
        "Deletion": "delete_sen",
        "Substitution": "sub_sen",
        "Insertion": "insert_sen",
        "Paraphrase": "para_sen"
    }

    try:
        original_message_str = json_obj.get("message")
        alpha = json_obj.get("alpha")
        msg_length = json_obj.get("msg_size")

        if not all([original_message_str, alpha is not None, msg_length is not None]):
            print(f"  - Warning: Skipping record due to missing 'message', 'alpha', or 'msg_size'.")
            return {name: (0, 0) for name in attack_fields}

        try:
            original_message_list = json.loads(original_message_str)
            original_message = np.array(original_message_list)
        except (json.JSONDecodeError, TypeError):
            print(f"  - Warning: Skipping record due to incorrect 'message' format: {original_message_str}")
            return {name: (0, 0) for name in attack_fields}

        total_bits = len(original_message)

        for attack_name, field_name in attack_fields.items():
            attacked_text = json_obj.get(field_name)

            if not attacked_text:
                print(f"  - {attack_name}: Warning - Attacked text field '{field_name}' not found.")
                results[attack_name] = (0, total_bits)
                continue

            vector = recover_bit(attacked_text, bit_length, device)

            if vector is None:
                print(f"  - {attack_name}: ❌ Recovery failed (sent_to_code returned None).")
                results[attack_name] = (0, total_bits)
                continue

            extracted_message = stc_extract(vector, alpha, msg_length, mat_height, stc_matrix_path)

            compare_len = min(len(original_message), len(extracted_message))
            matching_bits = np.sum(original_message[:compare_len] == extracted_message[:compare_len])

            results[attack_name] = (matching_bits, total_bits)

            if matching_bits == total_bits and len(original_message) == len(extracted_message):
                print(f"  - {attack_name}: ✅ Message perfectly recovered.")
            else:
                print(f"  - {attack_name}: ❌ Message mismatch ({matching_bits}/{total_bits} bits matched).")

    except Exception as e:
        print(f"  - Error: An unexpected error occurred while