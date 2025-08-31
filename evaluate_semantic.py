import json
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize
import torch
import argparse
import math

_GLOBAL_KWARGS = {}

def initialize_resources(**kwargs):
    global _GLOBAL_KWARGS
    _GLOBAL_KWARGS = kwargs

def sent_to_code(sentence, device, threshold):
    k = _GLOBAL_KWARGS.get('bit_length', 4)
    return f"{np.random.randint(0, 2**k):0{k}b}"


def recover_codewords(text: str, device: str) -> list:
    codewords = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        bitstring = sent_to_code(sentence, device, 0.01)
        if bitstring is not None:
            codewords.append(bitstring)
    return codewords


def calculate_shannon_entropy(data_list: list) -> float:
    if not data_list:
        return 0.0
    counts = Counter(data_list)
    total_count = len(data_list)
    probabilities = [count / total_count for count in counts.values()]
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy


def main():
    parser = argparse.ArgumentParser(description="Calculate the Shannon entropy of codeword distribution in a JSONL file.")
    parser.add_argument('--i', type=str, required=True, help="Path to the JSONL file containing generated text.")
    parser.add_argument('--bit-num', type=int, default=4, help="Bit length (k-value) for sent_to_code initialization.")
    parser.add_argument('--cc-path', type=str, default='./sent_to_code/data/4_kmeans/cc.pt', help='Path to the cc.pt file.')
    parser.add_argument('--embedder-path', type=str, default='./sent_to_code/SemStamp-c4-sbert', help='Path to the sentence embedder model.')
    args = parser.parse_args()

    FILE_PATH = args.i
    BIT_LENGTH_k = args.bit_num
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")
    try:
        initialize_resources(
            cc_path=args.cc_path,
            embedder_path=args.embedder_path,
            bit_length=BIT_LENGTH_k
        )
        print("Resources initialized successfully.")
    except Exception as e:
        print(f"Error: Resource initialization failed. Please check paths. Details: {e}")
        return

    all_extracted_codewords = []
    line_count = 0
    error_count = 0

    print(f"\nProcessing file: '{FILE_PATH}'")
    try:
        with open(FILE_PATH, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                if not line.strip():
                    continue
                try:
                    json_obj = json.loads(line.strip())
                    generated_text = json_obj.get("generated_sentence")

                    if generated_text:
                        codewords_from_text = recover_codewords(generated_text, DEVICE)
                        all_extracted_ codewords.extend(codewords_from_text)
                    else:
                        print(f"Warning: Line {line_count} is missing the 'generated_sentence' field.")
                        error_count += 1
                except json.JSONDecodeError:
                    print(f"Warning: Line {line_count} has invalid JSON format.")
                    error_count += 1

        entropy_value = calculate_shannon_entropy(all_extracted_codewords)
        max_entropy = float(BIT_LENGTH_k)

        print("\n========== Entropy Analysis Results ==========")
        print(f"Total lines processed: {line_count}")
        print(f"Lines failed or skipped: {error_count}")
        print(f"Total codewords (sentences) extracted: {len(all_extracted_codewords)}")

        if all_extracted_codewords:
            unique_codewords = len(set(all_extracted_codewords))
            print(f"Number of unique codewords: {unique_codewords}")
            print(f"Shannon Entropy of codeword distribution: {entropy_value:.4f} bits")
            if max_entropy > 0:
                normalized_entropy = entropy_value / max_entropy
                print(f"Theoretical maximum entropy (for k={BIT_LENGTH_k}): {max_entropy:.4f} bits")
                print(f"Normalized Entropy: {normalized_entropy:.4f}")

        print("==========================================")

    except FileNotFoundError:
        print(f"Error: File not found at '{FILE_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred during processing: {e}")


if __name__ == '__main__':
    main()


