import json
import time
import random
import nltk
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from nltk import pos_tag, sent_tokenize
from nltk.corpus import wordnet
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM


class TextEditor:
    """Base class for text editing."""

    def __init__(self) -> None:
        pass

    def edit(self, text: str, reference=None):
        return text


class GPTParaphraser(TextEditor):

    def __init__(self, openai_model: str = "gpt-4o-mini", prompt: str = None) -> None:
        """
        Initialize the GPT paraphraser.

        Parameters:
            openai_model (str): The OpenAI model to use for paraphrasing.
            prompt (str): The prompt to use for paraphrasing.
        """
        self.openai_model = openai_model
        self.prompt = prompt or "Please rewrite the following text and keep its original meaning: "
        self.client = OpenAI(
            base_url="",
            api_key=""
        )

    def edit(self, text):
        """Paraphrase text using GPT."""
        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for paraphrasing text."},
                    {"role": "user", "content": f"Rewrite the following text:\n{text}"}
                ],
                temperature=0.9,
            )
            generated_text = completion.choices[0].message.content
            return generated_text
        except Exception as e:
            print(f"GPT paraphrasing failed: {e}")
            return text


class WordDeletion(TextEditor):
    """Delete words randomly from the text."""

    def __init__(self, ratio: float) -> None:
        """
        Initialize the word deletion editor.

        Parameters:
            ratio (float): The ratio of words to delete.
        """
        self.ratio = ratio

    def edit(self, text: str, reference=None):
        """Delete words randomly from the text."""
        # Handle empty string input
        if not text:
            return text

        # Split the text into words and randomly delete each word based on the ratio
        word_list = text.split()
        edited_words = [word for word in word_list if random.random() >= self.ratio]

        # Join the words back into a single string
        deleted_text = ' '.join(edited_words)
        return deleted_text


class SynonymSubstitution(TextEditor):
    """Randomly replace words with synonyms from WordNet."""

    def __init__(self, ratio: float) -> None:
        """
        Initialize the synonym substitution editor.

        Parameters:
            ratio (float): The ratio of words to replace.
        """
        self.ratio = ratio
        # Ensure wordnet data is available
        nltk.download('wordnet', quiet=True)

    def edit(self, text: str, reference=None):
        """Randomly replace words with synonyms from WordNet."""
        words = text.split()
        num_words = len(words)

        # Dictionary to cache synonyms for words
        word_synonyms = {}

        # First pass: Identify replaceable words and cache their synonyms
        replaceable_indices = []
        for i, word in enumerate(words):
            if word not in word_synonyms:
                synonyms = [syn for syn in wordnet.synsets(word) if len(syn.lemmas()) > 1]
                word_synonyms[word] = synonyms
            if word_synonyms[word]:
                replaceable_indices.append(i)

        # Calculate the number of words to replace
        num_to_replace = min(int(self.ratio * num_words), len(replaceable_indices))

        # Randomly select words to replace
        if num_to_replace > 0:
            indices_to_replace = random.sample(replaceable_indices, num_to_replace)

            # Perform replacement
            for i in indices_to_replace:
                synonyms = word_synonyms[words[i]]
                chosen_syn = random.choice(synonyms)
                new_word = random.choice(chosen_syn.lemmas()[1:]).name().replace('_', ' ')
                words[i] = new_word

        # Join the words back into a single string
        replaced_text = ' '.join(words)
        return replaced_text


class WordInsertion(TextEditor):
    """
    Randomly insert contextually relevant words into the text using a masked language model.
    """

    def __init__(self, ratio: float, model_name: str = '', device: str = 'cuda'):
        """
        Initialize the word insertion editor.

        Parameters:
            ratio (float): The ratio of words to insert relative to the text length.
            model_name (str): The pre-trained masked language model to use.
            device (str): The device to run the model on ('cuda' or 'cpu').
        """
        super().__init__()
        self.ratio = ratio
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Initialize tokenizer and model from Hugging Face
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('')
            self.model = AutoModelForMaskedLM.from_pretrained('')
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode
            print("✅ WordInsertion model loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model '{model_name}'. Please ensure you have an internet connection.")
            print(f"   Error: {e}")
            raise

    def edit(self, text: str, reference=None):
        """
        Insert words into the text based on the specified ratio.
        """
        sentences = sent_tokenize(text)
        edited_sentences = []

        for sentence in sentences:
            words = sentence.split()
            if not words:
                continue

            # Calculate the number of words to insert in this sentence
            num_to_insert = int(len(words) * self.ratio)

            for _ in range(num_to_insert):
                if len(words) < 2:  # Need at least two words to insert between
                    break

                # 1. Identify a random insertion point
                insert_pos = random.randint(1, len(words) - 1)

                # 2. Generate candidate words using MLM
                # Create the sentence with a MASK token
                masked_sentence_parts = words[:insert_pos] + [self.tokenizer.mask_token] + words[insert_pos:]
                masked_sentence = " ".join(masked_sentence_parts)

                # Tokenize and get model predictions
                inputs = self.tokenizer(masked_sentence, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    logits = self.model(**inputs).logits

                # Find the masked token's position and get top predictions
                mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

                # Get top 5 predicted token IDs
                predicted_token_ids = torch.topk(logits[0, mask_token_index[0]], k=5).indices.tolist()

                # Decode tokens, filtering out subwords and special tokens
                candidate_words = []
                for token_id in predicted_token_ids:
                    word = self.tokenizer.decode(token_id).strip()
                    if word.isalnum():  # Simple filter for valid words
                        candidate_words.append(word)

                if not candidate_words:
                    continue  # Skip if no suitable word was found

                # 3. Select and insert a word
                word_to_insert = random.choice(candidate_words)
                words.insert(insert_pos, word_to_insert)

            edited_sentences.append(" ".join(words))

        return " ".join(edited_sentences)


def recover_bit(text: str, bit_num: int, device: str, sent_to_code_func):
    """Recover steganographic bits from text."""
    stego_bit = []
    for sentence in sent_tokenize(text):
        sentence = sentence.strip()
        if not sentence:
            continue

        bitstring = sent_to_code_func(sentence, device, 0.0001)
        if bitstring is None:
            return None

        stego_bit.extend(int(b) for b in bitstring)

    return stego_bit


# ============================================================================
# STEP 3: ROBUSTNESS EVALUATION FUNCTIONS
# ============================================================================

def calculate_bit_accuracy(a: Optional[List[int]], b: Optional[List[int]], k: int) -> Tuple[int, int]:
    total_bits = k
    if a is None or b is None or k == 0:
        return 0, total_bits

    original_prefix = a[:k]

    compare_len = min(len(original_prefix), len(b))

    matching_bits = sum(1 for i in range(compare_len) if original_prefix[i] == b[i])

    return matching_bits, total_bits

def str2bits(bit_str: Optional[str]) -> Optional[List[int]]:
    if bit_str is None:
        return None
    bit_str = bit_str.strip()
    if not bit_str:
        return None
    return [int(x) for x in bit_str.strip("[] ").split(",") if x.strip()]


def decode_bits(bits: List[int]) -> List[int]:

    return bits


def prefix_equal(a: Optional[List[int]], b: Optional[List[int]], k: int) -> bool:
    """Compare if a[:k] equals b[:k]"""
    return a is not None and b is not None and a[:k] == b[:k]


def evaluate_attack_robustness(records: List[Dict], field: str) -> Dict[str, float]:
    message_correct = total_records = 0
    total_bits = matching_bits = 0

    for rec in records:
        total_records += 1
        stego_bits = decode_bits(str2bits(rec.get("stego")))
        attack_bits = decode_bits(str2bits(rec.get(field)))
        k = int(rec.get("msg_size", 0))
        if prefix_equal(stego_bits, attack_bits, k):
            message_correct += 1
        match_count, total_count = calculate_bit_accuracy(stego_bits, attack_bits, k)
        matching_bits += match_count
        total_bits += total_count
    message_acc = message_correct / total_records if total_records else 0.0
    bit_acc = matching_bits / total_bits if total_bits else 0.0

    return {
        "message_correct": message_correct,
        "bit_correct": matching_bits,
        "total_records": total_records,
        "total_bits": total_bits,
        "message_accuracy": message_acc,
        "bit_accuracy": bit_acc
    }


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class SteganographyRobustnessEvaluator:
    """Integrated pipeline for steganography robustness evaluation."""

    def __init__(self,
                 deletion_ratio: float = 0.1,
                 substitution_ratio: float = 0.1,
                 insertion_ratio: float = 0.1,
                 device: str = "cuda"):
        """
        Initialize the evaluator.

        Parameters:
            deletion_ratio: Ratio of words to delete
            substitution_ratio: Ratio of words to substitute with synonyms
            device: Device for steganographic operations
        """
        self.device = device
        self.substitution_attack = SynonymSubstitution(ratio=substitution_ratio)
        self.deletion_attack = WordDeletion(ratio=deletion_ratio)
        self.insertion_attack = WordInsertion(ratio=insertion_ratio, device=device)
        self.para_attack = GPTParaphraser(
            openai_model='gpt-4o-mini',
            prompt='Please rewrite the following text and keep its original meaning: '
        )

        self.stego_initialized = False
        self.sent_to_code_func = None

    def initialize_steganography(self,
                                 cc_path: str = "./sent_to_code/data/4_kmeans/cc.pt",
                                 embedder_path: str = "./sent_to_code/SemStamp-c4-sbert",
                                 bit_length: int = 4):
        """Initialize steganography resources."""
        try:
            from sent_to_code.sent_to_code import initialize_resources, sent_to_code
            initialize_resources(
                cc_path=cc_path,
                embedder_path=embedder_path,
                bit_length=bit_length
            )
            self.sent_to_code_func = sent_to_code
            self.stego_initialized = True
            print("✅ Steganography resources initialized successfully")
        except ImportError as e:
            print(f"❌ Failed to import steganography modules: {e}")
            print("Please ensure sent_to_code package is installed and accessible")
            self.stego_initialized = False
        except Exception as e:
            print(f"❌ Failed to initialize steganography resources: {e}")
            self.stego_initialized = False

    def apply_attacks(self, input_path: str, output_path: str, num_samples: Optional[int] = None) -> None:
        in_path, out_path = Path(input_path), Path(output_path)
        records: list[dict]
        try:
            with in_path.open("r", encoding="utf-8-sig") as f:
                records = json.load(f)
            print("  - Successfully loaded as a standard JSON array file.")
        except json.JSONDecodeError:
            print("  - Could not parse as standard JSON, attempting to read as JSON Lines format.")
            records = []
            with in_path.open("r", encoding="utf-8-sig") as f:
                for line in f:
                    if line.strip(): records.append(json.loads(line))

        if num_samples is not None and num_samples > 0:
            if len(records) > num_samples:
                print(f"  - Applying sample limit: Processing the first {num_samples} of {len(records)} records.")
                records = records[:num_samples]
            else:
                print(
                    f"  - Sample limit ({num_samples}) is greater than or equal to total records ({len(records)}). Processing all records.")

        for i, item in enumerate(records):
            if i % 10 == 0: print(f"  Processing record {i + 1}/{len(records)}")
            gen_text = item.get("generated_sentence", "")
            if gen_text:
                item["delete_sen"] = self.deletion_attack.edit(gen_text)
                item["sub_sen"] = self.substitution_attack.edit(gen_text)
                item["insert_sen"] = self.insertion_attack.edit(gen_text)
                item['para_sen'] = self.para_attack.edit(gen_text)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"✅ Step 1 completed. Generated {len(records)} attacked records: {out_path.resolve()}")

    def recover_steganographic_bits(self, input_path: str, output_path: str) -> None:
        """Recover steganographic bits from attacked texts."""
        print("🔄 Step 2: Recovering steganographic bits...")

        if not self.stego_initialized:
            raise RuntimeError("Steganography resources not initialized. Call initialize_steganography() first.")

        in_path, out_path = Path(input_path), Path(output_path)
        raw = in_path.read_text("utf-8").strip()

        records = (
            json.loads(raw) if raw.startswith("[")
            else [json.loads(line) for line in raw.splitlines() if line]
        )

        for i, rec in enumerate(records):
            if i % 10 == 0:
                print(f"  Processing record {i + 1}/{len(records)}")

            bit_num = rec.get("msg_size")

            for fld, new_fld in [
                ("delete_sen", "delete_stego"),
                ("sub_sen", "sub_stego"),
                ("insert_sen", "insert_stego"),
                ("para_sen", "para_stego"),
            ]:
                txt = rec.get(fld, "")
                if not txt:
                    rec[new_fld] = None
                    continue

                bits = recover_bit(txt, bit_num, self.device, self.sent_to_code_func)

                if bits is None:
                    rec[new_fld] = None
                else:
                    rec[new_fld] = "[" + ",".join(map(str, bits)) + "]"

        # Save results
        out_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ Step 2 completed. Results saved: {out_path.resolve()}")


    def evaluate_robustness(self, input_path: str) -> Dict[str, Dict[str, float]]:
        print("🔄 Step 3: Evaluating robustness...")

        data_path = Path(input_path)
        if not data_path.is_file():
            raise FileNotFoundError(f"File not found: {data_path}")

        records = json.loads(data_path.read_text(encoding="utf-8"))

        test_fields = ["delete_stego", "sub_stego", "insert_stego","para_stego"]
        results = {}

        print(f"\n📊 Robustness Evaluation Results")
        print(f"File: {data_path}")
        print("-" * 65)
        print(f"{'Attack Type':<15} | {'Message Accuracy':<20} | {'Bit Accuracy':<20}")
        print("-" * 65)

        for fld in test_fields:
            stats = evaluate_attack_robustness(records, fld)
            results[fld] = stats

            msg_acc_str = f"{stats['message_accuracy']:.2%} ({stats['message_correct']}/{stats['total_records']})"
            bit_acc_str = f"{stats['bit_accuracy']:.2%} ({stats['bit_correct']}/{stats['total_bits']})"

            print(f"{fld:<15} | {msg_acc_str:<20} | {bit_acc_str:<20}")

        print("-" * 65)
        print("✅ Step 3 completed.")
        return results

    def run_full_pipeline(self, input_file: str, cc_path: str, embedder_path: str, bit_length: int,
                          keep_intermediate: bool = True, num_samples: Optional[int] = None) -> Dict[
        str, Dict[str, float]]:
        print("🚀 Starting Steganography Robustness Evaluation Pipeline")
        print("=" * 60)
        self.initialize_steganography(cc_path, embedder_path, bit_length)
        base_name = Path(input_file).stem
        attacked_file = f"{base_name}_attacked.json"
        final_file = f"{base_name}_with_bits.json"
        try:
            self.apply_attacks(input_file, attacked_file, num_samples=num_samples)
            self.recover_steganographic_bits(attacked_file, final_file)
            results = self.evaluate_robustness(final_file)
            if not keep_intermediate:
                Path(attacked_file).unlink(missing_ok=True)
                print(f"🗑️  Removed intermediate file: {attacked_file}")
            print("\n🎉 Pipeline completed successfully!")
            return results
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            Path(attacked_file).unlink(missing_ok=True)
            Path(final_file).unlink(missing_ok=True)
            raise

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Integrated steganography robustness evaluation pipeline")
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON file with generated_sentence field")

    parser.add_argument(
        "--num-samples",'--n',
        type=int,
        default=None,
        help="The maximum number of samples to process from the input file."
    )

    parser.add_argument("--cc-path", default="./sent_to_code/data/4_kmeans/cc.pt")
    parser.add_argument("--embedder-path", default="./sent_to_code/SemStamp-c4-sbert")
    parser.add_argument("--bit-length", type=int, default=4)
    parser.add_argument("--deletion-ratio", type=float, default=0.1)
    parser.add_argument("--substitution-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keep-intermediate", action="store_true")
    parser.add_argument("--step", choices=["attack", "recover", "evaluate", "full"], default="full")
    args = parser.parse_args()

    evaluator = SteganographyRobustnessEvaluator(
        device=args.device
    )

    if args.step == "full":
        results = evaluator.run_full_pipeline(
            input_file=args.input,
            cc_path=args.cc_path,
            embedder_path=args.embedder_path,
            bit_length=args.bit_length,
            keep_intermediate=args.keep_intermediate,
            num_samples=args.num_samples
        )
        summary_file = f"{Path(args.input).stem}_robustness_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"📋 Summary saved to: {summary_file}")

    elif args.step == "attack":
        output_file = f"{Path(args.input).stem}_attacked.json"
        evaluator.apply_attacks(args.input, output_file, num_samples=args.num_samples)


    elif args.step == "recover":
        input_file = f"{Path(args.input).stem}_attacked.json"
        output_file = f"{Path(args.input).stem}_with_bits.json"
        evaluator.initialize_steganography(args.cc_path, args.embedder_path, args.bit_length)
        evaluator.recover_steganographic_bits(input_file, output_file)

    elif args.step == "evaluate":
        input_file = f"{Path(args.input).stem}_with_bits.json"
        results = evaluator.evaluate_robustness(input_file)


if __name__ == "__main__":
    main()


