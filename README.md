## 📂 Project Structure

- **DMW/dataset/**  
  Contains datasets included C4 dataset and OpenGen dataset.  

- **DMW/sent_to_code/**  
  Data preprocessing and semantic encoding module, mapping sentences into bitstrings and Calculate the euclidean metric.  

- **generate_core.py**  
  Core script for watermark embedding.  

- **transform.py**  
  Data conversion utility that normalizes JSON files into JSONL format.  

- **evaluate.py**  
  Calculate the PPL of the results.  

- **evaluate_attack.py**  
  Evaluation under adversarial attacks such as match rate and bit accuracy.  

- **evaluate_robustness.py**  
  Script that applies adversarial attacks to generated text and produces attacked result files.
  

- **evaluate_semantic.py**  
  Semantic quality evaluation (Semantic Entropy).  

- **evaluate_success.py**  
  Calculate the match rate and bit accuracy.  
  

---

Before running any scripts, please download the following Hugging Face model:

Model: AbeHou/SemStamp-c4-sbert

Download the model folder and place it under:

./sent_to_code/


Your directory structure should look like:

sent_to_code/
    ├── SemStamp-c4-sbert/
    ├── data/
    ├── sent_to_code.py
    └── ...


This model is required for the semantic encoding and bitstring projection modules.


**Word Insertion Attack Note**.
For the word insertion adversarial attack, we use the pretrained masked language model **bert-base-uncased** to generate context-aware word insertions. Please download this model from Hugging Face and specify its local path in the corresponding attack implementation


## ⚙️ Environment Setup

It is recommended to use Conda:

```bash
conda create -n dmw python=3.10 -y
conda activate dmw

pip install -r requirements.txt


python generate_core.py \
  --input-file dataset/openGen/processed_OpenGen.jsonl \
  --sample-size 100 \
  --model-path path_to_your_model \
  --ppl-model-path path_yo_your_model \
  --stc-matrix-path stc_matrix.npy \
  --cc-path sent_to_code/data/4_kmeans/cc.pt \
  --embedder-path path_to_your_model \
  --msg 8 --alpha 0.5 --bit_num 4 --seg 8 \
  --wq 1.0 --we 1.0 --wr 1.0 \
  --h 6 --window-size 8 --device cuda
(For the embedder-path,we choose the SemStamp-c4-sbert)

python transform.py results.json

python evaluate.py --i results.json

python evaluate_success.py \
--i results.json \
--bit-num 4 \
--cc-path sent_to_code/data/4_kmeans/cc.pt \
--embedder-path path_to_your_model  \
--h your_h  \
--seg your_seg

python evaluate_robustness.py -i results.json

