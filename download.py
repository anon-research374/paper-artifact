# Load model directly
# set HF_ENDPOINT by python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

model_id = 'BAAI/bge-small-zh-v1.5'

snapshot_download(
    repo_id=model_id,
    max_workers=8,
    ignore_patterns=["tf*",  "flax*"],
    resume_download=True,
    local_dir="./my_model/bge-small-zh-v1.5"
)



from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModel.from_pretrained(model_id, local_files_only=True)

text = "Replace me by any text you'd like."sssssssss
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
