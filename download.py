# Load model directly
# set HF_ENDPOINT by python
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

model_id = 'AbeHou/SemStamp-c4-sbert'

snapshot_download(
    repo_id=model_id,
    max_workers=8,
    ignore_patterns=["tf*",  "flax*"],
    resume_download=True,
    local_dir="./sent_to_code/SemStamp-c4-sbert"
)



from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
model = AutoModel.from_pretrained(model_id, local_files_only=True)

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)

