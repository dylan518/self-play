import json
from datasets import Dataset, DatasetDict
import argparse
import json
import os
STORAGE_PATH = os.getenv("STORAGE_PATH")
HUGGINGFACENAME = os.getenv("HUGGINGFACENAME")
PUSH_HF = os.getenv("RZERO_PUSH_HF", "0") == "1"
print(STORAGE_PATH)
if PUSH_HF:
    from huggingface_hub import login
    with open('tokens.json', 'r') as f:
        token = json.load(f)['huggingface']
    login(token=token)
parser = argparse.ArgumentParser()
parser.add_argument("--repo_name", type=str, default="")
parser.add_argument("--max_score", type=float, default=0.7)
parser.add_argument("--min_score", type=float, default=0.3)
parser.add_argument("--experiment_name", type=str, default="Qwen_Qwen3-4B-Base_all")
args = parser.parse_args()

datas= []
for i in range(8):
    try:
        with open(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json', 'r') as f:
            data = json.load(f)
            datas.extend(data)
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue


for i in range(8):
    try:
        os.remove(f'{STORAGE_PATH}/generated_question/{args.experiment_name}_{i}_results.json')
    except:
        print(f"File {args.experiment_name}_{i}_results.json not found")
        continue

# Persist EVERYTHING from generation before any filtering/deletion (artifact trail).
art_dir = f"{STORAGE_PATH}/artifacts/{args.experiment_name}"
os.makedirs(art_dir, exist_ok=True)
with open(f"{art_dir}/all_questions.jsonl", "w", encoding="utf-8") as f:
    for d in datas:
        f.write(json.dumps(d) + "\n")
print(f"artifact: {art_dir}/all_questions.jsonl ({len(datas)} rows)")

scores = [data['score'] for data in datas]
#  print the distribution of scores
import matplotlib.pyplot as plt
plt.hist(scores, bins=11)
plt.savefig('scores_distribution.png')

#count the number  of score between 0.2 and 0.8 
if not args.repo_name == "":
    filtered_datas = [{'problem':data['question'],'answer':data['answer'],'score':data['score']} for data in datas if data['score'] >= args.min_score and data['score'] <= args.max_score and data['answer'] != '' and data['answer']!= 'None']
    print(len(filtered_datas))
    train_dataset = Dataset.from_list(filtered_datas)
    # Always save locally; verl reads the parquet directly (no Hub round-trip needed).
    local_path = f"{STORAGE_PATH}/generated_question/{args.experiment_name}.parquet"
    train_dataset.to_parquet(local_path)
    print(f"saved {len(filtered_datas)} rows to {local_path}")
    if PUSH_HF:
        dataset_dict = {"train": train_dataset}
        config_name = f"{args.experiment_name}"
        dataset = DatasetDict(dataset_dict)
        dataset.push_to_hub(f"{HUGGINGFACENAME}/{args.repo_name}",private=True,config_name=config_name)







