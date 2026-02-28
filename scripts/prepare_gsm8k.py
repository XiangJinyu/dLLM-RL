"""Prepare GSM8K dataset from parquet to JSON format."""

import json, os
import pyarrow.parquet as pq


def extract_answer(answer_text):
    lines = answer_text.strip().split("\n")
    last = lines[-1]
    if "####" in last:
        return last.split("####")[-1].strip().replace(",", "")
    return last.strip()


os.makedirs("/workspace/data", exist_ok=True)

for split in ["train", "test"]:
    path = f"/workspace/data/gsm8k_raw/main/{split}-00000-of-00001.parquet"
    table = pq.read_table(path)
    items = table.to_pydict()
    data = []
    for i in range(len(items["question"])):
        data.append(
            {
                "question": items["question"][i],
                "ground_truth_answer": extract_answer(items["answer"][i]),
            }
        )
    out = f"/workspace/data/gsm8k_{split}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    q0 = data[0]["question"][:80]
    a0 = data[0]["ground_truth_answer"]
    print(f"GSM8K {split}: {len(data)} samples")
    print(f"  Example Q: {q0}...")
    print(f"  Example A: {a0}")
