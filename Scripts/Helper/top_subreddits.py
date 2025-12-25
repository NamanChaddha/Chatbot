import json
from collections import Counter
from pathlib import Path

def get_top_subreddits(file_path, top_n=100):
    counts=Counter()
    with open(file_path,"r",encoding="utf-8") as file:
        for line in file:
            try:
                data=json.loads(line)
                subreddit=data.get("subreddit")
                if subreddit is not None:
                    counts[subreddit]+=1
            except json.JSONDecodeError:
                continue
    return counts.most_common(top_n)

file_path=Path(__file__).parent.parent/"Reddit_train.jsonl"
top_100=get_top_subreddits(file_path)
for i,(subreddit,count) in enumerate(top_100,1):
    print(f"{i}.r/{subreddit}-{count}")
