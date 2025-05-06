import csv
import json
import os

folder_path = "../data/eng"

corpus = []

def get_replies(comments):
    text = ""
    for comment in comments:
        text += comment['body']
        text += get_replies(comment['replies'])
        
    return text

import re

BOT_MESSAGES =[
    "Your post is getting popular and we just featured it on our Discord! "
    "[Come check it out!]()You've also been given a special flair for your contribution. "
    "We appreciate your post!*I am a bot and this action was performed automatically.*"    
]

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)

            title = data['data']['submission_metadata']['title']
            
            text = get_replies(data['data']['comments']).lstrip().replace("\n", "").replace("\r", "")
            
            text = re.sub(r'http[^\s)\]]+', '', text)
            
            for msg in BOT_MESSAGES:
                text = text.replace(msg, "")

            text = re.sub(r'Hey.*?concerns\.\*', '', text, flags=re.DOTALL)
            
            corpus.append({"title" : title, "text" : text})

csv_filename = "../data/eng_corpus.csv"

fieldnames = corpus[0].keys() if corpus else []

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(corpus)