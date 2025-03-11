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

for file in os.listdir(folder_path):
    if file.endswith(".json"):
        with open(os.path.join(folder_path, file), 'r') as f:
            data = json.load(f)

            title = data['data']['submission_metadata']['title']
            
            text = get_replies(data['data']['comments'])
            
            corpus.append({"title" : title, "text" : text})

csv_filename = "../data/eng_corpus.csv"

fieldnames = corpus[0].keys() if corpus else []

with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()

    writer.writerows(corpus)