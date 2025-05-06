import json
sub_name = 'OpenAI'
with open(f'../data/{sub_name}-top-100-results.json', 'r') as f:
    data = json.load(f)
    
    links = []
    for item in data['data']:
        links.append("https://www.reddit.com" + item['permalink'])
        # https://www.reddit.com/r/meme/comments/1icck05/i_cant_believe_chatgpt_lost_its_job_to_ai/
        # /r/DeepSeek/comments/1ibound/i_cant_believe_chatgpt_lost_its_job_to_ai/
        
with open(f'../data/{sub_name}-top-100-links.txt', 'w') as f:
    for link in links:
        f.write(link + '\n')