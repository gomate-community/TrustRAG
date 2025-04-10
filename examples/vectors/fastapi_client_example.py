import requests
import json

url = "http://10.208.63.29:6008/v1/embeddings"

headers = {
    "Authorization": "Bearer sk-aaabbbcccdddeeefffggghhhiiijjjkkk",
    "Content-Type": "application/json"
}

payload = {
    "input": ["This is a sample text to get embeddings for.",
              "Here's another example sentence."],
    "model": "bge-large-en-v1.5"
}

response = requests.post(url, headers=headers, json=payload)

if response.status_code == 200:
    # Parse and print the response
    data = response.json()
    print(data.keys())
    print(f"Model: {data['model']}")
    print(f"Number of embeddings: {len(data['data'])}")
    print(f"Dimension of first embedding: {len(data['data'][0]['embedding'])}")
    print(data['data'][0]['embedding'])
    print(f"Usage - Prompt tokens: {data['usage']['prompt_tokens']}")
    print(f"Usage - Total tokens: {data['usage']['total_tokens']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)


# dict_keys(['data', 'model', 'object', 'usage'])

# data为embedding
