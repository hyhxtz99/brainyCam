import sys
import os
import requests

API_KEY = MY_API_KEY
EMAIL = MY_EMAIL

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "HTTP-Referer": EMAIL,
    "Content-Type": "application/json"
}

def main():
    if len(sys.argv) < 3:
        print("Usage: python openrouter_claude_qa.py <context_file> \"<question>\"")
        return

    context_path = sys.argv[1]
    question = " ".join(sys.argv[2:])

    if not os.path.exists(context_path):
        print(f"[ERROR] Context file not found: {context_path}")
        return

    with open(context_path, 'r', encoding='utf-8') as f:
        context = f.read()

    prompt = f"""Context:
{context}

Question: {question}"""

    data = {
        "model": "anthropic/claude-3-sonnet",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.2
    }

    
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)

    if res.status_code != 200:
        print(f"[ERROR] HTTP {res.status_code}: {res.text}")
        return

    output = res.json()
    print(output["choices"][0]["message"]["content"].strip())

if __name__ == "__main__":
    main()
