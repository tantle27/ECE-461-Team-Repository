import requests

url = "https://genai.rcac.purdue.edu/api/chat/completions"
headers = {
    "Authorization": f"Bearer <insert_api_key_here>",
    "Content-Type": "application/json"
}
body = {
    "model": "llama3.1:latest",
    "messages": [
    {
        "role": "user",
        "content": "What is your name?"
    }
    ],
    # "stream": True
}
response = requests.post(url, headers=headers, json=body)
if response.status_code == 200:
    with open("response_output.txt", "w", encoding="utf-8") as f:
        f.write(response.text)
else:
    raise Exception(f"Error: {response.status_code}, {response.text}")