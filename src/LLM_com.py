import requests
import sys


#arguement assignments
if len(sys.argv) < 3:
    raise Exception("Missing file arguement")

input_file = sys.argv[1]
with open(input_file, "r", encoding="utf-8") as f:
    prompt_file = f.read().strip()

purpose = sys.argv[2]
if purpose == "dataset":
    prompt = ""
elif purpose == "code":
    prompt = ""
elif purpose == "performance":
    prompt = ""
else:
    raise Exception("Invalid purpose argument")

#API call (LLaMA 3.1 being used)
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
        "content": prompt + ": " + prompt_file
    }
    ],
    # "stream": True
}

#Writing the LLM response to a file based on purpose
response = requests.post(url, headers=headers, json=body)
if response.status_code == 200:
    if purpose == "dataset":
        with open("dataset_output.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
    elif purpose == "code":
        with open("code_quality_output.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
    elif purpose == "performance":
        with open("performance_output.txt", "w", encoding="utf-8") as f:
            f.write(response.text)
else:
    raise Exception(f"Error: {response.status_code}, {response.text}")