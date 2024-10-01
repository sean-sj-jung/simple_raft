import os
from openai import OpenAI
api_key = "sk-youropenaiapikey"


def chat(text):
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": text,
        }],
        model="gpt-4o-mini-2024-07-18",
    )
    return response