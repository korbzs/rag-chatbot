import os
from openai import OpenAI

def check_openai_moderation(text: str) -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")

    client = OpenAI()
    
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
    )
    
    return response.results[0].flagged