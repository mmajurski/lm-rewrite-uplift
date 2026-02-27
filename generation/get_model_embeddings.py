"""Compute embeddings for questions using a vLLM embedding endpoint."""

import os
import httpx
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

GRADER_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
GRADER_MODEL_BASE_URL = os.getenv("EMBEDDING_MODEL_BASE_URL")
GRADER_MODEL_API_KEY = os.getenv("VLLM_API_KEY", "sk-no-key-required")

if GRADER_MODEL_BASE_URL is None:
    raise ValueError("EMBEDDING_MODEL_BASE_URL environment variable is not set. See .env.example.")


client = openai.OpenAI(
            base_url=GRADER_MODEL_BASE_URL, 
            api_key=GRADER_MODEL_API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0)
        )

# response = await client.responses.create(
#                             model=model,
#                             instructions=instructions,
#                             input=prompt,
#                             max_output_tokens=32000, #8192,
#                             temperature=0.7,
#                             stream=False,
#                             reasoning={'effort': reasoning_effort},
#                         )

input_paragraph = "This a test of embeddings... pls work"
response = client.embeddings.create(
        model=GRADER_MODEL,
        input=input_paragraph
    )

print(response)
print('here')