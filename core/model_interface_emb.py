"""Async embedding model client using the OpenAI-compatible embeddings API."""

import os
import httpx
import openai
import asyncio
from openai import AsyncOpenAI
import time


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def translate_remote(remote: str) -> tuple[str, str]:
    """Resolve a remote server identifier to (url, api_key).

    Pass either:
      - "openai": uses api.openai.com and OPENAI_API_KEY env var
      - A full URL (e.g. "https://your-server:8443/v1"): uses VLLM_API_KEY env var
    """
    if remote == "openai":
        url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    elif "://" in remote:
        # Full URL provided directly
        url = remote
        api_key = os.environ.get("VLLM_API_KEY", "sk-no-key-required")
    else:
        raise ValueError(
            f"Unknown remote '{remote}'. Pass a full URL (e.g. 'https://your-server:8443/v1') "
            "or 'openai'."
        )

    if api_key is None:
        raise ValueError(f"API key is not set for remote: {remote}")

    return url, api_key


class SglModelAsyncEmb:
    """Async embedding client for OpenAI-compatible embedding APIs (vLLM, OpenAI, etc.)."""

    def __init__(self, model: str, remote: str, connection_parallelism: int = 64):
        self.model = model
        self.remote = remote
        self.url, self.api_key = translate_remote(remote)

        self.connection_parallelism = connection_parallelism
        if 'openai' in self.url:
            print("OpenAI detected, setting connection parallelism to 10")
            self.connection_parallelism = 10

        print(f"Using model: {self.model} on remote: {self.remote} at {self.url}")

        self.client = openai.AsyncOpenAI(
            base_url=self.url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(verify=False, timeout=120.0)
        )

    @staticmethod
    async def generate_text_async(model, client, prompt, request_id):
        start_time = time.time()
        try:
            response = await client.embeddings.create(
                model=model,
                input=prompt
            )

            elapsed = time.time() - start_time
            emb_vec = response.data[0].embedding

            result = {
                "request_id": request_id,
                "embedding": emb_vec,
                "error": None,
                "elapsed_time": elapsed,
            }
            return result
        except Exception as e:
            result = {
                "request_id": request_id,
                "embedding": None,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "prompt": prompt
            }
            return result


    async def process_all_batches(self, model_prompts):
        """Process all batches with a single client instance, in chunks of 64."""

        total_start_time = time.time()
        results = []

        # Process prompts in batches
        batch_size = self.connection_parallelism
        for i in range(0, len(model_prompts), batch_size):
            batch_prompts = model_prompts[i:i+batch_size]
            current_time = time.strftime("%H:%M:%S")
            print(f"  ({current_time}) remote batch {i//batch_size + 1}/{(len(model_prompts)-1)//batch_size + 1} ({len(batch_prompts)} prompts per batch)")

            # Create tasks for this batch with the shared client
            batch_tasks = [
                self.generate_text_async(self.model, self.client, prompt, i+j)
                for j, prompt in enumerate(batch_prompts)
            ]

            # Execute this batch concurrently and gather results
            batch_results = await asyncio.gather(*batch_tasks)
            failed_flag = False
            for res in batch_results:
                if res['error'] is not None:
                    failed_flag = True
                    break
            if failed_flag:
                print(80*"=")
                print(f"Failed to generate response for some prompts")
                for idx, res in enumerate(batch_results):
                    if res['error'] is not None:
                        print(f"Error in prompt {i + idx}: {res['error']}")
                        print(f"Prompt: \n\n{res['prompt']}")
                        break
                print(80*"=")
                raise Exception("Failed to generate response for some prompts")
            results.extend(batch_results)

        total_time = time.time() - total_start_time

        return results, total_time


    def generate(self, model_prompts: list[str]):
        """Synchronously generate embeddings for a list of inputs (async under the hood)."""
        print("Remote model generating %d prompts (asynchronously)" % len(model_prompts))
        return asyncio.run(self.process_all_batches(model_prompts))
