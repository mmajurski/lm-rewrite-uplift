"""Async LLM API client using the OpenAI-compatible API (responses or chat completions)."""

import os
import httpx
import openai
import asyncio
from asyncio import Semaphore
from openai import AsyncOpenAI, OpenAI
import time
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem


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


class SglModelAsync:
    """Async LLM client for OpenAI-compatible APIs (vLLM, OpenAI, etc.).

    Supports both the Responses API and the Chat Completions API.
    Manages a semaphore to limit concurrent connections.
    """

    def __init__(self, model: str, remote: str, reasoning_effort: str = 'medium',
                 connection_parallelism: int = 64, chat_completion_api: bool = False):
        """Initialize the async LLM client.

        Args:
            model: Model identifier string (e.g. "v_llm/gpt-oss-120b").
            remote: Server URL or "openai". See translate_remote() for details.
            reasoning_effort: Effort level passed to reasoning-capable models.
            connection_parallelism: Max concurrent in-flight requests.
            chat_completion_api: If True, use chat.completions; otherwise use responses.
        """
        self.model = model
        self.remote = remote
        self.url, self.api_key = translate_remote(remote)
        self.reasoning_effort = reasoning_effort
        self.chat_completion_api = chat_completion_api
        print(f"SglModelAsync selected reasoning_effort: {reasoning_effort}")

        self.connection_parallelism = connection_parallelism
        if 'openai' in self.url:
            print("OpenAI detected, setting connection parallelism to 10")
            self.connection_parallelism = 10

        print(f"Using model: {self.model} on remote: {self.remote} at {self.url}")

        self.client = openai.AsyncOpenAI(
            base_url=self.url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(verify=False, timeout=600.0)
        )

    @staticmethod
    async def generate_text_async(model, client, prompt, request_id, reasoning_effort='medium', chat_completion_api=False):
        start_time = time.time()
        try:
            # System > Developer > User > Assistant > Tool.
            instructions = "You are an assistant that doesn't make mistakes. If a reference format is presented to you, you follow it perfectly without making errors."

            retry_count = 0
            latest_error = None
            retry_limit = 4
            while retry_count < retry_limit:
                try:
                    if chat_completion_api:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": instructions},
                                {"role": "user", "content": prompt},
                            ],
                            max_tokens=64000,
                            temperature=0.7,
                            stream=False,
                            reasoning_effort=reasoning_effort
                        )
                    else:
                        response = await client.responses.create(
                            model=model,
                            instructions=instructions,
                            input=prompt,
                            max_output_tokens=64000,
                            temperature=0.7,
                            stream=False,
                            reasoning={'effort': reasoning_effort},
                        )

                    if retry_count > 0:
                        print(f"Model took {retry_count} retries, but succeeded")
                    break
                except Exception as e:
                    latest_error = e
                    print(f"Generate try {retry_count} error: {latest_error}")
                    retry_count += 1
            if retry_count >= retry_limit:
                raise Exception(f"Failed (retry count {retry_count}) to generate response: {latest_error}")

            elapsed = time.time() - start_time

            content = ""
            scratchpad = ""
            if chat_completion_api:
                content = response.choices[0].message.content
                if 'reasoning_content' in response.choices[0].message.model_extra:
                    scratchpad = response.choices[0].message.model_extra['reasoning_content']
            else:
                for c in response.output:
                    if isinstance(c, ResponseOutputMessage):
                        content = c.content[0].text
                    if isinstance(c, ResponseReasoningItem):
                        scratchpad = c.content[0].text

            result = {
                "request_id": request_id,
                "content": content,
                "scratchpad": scratchpad,
                "error": None,
                "elapsed_time": elapsed,
            }
            if content.isspace():
                result['error'] = "Empty response"
            return result
        except Exception as e:
            result = {
                "request_id": request_id,
                "content": None,
                "scratchpad": None,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "prompt": prompt
            }
            return result


    async def process_all_batches(self, model_prompts):
        """Process all prompts with a rolling buffer of N concurrent connections."""

        total_start_time = time.time()
        results = []

        # Create a semaphore to limit concurrent connections
        semaphore = Semaphore(self.connection_parallelism)

        # Track active tasks
        active_tasks = 0

        async def process_single_prompt(prompt, index):
            """Process a single prompt with semaphore limiting."""
            nonlocal active_tasks
            async with semaphore:  # This will wait if we're at the limit
                active_tasks += 1
                try:
                    return await self.generate_text_async(
                        self.model,
                        self.client,
                        prompt,
                        index,
                        self.reasoning_effort,
                        self.chat_completion_api
                    )
                finally:
                    active_tasks -= 1

        # Create all tasks at once - they'll be limited by the semaphore
        tasks = [
            asyncio.create_task(process_single_prompt(prompt, i))
            for i, prompt in enumerate(model_prompts)
        ]

        # Process with progress tracking
        completed = 0
        total = len(tasks)

        # Use as_completed to process results as they finish
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1

                # Progress reporting
                if completed % 10 == 0 or completed == total:
                    current_time = time.strftime("%H:%M:%S")
                    print(f"  ({current_time}) Completed {completed}/{total} prompts")
                    print(f"    Active connections: {active_tasks} (max allowed: {self.connection_parallelism})")

                # Check for errors and handle them
                if result['error'] is not None:
                    print(80*"=")
                    print(f"Failed to generate response for prompt {result['request_id']}")
                    print(f"Error: {result['error']}")
                    print(f"Prompt: \n\n\n{result.get('prompt', 'N/A')}\n\n\n")
                    print(80*"=")
                    raise Exception(f"Failed to generate response for prompt {result['request_id']}: {result['error']}")

            except Exception as e:
                print(f"Error processing task: {e}")
                raise

        # Sort results by request_id to maintain original order
        results.sort(key=lambda x: x['request_id'])

        total_time = time.time() - total_start_time
        return results, total_time


    def generate(self, model_prompts: list[str]):
        """Synchronously generate responses for a list of prompts (async under the hood)."""
        print("Remote model generating %d prompts (asynchronously)" % len(model_prompts))
        return asyncio.run(self.process_all_batches(model_prompts))
