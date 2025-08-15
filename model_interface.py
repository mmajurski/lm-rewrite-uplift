import os
import httpx
import openai
import asyncio
from openai import AsyncOpenAI
import time
from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem


from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def translate_remote(remote:str) -> tuple[str, str]:
    if ":" in remote:
        remote, port_num = remote.split(":")
        port_num = int(port_num)
    else:
        port_num = 8443
    if remote == "sierra":
        url=f"https://pn131285.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "oscar":
        url=f"https://pn131274.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "papa":
        url=f"https://pn131275.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "echo":
        url=f"https://pn125915.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "foxtrot":
        url=f"https://pn125916.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "golf":
        url=f"https://pn125917.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "hotel":
        url=f"https://pn125918.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "redwing":
        url=f"https://redwing.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"
    elif remote == "rchat":
        url=f"https://rchat.nist.gov/api"
        api_key = os.environ.get("RCHAT_API_KEY")
    elif remote == "openai":
        url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        url=f"https://{remote}.nist.gov:{port_num}/v1"
        api_key = "sk-no-key-required"

        
    
    if api_key is None:
        raise ValueError("API key is not set for remote: %s" % remote)
    if url is None:
        raise ValueError("URL is not set for remote: %s" % remote)
    
    return url, api_key


    


class SglModelAsync:
    def __init__(self, model:str, remote:str, reasoning_effort:str='medium', connection_parallelism:int=64):
        self.model = model
        self.remote = remote
        self.url, self.api_key = translate_remote(remote)
        self.reasoning_effort = reasoning_effort
        print(f"SglModelAsync selected reasoning_effort: {reasoning_effort}")

        self.connection_parallelism = connection_parallelism
        if 'openai' in self.url:
            print("OpenAI detected, setting connection parallelism to 10")
            self.connection_parallelism = 10

        print(f"Using model: {self.model} on remote: {self.remote} at {self.url}")

        # self.test_connection()

        self.client = openai.AsyncOpenAI(
            base_url=self.url, 
            api_key=self.api_key,
            http_client=httpx.AsyncClient(verify=False)
        )
        
    @staticmethod
    async def generate_text_async(model, client, prompt, request_id, reasoning_effort='medium'):
        start_time = time.time()
        try:
            
        
            # https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
            # System > Developer > User > Assistant > Tool.

            instructions = "You are an assistant that doesn't make mistakes. If a reference format is presented to you, you follow it perfectly without making errors."
            
            if 'gpt-oss' in model:
                retry_count_for_gpt_oss = 0
                latest_error = None
                while retry_count_for_gpt_oss < 20:
                    try:
                        response = await client.responses.create(
                            model=model,
                            instructions=instructions,
                            input=prompt,
                            max_output_tokens=32000, #8192,
                            temperature=0.7,
                            stream=False,
                            reasoning={'effort': reasoning_effort},
                        )
                        if retry_count_for_gpt_oss > 0:
                            print(f"GPT-OSS took {retry_count_for_gpt_oss} retries, but succeeded")
                        break
                    except Exception as e:
                        latest_error = e
                        retry_count_for_gpt_oss += 1
                        time.sleep(0.2)
                if retry_count_for_gpt_oss == 3:
                    raise Exception(f"Failed to generate response for GPT-OSS: {latest_error}")
                
                
            else:
                response = await client.responses.create(
                        model=model,
                        instructions=instructions,
                        input=prompt,
                        max_output_tokens=32000,
                        temperature=0.7,
                        stream=False,
                        reasoning={'effort': reasoning_effort},
                    )
                    
            
            
            elapsed = time.time() - start_time
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            if input_tokens is None:
                input_tokens = 0
            output_tokens = getattr(response.usage, 'output_tokens', 0) 
            if output_tokens is None:
                output_tokens = 0
            total_tokens = getattr(response.usage, 'total_tokens', 0)
            if total_tokens is None:
                total_tokens = 0

            
            content = ""
            scratchpad = ""
            for c in response.output:
                if isinstance(c, ResponseOutputMessage):
                    content = c.content[0].text
                if isinstance(c, ResponseReasoningItem):
                    scratchpad = c.content[0].text


            # instructions = f"You are an assistant that doesn't make mistakes. If a reference format is presented to you, you follow it perfectly without making errors. Reasoning: {reasoning_effort}."

            # response = await client.chat.completions.create(
            #     model=model,
            #     messages=[
            #         {"role": "system", "content": instructions},
            #         {"role": "user", "content": prompt}
            #     ],
            #     max_tokens=8192,
            #     temperature=0.7
            # )

            # elapsed = time.time() - start_time
            # input_tokens = response.usage.prompt_tokens
            # output_tokens = response.usage.completion_tokens
            # total_tokens = response.usage.total_tokens

            
            # content = response.choices[0].message.content
            
            
            result = {
                "request_id": request_id,
                "content": content,
                "scratchpad": scratchpad,
                "error": None,
                "elapsed_time": elapsed,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                # "prompt": prompt
            }
            if content.isspace():
                result['error'] = "Empty response"
            return result
        except Exception as e:
            # optional handling here, but for now nothing to do
            result = {
                "request_id": request_id,
                "content": None,
                "scratchpad": None,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "prompt": prompt
            }
            # raise e
            return result
        

    async def process_all_batches(self, model_prompts):
        """Process all batches with a single client instance, in chunks of 64"""

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
                self.generate_text_async(self.model, self.client, prompt, i+j, self.reasoning_effort) 
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
                # Find the first failed result and print its error
                for idx, res in enumerate(batch_results):
                    if res['error'] is not None:
                        print(f"Error in prompt {i + idx}: {res['error']}")
                        print(f"Prompt: \n\n\n{res['prompt']}\n\n\n")
                        break
                print(80*"=")
                raise Exception("Failed to generate response for some prompts")
            results.extend(batch_results)
        
        total_time = time.time() - total_start_time
        
        return results, total_time
    
    
    def generate(self, model_prompts: list[str]):
        print("Remote model generating %d prompts (asynchronously)" % len(model_prompts))
        return asyncio.run(self.process_all_batches(model_prompts))
    

    


# Example usage
if __name__ == "__main__":
    async def main():
        model = SglModelAsync()
        import json
        # load the squadv2 dataset subset
        with open('squadv2_subset.json', 'r') as f:
            dataset = json.load(f)
        dataset = dataset[:32]

        contexts = [item['context'] for item in dataset]

        # Async usage
        results, total_time = await model.generate_async(contexts)
        
        # Or synchronous usage
        # results, total_time = model.generate(contexts)
        
        res = results[0]
        print(res)
        print(f"in total took: {total_time} seconds")
        print(f"per question took: {total_time / len(results)} seconds for {len(results)} questions")

    asyncio.run(main())
    
    