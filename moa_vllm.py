"""prerequisite

!git lfs clone https://huggingface.co/Qwen/Qwen2-0.5B-Instruct
!git lfs clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct
!git lfs clone https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
!pip install -q vllm
"""

import time
import uuid
import torch
import asyncio
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("vllm").setLevel(logging.ERROR)

from typing import List, Dict, AsyncGenerator
from vllm import AsyncLLMEngine, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs


user_prompt = "What are 3 fun things to do in SF?"
reference_models = [
    "./Qwen2-0.5B-Instruct",
    "./Phi-3-mini-4k-instruct",
    "./Meta-Llama-3.1-8B-Instruct"
]
aggregator_model = "./Meta-Llama-3.1-8B-Instruct"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


def print_section(title: str, content: str, separator: str = "="):
    print(f"\n{separator * 50}")
    print(f"{title}:")
    print(f"{separator * 50}")
    print(f"{content}")


async def generate(engine: AsyncLLMEngine, prompt: str, id: uuid.UUID) -> str:
    sampling_params = SamplingParams(temperature=0.8, max_tokens=512)
    try:
        results_generator = engine.generate(prompt, sampling_params, id)
        async for request_output in results_generator:
            pass
        return request_output.outputs[0].text
    except Exception as e:
        print_section("Error", f"Error generating response: {e}", ">")
        return ""


async def generate_stream(engine: AsyncLLMEngine, prompt: str, id: uuid.UUID) -> AsyncGenerator[str, None]:
    sampling_params = SamplingParams(temperature=0.8, max_tokens=512)
    try:
        results_generator = engine.generate(prompt, sampling_params, id)
        async for request_output in results_generator:
            if request_output.outputs:
                yield request_output.outputs[0].text
    except Exception as e:
        print_section("Error", f"Error generating response: {e}", ">")
        yield ""


async def create_engine(model: str) -> AsyncLLMEngine:
    engine_args = AsyncEngineArgs(
        model=model,
        dtype="half",
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        swap_space=4,
        max_model_len=1024,
        kv_cache_dtype="fp8_e5m2",
        tensor_parallel_size=1,
        disable_log_requests=True,
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def generate_responses(engines: Dict[str, AsyncLLMEngine]) -> List[str]:
    tasks = [generate(engine, user_prompt, uuid.uuid4()) for engine in engines.values()]
    return await asyncio.gather(*tasks)


async def main():
    engines = {}
    try:
        for model in reference_models:
            engines[model] = await create_engine(model)

        reference_start_time = time.time()
        reference_results = await generate_responses(engines)
        reference_latency = time.time() - reference_start_time

        for model, result in zip(reference_models, reference_results):
            print_section(f"Result for model {model}", result, ">")

        aggregator_prompt = (
            aggreagator_system_prompt
            + "\n"
            + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(reference_results)])
        )
        print_section("Aggregator Prompt", aggregator_prompt)

        aggregator_engine = await create_engine(aggregator_model)

        aggregator_start_time = time.time()
        final_result = await generate(aggregator_engine, aggregator_prompt, uuid.uuid4())
        aggregator_latency = time.time() - aggregator_start_time

        print_section("Final Aggregated Result", final_result)

        total_latency = reference_latency + aggregator_latency
        print_section("Total Latency", f"{total_latency:.2f} seconds")

    except Exception as e:
        print_section("Error", f"An error occurred: {e}", ">")
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    asyncio.run(main())