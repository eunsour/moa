"""
Prerequisite:
pip install -q transformers
"""

import time
import torch
import asyncio
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline


user_prompt = "What are 3 fun things to do in SF?"
reference_models = [
    "Qwen/Qwen2-0.5B-Instruct",
    "google/gemma-2-2b-it"
    "microsoft/Phi-3.5-mini-instruct",
]
aggregator_model = "microsoft/Phi-3.5-mini-instruct"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""


def print_section(title: str, content: str, separator: str = "="):
    print(f"\n{separator * 50}")
    print(f"{title}:")
    print(f"{separator * 50}")
    print(f"{content}")


async def generate(pipeline: TextGenerationPipeline, prompt: str) -> str:
    try:
        result = pipeline(prompt, max_new_tokens=512, do_sample=True, temperature=0.8)
        return result[0]["generated_text"]
    except Exception as e:
        print_section("Error", f"Error generating response: {e}", ">")
        return ""


async def create_pipeline(model_name: str) -> TextGenerationPipeline:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    if not hasattr(model, "is_loaded_in_8bit") and not hasattr(model, "is_loaded_in_4bit"):
        model = model.to("cuda")

    return TextGenerationPipeline(model=model, tokenizer=tokenizer)


async def generate_responses(pipelines: Dict[str, TextGenerationPipeline]) -> List[str]:
    tasks = [generate(pipeline, user_prompt) for pipeline in pipelines.values()]
    return await asyncio.gather(*tasks)


async def main():
    pipelines = {}
    try:
        for model in reference_models:
            pipelines[model] = await create_pipeline(model)

        reference_start_time = time.time()
        reference_results = await generate_responses(pipelines)
        reference_latency = time.time() - reference_start_time

        for model, result in zip(reference_models, reference_results):
            print_section(f"Result for model {model}", result, ">")

        aggregator_prompt = (
            aggreagator_system_prompt
            + "\n"
            + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(reference_results)])
        )
        print_section("Aggregator Prompt", aggregator_prompt)

        aggregator_pipeline = await create_pipeline(aggregator_model)

        aggregator_start_time = time.time()
        final_result = await generate(aggregator_pipeline, aggregator_prompt)
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
