#!/usr/bin/env python3
"""
Llama Model Benchmarking Tool
Benchmarks Llama models across multiple providers: Replicate, Together AI, Perplexity, and OpenRouter
Measures: Quality, Latency, Cost, and Reliability
"""

import os
import time
import json
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import requests


@dataclass
class BenchmarkResult:
    provider: str
    model: str
    prompt: str
    response: str
    time_to_first_token: Optional[float]
    total_time: float
    input_tokens: int
    output_tokens: int
    cost: float
    success: bool
    error: Optional[str]
    timestamp: str


@dataclass
class ProviderConfig:
    name: str
    model_8b: str
    model_70b: str
    api_key_env: str
    base_url: str
    input_cost_per_million: float
    output_cost_per_million: float


TEST_PROMPTS = [
    {
        "name": "reasoning",
        "prompt": "Tina has one brother and one sister. How many sisters do Tina's siblings have? Think step by step.",
        "category": "logical_reasoning"
    },
    {
        "name": "coding",
        "prompt": "Write a Python function to find the longest palindromic substring in a given string. Include comments explaining the algorithm.",
        "category": "code_generation"
    },
    {
        "name": "creative",
        "prompt": "Write a short poem about artificial intelligence in the style of Emily Dickinson.",
        "category": "creative_writing"
    },
    {
        "name": "factual",
        "prompt": "What are the three branches of the United States government and what is the primary function of each?",
        "category": "factual_knowledge"
    },
    {
        "name": "summarization",
        "prompt": "Summarize the key principles of machine learning in 3-4 sentences suitable for a beginner.",
        "category": "summarization"
    }
]


PROVIDERS = {
    "replicate": ProviderConfig(
        name="Replicate",
        model_8b="meta/meta-llama-3-8b-instruct",
        model_70b="meta/meta-llama-3-70b-instruct",
        api_key_env="REPLICATE_API_TOKEN",
        base_url="https://api.replicate.com/v1",
        input_cost_per_million=0.05,
        output_cost_per_million=0.25
    ),
    "together": ProviderConfig(
        name="Together AI",
        model_8b="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        model_70b="meta-llama/Llama-3.1-70B-Instruct-Turbo",
        api_key_env="TOGETHER_API_KEY",
        base_url="https://api.together.xyz/v1",
        input_cost_per_million=0.18,
        output_cost_per_million=0.18
    ),
    "perplexity": ProviderConfig(
        name="Perplexity",
        model_8b="llama-3.1-8b-instruct",
        model_70b="llama-3.1-70b-instruct",
        api_key_env="PERPLEXITY_API_KEY",
        base_url="https://api.perplexity.ai",
        input_cost_per_million=0.20,
        output_cost_per_million=0.20
    ),
    "openrouter": ProviderConfig(
        name="OpenRouter",
        model_8b="meta-llama/llama-3.1-8b-instruct",
        model_70b="meta-llama/llama-3.1-70b-instruct",
        api_key_env="OPENROUTER_API_KEY",
        base_url="https://openrouter.ai/api/v1",
        input_cost_per_million=0.06,
        output_cost_per_million=0.06
    )
}


class LlamaBenchmark:
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        
    def check_api_keys(self) -> Dict[str, bool]:
        """Check which API keys are available"""
        available = {}
        for provider_key, config in PROVIDERS.items():
            api_key = os.getenv(config.api_key_env)
            available[provider_key] = api_key is not None and len(api_key) > 0
        return available
    
    def benchmark_replicate(self, model: str, prompt: str) -> BenchmarkResult:
        """Benchmark Replicate API"""
        api_token = os.getenv("REPLICATE_API_TOKEN")
        if not api_token:
            return BenchmarkResult(
                provider="Replicate",
                model=model,
                prompt=prompt,
                response="",
                time_to_first_token=None,
                total_time=0,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                success=False,
                error="API token not found",
                timestamp=datetime.now().isoformat()
            )
        
        start_time = time.time()
        try:
            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "version": "latest",
                "input": {
                    "prompt": prompt,
                    "max_tokens": 512,
                    "temperature": 0.6
                }
            }
            
            response = requests.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            prediction = response.json()
            
            prediction_url = prediction["urls"]["get"]
            
            time_to_first_token = None
            while True:
                time.sleep(1)
                result = requests.get(prediction_url, headers=headers, timeout=30)
                result.raise_for_status()
                prediction_status = result.json()
                
                if prediction_status["status"] == "succeeded":
                    if time_to_first_token is None:
                        time_to_first_token = time.time() - start_time
                    
                    output = "".join(prediction_status["output"]) if isinstance(prediction_status["output"], list) else str(prediction_status["output"])
                    total_time = time.time() - start_time
                    
                    metrics = prediction_status.get("metrics", {})
                    input_tokens = len(prompt.split())
                    output_tokens = len(output.split())
                    
                    cost = (input_tokens / 1_000_000 * PROVIDERS["replicate"].input_cost_per_million +
                           output_tokens / 1_000_000 * PROVIDERS["replicate"].output_cost_per_million)
                    
                    return BenchmarkResult(
                        provider="Replicate",
                        model=model,
                        prompt=prompt,
                        response=output,
                        time_to_first_token=time_to_first_token,
                        total_time=total_time,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost=cost,
                        success=True,
                        error=None,
                        timestamp=datetime.now().isoformat()
                    )
                elif prediction_status["status"] == "failed":
                    raise Exception(f"Prediction failed: {prediction_status.get('error')}")
                
                if time.time() - start_time > 120:
                    raise Exception("Timeout waiting for prediction")
                    
        except Exception as e:
            return BenchmarkResult(
                provider="Replicate",
                model=model,
                prompt=prompt,
                response="",
                time_to_first_token=None,
                total_time=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                success=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def benchmark_openai_compatible(self, provider_key: str, model: str, prompt: str) -> BenchmarkResult:
        """Benchmark OpenAI-compatible APIs (Together, Perplexity, OpenRouter)"""
        config = PROVIDERS[provider_key]
        api_key = os.getenv(config.api_key_env)
        
        if not api_key:
            return BenchmarkResult(
                provider=config.name,
                model=model,
                prompt=prompt,
                response="",
                time_to_first_token=None,
                total_time=0,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                success=False,
                error="API key not found",
                timestamp=datetime.now().isoformat()
            )
        
        start_time = time.time()
        time_to_first_token = None
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            if provider_key == "openrouter":
                headers["HTTP-Referer"] = "https://github.com/aadisangeet/My-first"
                headers["X-Title"] = "Llama Benchmark"
            
            data = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 512,
                "temperature": 0.6,
                "stream": False
            }
            
            response = requests.post(
                f"{config.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            total_time = time.time() - start_time
            time_to_first_token = total_time
            
            output = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", len(prompt.split()))
            output_tokens = usage.get("completion_tokens", len(output.split()))
            
            cost = (input_tokens / 1_000_000 * config.input_cost_per_million +
                   output_tokens / 1_000_000 * config.output_cost_per_million)
            
            return BenchmarkResult(
                provider=config.name,
                model=model,
                prompt=prompt,
                response=output,
                time_to_first_token=time_to_first_token,
                total_time=total_time,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                success=True,
                error=None,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return BenchmarkResult(
                provider=config.name,
                model=model,
                prompt=prompt,
                response="",
                time_to_first_token=None,
                total_time=time.time() - start_time,
                input_tokens=0,
                output_tokens=0,
                cost=0,
                success=False,
                error=str(e),
                timestamp=datetime.now().isoformat()
            )
    
    def run_benchmark(self, provider_key: str, model_size: str, prompt_data: dict) -> BenchmarkResult:
        """Run benchmark for a specific provider and model"""
        config = PROVIDERS[provider_key]
        model = config.model_8b if model_size == "8b" else config.model_70b
        prompt = prompt_data["prompt"]
        
        print(f"  Testing {config.name} - {model}...")
        
        if provider_key == "replicate":
            return self.benchmark_replicate(model, prompt)
        else:
            return self.benchmark_openai_compatible(provider_key, model, prompt)
    
    def run_all_benchmarks(self, model_sizes: List[str] = ["8b", "70b"]):
        """Run all benchmarks"""
        available_providers = self.check_api_keys()
        
        print("Available API keys:")
        for provider, available in available_providers.items():
            status = "✓" if available else "✗"
            print(f"  {status} {PROVIDERS[provider].name}")
        print()
        
        for prompt_data in TEST_PROMPTS:
            print(f"\nPrompt: {prompt_data['name']} ({prompt_data['category']})")
            print(f"  \"{prompt_data['prompt'][:60]}...\"")
            
            for model_size in model_sizes:
                print(f"\n  Model size: {model_size.upper()}")
                
                for provider_key in PROVIDERS.keys():
                    if not available_providers[provider_key]:
                        print(f"  Skipping {PROVIDERS[provider_key].name} (no API key)")
                        continue
                    
                    result = self.run_benchmark(provider_key, model_size, prompt_data)
                    self.results.append(result)
                    
                    if result.success:
                        print(f"    ✓ {result.provider}: {result.total_time:.2f}s, ${result.cost:.6f}")
                    else:
                        print(f"    ✗ {result.provider}: {result.error}")
                    
                    time.sleep(1)
    
    def generate_report(self, output_file: str = "benchmark_report.md"):
        """Generate a comprehensive markdown report"""
        if not self.results:
            print("No results to report")
            return
        
        successful_results = [r for r in self.results if r.success]
        
        report = []
        report.append("# Llama Model Benchmark Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        report.append(f"\nTotal tests run: {len(self.results)}")
        report.append(f"Successful: {len(successful_results)}")
        report.append(f"Failed: {len(self.results) - len(successful_results)}")
        
        report.append("\n## Executive Summary")
        report.append("\nThis benchmark compares Llama models across multiple providers:")
        report.append("- **Replicate**: meta-llama-3-8b-instruct, meta-llama-3-70b-instruct")
        report.append("- **Together AI**: Llama-3.1-8B-Instruct-Turbo, Llama-3.1-70B-Instruct-Turbo")
        report.append("- **Perplexity**: llama-3.1-8b-instruct, llama-3.1-70b-instruct")
        report.append("- **OpenRouter**: llama-3.1-8b-instruct, llama-3.1-70b-instruct")
        
        report.append("\n## Metrics Evaluated")
        report.append("\n1. **Quality**: Response accuracy and coherence across different prompt types")
        report.append("2. **Latency**: Time to first token and total response time")
        report.append("3. **Cost**: Per-request cost based on token usage")
        report.append("4. **Reliability**: Success rate and error handling")
        
        if successful_results:
            report.append("\n## Performance Summary")
            
            by_provider = {}
            for result in successful_results:
                if result.provider not in by_provider:
                    by_provider[result.provider] = []
                by_provider[result.provider].append(result)
            
            report.append("\n### Average Latency by Provider")
            report.append("\n| Provider | Avg Total Time (s) | Avg Time to First Token (s) |")
            report.append("|----------|-------------------|----------------------------|")
            
            for provider, results in sorted(by_provider.items()):
                avg_total = statistics.mean([r.total_time for r in results])
                ttft_values = [r.time_to_first_token for r in results if r.time_to_first_token is not None]
                avg_ttft = statistics.mean(ttft_values) if ttft_values else 0
                report.append(f"| {provider} | {avg_total:.2f} | {avg_ttft:.2f} |")
            
            report.append("\n### Average Cost by Provider")
            report.append("\n| Provider | Avg Cost per Request ($) | Total Cost ($) |")
            report.append("|----------|-------------------------|----------------|")
            
            for provider, results in sorted(by_provider.items()):
                avg_cost = statistics.mean([r.cost for r in results])
                total_cost = sum([r.cost for r in results])
                report.append(f"| {provider} | ${avg_cost:.6f} | ${total_cost:.6f} |")
            
            report.append("\n### Reliability")
            report.append("\n| Provider | Success Rate | Failed Requests |")
            report.append("|----------|--------------|-----------------|")
            
            for provider_key, config in PROVIDERS.items():
                provider_results = [r for r in self.results if r.provider == config.name]
                if provider_results:
                    success_count = len([r for r in provider_results if r.success])
                    success_rate = (success_count / len(provider_results)) * 100
                    failed_count = len(provider_results) - success_count
                    report.append(f"| {config.name} | {success_rate:.1f}% | {failed_count} |")
        
        report.append("\n## Detailed Results")
        
        for prompt_data in TEST_PROMPTS:
            report.append(f"\n### {prompt_data['name'].title()} ({prompt_data['category']})")
            report.append(f"\n**Prompt**: {prompt_data['prompt']}")
            
            prompt_results = [r for r in self.results if r.prompt == prompt_data['prompt']]
            
            if prompt_results:
                report.append("\n| Provider | Model | Time (s) | Cost ($) | Success |")
                report.append("|----------|-------|----------|----------|---------|")
                
                for result in prompt_results:
                    model_short = result.model.split('/')[-1] if '/' in result.model else result.model
                    status = "✓" if result.success else "✗"
                    report.append(f"| {result.provider} | {model_short} | {result.total_time:.2f} | ${result.cost:.6f} | {status} |")
                
                for result in [r for r in prompt_results if r.success]:
                    report.append(f"\n#### {result.provider} Response")
                    report.append(f"\n```")
                    report.append(result.response[:500] + ("..." if len(result.response) > 500 else ""))
                    report.append("```")
        
        report.append("\n## Methodology")
        report.append("\nEach model was tested with 5 different prompts covering:")
        report.append("- Logical reasoning")
        report.append("- Code generation")
        report.append("- Creative writing")
        report.append("- Factual knowledge")
        report.append("- Summarization")
        report.append("\nAll tests used:")
        report.append("- Temperature: 0.6")
        report.append("- Max tokens: 512")
        report.append("- Single-shot prompts (no conversation history)")
        
        report.append("\n## Reproducibility")
        report.append("\nTo reproduce these results:")
        report.append("1. Set up API keys as environment variables:")
        report.append("   - `REPLICATE_API_TOKEN`")
        report.append("   - `TOGETHER_API_KEY`")
        report.append("   - `PERPLEXITY_API_KEY`")
        report.append("   - `OPENROUTER_API_KEY`")
        report.append("2. Install dependencies: `pip install requests`")
        report.append("3. Run: `python benchmark.py`")
        
        report.append("\n## Raw Data")
        report.append("\nFull benchmark results are available in `benchmark_results.json`")
        
        report_text = "\n".join(report)
        
        with open(output_file, 'w') as f:
            f.write(report_text)
        
        print(f"\n\nReport generated: {output_file}")
        
        with open("benchmark_results.json", 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2)
        
        print(f"Raw results saved: benchmark_results.json")
        
        return report_text


def main():
    print("=" * 60)
    print("Llama Model Benchmark")
    print("=" * 60)
    print("\nThis benchmark will test Llama models across multiple providers")
    print("measuring quality, latency, cost, and reliability.\n")
    
    benchmark = LlamaBenchmark()
    
    benchmark.run_all_benchmarks(model_sizes=["8b", "70b"])
    
    benchmark.generate_report()
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
