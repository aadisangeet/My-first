# Llama Model Benchmark

A comprehensive benchmarking tool for comparing Llama models across multiple AI providers.

## Overview

This benchmark compares the **latest Llama 3.1 models** (8B and 70B variants) across four major providers:

- **Replicate**: Uses Llama 3 models (meta-llama-3-8b-instruct, meta-llama-3-70b-instruct)
- **Together AI**: Uses Llama 3.1 Turbo models
- **Perplexity**: Uses Llama 3.1 models
- **OpenRouter**: Uses Llama 3.1 models

### Metrics Evaluated

1. **Quality**: Response accuracy and coherence across different prompt types
2. **Latency**: Time to first token and total response time
3. **Cost**: Per-request cost based on token usage
4. **Reliability**: Success rate and error handling

## Setup

### Prerequisites

- Python 3.7 or higher
- API keys for the providers you want to test

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys as environment variables:
```bash
export REPLICATE_API_TOKEN="your_replicate_token"
export TOGETHER_API_KEY="your_together_key"
export PERPLEXITY_API_KEY="your_perplexity_key"
export OPENROUTER_API_KEY="your_openrouter_key"
```

Note: You can test with any subset of providers - the benchmark will skip providers without API keys.

## Usage

Run the benchmark:
```bash
python benchmark.py
```

The script will:
1. Test each available provider with 5 different prompts
2. Measure latency, cost, and reliability for each request
3. Generate a comprehensive report in `benchmark_report.md`
4. Save raw results to `benchmark_results.json`

## Test Prompts

The benchmark uses 5 carefully selected prompts covering different capabilities:

1. **Logical Reasoning**: Tests step-by-step problem solving
2. **Code Generation**: Tests programming ability with comments
3. **Creative Writing**: Tests creative output in a specific style
4. **Factual Knowledge**: Tests accuracy of factual information
5. **Summarization**: Tests ability to condense information

## Output

### benchmark_report.md
A comprehensive markdown report including:
- Executive summary with provider comparison
- Average latency by provider
- Average cost by provider
- Reliability metrics (success rates)
- Detailed results for each prompt
- Sample responses from each provider
- Methodology and reproducibility instructions

### benchmark_results.json
Raw JSON data containing all benchmark results for further analysis.

## Pricing Information (as of benchmark creation)

| Provider | Input (per 1M tokens) | Output (per 1M tokens) |
|----------|----------------------|------------------------|
| Replicate | $0.05 | $0.25 |
| Together AI | $0.18 | $0.18 |
| Perplexity | $0.20 | $0.20 |
| OpenRouter | $0.06 | $0.06 |

Note: Prices may vary. Check provider documentation for current rates.

## Model Availability Notes

- **Replicate**: Currently offers Llama 3 (not 3.1) for 8B and 70B sizes
- **Together AI**: Offers Llama 3.1 Turbo variants optimized for speed
- **Perplexity**: Offers standard Llama 3.1 models
- **OpenRouter**: Offers standard Llama 3.1 models with routing to multiple providers

When exact Llama 3.1 variants aren't available, the benchmark uses the closest available model and notes this in the report.

## Customization

You can customize the benchmark by modifying:

- `TEST_PROMPTS`: Add or modify test prompts
- `PROVIDERS`: Update model names, pricing, or add new providers
- `model_sizes`: Test only specific model sizes (e.g., just "8b")

## Limitations

- Token counting is approximate for providers that don't return exact counts
- Streaming is not currently tested (all requests use non-streaming mode)
- Quality evaluation is qualitative (requires manual review of responses)
- Network conditions may affect latency measurements

## Contributing

To add a new provider:

1. Add provider configuration to `PROVIDERS` dictionary
2. Implement provider-specific API call if needed (or use OpenAI-compatible method)
3. Update pricing information
4. Test thoroughly

## License

This benchmark tool is provided as-is for evaluation purposes.
