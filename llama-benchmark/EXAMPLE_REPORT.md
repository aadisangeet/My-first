# Llama Model Benchmark Report - Example

Generated: 2025-11-04 17:45:00 UTC

Total tests run: 40
Successful: 36
Failed: 4

## Executive Summary

This benchmark compares Llama models across multiple providers:
- **Replicate**: meta-llama-3-8b-instruct, meta-llama-3-70b-instruct
- **Together AI**: Llama-3.1-8B-Instruct-Turbo, Llama-3.1-70B-Instruct-Turbo
- **Perplexity**: llama-3.1-8b-instruct, llama-3.1-70b-instruct
- **OpenRouter**: llama-3.1-8b-instruct, llama-3.1-70b-instruct

## Metrics Evaluated

1. **Quality**: Response accuracy and coherence across different prompt types
2. **Latency**: Time to first token and total response time
3. **Cost**: Per-request cost based on token usage
4. **Reliability**: Success rate and error handling

## Performance Summary

### Average Latency by Provider

| Provider | Avg Total Time (s) | Avg Time to First Token (s) |
|----------|-------------------|----------------------------|
| OpenRouter | 2.34 | 0.45 |
| Perplexity | 2.67 | 0.52 |
| Replicate | 3.12 | 0.89 |
| Together AI | 2.45 | 0.41 |

### Average Cost by Provider

| Provider | Avg Cost per Request ($) | Total Cost ($) |
|----------|-------------------------|----------------|
| OpenRouter | $0.000234 | $0.002106 |
| Perplexity | $0.000312 | $0.002808 |
| Replicate | $0.000189 | $0.001701 |
| Together AI | $0.000267 | $0.002403 |

### Reliability

| Provider | Success Rate | Failed Requests |
|----------|--------------|-----------------|
| OpenRouter | 90.0% | 1 |
| Perplexity | 100.0% | 0 |
| Replicate | 80.0% | 2 |
| Together AI | 90.0% | 1 |

## Key Findings

### Latency
- **Fastest**: Together AI and OpenRouter show the best latency performance
- **Time to First Token**: Together AI leads with ~0.41s average
- **Total Response Time**: OpenRouter provides the fastest complete responses

### Cost
- **Most Economical**: Replicate offers the lowest per-request cost
- **Best Value**: OpenRouter provides good balance of speed and cost
- **Premium Options**: Perplexity and Together AI cost slightly more but offer reliability

### Reliability
- **Most Reliable**: Perplexity achieved 100% success rate
- **Consistent Performance**: All providers maintained >80% success rate
- **Error Handling**: Most failures were due to rate limiting or temporary unavailability

### Quality (Qualitative Assessment)
- **Reasoning Tasks**: All models performed well on logical reasoning
- **Code Generation**: 70B models produced more detailed, well-commented code
- **Creative Writing**: Noticeable quality difference between 8B and 70B variants
- **Factual Accuracy**: All models provided accurate factual information
- **Summarization**: Consistent quality across all providers

## Detailed Results

### Reasoning (logical_reasoning)

**Prompt**: Tina has one brother and one sister. How many sisters do Tina's siblings have? Think step by step.

| Provider | Model | Time (s) | Cost ($) | Success |
|----------|-------|----------|----------|---------|
| OpenRouter | llama-3.1-8b-instruct | 2.12 | $0.000198 | ✓ |
| OpenRouter | llama-3.1-70b-instruct | 2.45 | $0.000234 | ✓ |
| Perplexity | llama-3.1-8b-instruct | 2.34 | $0.000267 | ✓ |
| Perplexity | llama-3.1-70b-instruct | 2.89 | $0.000312 | ✓ |
| Replicate | meta-llama-3-8b-instruct | 2.78 | $0.000156 | ✓ |
| Replicate | meta-llama-3-70b-instruct | 3.23 | $0.000189 | ✓ |
| Together AI | Llama-3.1-8B-Instruct-Turbo | 2.01 | $0.000223 | ✓ |
| Together AI | Llama-3.1-70B-Instruct-Turbo | 2.56 | $0.000278 | ✓ |

#### Sample Response (Together AI - 8B)

```
Let me think through this step by step:

1. Tina has one brother and one sister
2. From the brother's perspective:
   - He has Tina as a sister
   - He has Tina's sister as another sister
   - So the brother has 2 sisters

3. From the sister's perspective:
   - She has Tina as a sister
   - So the sister has 1 sister

Therefore, Tina's siblings have either 2 sisters (the brother) or 1 sister (the sister).
```

### Coding (code_generation)

**Prompt**: Write a Python function to find the longest palindromic substring in a given string. Include comments explaining the algorithm.

| Provider | Model | Time (s) | Cost ($) | Success |
|----------|-------|----------|----------|---------|
| OpenRouter | llama-3.1-8b-instruct | 2.67 | $0.000245 | ✓ |
| OpenRouter | llama-3.1-70b-instruct | 3.12 | $0.000289 | ✓ |
| Perplexity | llama-3.1-8b-instruct | 2.89 | $0.000312 | ✓ |
| Perplexity | llama-3.1-70b-instruct | 3.45 | $0.000367 | ✓ |
| Replicate | meta-llama-3-8b-instruct | 3.34 | $0.000198 | ✓ |
| Replicate | meta-llama-3-70b-instruct | 3.89 | $0.000234 | ✓ |
| Together AI | Llama-3.1-8B-Instruct-Turbo | 2.45 | $0.000267 | ✓ |
| Together AI | Llama-3.1-70B-Instruct-Turbo | 3.01 | $0.000312 | ✓ |

## Recommendations

### For Production Use
- **Best Overall**: Together AI offers excellent balance of speed, cost, and reliability
- **Budget-Conscious**: Replicate provides lowest cost with acceptable performance
- **Mission-Critical**: Perplexity's 100% success rate makes it ideal for critical applications

### For Development/Testing
- **Rapid Iteration**: OpenRouter's fast response times accelerate development
- **Cost-Effective Testing**: Replicate's low pricing is ideal for high-volume testing

### Model Size Selection
- **8B Models**: Suitable for most tasks, significantly faster and cheaper
- **70B Models**: Recommended for complex reasoning, detailed code generation, and creative tasks

## Methodology

Each model was tested with 5 different prompts covering:
- Logical reasoning
- Code generation
- Creative writing
- Factual knowledge
- Summarization

All tests used:
- Temperature: 0.6
- Max tokens: 512
- Single-shot prompts (no conversation history)

## Reproducibility

To reproduce these results:
1. Set up API keys as environment variables:
   - `REPLICATE_API_TOKEN`
   - `TOGETHER_API_KEY`
   - `PERPLEXITY_API_KEY`
   - `OPENROUTER_API_KEY`
2. Install dependencies: `pip install requests`
3. Run: `python benchmark.py`

## Raw Data

Full benchmark results are available in `benchmark_results.json`

---

*Note: This is an example report showing the expected format and structure. Actual results will vary based on API availability, network conditions, and model updates.*
