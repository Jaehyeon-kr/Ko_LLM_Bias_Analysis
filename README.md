# Korean LLM Bias Analysis

A comprehensive evaluation framework for analyzing bias in Korean Large Language Models (LLMs).

## 📋 Overview

This project provides a complete toolkit for evaluating and visualizing bias patterns in Korean language models, with support for entropy-based confidence analysis and multiple dataset formats.

## ✨ Features

- **Multi-Model Support**: Evaluate EXAONE, Tri, Kanana, and other Korean LLMs
- **Entropy Analysis**: Shannon entropy calculation for full vocabulary and answer choices
- **Auto-Format Detection**: Works with both KoBBQ and Context3 dataset formats
- **Comprehensive Metrics**: Accuracy, bias rate, entropy, and probability distributions
- **Rich Visualizations**: 10+ visualization methods for result analysis
- **GPU/CPU Support**: Automatic device detection and optimization

## 🎯 Supported Models

- **EXAONE 3.5** (LG AI Research): 2.4B, 7.8B, 32B parameters
- **Trillion Labs**: Tri-7B, Tri-21B
- **Kanana** (Kakao): 2.1B parameters
- **Custom models**: Easy to add new models via configuration

## 📊 Evaluation Metrics

- **Accuracy**: Percentage of correct answers
- **Bias Rate**: Percentage of biased responses
- **Full Entropy**: Shannon entropy over entire vocabulary
- **Choice Entropy**: Shannon entropy over answer choices only
- **Choice Probabilities**: Probability distribution for each choice

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from llm_evaluate import LLMBenchmark

# Initialize benchmark
benchmark = LLMBenchmark("exaone-3.5-2.4b")

# Evaluate dataset
results = benchmark.evaluate_dataset("data_context3.json")

# Results saved to benchmark_results_context3/
```

### Visualization

```python
from visualization_english import BenchmarkVisualizer

# Load results
viz = BenchmarkVisualizer("benchmark_results_context3")
viz.load_results()

# Generate all visualizations
viz.generate_all_plots(output_dir="visualizations")
```

### Interactive Notebook

See [llm_evaluation.ipynb](llm_evaluation.ipynb) for detailed usage with step-by-step examples.

## 📁 File Structure

```
Ko_LLM_Bias_Analysis/
├── llm_evaluation.ipynb          # Interactive evaluation notebook
├── llm_evaluate.py                # Main evaluation script
├── visualization_english.py       # Visualization module
├── llm_benchmark_methods.py       # Helper methods reference
├── data_context3.json             # Sample dataset
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 📈 Visualization Methods

1. **Model Comparison**: Compare accuracy, bias rate, and entropy across models
2. **Entropy Distribution**: Histogram of entropy values
3. **Correct vs Incorrect**: Entropy comparison by correctness
4. **Choice Probabilities**: Probability distribution visualization
5. **Category Performance**: Performance breakdown by category
6. **Confusion Matrix**: Prediction vs ground truth heatmap
7. **Radar Chart**: Multi-metric comparison
8. **Entropy-Accuracy Scatter**: Correlation analysis
9. **Bias Pattern Analysis**: Detailed bias behavior analysis
10. **Difficulty Analysis**: Sample difficulty based on entropy

## 🔧 Configuration

Edit `MODEL_CONFIGS` in [llm_evaluate.py](llm_evaluate.py) to add new models:

```python
MODEL_CONFIGS = {
    "your-model": {
        "model_id": "huggingface/model-name",
        "type": "causal",
        "max_length": 2048
    }
}
```

## 📝 Dataset Format

### KoBBQ Format
```json
{
  "context": "상황 설명",
  "question": "질문",
  "A": "선택지 A",
  "B": "선택지 B",
  "C": "선택지 C",
  "ground_truth": "A"
}
```

### Context3 Format
```json
{
  "question": "질문",
  "choices": ["선택지1", "선택지2", "선택지3"],
  "answer": 0
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is for academic and research purposes.
