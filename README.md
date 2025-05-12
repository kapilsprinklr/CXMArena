# CXM Arena Data Loader & Evaluator

This repository provides tools to **download, prepare, and evaluate datasets** for the [CXM Arena](https://huggingface.co/datasets/sprinklr-huggingface/CXM_Arena) tasks. It includes two main modules:

- `cxm_downloader.py`: Download and prepare data for various CXM Arena tasks.
- `cxm_evaluator.py`: Evaluate model predictions using task-specific metrics.

## Features
- Easy access to multiple CXM Arena tasks (Agent Quality, KB Refinement, Article Search, Intent Prediction, Multi-Turn RAG, Tool Calling)
- Consistent data loading interface
- Task-specific evaluation metrics

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/cxm-arena-utils.git
   cd cxm-arena-utils
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
- `pandas`
- `datasets`
- `numpy`

## Usage

### Data Loading
```python
from cxm_downloader import CXMDataLoader

data = CXMDataLoader.load("AQM")  # or "KB_REFINEMENT", "ARTICLE_SEARCH", etc.
print(data.keys())
```

### Evaluation
```python
from cxm_evaluator import CXMEvaluator

evaluator = CXMEvaluator()
score = evaluator.evaluate("AQM", data, model_outputs)
print("Score:", score)
```

## File Structure
```
.
├── cxm_downloader.py
├── cxm_evaluator.py
├── README.md
└── requirements.txt
```