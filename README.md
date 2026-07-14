# Transformer - MoE Infinity Experiments

This folder contains the MoE (Mixture of Experts) Infinity implementation with custom transformers modifications for efficient expert prediction and prefetching.

## Overview

The project implements an optimized MoE inference system with:
- **Expert Prediction**: Predict which experts will be activated for upcoming tokens
- **Expert Prefetching**: Preload experts before they're needed to reduce latency
- **IVF Index**: Inverted File Index for efficient expert lookup
- **Trace-based Analysis**: Uses execution traces to optimize expert loading

## Directory Structure

```
transformer/
├── POC-inference-cuda-bench.ipynb    # Main benchmark notebook
├── requirements.txt                   # Python dependencies
├── utils.py                           # Utility functions
├── transformers/                      # Modified transformers library
│   └── moe_infinity/                  # MoE Infinity implementation
│       ├── expert_tracer.py           # Trace collection and management
│       ├── expert_predictor.py        # Expert prediction logic
│       ├── expert_prefetcher.py       # Expert prefetching
│       ├── expert_prediction_mixin.py # Generation mixin
│       └── ...
├── LLaMA-MoE-v1-3_5B-4_16_cuda_finemoe_eval/  # Model weights
├── dedup_data_eps0.1/                 # Deduplicated trace data
├── HNSW_python_implementation/        # HNSW index implementation
└── IVF_python_implementation/         # IVF index implementation
```

## Setup Instructions

### 1. Clone/Download

```bash
cd /path/to/genai/genai_moe_loader/experiments/transformer
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

**For Linux:**
```bash
pip install -r requirements.txt
```

**For Windows:**
```bash
pip install -r requirements_win.txt
```

The `requirements_win.txt` includes additional CUDA bindings and Windows-specific dependencies.

### 4. Download Model Weights

The model weights are stored in `LLaMA-MoE-v1-3_5B-4_16_cuda_finemoe_eval/`. If not included in the repository, download them separately.

### 5. Prepare Trace Data

Place your deduplicated trace data in `dedup_data_eps0.1/` folder. The trace files should be in `.npy` format.

## Running the Benchmark

### Option 1: Using Jupyter Notebook (Recommended)

```bash
# Activate venv first
source venv/bin/activate

# Start Jupyter
jupyter notebook POC-inference-cuda-bench.ipynb
```

Then run all cells in the notebook.

### Option 2: Using nbconvert (Headless)

```bash
# Activate venv first
source venv/bin/activate

# Execute notebook and save output
python -m nbconvert --to notebook --execute POC-inference-cuda-bench.ipynb --output POC-inference-cuda-bench-executed.ipynb
```

### Option 3: Run Python Script Directly

Convert the notebook to a Python script or run individual components:

```bash
source venv/bin/activate
python diagnose_peak_memory.py
```

## Configuration

### Path Configuration

Update the following paths in `POC-inference-cuda-bench.ipynb` if your folder structure differs:

```python
# transformers library path
sys.path.insert(0, '/home/appfw/hardik/genai/genai_moe_loader/experiments/transformer/transformers')

# Trace data path (used in generate() calls)
trace_folder="/home/appfw/hardik/genai/genai_moe_loader/experiments/transformer/dedup_data_eps0.1"
```

### Memory Optimization

The notebook sets `MALLOC_ARENA_MAX=2` to reduce memory fragmentation during expert loading. This is critical for large MoE models.

## Key Components

### MoE Infinity (`transformers/moe_infinity/`)

| File | Description |
|------|-------------|
| `expert_tracer.py` | Collects and manages execution traces for expert activation patterns |
| `expert_predictor.py` | Predicts which experts will be needed based on traces |
| `expert_prefetcher.py` | Handles async prefetching of experts from disk/swap |
| `expert_prediction_mixin.py` | Mixin class that patches generation for expert prediction |
| `expert_entry.py` | Data structures for expert entries |
| `expert_cache.py` | Caching mechanisms for loaded experts |
| `expert_priority_score.py` | Priority scoring for expert eviction |

### Vector Search Implementations

- **IVF (Inverted File Index)**: Fast approximate nearest neighbor search
- **HNSW (Hierarchical Navigable Small World)**: Alternative graph-based index

## Benchmark Metrics

The benchmark notebook measures:

1. **Model Load Time**: Time to load model weights
2. **Offline Processing**: IVF index building time (excluded from inference metrics)
3. **Inference Performance**:
   - TTFT (Time To First Token)
   - Average token generation time
   - Tokens per second throughput
4. **Memory Usage**:
   - RSS (Resident Set Size) before/after inference
   - Peak RSS during inference
   - CUDA memory delta (if GPU available)

## Troubleshooting

### FileNotFoundError for trace data

Ensure the `dedup_data_eps0.1/` folder contains `.npy` trace files and the path in the notebook is correct.

### Out of Memory Errors

1. Reduce `max_new_tokens` in the generate call
2. Ensure `MALLOC_ARENA_MAX=2` is set before importing torch/numpy
3. Close other memory-intensive applications

### CUDA not available

The benchmark can run on CPU. CUDA is optional but recommended for faster inference.

## Dependencies

Key dependencies (see `requirements.txt` for full list):

- `torch>=2.0.0` - PyTorch
- `transformers>=4.50.0` - Hugging Face Transformers (modified)
- `numpy>=1.24.0` - Numerical computing
- `psutil>=5.9.0` - Process and memory monitoring
- `jupyter>=1.0.0` - Notebook environment
- `nbconvert>=7.0.0` - Notebook execution

## License

This project is for research and experimental purposes.

## Contact

For issues or questions, please open an issue in the repository.
