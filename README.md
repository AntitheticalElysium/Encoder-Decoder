# Encoder-Decoder with Attention

English → French neural machine translation built from scratch in CuPy.

## Architecture

- **Encoder**: 2-layer bidirectional GRU
- **Decoder**: 2-layer GRU with Bahdanau attention
- **Tokenization**: BPE (1000 merges)
- **Optimizer**: Adam with gradient clipping
- **Implementation**: Pure CuPy (no PyTorch/TensorFlow)

All layers, backpropagation, and attention mechanisms implemented manually.

## What I Learned

- Implementing GRU cells and recurrent backpropagation through time
- Building attention mechanisms from scratch (projection, alignment scoring, context vectors)
- Manual gradient computation for complex architectures
- BPE tokenization and optimization (reduced O(n²) complexity with caching)
- Training dynamics: gradient clipping, learning rate scheduling, checkpoint management
- GPU programming with CuPy for deep learning primitives

## Pipeline

### 1. Prepare Data
```bash
python -m src.data
```
Downloads fra-eng corpus, trains BPE, builds vocabularies, and preprocesses ~163k sentence pairs.

### 2. Train Model
```bash
python -m src.train
```
Trains for 5000 iterations (~20-30 min on GPU). Saves checkpoints to `models/seq2seq.pkl`.

### 3. Evaluate
```bash
python -m src.eval
```
Translates test phrases using the trained model.

## Requirements

```bash
pip install numpy cupy d2l torch
```

Requires CUDA-compatible GPU.
