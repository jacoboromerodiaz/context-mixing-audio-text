# Context Mixing: Audio + Text Attribution

This repository provides tools to analyze **context mixing** in multimodal Speech-to-Text Translation (S2TT) systems, specifically how **audio** and **textual transcripts** contribute to model predictions.  
It uses **Captum** to compute token-level attributions over both modalities.

---

## Overview

The project includes:
- **Attribution analysis** (`src/cmix/main.py`): unified pipeline for gradient- and perturbation-based interpretability.
- **Data management** (`src/cmix/data.py`): utilities for loading model outputs, alignments, and templates.
- **Utility functions** (`src/cmix/utils.py`): plotting, normalization, and grouped attribution scoring.
- **Optional forced alignment** (`src/cmix/alignment/forced_alignment.py`): aligns HuBERT tokens to word spans using *torchaudio* MMS.

---

## Motivation

When evaluating multimodal models that mix **audio** and **text**, it is crucial to understand *which modality the model actually relies on*.  
This repository provides a reproducible framework to:
- Quantify audio vs. text influence via attribution maps.
- Visualize token-level importance and word-grouped scores.
- Compare different architectures (e.g., Chain-of-Thought vs. Cascade).

---

## Repository Structure

```
context-mixing-audio-text/
│
├── config/
│   ├── attribution/
│   │   └── fleurs_iber_en_xx.yaml          # Example attribution config
│   └── alignment/
│       └── fleurs_iber_alignment.yaml      # Example forced alignment config
│
├── scripts/                                # SLURM or run scripts
│
├── src/
│   └── cmix/
│       ├── main.py                         # Main attribution runner
│       ├── data.py                         # Data loading & prompt handling
│       ├── utils.py                        # Attribution helpers & plotting
│       ├── distribute.py                   # Optional distributed execution
│       └── alignment/
│           ├── forced_alignment.py         # Optional audio-token alignment
│           └── README.md                   # Alignment documentation
│
└── README.md                               # Project documentation
```

---

## Configuration

All runs are driven by **YAML configuration files**, located in `config/`.

### Example (Attribution)

```yaml
checkpoint_path: /path/to/llm/checkpoint/
results_dir: /path/to/entire_sequence/files/
aligned_huberts_dir: /path/to/aligned_huberts/
output_dir: /path/to/outputs/

attr_params:
  method: shapley-values
  skip_tokens: ["<|im_end|>", "<|im_start|>", "user", "assistant"]
  attribution_kwargs: {"n_samples": 25}

data:
  lang_pairs: ["en_es", "en_ca", "en_pt"]
  chunk_size: 1
  input_translation: true
```

---

## Running Attribution

```bash
python -m cmix.main --config config/attribution/fleurs_iber_en_xx.yaml
```

The script will:
1. Load the model checkpoint and tokenizer.
2. Parse generation results (`.entire_sequence.txt`) and aligned HuBERT files.
3. Compute attribution maps using the chosen Captum method.
4. Save results (`scores.tsv`, per-language) and plots under `output_dir`.

---

## Optional: Forced Alignment

If your S2TT system does not provide aligned or deduplicated HuBERT tokens,  
you can run the **forced alignment** preprocessing stage to build `.aligned_huberts.txt` files.

```bash
python -m cmix.alignment.forced_alignment --config config/alignment/fleurs_iber_alignment.yaml
```

See [`src/cmix/alignment/README.md`](src/cmix/alignment/README.md) for detailed usage and examples.

---

## Outputs

After running the attribution pipeline, each `{lang_pair}` directory will include:

```
scores.tsv                # aggregated audio/text contribution scores
plots/sample_XX.png       # token-level heatmaps
```

Mean values per modality (audio, transcript, translation, etc.) are printed to the console.

---

## Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.1  
- Torchaudio ≥ 2.1  
- Captum ≥ 0.7  
- Transformers ≥ 4.40  
- tqdm, matplotlib, pyyaml

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Project Context

This repository is used for some of the experiments described in:

> Romero-Díaz, J. *Listening or Reading? Evaluating Speech Awareness in Chain-of-Thought Speech-to-Text Translation.*  
> ICASSP 2026 (under review)

It forms part of ongoing research at the **Barcelona Supercomputing Center (BSC-CNS)** Language Technologies Lab.

---

## Citation

If you use this work, please cite:

```
@article{romerodiaz2025cot_s2tt,
  title   = {Listening or Reading? Evaluating Speech Awareness in Chain-of-Thought Speech-to-Text Translation},
  author  = {Romero-Díaz, Jacobo and Gállego, Gerard I. and Pareras, Oriol and Costa, Federico and Hernando, Javier and España-Bonet, Cristina},
  journal = {arXiv preprint arXiv:2510.03115},
  year    = {2025}
}
```

Available at: [https://arxiv.org/abs/2510.03115](https://arxiv.org/abs/2510.03115)
