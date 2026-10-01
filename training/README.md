# L2S1 Training

Installable Jev-type LoRA tools for SmolLM2, Qwen3, Gemma3, TinyLlama and Gemma4.
The `l2s1-train` command prepares data, downloads pinned checkpoints on request,
runs serial training/evaluation, and produces Choice/Noul/Score JSON answers.
The package imports without PyTorch; CUDA training is an optional dependency.

Install training tools 0.1.1 from the released SDK v0.2.2 source tag:

```sh
python -m pip install 'l2s1-training @ git+https://github.com/LuticaCANARD/L2S1.git@v0.2.2#subdirectory=training'
l2s1-train --version
```

For a local checkout, run from the repository root:

```sh
python -m pip install ./training
l2s1-train --help
l2s1-train profiles
l2s1-train doctor

# In a separate training environment with a suitable CUDA PyTorch wheel:
python -m pip install './training[cuda,parquet]'
l2s1-train doctor --cuda
```

This is a source/wheel installation, not a claim that the package is published
on PyPI. Python 3.11+ is required. CUDA training uses the pinned dependency set
in `pyproject.toml`; CPU-only installations can prepare JSONL and generate reports.
The registry is extensible. Only Gemma4 has completed the real-weight pilot;
other bundled profiles have tiny random architecture/gradient checks.

```sh
l2s1-train prepare --train training/examples/train.jsonl \
  --development training/examples/development.jsonl \
  --test training/examples/test.jsonl --output results/jev-example
```

The three example cases exercise the format only. Supply meaningful independent
splits for training; the examples cannot establish model quality.
See [the operational guide](https://github.com/LuticaCANARD/L2S1/blob/main/docs/JEV_LORA.md)
for checkpoint download, native binaries, full matrix execution and evidence limits.

The CLI and schema version 1 are supported repository interfaces. Historical
`scripts/*jev*.py` entry points delegate to this package. A changed dataset,
training implementation or checkpoint requires a fresh run and smoke binding.
Runs never overwrite existing output directories. No automatic adapter promotion,
resumption, threshold fitting, or model weight publication occurs.
