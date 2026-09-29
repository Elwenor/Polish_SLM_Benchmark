# Polish SLM Benchmark

Minimal runner for evaluating small Polish causal language models on 10 OpenPL tasks.

Current scorer version: **1.0.1**

The benchmark uses a pinned revision of the SpeakLeash fork of
`lm-evaluation-harness`, which contains the OpenPL task definitions required
by the scorer.

## Installation

Python **3.12** is recommended.

Clone the repository:

```bash
git clone https://github.com/Elwenor/Polish_SLM_Benchmark.git
cd Polish_SLM_Benchmark
```

Create a clean virtual environment.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

## PyTorch and accelerator support

Install PyTorch for your hardware **before** installing the remaining benchmark dependencies.

The correct PyTorch build depends on your platform and accelerator:

- **NVIDIA GPU**: install a CUDA-enabled PyTorch build
- **AMD GPU on supported Linux systems**: install a ROCm-enabled PyTorch build
- **CPU-only**: install the standard CPU build
- **macOS / Apple Silicon**: PyTorch may use the MPS backend where supported

Use the official PyTorch installation instructions for the appropriate command:

https://pytorch.org/get-started/locally/

### NVIDIA / CUDA

After installing PyTorch, verify that the current Python environment can see your NVIDIA GPU:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

For GPU evaluation with:

```text
--device cuda:0
```

you should see:

```text
CUDA available: True
```

A result such as:

```text
torch: 2.x.x+cpu
CUDA available: False
CUDA runtime: None
```

means that the current Python environment contains a CPU-only PyTorch build,
even if NVIDIA drivers or the CUDA Toolkit are installed system-wide.

Installing CUDA on the operating system is not enough by itself. The Python
environment running the benchmark must also contain a CUDA-enabled PyTorch build.

### AMD / ROCm

On supported Linux systems, AMD GPUs can be used through a ROCm-enabled PyTorch build.

Verify the installation with:

```bash
python -c "import torch; print('torch:', torch.__version__); print('HIP:', torch.version.hip); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

PyTorch uses the `torch.cuda` API for both CUDA and ROCm devices, so AMD GPUs running through ROCm are typically still addressed as:

```text
--device cuda:0
```

This naming is inherited from PyTorch and does not mean that an AMD GPU is using NVIDIA CUDA.

ROCm support depends on the operating system, GPU generation, driver stack and PyTorch build. Check the current PyTorch/ROCm documentation for supported configurations.

### CPU

CPU execution is supported with:

```text
--device cpu
```

but full benchmark runs will be substantially slower.

## Install benchmark dependencies

After installing a suitable PyTorch build:

```bash
pip install -r requirements.txt
```

The required OpenPL tasks are provided by the SpeakLeash fork of `lm-evaluation-harness`, pinned to:

```text
21d0ea9cf4fd6153dfff4d84d6ad0aab5488f302
```

No separate clone of `lm-evaluation-harness` is required.

## Verify the environment

Before running an evaluation:

```bash
python check_environment.py
```

A compatible installation should end with:

```text
Environment OK: all 10 required OpenPL tasks found.
```

The checker verifies:

- `lm_eval`
- `TaskManager`
- Hugging Face `HFLM`
- all 10 required OpenPL tasks
- installed package versions
- accelerator availability reported by PyTorch
- the pinned `lm-evaluation-harness` revision when available from package metadata

The checker does **not** require a GPU. CPU-only environments are valid, but GPU-specific runs will only work if the corresponding PyTorch backend is available in the current Python environment.

## Reference environment

The benchmark has been validated in the following reference environment:

```text
Python        3.12.2
lm_eval       0.4.2
Transformers  5.5.0
datasets      3.6.0
accelerate    1.13.0
PyTorch       2.9.1+cu130
CUDA runtime  13.0
```

This is a reference configuration, not a mandatory hardware requirement.

See `requirements-tested.txt` for the package versions used in the reference environment.

Exact floating-point results may vary slightly across hardware, accelerator backends, drivers, PyTorch versions and numerical precision settings.

## Usage

The examples below use `cuda:0`.

- For CPU evaluation, use `--device cpu`.
- On ROCm-enabled AMD systems, PyTorch typically also exposes the GPU through the `cuda:*` device namespace.

Example evaluation of a Hugging Face causal language model:

```powershell
python .\Polish_SLM_Benchmark_v1.0.1.py `
  --source hf `
  --model "SlayerLab/GoLLeM-110M-PL-v3" `
  --device cuda:0 `
  --dtype bf16 `
  --batch-size 8 `
  --out ".\results_gollem_v3"
```

Linux / macOS equivalent:

```bash
python Polish_SLM_Benchmark_v1.0.1.py \
  --source hf \
  --model "SlayerLab/GoLLeM-110M-PL-v3" \
  --device cuda:0 \
  --dtype bf16 \
  --batch-size 8 \
  --out "./results_gollem_v3"
```

A local Transformers model can be evaluated by passing its local directory to `--model`.

For a quick smoke test, use `--limit`:

```powershell
python .\Polish_SLM_Benchmark_v1.0.1.py `
  --source hf `
  --model "SlayerLab/GoLLeM-110M-PL-v3" `
  --device cuda:0 `
  --dtype bf16 `
  --batch-size 8 `
  --limit 2 `
  --out ".\smoke_test"
```

`--limit` is intended for debugging only. Results obtained with a limit should not be reported as full benchmark scores.

## Custom models and checkpoints

Models that are not directly loadable through Hugging Face Transformers can be evaluated through an adapter.

Example:

```powershell
python .\Polish_SLM_Benchmark_v1.0.1.py `
  --source adapter `
  --adapter-file ".\my_openpl_adapter.py" `
  --checkpoint ".\checkpoints\model.pt" `
  --device cuda:0 `
  --batch-size 8 `
  --dtype bf16 `
  --out ".\results_custom_model"
```

The adapter must expose:

```python
build_lm(checkpoint, device, batch_size, dtype, args)
```

and return an object compatible with the evaluation harness, including at least:

```text
loglikelihood
tokenizer
```

This makes the scorer independent of the training framework. The benchmark is an **evaluation tool**, not a training framework.

## OpenPL tasks

The benchmark evaluates the following 10 tasks:

```text
polemo2_in_multiple_choice
polemo2_out_multiple_choice
polish_8tags_multiple_choice
polish_belebele_mc
polish_cbd_multiple_choice
polish_dyk_multiple_choice
polish_klej_ner_multiple_choice
polish_polqa_reranking_multiple_choice
polish_ppc_multiple_choice
polish_psc_multiple_choice
```

## Scoring

Primary scoring:

```text
PolEmo2 IN / OUT  domain PMI + accuracy
8Tags             domain PMI + accuracy
Belebele          raw likelihood + accuracy
CBD               domain PMI + macro-F1
DYK               domain PMI + binary F1
KLEJ NER          domain PMI + accuracy
PolQA             raw likelihood + accuracy
PPC               domain PMI + accuracy
PSC               domain PMI + binary F1
```

The final benchmark score is the **unweighted mean of the 10 primary task scores**.

It is **not** the original OpenPL `AVG acc_norm`.

The scorer also records the original `lm-evaluation-harness` metrics for diagnostic purposes.

## Sanity checks

For every task, the scorer reconstructs raw accuracy from the captured log-likelihood requests and compares it with the corresponding accuracy reported by `lm-evaluation-harness`.

If the reconstructed score differs from the harness result by more than the configured tolerance, the task fails the sanity check and its final benchmark score is not trusted.

This is intended to catch problems such as:

- incompatible task definitions
- incorrect label reconstruction
- unexpected harness behavior
- scorer/task version mismatches

## Output

Each run creates an output directory containing:

```text
final_results.json
final_results.csv
official_lm_eval_results.json
captured_requests.jsonl

<task>/
  final_task_result.json
  per_example.jsonl
```

`final_results.json` contains the benchmark summary and per-task scores.

The per-example files are useful for auditing label mappings, likelihoods, PMI corrections and unexpected model behavior.

## Reproducibility

The benchmark depends on the OpenPL task definitions present in the pinned SpeakLeash `lm-evaluation-harness` revision.

The reference revision is:

```text
https://github.com/speakleash/lm-evaluation-harness
commit: 21d0ea9cf4fd6153dfff4d84d6ad0aab5488f302
```

Using another revision of `lm-evaluation-harness` may change:

- task prompts
- dataset configuration
- label mappings
- request construction
- metrics
- final scores

For comparable benchmark results, use the dependencies from this repository and verify the environment with:

```bash
python check_environment.py
```

GitHub Actions also runs a clean-environment smoke test on every push and pull request.

## v1.0.1

Version 1.0.1 fixes domain-PMI blanking for:

```text
CBD  -> TEXT
PPC  -> sentence_A + sentence_B
PSC  -> extract_text + summary_text
```

An independent GoLLeM evaluation highlighted discrepancies in these tasks and prompted a re-audit of the scorer.

Thanks to **Maggio33 / SlayerLab** for the independent evaluation:

https://huggingface.co/Maggio33/GoLLeM-110M-PL-v3

## Reporting results

When publishing benchmark results, please report at least:

```text
model name / checkpoint
model revision, if applicable
scorer version
final composite score
per-task scores
number of tasks passing the sanity check
```

For Hugging Face models, using an immutable model revision or commit hash is recommended when exact reproducibility is important.

## Issues

If you find a problem with:

- label mapping
- prompt blanking
- PMI baselines
- task metrics
- OpenPL reconstruction
- dependency compatibility

please open an issue with the affected task and enough information to reproduce the problem.

Useful reproduction information includes:

```text
Python version
PyTorch version
Transformers version
lm_eval version
CUDA / ROCm version, if applicable
GPU model, if applicable
model / checkpoint
scorer version
error message or affected task
```

Please do not include local filesystem paths, credentials, private model locations or other machine-specific information unless it is necessary to reproduce the issue.

## Citation

```bibtex
@misc{PolishSLMBenchmark,
  author = {Aleksander Ogrodzki},
  title  = {Polish SLM Benchmark},
  year   = {2026}
}
```
