# Polish SLM Benchmark

Minimal runner for evaluating small Polish causal language models on 10 OpenPL tasks.

Current scorer version: **1.0.1**

## Usage

```powershell
python .\Polish_SLM_Benchmark_v1.0.1.py `
  --source hf `
  --model "SlayerLab/GoLLeM-110M-PL-v3" `
  --device cuda:0 `
  --dtype bf16 `
  --batch-size 8 `
  --out ".\results_gollem_v3"
```

A local Transformers model can be used by passing its local directory to `--model`.

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

The final score is the unweighted mean of the 10 primary task scores.

It is **not** the original OpenPL `AVG acc_norm`.

## v1.0.1

Version 1.0.1 fixes domain-PMI blanking for:

```text
CBD  -> TEXT
PPC  -> sentence_A + sentence_B
PSC  -> extract_text + summary_text
```

An independent GoLLeM evaluation highlighted discrepancies in these tasks and prompted a re-audit of the scorer.

Thanks to **Maggio33 / SlayerLab**:

https://huggingface.co/Maggio33/GoLLeM-110M-PL-v3

## Issues

If you find a problem with label mapping, prompt blanking, PMI baselines, task metrics, or OpenPL reconstruction, please open an issue with the affected task and reproduction details.

## Citation

```bibtex
@misc{PolishSLMBenchmark,
  author = {Aleksander Ogrodzki},
  title  = {Polish SLM Benchmark},
  year   = {2026}
}
```
