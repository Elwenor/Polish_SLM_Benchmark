#!/usr/bin/env python3
"""Polish SLM Benchmark — OpenPL 10-task scorer."""

from __future__ import annotations
import argparse, csv, importlib.util, json, math, re, time
from collections import Counter, defaultdict
from pathlib import Path
import torch
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM

SCORER_VERSION = "1.0.1"

OPENPL_TASKS = [
    "polemo2_in_multiple_choice", "polemo2_out_multiple_choice",
    "polish_8tags_multiple_choice", "polish_belebele_mc",
    "polish_cbd_multiple_choice", "polish_dyk_multiple_choice",
    "polish_klej_ner_multiple_choice", "polish_polqa_reranking_multiple_choice",
    "polish_ppc_multiple_choice", "polish_psc_multiple_choice",
]

# -- scoring --
SCORING = {
    "polemo2_in_multiple_choice": ("pmi", "accuracy"),
    "polemo2_out_multiple_choice": ("pmi", "accuracy"),
    "polish_8tags_multiple_choice": ("pmi", "accuracy"),
    "polish_belebele_mc": ("raw", "accuracy"),
    "polish_cbd_multiple_choice": ("pmi", "macro_f1_6"),
    "polish_dyk_multiple_choice": ("pmi", "binary_f1"),
    "polish_klej_ner_multiple_choice": ("pmi", "accuracy"),
    "polish_polqa_reranking_multiple_choice": ("raw", "accuracy"),
    "polish_ppc_multiple_choice": ("pmi", "accuracy"),
    "polish_psc_multiple_choice": ("pmi", "binary_f1"),
}

POLEMO_LABELS = {"__label__meta_zero": 0, "__label__meta_minus_m": 1,
                  "__label__meta_plus_m": 2, "__label__meta_amb": 3}
# -- etykiety --
KLEJ_NER_LABELS = {"noEntity": 0, "placeName": 1, "persName": 2, "orgName": 3,
                    "time": 4, "timeName": 4, "date": 4, "dateName": 4, "geogName": 5}
CBD_LABELS = {"szyderstwo": 1, "obelga": 2, "insynuacja": 3,
              "grozba": 4, "groźba": 4, "molestowanie": 5}
CONTENT_FIELDS = ("sentence", "text", "title", "review", "opinion",
                   "question", "query", "passage", "premise", "hypothesis")

PMI_CONTENT_FIELDS = {
    "polish_cbd_multiple_choice": ("TEXT",),
    "polish_ppc_multiple_choice": ("sentence_A", "sentence_B"),
    "polish_psc_multiple_choice": ("extract_text", "summary_text"),
}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source", choices=("hf", "adapter"), default="hf")
    p.add_argument("--model", default=None)
    p.add_argument("--revision", default=None)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--adapter-file", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32", "auto"), default="bf16")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=float, default=None, help="tylko do debugowania")
    p.add_argument("--raw-match-tolerance-pp", type=float, default=0.05)
    p.add_argument("--out", default="./openpl_final_run")
    return p.parse_args()


# -- helpers --

def pct(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.2f}%"

def json_default(o):
    if isinstance(o, Path): return str(o)
    if isinstance(o, torch.Tensor): return o.item() if o.numel() == 1 else o.tolist()
    if isinstance(o, set): return sorted(o)
    return str(o)

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")

def argmax(xs):
    return max(range(len(xs)), key=lambda i: xs[i])

def mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# -- model --

def build_lm(args):
    if args.source == "adapter":
        if not args.adapter_file or not args.checkpoint:
            raise SystemExit("--source adapter wymaga --adapter-file i --checkpoint")
        path = Path(args.adapter_file).expanduser().resolve()
        spec = importlib.util.spec_from_file_location("openpl_adapter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        lm = module.build_lm(checkpoint=Path(args.checkpoint).expanduser().resolve(),
                              device=args.device, batch_size=args.batch_size,
                              dtype=args.dtype, args=args)
        assert hasattr(lm, "loglikelihood") and hasattr(lm, "tokenizer")
        return lm

    if not args.model:
        raise SystemExit("--source hf wymaga --model")
    dtype = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "auto": "auto"}[args.dtype]
    kwargs = dict(pretrained=args.model, device=args.device, batch_size=args.batch_size,
                  dtype=dtype, trust_remote_code=args.trust_remote_code)
    if args.revision:
        kwargs["revision"] = args.revision
    try:
        return HFLM(**kwargs)
    except TypeError as e:
        # -- starszy HFLM --
        if "trust_remote_code" not in str(e):
            raise
        kwargs.pop("trust_remote_code")
        print("[model] ten fork HFLM nie przyjmuje trust_remote_cod)")
        return HFLM(**kwargs)


# -- requests --

class PairRequest:
    def __init__(self, context, continuation):
        self.args = (context, continuation)


def run_full_suite(args, lm):
    original_ll = lm.loglikelihood
    captured = []

    def wrapped(requests):
        reqs = list(requests)
        outs = original_ll(reqs)
        for req, out in zip(reqs, outs):
            args_ = getattr(req, "args", None) or req.arguments
            ll = float(out[0] if isinstance(out, (tuple, list)) else out)
            meta = getattr(req, "metadata", {}) or {}
            captured.append({
                "task_name": getattr(req, "task_name", None) or meta.get("task_name"),
                "doc_id": getattr(req, "doc_id", None) if getattr(req, "doc_id", None) is not None else meta.get("doc_id"),
                "doc": getattr(req, "doc", None),
                "context": str(args_[0]), "continuation": str(args_[1]), "ll": ll,
            })
        return outs

    lm.loglikelihood = wrapped
    t0 = time.perf_counter()
    try:
        official = evaluator.simple_evaluate(
            model=lm, tasks=OPENPL_TASKS, num_fewshot=args.num_fewshot,
            batch_size=args.batch_size, device=args.device, limit=args.limit,
            log_samples=False, task_manager=tasks.TaskManager(),
        )
    finally:
        lm.loglikelihood = original_ll
    return original_ll, captured, official, time.perf_counter() - t0


def group_by_example(items):
    if not items:
        return []
    if all(x["doc_id"] is not None for x in items):
        grouped, order = defaultdict(list), []
        for x in items:
            if x["doc_id"] not in grouped:
                order.append(x["doc_id"])
            grouped[x["doc_id"]].append(x)
        groups = [grouped[k] for k in order]
    else:
        groups, cur, last_ctx = [], [], None
        for x in items:
            if cur and x["context"] != last_ctx:
                groups.append(cur); cur = []
            cur.append(x); last_ctx = x["context"]
        if cur:
            groups.append(cur)

    examples = []
    for i, g in enumerate(groups):
        if len({x["context"] for x in g}) != 1:
            continue
        examples.append({
            "doc_id": g[0]["doc_id"] if g[0]["doc_id"] is not None else i,
            "doc": g[0]["doc"], "context": g[0]["context"],
            "choices": [x["continuation"] for x in g], "lls": [x["ll"] for x in g],
        })
    return examples


# -- gold --

def exact_gold(task, ex):
    doc = ex.get("doc") or {}

    if task in ("polemo2_in_multiple_choice", "polemo2_out_multiple_choice"):
        return POLEMO_LABELS.get(doc.get("target"))
    if task == "polish_8tags_multiple_choice":
        return int(doc["label"]) if "label" in doc else None
    if task == "polish_belebele_mc":
        n = len(ex["choices"])
        for k in ("correct_answer_num", "answer", "label"):
            if k in doc:
                v = int(doc[k])
                # -- indeks od 1 --
                if 1 <= v <= n: return v - 1
                if 0 <= v < n: return v
        return None
    if task == "polish_cbd_multiple_choice":
        v = doc.get("CATEGORIES")
        if v is None:
            # -- klasa czysta --
            return 0
        return CBD_LABELS.get(str(v).strip().lower())
    if task == "polish_dyk_multiple_choice":
        for k in ("target", "label"):
            if k in doc: return int(doc[k])
        return None
    if task == "polish_klej_ner_multiple_choice":
        return KLEJ_NER_LABELS.get(doc.get("target"))
    if task == "polish_polqa_reranking_multiple_choice":
        for k in ("relevant", "label", "target"):
            if k in doc: return int(doc[k])
        return None
    if task == "polish_ppc_multiple_choice":
        # -- indeks od 1 --
        return int(doc["label"]) - 1 if "label" in doc else None
    if task == "polish_psc_multiple_choice":
        for k in ("label", "target"):
            if k in doc: return int(doc[k])
        return None
    return None


# -- domain PMI --

def blank_context(task, context, doc):
    fields = PMI_CONTENT_FIELDS.get(task)

    if fields:
        values = []
        complete = True
        for key in fields:
            value = doc.get(key)
            if not isinstance(value, str) or not value.strip() or value not in context:
                complete = False
                break
            values.append(value)

        if complete:
            blank = context
            for value in values:
                blank = blank.replace(value, "", 1)
            return blank, False

    candidates = sorted(
        ((len(doc[k]), doc[k]) for k in CONTENT_FIELDS if isinstance(doc.get(k), str) and doc[k].strip()),
        reverse=True,
    )
    for _, value in candidates:
        if value in context:
            return context.replace(value, "", 1), False

    # -- fallback PMI --
    replaced = re.sub(r'(")(.*?)(\")', r"\1\3", context, count=1, flags=re.S)
    if replaced != context:
        return replaced, True
    lines = context.splitlines()
    return ("\n" + "\n".join(lines[1:]) if len(lines) > 1 else ""), True


def score_unique_pairs(original_ll, pairs, batch_size):
    unique = list(dict.fromkeys(pairs))
    scores = {}
    for i in range(0, len(unique), batch_size):
        chunk = unique[i:i + batch_size]
        outs = original_ll([PairRequest(c, k) for c, k in chunk])
        for pair, out in zip(chunk, outs):
            scores[pair] = float(out[0] if isinstance(out, (tuple, list)) else out)
    return scores


def attach_scores(args, original_ll, task, examples):
    """Zwraca (dropped_gold_none, pmi_fallback_n) — ile przykladow odrzucono bo
    gold sie nie odtworzyl"""
    dropped_gold_none = 0
    for ex in examples:
        ex["gold"] = exact_gold(task, ex)
        if ex["gold"] is None:
            dropped_gold_none += 1

    if SCORING[task][0] != "pmi":
        for ex in examples:
            ex["scores"] = {"raw": ex["lls"]}
        return dropped_gold_none, 0

    pmi_fallback_n = 0
    for ex in examples:
        ex["blank_context"], used_fallback = blank_context(task, ex["context"], ex.get("doc") or {})
        pmi_fallback_n += int(used_fallback)
    pairs = [(ex["blank_context"], c) for ex in examples for c in ex["choices"]]
    baseline = score_unique_pairs(original_ll, pairs, args.batch_size)

    for ex in examples:
        ex["scores"] = {
            "raw": ex["lls"],
            "pmi": [ll - baseline[(ex["blank_context"], c)] for ll, c in zip(ex["lls"], ex["choices"])],
        }

    return dropped_gold_none, pmi_fallback_n


# -- metrics --

def accuracy(y_true, y_pred):
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true) if y_true else None

def binary_f1(y_true, y_pred, pos=1):
    tp = sum(y == pos and p == pos for y, p in zip(y_true, y_pred))
    fp = sum(y != pos and p == pos for y, p in zip(y_true, y_pred))
    fn = sum(y == pos and p != pos for y, p in zip(y_true, y_pred))
    return 0.0 if 2*tp+fp+fn == 0 else 2*tp / (2*tp+fp+fn)

def macro_f1(y_true, y_pred, classes):
    return mean(binary_f1(y_true, y_pred, pos=c) for c in classes)

def balanced_accuracy(y_true, y_pred):
    recalls = []
    for c in sorted(set(y_true)):
        support = sum(y == c for y in y_true)
        if support:
            recalls.append(sum(y == c and p == c for y, p in zip(y_true, y_pred)) / support)
    return mean(recalls)

def official_metric(official, task, metric):
    for k, v in official.get("results", {}).get(task, {}).items():
        if k.split(",", 1)[0] == metric:
            return float(v)
    return None


def score_task(task, examples):
    rule, metric = SCORING[task]
    y_true, y_pred = [], []
    for ex in examples:
        gold, scores = ex.get("gold"), ex.get("scores", {}).get(rule)
        if gold is None or scores is None or not (0 <= gold < len(scores)):
            continue
        y_true.append(int(gold))
        y_pred.append(argmax(scores))

    final = {"accuracy": accuracy, "binary_f1": binary_f1,
              "macro_f1_6": lambda t, p: macro_f1(t, p, range(6))}[metric](y_true, y_pred)

    return {"task": task, "prediction_rule": rule, "metric": metric, "final_score": final,
            "n": len(y_true), "accuracy": accuracy(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy(y_true, y_pred),
            "gold_histogram": dict(sorted(Counter(y_true).items())),
            "prediction_histogram": dict(sorted(Counter(y_pred).items()))}


# -- main --

def main():
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"scorer={SCORER_VERSION} source={args.source} model={args.model or args.checkpoint} out={out_dir}")

    t0 = time.perf_counter()
    lm = build_lm(args)
    print(f"[model] zaladowany w {time.perf_counter()-t0:.2f}s")

    original_ll, captured, official, eval_s = run_full_suite(args, lm)
    write_json(out_dir / "official_lm_eval_results.json", official)
    write_jsonl(out_dir / "captured_requests.jsonl", captured)
    print(f"[official] requests={len(captured):,} time={eval_s:.2f}s")

    by_task = defaultdict(list)
    for item in captured:
        if item["task_name"] in OPENPL_TASKS:
            by_task[item["task_name"]].append(item)

    rows = []
    for task in OPENPL_TASKS:
        print(f"\n{task}")
        examples = group_by_example(by_task.get(task, []))
        dropped_gold_none, pmi_fallback_n = attach_scores(args, original_ll, task, examples)

        raw_true = [int(ex["gold"]) for ex in examples if ex.get("gold") is not None]
        raw_pred = [argmax(ex["lls"]) for ex in examples if ex.get("gold") is not None]
        reconstructed = accuracy(raw_true, raw_pred)
        official_acc = official_metric(official, task, "acc")
        official_acc_norm = official_metric(official, task, "acc_norm")
        diff_pp = abs(reconstructed - official_acc) * 100 if reconstructed is not None and official_acc is not None else None
        sanity_ok = diff_pp is not None and diff_pp <= args.raw_match_tolerance_pp

        result = score_task(task, examples)
        if not sanity_ok:
            print(f"  [SANITY FAIL] raw={pct(reconstructed)} official={pct(official_acc)} diff={diff_pp} -> FINAL=None")
            result["final_score"] = None
        result.update(official_acc=official_acc, official_acc_norm=official_acc_norm,
                       reconstructed_raw_acc=reconstructed, sanity_ok=sanity_ok,
                       dropped_gold_none=dropped_gold_none, pmi_fallback_n=pmi_fallback_n)

        print(f"  official ACC={pct(official_acc)} ACC_NORM={pct(official_acc_norm)} "
              f"FINAL={pct(result['final_score'])} n={result['n']}"
              + (f" dropped={dropped_gold_none}" if dropped_gold_none else "")
              + (f" pmi_fallback={pmi_fallback_n}" if pmi_fallback_n else ""))

        task_dir = out_dir / task
        task_dir.mkdir(exist_ok=True)
        write_jsonl(task_dir / "per_example.jsonl", examples)
        write_json(task_dir / "final_task_result.json", result)
        rows.append(result)

    # -- composite z 10 zadan --
    n_tasks_ok = sum(1 for r in rows if r["final_score"] is not None)
    final_macro = sum((r["final_score"] or 0.0) for r in rows) / len(rows) if rows else None
    # -- diagnostyka --
    final_macro_ok_only = mean(r["final_score"] for r in rows)
    official_norm_macro = mean(r["official_acc_norm"] for r in rows)

    summary = {"scorer_version": SCORER_VERSION,
               "model": args.model, "checkpoint": args.checkpoint,
               "official_acc_norm_macro": official_norm_macro,
               "final_composite_macro": final_macro,
               "n_tasks_ok": n_tasks_ok, "n_tasks_total": len(rows),
               "final_composite_macro_ok_tasks_only": final_macro_ok_only,
               "tasks": rows}
    write_json(out_dir / "final_results.json", summary)

    with (out_dir / "final_results.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["task", "official_acc", "official_acc_norm", "prediction_rule", "metric",
                   "final_score", "accuracy", "balanced_accuracy", "n", "sanity_ok"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)

    print(f"\n{'task':38s}{'official':>10s}{'FINAL':>10s}{'n':>7s}")
    for r in rows:
        print(f"{r['task'][:38]:38s}{pct(r['official_acc_norm']):>10s}{pct(r['final_score']):>10s}{r['n']:>7d}")
    print(f"{'MACRO':38s}{pct(official_norm_macro):>10s}{pct(final_macro):>10s}")
    if n_tasks_ok < len(rows):
        print(f"  UWAGA: tylko {n_tasks_ok}/{len(rows)} zadan przeszlo sanity check; "
              f"pozostale liczone jako 0 w FINAL (surowa srednia z {n_tasks_ok} "
              f"wiarygodnych zadan to {pct(final_macro_ok_only)})")
    print(f"\nzapisano: {out_dir / 'final_results.json'}")


if __name__ == "__main__":
    main()