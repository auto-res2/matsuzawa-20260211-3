"""
Inference script for VMSCA and SC-majority methods.
Single run executor invoked by main.py as subprocess.
"""
import os
import sys
import json
import re
import math
import ast
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from omegaconf import DictConfig, OmegaConf
import wandb

from preprocess import (
    load_gsm8k,
    check_answer_correctness,
    build_few_shot_prompt,
    format_problem_prompt,
    normalize_numeric_answer
)


# ============ VMSCA Aggregation Logic (from experimental_code) ============

ANS_RE = re.compile(r"^\s*Answer\s*:\s*(.+?)\s*$", re.M)
SCORE_RE = re.compile(r"^\s*Score\s*:\s*([01](?:\.\d+)?)\s*$", re.M)
CHECK_RE = re.compile(r"^\s*Check\s*:\s*(.+?)\s*$", re.M)

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Tuple, ast.List
)


def safe_arith_eval(expr: str):
    """Evaluate a pure arithmetic expression safely (no variables, no function calls)."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed node: {type(node).__name__}")
    return eval(compile(tree, "<check>", "eval"), {"__builtins__": {}}, {})


def parse_fields(text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    """Parse Answer/Score/Check from completion."""
    a = ANS_RE.search(text)
    s = SCORE_RE.search(text)
    c = CHECK_RE.search(text)
    ans = a.group(1).strip() if a else None
    score = float(s.group(1)) if s else None
    if score is not None:
        score = max(0.0, min(1.0, score))
    check = c.group(1).strip() if c else None
    return ans, score, check


def _coerce_number(x: str):
    """Try int then float; else return original string."""
    try:
        if re.fullmatch(r"[-+]?\d+", x.strip()):
            return int(x)
        return float(x)
    except Exception:
        return x.strip()


def verify_check(ans_str: str, check_expr: str, tol: float = 1e-6) -> bool:
    """Verify that check expression matches answer."""
    if ans_str is None or check_expr is None:
        return False
    ans = _coerce_number(ans_str)
    try:
        val = safe_arith_eval(check_expr)
    except Exception:
        return False
    # numeric match if both numeric
    if isinstance(ans, (int, float)) and isinstance(val, (int, float)):
        return abs(float(ans) - float(val)) <= tol
    return str(ans).strip() == str(val).strip()


@dataclass
class SampleResult:
    """Individual sample metadata."""
    answer: Optional[str]
    score: Optional[float]
    check: Optional[str]
    verified: bool
    weight: float
    text: str


def vmsca_aggregate(samples: List[str], beta: float, delta: float, default_score: float) -> Tuple[Optional[str], float, List[SampleResult], Dict]:
    """
    Aggregate samples using VMSCA weighting.
    
    Returns:
        (predicted_answer, confidence, sample_results, metrics_dict)
    """
    tally = defaultdict(float)
    total_w = 0.0
    results = []
    
    verified_count = 0
    total_samples = len(samples)
    
    for text in samples:
        ans, score, check = parse_fields(text)
        
        if ans is None:
            # Cannot extract answer -> skip
            results.append(SampleResult(None, None, None, False, 0.0, text))
            continue
        
        if score is None:
            score = default_score
        
        v = verify_check(ans, check)
        if v:
            verified_count += 1
        
        w = math.exp(beta * score) * (1.0 if v else delta)
        tally[ans] += w
        total_w += w
        
        results.append(SampleResult(ans, score, check, v, w, text))
    
    if not tally:
        return None, 0.0, results, {
            "total_samples": total_samples,
            "verified_count": 0,
            "verification_pass_rate": 0.0,
            "ess": 0.0
        }
    
    best_ans, best_w = max(tally.items(), key=lambda kv: kv[1])
    conf = best_w / max(1e-12, sum(tally.values()))
    
    # Compute ESS
    weights = [r.weight for r in results if r.weight > 0]
    if weights:
        total = sum(weights)
        probs = [w / total for w in weights]
        ess = 1.0 / sum(p * p for p in probs)
    else:
        ess = 0.0
    
    metrics = {
        "total_samples": total_samples,
        "verified_count": verified_count,
        "verification_pass_rate": verified_count / max(1, total_samples),
        "ess": ess
    }
    
    return best_ans, conf, results, metrics


def sc_majority_aggregate(samples: List[str]) -> Tuple[Optional[str], float, List[str]]:
    """
    Simple majority vote for SC-majority baseline.
    
    Returns:
        (predicted_answer, confidence, parsed_answers)
    """
    answers = []
    for text in samples:
        # Parse Answer field (same regex)
        a = ANS_RE.search(text)
        if a:
            answers.append(a.group(1).strip())
        else:
            # Fallback: extract first number
            try:
                # Try to extract any numeric token
                match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
                if match:
                    answers.append(match.group(0))
            except:
                pass
    
    if not answers:
        return None, 0.0, []
    
    # Count votes
    counts = defaultdict(int)
    for ans in answers:
        counts[ans] += 1
    
    best_ans, best_count = max(counts.items(), key=lambda kv: kv[1])
    conf = best_count / len(answers)
    
    return best_ans, conf, answers


# ============ Model Inference ============

def load_model_and_tokenizer(model_name: str, cache_dir: str):
    """Load HuggingFace model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


def generate_samples(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_new_tokens: int,
    adaptive_stop: bool = False,
    tau: float = 0.85,
    beta: float = 2.0,
    delta: float = 0.05,
    default_score: float = 0.5
) -> List[str]:
    """
    Generate multiple samples for a single problem.
    
    If adaptive_stop is True (VMSCA), stop early when leading answer dominates.
    Otherwise generate all n_samples (SC-majority).
    """
    samples = []
    tally = defaultdict(float)
    total_w = 0.0
    
    for i in range(n_samples):
        # Generate one sample
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        samples.append(text)
        
        # Adaptive stopping check (VMSCA only)
        if adaptive_stop:
            ans, score, check = parse_fields(text)
            if ans is not None:
                if score is None:
                    score = default_score
                v = verify_check(ans, check)
                w = math.exp(beta * score) * (1.0 if v else delta)
                tally[ans] += w
                total_w += w
                
                if tally:
                    best_ans, best_w = max(tally.items(), key=lambda kv: kv[1])
                    conf = best_w / max(1e-12, sum(tally.values()))
                    if conf >= tau:
                        # Early stop
                        break
    
    return samples


# ============ Main Inference Loop ============

def run_inference(cfg: DictConfig):
    """
    Main inference loop for a single run.
    """
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
            mode=cfg.wandb.mode
        )
        print(f"WandB run URL: {wandb.run.get_url()}")
    else:
        wandb.init(mode="disabled")
    
    # Extract config
    method_name = cfg.run.method.name
    model_name = cfg.run.method.model_name
    cache_dir = cfg.run.dataset.cache_dir
    
    # Load dataset
    dataset_split = cfg.run.dataset.split
    max_samples = cfg.run.dataset.max_samples
    
    print(f"Loading {cfg.run.dataset.name} ({dataset_split}, max_samples={max_samples})...")
    items = load_gsm8k(dataset_split, max_samples, cache_dir)
    print(f"Loaded {len(items)} items")
    
    # Load model
    print(f"Loading model {model_name}...")
    model, tokenizer = load_model_and_tokenizer(model_name, cache_dir)
    print("Model loaded")
    
    # Build few-shot prompt
    require_format = cfg.run.method.get("require_format", False)
    few_shot_examples = cfg.run.method.get("few_shot_examples", 3)
    few_shot_prefix = build_few_shot_prompt(few_shot_examples, require_format)
    
    # Inference parameters
    temperature = cfg.run.method.temperature
    max_new_tokens = cfg.run.method.max_new_tokens
    
    # Method-specific parameters
    is_vmsca = (method_name == "VMSCA")
    
    if is_vmsca:
        m_max = cfg.run.method.m_max
        beta = cfg.run.method.beta
        delta = cfg.run.method.delta
        tau = cfg.run.method.tau
        default_score = cfg.run.method.default_score
        adaptive_stop = True
    else:
        # SC-majority
        m_max = cfg.run.method.m
        adaptive_stop = False
        beta = delta = tau = default_score = None
    
    # Results storage
    results = []
    
    # Metrics accumulators
    correct_count = 0
    confidently_wrong_count = 0
    total_samples_used = 0
    total_verified = 0
    total_generated = 0
    ess_values = []
    
    # Iterate over dataset
    for idx, item in enumerate(items):
        question = item["question"]
        gold = item["gold_numeric"]
        
        # Build prompt
        prompt = format_problem_prompt(question, few_shot_prefix)
        
        # Generate samples
        samples = generate_samples(
            model, tokenizer, prompt,
            n_samples=m_max,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            adaptive_stop=adaptive_stop,
            tau=tau if is_vmsca else 0.85,
            beta=beta if is_vmsca else 2.0,
            delta=delta if is_vmsca else 0.05,
            default_score=default_score if is_vmsca else 0.5
        )
        
        samples_used = len(samples)
        total_samples_used += samples_used
        total_generated += samples_used
        
        # Aggregate
        if is_vmsca:
            predicted, conf, sample_results, agg_metrics = vmsca_aggregate(samples, beta, delta, default_score)
            total_verified += agg_metrics["verified_count"]
            ess_values.append(agg_metrics["ess"])
        else:
            predicted, conf, parsed_answers = sc_majority_aggregate(samples)
            agg_metrics = {}
        
        # Check correctness
        if predicted is not None:
            correct = check_answer_correctness(predicted, gold)
        else:
            correct = False
        
        if correct:
            correct_count += 1
        
        # Confidently wrong
        if not correct and conf > 0.8:
            confidently_wrong_count += 1
        
        # Store result
        result = {
            "item_id": item["id"],
            "question": question,
            "gold": gold,
            "predicted": predicted,
            "confidence": conf,
            "correct": correct,
            "samples_used": samples_used,
        }
        
        if is_vmsca:
            result["verification_pass_rate"] = agg_metrics["verification_pass_rate"]
            result["ess"] = agg_metrics["ess"]
        
        results.append(result)
        
        # Log progress
        if (idx + 1) % 10 == 0:
            acc_so_far = correct_count / (idx + 1)
            print(f"Progress: {idx+1}/{len(items)} | Accuracy: {acc_so_far:.3f}")
    
    # Compute final metrics
    n = len(items)
    accuracy = correct_count / n
    cw_rate = confidently_wrong_count / n
    avg_samples_used = total_samples_used / n
    
    metrics = {
        "accuracy": accuracy,
        "confidently_wrong_rate": cw_rate,
        "average_samples_used": avg_samples_used,
    }
    
    if is_vmsca:
        metrics["verification_pass_rate"] = total_verified / max(1, total_generated)
        metrics["ess_mean"] = sum(ess_values) / max(1, len(ess_values)) if ess_values else 0.0
    
    print(f"\n=== Final Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Log to WandB
    wandb.log(metrics)
    for k, v in metrics.items():
        wandb.summary[k] = v
    
    # Save results to disk
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "predictions.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {results_file}")
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    wandb.finish()
    
    return metrics


if __name__ == "__main__":
    # Expect cfg to be passed via Hydra from main.py
    # For standalone testing, load manually
    import hydra
    from hydra import compose, initialize_config_dir
    
    # This is invoked as a subprocess by main.py with resolved config
    # Parse command-line args if needed
    pass
