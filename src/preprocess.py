"""
Dataset preprocessing for GSM8K arithmetic reasoning.
Loads datasets with caching and provides utilities for answer extraction.
"""
import re
from typing import Dict, List, Tuple
from datasets import load_dataset


def load_gsm8k(split: str, max_samples: int = None, cache_dir: str = ".cache") -> List[Dict]:
    """
    Load GSM8K dataset.
    
    Args:
        split: 'train' or 'test'
        max_samples: limit to first N samples (for fast experimentation)
        cache_dir: HuggingFace cache directory
    
    Returns:
        List of dicts with 'question' and 'answer' keys
    """
    dataset = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir, trust_remote_code=True)
    
    items = []
    for i, example in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        question = example["question"].strip()
        # GSM8K answer format: "#### 42" at the end
        answer_str = example["answer"].strip()
        gold_numeric = extract_gold_answer(answer_str)
        
        items.append({
            "id": i,
            "question": question,
            "answer_str": answer_str,
            "gold_numeric": gold_numeric
        })
    
    return items


def extract_gold_answer(answer_str: str) -> float:
    """
    Extract numeric gold answer from GSM8K answer field.
    GSM8K format: solution steps followed by #### <number>
    
    Returns:
        Numeric answer (int or float)
    """
    # GSM8K uses "#### <answer>" separator
    match = re.search(r"####\s*([-+]?\d+(?:,\d{3})*(?:\.\d+)?)", answer_str)
    if match:
        num_str = match.group(1).replace(",", "")
        if "." in num_str:
            return float(num_str)
        else:
            return int(num_str)
    
    # Fallback: extract last number
    numbers = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", answer_str)
    if numbers:
        num_str = numbers[-1].replace(",", "")
        if "." in num_str:
            return float(num_str)
        else:
            return int(num_str)
    
    raise ValueError(f"Could not extract numeric answer from: {answer_str}")


def normalize_numeric_answer(answer_str: str) -> float:
    """
    Normalize a model prediction string to numeric form.
    Used for accuracy calculation.
    
    Args:
        answer_str: raw string from model (e.g., "42", "3.14", "$100")
    
    Returns:
        Numeric value (int or float)
    """
    # Remove common non-numeric characters
    cleaned = re.sub(r"[,$%]", "", answer_str.strip())
    
    # Extract first number-like token
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if match:
        num_str = match.group(0)
        if "." in num_str:
            return float(num_str)
        else:
            return int(num_str)
    
    raise ValueError(f"Could not parse numeric answer from: {answer_str}")


def check_answer_correctness(predicted: str, gold: float, tol: float = 1e-6) -> bool:
    """
    Check if predicted answer matches gold.
    
    Args:
        predicted: model output string
        gold: ground-truth numeric value
        tol: tolerance for float comparison
    
    Returns:
        True if correct
    """
    try:
        pred_val = normalize_numeric_answer(predicted)
        gold_val = float(gold)
        
        # Both are effectively integers
        if isinstance(pred_val, int) and isinstance(gold, int):
            return pred_val == gold
        
        # Float comparison with tolerance
        return abs(pred_val - gold_val) <= max(tol, tol * abs(gold_val))
    
    except Exception:
        return False


def build_few_shot_prompt(n_examples: int = 3, require_format: bool = False) -> str:
    """
    Build few-shot CoT prompt with optional VMSCA format enforcement.
    
    Args:
        n_examples: number of exemplars
        require_format: if True, include Answer/Score/Check format instructions
    
    Returns:
        Prompt string with instructions and examples
    """
    base_instruction = """Solve the following math word problem step by step. Show your reasoning clearly."""
    
    if require_format:
        format_instruction = """
After solving, perform a quick self-check. Output EXACTLY in this format:

Answer: <final numeric answer>
Score: <your confidence from 0.0 to 1.0>
Check: <a pure arithmetic Python expression that computes the same answer>

Example Check expressions: "50 + 30 * 2", "100 / 4 + 25", "(12 * 3) - 8"
"""
        base_instruction += format_instruction
    
    # Few-shot exemplars (simplified for brevity; in practice use 2-3 full examples)
    exemplars = []
    
    if require_format:
        exemplar_1 = """
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends with four. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day?

Solution:
Janet starts with 16 eggs.
She eats 3 eggs, leaving 16 - 3 = 13 eggs.
She bakes muffins with 4 eggs, leaving 13 - 4 = 9 eggs.
She sells 9 eggs at $2 each: 9 * 2 = 18 dollars.

Answer: 18
Score: 0.95
Check: (16 - 3 - 4) * 2
"""
        exemplars.append(exemplar_1.strip())
    else:
        exemplar_1 = """
Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins for her friends with four. She sells the remainder at the farmers' market for $2 per egg. How much does she make every day?

Solution:
Janet starts with 16 eggs.
She eats 3 eggs, leaving 16 - 3 = 13 eggs.
She bakes muffins with 4 eggs, leaving 13 - 4 = 9 eggs.
She sells 9 eggs at $2 each: 9 * 2 = 18 dollars.

Answer: 18
"""
        exemplars.append(exemplar_1.strip())
    
    # Limit to n_examples
    exemplars = exemplars[:n_examples]
    
    full_prompt = base_instruction + "\n\n" + "\n\n---\n\n".join(exemplars)
    return full_prompt


def format_problem_prompt(question: str, few_shot_prefix: str) -> str:
    """
    Format a single problem into inference prompt.
    
    Args:
        question: problem text
        few_shot_prefix: pre-built few-shot examples
    
    Returns:
        Full prompt ready for model
    """
    return f"{few_shot_prefix}\n\n---\n\nProblem: {question}\n\nSolution:\n"
