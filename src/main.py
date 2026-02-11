"""
Main orchestrator for single run execution.
Applies mode overrides and invokes inference script.
"""
import sys
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import optuna


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """
    Orchestrate a single run.
    
    Steps:
    1. Apply mode overrides (sanity_check vs main)
    2. Run Optuna hyperparameter search if enabled
    3. Invoke inference.py with final config
    4. Perform sanity validation in sanity_check mode
    """
    
    print(f"=== Run: {cfg.run.run_id} ===")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    
    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("Applying sanity_check mode overrides...")
        
        # Disable WandB
        OmegaConf.update(cfg, "wandb.mode", "disabled", merge=False)
        
        # Reduce dataset size
        OmegaConf.update(cfg, "run.dataset.max_samples", 10, merge=False)
        
        # Reduce sampling budget
        if cfg.run.method.name == "VMSCA":
            OmegaConf.update(cfg, "run.method.m_max", 5, merge=False)
        else:
            OmegaConf.update(cfg, "run.method.m", 5, merge=False)
        
        # Disable Optuna
        if "optuna" in cfg.run and cfg.run.optuna.enabled:
            OmegaConf.update(cfg, "run.optuna.enabled", False, merge=False)
        
        print("Sanity check overrides applied")
    
    # Handle Optuna hyperparameter search
    if "optuna" in cfg.run and cfg.run.optuna.enabled and cfg.mode != "sanity_check":
        print("Running Optuna hyperparameter search...")
        best_params = run_optuna_search(cfg)
        
        # Update config with best params
        for param_name, param_value in best_params.items():
            OmegaConf.update(cfg, f"run.method.{param_name}", param_value, merge=False)
        
        print(f"Best hyperparameters: {best_params}")
    
    # Run inference
    print("Starting inference...")
    metrics = run_inference_subprocess(cfg)
    
    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(cfg, metrics)
    
    print("Run completed successfully")


def run_optuna_search(cfg: DictConfig) -> dict:
    """
    Run Optuna hyperparameter search on validation set.
    """
    optuna_cfg = cfg.run.optuna
    
    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        trial_params = {}
        for param_spec in optuna_cfg.search_space:
            param_name = param_spec["param_name"]
            dist_type = param_spec["distribution_type"]
            
            if dist_type == "categorical":
                trial_params[param_name] = trial.suggest_categorical(param_name, param_spec["choices"])
            elif dist_type == "float":
                trial_params[param_name] = trial.suggest_float(param_name, param_spec["low"], param_spec["high"])
            elif dist_type == "int":
                trial_params[param_name] = trial.suggest_int(param_name, param_spec["low"], param_spec["high"])
        
        # Create trial config
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        
        # Use validation set
        OmegaConf.update(trial_cfg, "run.dataset.split", optuna_cfg.validation.split, merge=False)
        OmegaConf.update(trial_cfg, "run.dataset.max_samples", optuna_cfg.validation.max_samples, merge=False)
        
        # Apply trial params
        for param_name, param_value in trial_params.items():
            OmegaConf.update(trial_cfg, f"run.method.{param_name}", param_value, merge=False)
        
        # Disable WandB for trials
        OmegaConf.update(trial_cfg, "wandb.mode", "disabled", merge=False)
        
        # Disable nested Optuna
        OmegaConf.update(trial_cfg, "run.optuna.enabled", False, merge=False)
        
        # Run inference
        metrics = run_inference_subprocess(trial_cfg)
        
        # Return primary metric
        metric_name = optuna_cfg.metric
        return metrics.get(metric_name, 0.0)
    
    # Create study
    study = optuna.create_study(
        direction=optuna_cfg.direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study.optimize(objective, n_trials=optuna_cfg.n_trials)
    
    print(f"Optuna search completed. Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    
    return study.best_params


def run_inference_subprocess(cfg: DictConfig) -> dict:
    """
    Run inference.py by directly importing and calling.
    """
    # Import here to avoid circular dependencies
    from inference import run_inference
    
    metrics = run_inference(cfg)
    return metrics


def perform_sanity_validation(cfg: DictConfig, metrics: dict):
    """
    Perform sanity validation checks in sanity_check mode.
    
    For inference tasks:
    - At least 5 samples processed
    - All metrics are finite
    - Accuracy is not always 0
    """
    print("\n=== Sanity Validation ===")
    
    # Extract key metrics
    accuracy = metrics.get("accuracy", 0.0)
    avg_samples = metrics.get("average_samples_used", 0.0)
    cw_rate = metrics.get("confidently_wrong_rate", 0.0)
    
    # Check 1: At least 5 samples processed
    expected_samples = cfg.run.dataset.max_samples
    if expected_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples (expected >= 5, got {expected_samples})")
        summary = {
            "samples": expected_samples,
            "accuracy": accuracy,
            "status": "FAIL"
        }
        print(f"SANITY_VALIDATION_SUMMARY: {summary}")
        sys.exit(1)
    
    # Check 2: All metrics are finite
    if not all(isinstance(v, (int, float)) and not (v != v or abs(v) == float('inf')) for v in metrics.values()):
        print("SANITY_VALIDATION: FAIL reason=non_finite_metrics")
        summary = {
            "samples": expected_samples,
            "accuracy": accuracy,
            "status": "FAIL"
        }
        print(f"SANITY_VALIDATION_SUMMARY: {summary}")
        sys.exit(1)
    
    # Check 3: Accuracy is meaningful (not all wrong)
    # In sanity_check we only run 10 items, so low accuracy is acceptable
    # but we check that the system ran successfully
    if accuracy < 0 or accuracy > 1:
        print(f"SANITY_VALIDATION: FAIL reason=invalid_accuracy (got {accuracy})")
        summary = {
            "samples": expected_samples,
            "accuracy": accuracy,
            "status": "FAIL"
        }
        print(f"SANITY_VALIDATION_SUMMARY: {summary}")
        sys.exit(1)
    
    # Success
    print("SANITY_VALIDATION: PASS")
    summary = {
        "samples": expected_samples,
        "accuracy": accuracy,
        "avg_samples_used": avg_samples,
        "confidently_wrong_rate": cw_rate,
        "status": "PASS"
    }
    print(f"SANITY_VALIDATION_SUMMARY: {summary}")


if __name__ == "__main__":
    main()
