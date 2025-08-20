# Unified Model Evaluation and Release Pipeline

A comprehensive pipeline for filtering, evaluating, and preparing model release candidates from training runs.

## Overview

This script automates the complete model evaluation workflow:
1. **Filter**: Scans training directories and selects top-performing models
2. **Stage**: Copies and renames selected models to staging area
3. **Evaluate**: Runs comprehensive evaluation on test data
4. **Release**: Packages best models with results and configuration

## Quick Start

### 1. Configure Source Directories
Update the `source_directories` list in `main()` with your training run paths:

```python
source_directories = [
    "training_runs/osnet_x0_25_experiment",
    "training_runs/osnet_x1_0_experiment", 
    "training_runs/osnet_x0_5_experiment",
    "previous_releases/checkpoints"
]
```

### 2. Run the Pipeline
```bash
python unified_evaluation_script.py
```

### 3. Check Results
Results are automatically saved to `release_candidates/rc_MMDDYYYY/`

## Configuration

### Key Parameters
- **`threshold_percent`**: Percentage of top models to evaluate (default: 10%)
- **`source_directories`**: List of directories to scan for checkpoints
- **Output paths**: Automatically generated with today's date

### Requirements
- PyTorch checkpoints with `config`, `epoch`, and performance metrics
- Checkpoint structure: `{config: {model: {params: {variant: "x0_25"}}}}`
- Test dataloader accessible via `get_reid_datasets(cfg.dataset)`

## Output Structure

```
staging/candidates_MMDDYYYY/           # Intermediate staging
├── osnet_x0_25_20250820_e19.pth.tar
├── osnet_x1_0_20250820_e25.pth.tar
└── ...

release_candidates/rc_MMDDYYYY/        # Final release package
├── config.yaml                       # Extracted configuration
├── evaluation_results.csv            # Complete evaluation results
├── best_by_variant.csv               # Best model per variant comparison
├── best_overall_osnet_x0_25_...      # Best performing model
├── best_x0_25_osnet_x0_25_...        # Best x0_25 variant
├── best_x1_0_osnet_x1_0_...          # Best x1_0 variant
└── best_x0_5_osnet_x0_5_...          # Best x0_5 variant
```

## Pipeline Stages

### Stage 1: Filtering and Staging
- **Input**: Training run directories with `.pth` and `.tar` files
- **Process**: 
  - Recursively scans all source directories
  - Extracts performance metrics (`best_score` or validation loss)
  - Selects top N% of models based on performance
  - Copies and renames models with standardized format
- **Output**: Staged models in `staging/candidates_MMDDYYYY/`

### Stage 2: Comprehensive Evaluation
- **Input**: Staged model candidates
- **Process**:
  - Loads test dataloader once for efficiency
  - Evaluates each model on test data
  - Generates comprehensive metrics (AUC, accuracy, F1, etc.)
  - Identifies best overall and best per variant
- **Output**: Complete release package with results and models

## Model Selection Logic

### Filtering Criteria
1. **Score-based**: Models with `best_score >= threshold` (higher is better)
2. **Loss-based**: Models with `val_loss <= threshold` (lower is better)
3. **Threshold calculation**: `top_score * (1 - threshold_percent/100)`

### Variant Analysis
- Automatically detects model variants (x0_25, x1_0, x0_5, etc.)
- Finds best performing model for each variant
- Provides variant comparison table
- Copies best model per variant to release directory

## File Naming Convention

### Staged Models
Format: `osnet_{variant}_{YYYYMMDD}_e{epoch}.pth.tar`
- Example: `osnet_x0_25_20250820_e19.pth.tar`

### Release Models
- **Best Overall**: `best_overall_{original_name}.pth.tar`
- **Best by Variant**: `best_{variant}_osnet_{variant}_{date}_e{epoch}.pth.tar`

## Results Files

### evaluation_results.csv
Complete evaluation results for all models with columns:
- `model`: Model filename
- `auc`: Area Under Curve score
- `accuracy`: Classification accuracy
- `f1_score`: F1 score
- `precision`, `recall`, `specificity`: Additional metrics
- Performance analysis metrics

### best_by_variant.csv
Comparison of best model per variant:
- `variant`: Model variant (x0_25, x1_0, etc.)
- `model`: Best model filename for this variant
- `auc`, `accuracy`, `f1_score`: Performance metrics
- `epoch`: Training epoch

## Troubleshooting

### Common Issues
1. **"No checkpoints found"**: Check source directory paths and file extensions
2. **"Missing required info"**: Ensure checkpoints contain `config`, `epoch`, and metrics
3. **"Failed to load dataloader"**: Verify dataset configuration and paths
4. **Permission errors**: Check write permissions for output directories

### Debug Tips
- Check console output for detailed progress and error messages
- Verify checkpoint structure matches expected format
- Ensure all source directories exist and contain `.pth.tar` files
- Test with a small subset first by adjusting `threshold_percent`

## Customization

### Adding New Metrics
Modify the metric extraction in `extract_checkpoint_info()`:
```python
# Add new metric sources
possible_score_keys = ['best_score', 'best_val_acc', 'your_metric']
```

### Changing Selection Criteria
Adjust filtering logic in `filter_and_stage_checkpoints()`:
```python
# Modify threshold calculation
score_threshold = best_score * 0.95  # Top 5%
```

### Custom Naming
Update filename generation in the copy functions:
```python
new_filename = f"custom_{variant}_{date}_e{epoch}.pth.tar"
```

## Dependencies

- `torch`: PyTorch for checkpoint loading
- `pandas`: Data analysis and CSV handling
- `omegaconf`: Configuration management
- `pathlib`: Path handling
- Custom modules: `src.eval.HydraCheckpointEvaluator`, `src.datasets.reid_dataset`
