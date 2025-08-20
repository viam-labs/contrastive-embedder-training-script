import torch
from omegaconf import OmegaConf
from pathlib import Path
import pandas as pd
import shutil
from datetime import datetime
from src.eval import HydraCheckpointEvaluator
from src.datasets.reid_dataset import get_reid_datasets

def extract_checkpoint_info_for_renaming(checkpoint_path):
    """Extract info needed for renaming from checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        epoch = checkpoint.get('epoch')
        variant = checkpoint.get('config', {}).get('model', {}).get('params', {}).get('variant')
        return epoch, variant
    except Exception as e:
        print(f"Warning: Could not extract info from {checkpoint_path}: {e}")
        return None, None

def analyze_best_by_variant(results_df):
    """
    Analyzes results to find the best model for each variant and creates comparison table
    """
    try:
        # Extract variant info for each model
        variant_data = []
        for idx, row in results_df.iterrows():
            epoch, variant = extract_checkpoint_info_for_renaming(Path(row['full_path']))
            if variant:
                variant_data.append({
                    'idx': idx,
                    'variant': variant,
                    'folder': row['folder'],
                    'model': row['model'],
                    'full_path': row['full_path'],
                    'auc': row['auc'],
                    'accuracy': row['accuracy'],
                    'f1_score': row['f1_score'],
                    'epoch': epoch
                })
        
        if not variant_data:
            return None
            
        variant_df = pd.DataFrame(variant_data)
        
        # Find best model for each variant (by AUC)
        best_by_variant = variant_df.loc[variant_df.groupby('variant')['auc'].idxmax()]
        
        # Create summary table
        summary_table = best_by_variant[['variant', 'folder', 'model', 'auc', 'accuracy', 'f1_score', 'epoch']].copy()
        summary_table = summary_table.sort_values('auc', ascending=False)
        
        print(f"\nFound {len(summary_table)} unique variants: {list(summary_table['variant'].unique())}")
        
        return {
            'summary_table': summary_table,
            'best_models': best_by_variant.to_dict('records'),
            'variant_df': variant_df
        }
        
    except Exception as e:
        print(f"Error analyzing variants: {e}")
        return None

def copy_best_by_variant(best_models, destination_folder):
    """
    Copies the best model for each variant to the destination folder
    """
    destination_path = Path(destination_folder)
    destination_path.mkdir(parents=True, exist_ok=True)
    
    for model_info in best_models:
        try:
            source_path = Path(model_info['full_path'])
            variant = model_info['variant']
            epoch = model_info['epoch']
            
            # Get file modification date
            mod_time = source_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mod_time).strftime('%Y%m%d')
            
            # Create filename with variant prefix
            new_filename = f"best_{variant}_osnet_{variant}_{date_str}_e{epoch}.pth.tar"
            destination_file_path = destination_path / new_filename
            
            shutil.copy(source_path, destination_file_path)
            print(f"  ✓ {variant}: {model_info['model']} -> {new_filename} (AUC: {model_info['auc']:.6f})")
            
        except Exception as e:
            print(f"  ✗ Failed to copy {model_info['variant']}: {e}")

def evaluate_all_models(candidates_folders, cfg, output_dir):
    """
    Evaluates checkpoints from multiple folders, saves results to CSV,
    and copies the best model with proper renaming.
    """
    
    # Auto-generate paths with today's date
    today_str = datetime.now().strftime('%m%d%Y')
    if output_dir is None:
        output_dir = f"release_candidates/rc_{today_str}"

    # --- 1. Load Data ONCE ---
    print("Loading test dataloader...")
    try:
        _, _, test_dataloader = get_reid_datasets(cfg.dataset)
        print("Dataloader loaded successfully.")
    except Exception as e:
        print(f"Error: Failed to load dataloader. {e}")
        return

    # --- 2. Collect all candidate models from all folders ---
    all_checkpoints = []
    
    print(f"\n--- Scanning {len(candidates_folders)} folders for models ---")
    for folder in candidates_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            print(f"Warning: Folder '{folder}' does not exist. Skipping.")
            continue
            
        checkpoints = list(folder_path.glob('*.pth.tar')) + list(folder_path.glob('*.pth'))
        print(f"Found {len(checkpoints)} models in '{folder}'")
        
        # Add folder info to each checkpoint
        for checkpoint in checkpoints:
            all_checkpoints.append({
                'path': checkpoint,
                'folder': folder_path.name,  # Just the folder name, not full path
                'full_folder_path': folder_path
            })

    if not all_checkpoints:
        print("Error: No checkpoint models found in any of the specified folders.")
        return

    print(f"\nTotal found: {len(all_checkpoints)} models across all folders.")
    
    all_results = []

    # --- 3. Loop through each model and evaluate ---
    for i, checkpoint_info in enumerate(all_checkpoints, 1):
        checkpoint_path = checkpoint_info['path']
        folder_name = checkpoint_info['folder']
        
        print(f"\n--- [{i}/{len(all_checkpoints)}] Evaluating: {folder_name}/{checkpoint_path.name} ---")
        try:
            evaluator = HydraCheckpointEvaluator(
                checkpoint_path=str(checkpoint_path),
                cfg=cfg,
                test_dataloader=test_dataloader
            )
            metrics = evaluator.evaluate()
            
            # Add folder info to results
            result_entry = {
                'folder': folder_name,
                'model': checkpoint_path.name,
                'full_path': str(checkpoint_path),
                **metrics
            }
            all_results.append(result_entry)
            evaluator.print_results(metrics)
            
        except Exception as e:
            print(f"  - FAILED to evaluate {checkpoint_path.name}. Reason: {e}")

    # --- 4. Print and Process the Final Summary ---
    if not all_results:
        print("Evaluation complete, but no results were generated.")
        return
        
    print("\n" + "="*70)
    print("                    FINAL RESULTS SUMMARY")
    print("="*70)
    
    try:
        results_df = pd.DataFrame(all_results)
        print(f"DataFrame created successfully with {len(results_df)} rows and {len(results_df.columns)} columns")
        
        primary_metric = 'auc'  # Using AUC as primary metric
        if primary_metric in results_df.columns:
            results_df = results_df.sort_values(by=primary_metric, ascending=False)
        else:
            print(f"Warning: Primary metric '{primary_metric}' not found. Available columns: {list(results_df.columns)}")
        
        # Print results grouped by folder for better readability
        print("\n--- RESULTS BY FOLDER ---")
        for folder in results_df['folder'].unique():
            folder_results = results_df[results_df['folder'] == folder]
            print(f"\n{folder}:")
            print(folder_results[['model', 'auc', 'accuracy', 'f1_score']].to_string(index=False))
        
        print(f"\n--- TOP 10 OVERALL MODELS ---")
        top_models = results_df.head(10)
        print(top_models[['folder', 'model', 'auc', 'accuracy', 'f1_score']].to_string(index=False))
        
        # --- BEST BY VARIANT ANALYSIS ---
        print(f"\n--- BEST MODEL BY VARIANT ---")
        variant_analysis = analyze_best_by_variant(results_df)
        if variant_analysis:
            print(variant_analysis['summary_table'].to_string(index=False))
        else:
            print("Could not extract variant information from checkpoints.")
        
    except Exception as e:
        print(f"Error creating or processing DataFrame: {e}")
        print(f"Raw results data: {all_results}")
        return

    # --- 5. Save summary to CSV and copy the best models ---
    try:
        # Ensure the results directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nAttempting to save results to: {output_path.resolve()}")
        
        # Save the full results table to a CSV file
        results_csv_path = output_path / "evaluation_results.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"✓ Full results saved to '{results_csv_path}'")
        
        # Save variant comparison if available
        if variant_analysis:
            variant_csv_path = output_path / "best_by_variant.csv"
            variant_analysis['summary_table'].to_csv(variant_csv_path, index=False)
            print(f"✓ Best by variant saved to '{variant_csv_path}'")

        # Identify the best model from the sorted DataFrame
        if len(results_df) == 0:
            print("No results to process for best model selection.")
            return
            
        best_row = results_df.iloc[0]
        best_model_name = best_row['model']
        best_folder = best_row['folder']
        source_path = Path(best_row['full_path'])
        
        if not source_path.exists():
            print(f"Error: Best model file '{source_path}' does not exist.")
            return
        
        # Extract checkpoint info for renaming
        epoch, variant = extract_checkpoint_info_for_renaming(source_path)
        
        # Generate new filename with proper format
        if epoch is not None and variant is not None:
            # Get file modification date
            mod_time = source_path.stat().st_mtime
            date_str = datetime.fromtimestamp(mod_time).strftime('%Y%m%d')
            
            # Create new filename: osnet_{variant}_{date}_e{epoch}.pth.tar
            new_filename = f"osnet_{variant}_{date_str}_e{epoch}.pth.tar"
        else:
            # Fallback to original name if we can't extract info
            new_filename = best_model_name
            print(f"Warning: Could not extract epoch/variant info, using original filename")
        
        destination_path = output_path / new_filename

        print(f"\n--- BEST MODEL SELECTION ---")
        print(f"Best model: {best_folder}/{best_model_name}")
        print(f"AUC: {best_row['auc']:.6f}, Accuracy: {best_row['accuracy']:.4f}")
        print(f"Copying to: '{new_filename}' in '{output_dir}'...")
        
        # Copy the best overall model
        shutil.copy(source_path, destination_path)
        print(f"✓ Copy complete. Best overall model saved as: {new_filename}")
        
        # Copy best models by variant
        if variant_analysis and 'best_models' in variant_analysis:
            print(f"\n--- COPYING BEST MODELS BY VARIANT ---")
            copy_best_by_variant(variant_analysis['best_models'], output_path)
        
        print(f"✓ Release candidate folder: {output_dir}")

    except PermissionError as e:
        print(f"Permission Error: Cannot write to '{output_dir}'. Check if:")
        print("  - The directory has write permissions")
        print("  - The file isn't open in another program (like Excel)")
        print(f"  - Specific error: {e}")
    except FileNotFoundError as e:
        print(f"File/Directory Error: {e}")
        print(f"  - Check if parent directory exists for: {output_dir}")
    except Exception as e:
        print(f"Unexpected error while saving results or copying best model: {e}")
        print(f"  - Output dir: {output_dir}")
        import traceback
        traceback.print_exc()

def load_config_from_checkpoint(checkpoint_path):
    """
    Loads configuration from a checkpoint file
    """
    try:
        print(f"Loading config from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'config' not in checkpoint:
            raise ValueError(f"No config found in checkpoint: {checkpoint_path}")
            
        cfg = OmegaConf.create(checkpoint['config'])
        print(f"✓ Config loaded successfully")
        return cfg
        
    except Exception as e:
        print(f"Error loading config from {checkpoint_path}: {e}")
        return None

def save_config_to_folder(cfg, output_folder, checkpoint_source=None):
    """
    Saves the configuration to the output folder as a YAML file
    """
    try:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config_file_path = output_path / "config.yaml"
        
        # Add metadata about source
        metadata = {
            "extraction_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        if checkpoint_source:
            metadata["config_extracted_from"] = str(checkpoint_source)
        
        # Create config with metadata
        config_with_meta = OmegaConf.create({**metadata, **cfg})
        
        # Save as YAML
        with open(config_file_path, 'w') as f:
            OmegaConf.save(config=config_with_meta, f=f)
        
        print(f"✓ Config saved to: {config_file_path}")
        return config_file_path
        
    except Exception as e:
        print(f"Error saving config: {e}")
        return None

def main():
    """
    Simple evaluation pipeline: Load → Evaluate → Save Best → Release
    """
    print("="*70)
    print("           MODEL EVALUATION AND RELEASE PIPELINE")
    print("="*70)
    
    # Define your candidate folders - ADD YOUR ACTUAL PATHS HERE
    candidates_folders = [
        "src/checkpoints/osnet_x0_25_optimal",
        "src/model_candidates",
        "src/checkpoints"  # Add any other folders with your model variants
    ]
    
    # Auto-generate today's release candidate folder
    today_str = datetime.now().strftime('%m%d%Y')
    output_dir = f"release_candidates/rc_{today_str}"
    
    print(f"Candidate folders: {len(candidates_folders)}")
    for i, folder in enumerate(candidates_folders, 1):
        print(f"  {i}. {folder}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Find any checkpoint to extract config from
        config_checkpoint = None
        for folder in candidates_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                checkpoints = list(folder_path.glob('*.pth.tar')) + list(folder_path.glob('*.pth'))
                if checkpoints:
                    config_checkpoint = checkpoints[0]
                    break
        
        if config_checkpoint is None:
            print("Error: No checkpoints found in any candidate folder!")
            return
        
        # Load configuration
        print(f"\n--- Loading Configuration ---")
        cfg = load_config_from_checkpoint(config_checkpoint)
        if cfg is None:
            print("Failed to load configuration. Exiting.")
            return
        
        print(f"Dataset configuration: {cfg.dataset}")
        
        # Save config to output directory
        save_config_to_folder(cfg, output_dir, config_checkpoint)
        
        # Run evaluation
        print(f"\n--- Running Evaluation ---")
        evaluate_all_models(
            candidates_folders=candidates_folders,
            cfg=cfg,
            output_dir=output_dir
        )
        
        print(f"\nEvaluation completed successfully!")
        print(f"Check results in: {output_dir}")
        print(f"Config saved as: {output_dir}/config.yaml")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()