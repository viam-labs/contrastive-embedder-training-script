# Contrastive Embedder Training Template

This template provides a modular training framework for contrastive learning with pair-based datasets.

## Dataset Integration

### Supported Data Format

The template supports pair-based contrastive learning with the following format:

**Direct Pairs**: `[(x_i, x_j, label_int)]`
- `x_i`, `x_j`: Image paths or PIL Images
- `label_int`: 1 for same person, 0 for different person

### Creating Your Own Dataset

1. **Copy the template**:
   ```bash
   cp src/datasets/generic_dataset.py src/datasets/my_dataset.py
   ```

2. **Implement the required methods** in `my_dataset.py`:

   ```python
   class MyPairDataset(GenericPairDataset):
       def __init__(self):
           # Load your dataset data here
           self.data = [
               ("path/to/img1.jpg", "path/to/img2.jpg", 1),  # same person
               ("path/to/img3.jpg", "path/to/img4.jpg", 0),  # different person
               # ... more pairs
           ]
       
       def __getitem__(self, idx):
           # Get a pair of images and their label
           img1_path, img2_path, label = self.data[idx]
           
           # Load and return the images
           from PIL import Image
           img1 = Image.open(img1_path).convert('RGB')
           img2 = Image.open(img2_path).convert('RGB')
           
           return img1, img2, torch.tensor(label, dtype=torch.float32)
       
       def __len__(self):
           return len(self.data)
   ```

3. **Create your dataset function**:

   ```python
   def get_my_datasets(cfg):
       transform = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
       ])
       
       train_dataset = MyPairDataset(
           transform=transform
       )
       
       # Create dataloaders...
       return train_dataloader, val_dataloader
   ```

4. **Update the training script**:
   ```python
   def _get_dataset(self):
       dataset_name = self.cfg.dataset.name
       if dataset_name == "my_dataset":
           dataset_config = self.cfg.dataset
           return get_my_datasets(dataset_config)
       # ... other datasets
   ```

### Configuration

Create a config file for your dataset:

```yaml:configs/dataset/my_dataset.yaml
name: my_dataset
batch_size: 32
num_workers: 4
pin_memory: true
num_pairs_per_epoch: 10000
```

### Configuration Options

You have two ways to configure your dataset:

**Option 1: Set default dataset in config file**

Update the main config to use your dataset by default:

```yaml:configs/config.yaml
defaults:
  - model: lightweight_embedder
  - dataset: my_dataset  # Change this to your dataset
  - trainer: default
  - loss: contrastive
  - _self_

seed: 42
```

Then train with the default configuration:
```bash
python src/train.py
```

**Option 2: Specify dataset via command line**

Keep the default config unchanged and specify your dataset when running:

```bash
python src/train.py dataset=my_dataset
```

This approach allows you to easily switch between different datasets without modifying the config file.


# Evaluation 
Evaluation
Evaluate trained models directly from MLflow runs. 
```bash
# Evaluate latest model from experiment
python evaluate.py --experiment-name "osnet_ain_contrastive_reid_training"
```

# Evaluate specific run
```bash
python evaluate.py --run-id abc123def456
```

# List available experiments/runs
```bash
python evaluate.py --list-experiments
```
```bash
python evaluate.py --list-runs "experiment_name"
```

Options
```bash
--run-id TEXT              Specific MLflow run ID
--experiment-name TEXT     Experiment name (uses latest run)
--tracking-uri TEXT        Custom MLflow server
--no-plots                Skip saving plots
--no-mlflow-log           Don't log results to MLflow
```

Output

Metrics: AUC, accuracy, optimal threshold, distance statistics
Plots: ROC curve, distance distributions, box plots
MLflow Logging: Results automatically logged for tracking