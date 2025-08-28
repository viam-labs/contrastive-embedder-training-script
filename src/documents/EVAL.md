Evaluating with eval.py
The eval.py script runs evaluation on trained contrastive embedding models. It loads the specified model checkpoint and dataset, then outputs accuracy and other metrics.

Supported Operations
Evaluate the latest or specific model checkpoint

Use any dataset supported by the training pipeline

Log evaluation metrics via MLflow

How to Run

1. Evaluate the Latest Checkpoint
   bash
   Copy
   Edit
   python src/eval.py
2. Evaluate a Specific Checkpoint
   bash
   Copy
   Edit
   python src/eval.py eval.checkpoint_path="path/to/checkpoint.pth.tar"
3. Evaluate with a Custom Dataset
   bash
   Copy
   Edit
   python src/eval.py dataset=my_dataset
4. Evaluate with a Custom Model
   bash
   Copy
   Edit
   python src/eval.py model=my_model
5. Combine Dataset, Model, and Checkpoint
   bash
   Copy
   Edit
   python src/eval.py dataset=my_dataset model=my_model eval.checkpoint_path="path/to/checkpoint.pth.tar"
   Configuration
   The script uses Hydra for configuration management.
   Key config files:

configs/config.yaml — default model, dataset, and evaluation settings

configs/dataset/\*.yaml — dataset-specific configs

configs/model/\*.yaml — model-specific configs

Override config options from the command line:

bash
Copy
Edit
python src/eval.py dataset=my_dataset eval.batch_size=64
Notes
Checkpoints are stored in outputs/<date>/<time>/checkpoints/ by default.

Evaluation logs and metrics are tracked via MLflow.

All dataset and model names must match entries in eval.py’s registries.

For training instructions, see:

Training with train.py

Datasets

Models
