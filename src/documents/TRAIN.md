Training with train.py
The train.py script is the main entry point for training and fine-tuning contrastive embedding models. It integrates models, datasets, and transfer learning configurations into a single workflow.

Supported Operations
Train from scratch using default configs

Train with a custom dataset or model

Perform transfer learning with pruning

Log and track experiments via MLflow

How to Run

1. Train with Default Configuration
   bash
   Copy
   Edit
   python src/train.py
2. Specify a Custom Dataset
   bash
   Copy
   Edit
   python src/train.py dataset=my_dataset
3. Specify a Custom Model
   bash
   Copy
   Edit
   python src/train.py model=my_model
4. Combine Custom Dataset and Model
   bash
   Copy
   Edit
   python src/train.py dataset=my_dataset model=my_model
5. Run Transfer Learning
   bash
   Copy
   Edit
   python src/train.py --config-name=config_transfer_learning
   Configuration
   The script uses Hydra for configuration management.
   Key config files:

configs/config.yaml — default model, dataset, trainer, and loss

configs/dataset/\*.yaml — dataset-specific configs

configs/model/\*.yaml — model-specific configs

configs/config_transfer_learning.yaml — transfer learning setup

To override config options from the command line:

bash
Copy
Edit
python src/train.py dataset=my_dataset model=my_model trainer.batch_size=64
Notes
All datasets and models must be defined in their respective registries in train.py.

MLflow logging is enabled by default; see evaluate.py for evaluation instructions.

For customization guides, see:

Datasets

Models

Transfer Learning
