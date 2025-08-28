Models
This template provides a modular approach to integrating new model architectures. The training script currently supports several OSNet-based models, and here's how to add a new one.

Supported Models
osnet_ain: A family of OSNet models with Adaptive Instance Normalization, including variants like x1_0, x0_75, x0_5, and x0_25.

simple_cnn: A basic CNN model for quick testing and debugging.

lightweight_embedder: A simple lightweight embedder model.

How to Add a New Model
Step 1: Create the Model Class
Create a new Python file for your model in the src/models directory. The class should inherit from torch.nn.Module and implement a forward method.

Python

# src/models/my_model.py

import torch.nn as nn

class MyModel(nn.Module):
def **init**(self, num_classes=1000):
super(MyModel, self).**init**() # Define your model layers here
self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
self.fc = nn.Linear(16 _ 224 _ 224, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc(x)
        return x

Step 2: Update the Training Script
Update the \_get_model method in src/train.py to recognize your new model name and instantiate its class.

Python

# In EnhancedReIDTrainer.\_get_model

def \_get_model(self):
model_name = self.cfg.model.name

    if model_name == "my_model":
        from models.my_model import MyModel
        model = MyModel(**self.cfg.model.params)
    elif model_name == "simple_cnn":
        from models.simple_cnn import SimpleCNN
        model = SimpleCNN(**self.cfg.model.params)
    # ... other models
    else:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(self.device)

Step 3: Create the Configuration File
Create a new YAML configuration file for your model in the configs/model directory.

Code snippet

name: my_model
params:
num_classes: 10
Step 4: Run with Your New Model
Use a command-line override to specify your new model.

Bash

python src/train.py model=my_model
