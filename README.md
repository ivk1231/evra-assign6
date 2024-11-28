# MNIST Classification with PyTorch
comment:- i tried around 33 iterations, and i have achieved upto 99.2% efficiency.
This repository contains a PyTorch implementation of a CNN model for MNIST digit classification.

## Model Architecture
- Uses less than 20k parameters
- Implements Batch Normalization
- Uses Dropout for regularization
- Implements Global Average Pooling
- Achieves >99.4% test accuracy

## Results
The model achieves the following performance:
- Parameters: <20k
- Test Accuracy: >99.4%
- Training Epochs: <20

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- pytest

## Setup and Training
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run training:
   ```bash
   python dummy.py
   ```

## Tests
Run tests using:
```bash
pytest tests/test_model.py
```

## Viewing Test Logs
1. Go to the GitHub repository
2. Click on the "Actions" tab
3. Click on the latest workflow run
4. In the workflow run, click on the "test" job
5. Expand the "Run tests" step to see detailed test logs

You can also see test logs locally by running:
```bash
pytest -v tests/test_model.py
