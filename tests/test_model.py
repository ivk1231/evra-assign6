import torch
from torchvision import datasets, transforms
from dummy import Net, evaluate
from torch.utils.data import Subset
import numpy as np

def setup_module(module):
    """Download dataset before running tests"""
    datasets.MNIST('../data', train=True, download=True)
    datasets.MNIST('../data', train=False, download=True)

def test_parameter_count():
    model = Net()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 20000, f"Model has {total_params} parameters, should be less than 20000"

def test_batch_norm_usage():
    model = Net()
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    assert has_bn, "Model should use Batch Normalization"

def test_dropout_usage():
    model = Net()
    has_dropout = any(isinstance(m, torch.nn.Dropout) for m in model.modules())
    assert has_dropout, "Model should use Dropout"

def test_gap_usage():
    model = Net()
    has_gap = any(isinstance(m, torch.nn.AvgPool2d) for m in model.modules())
    assert has_gap, "Model should use Global Average Pooling"

def test_model_accuracy():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    # Load the full test dataset
    test_dataset = datasets.MNIST('../data', train=False, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    
    # Split into validation (50) and test (10k) sets
    indices = list(range(len(test_dataset)))
    val_indices = indices[:50]  # First 50 samples for validation
    test_indices = indices[50:10050]  # Next 10k samples for testing
    
    val_dataset = Subset(test_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)
    
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=50, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Train the model
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomRotation((-7.0, 7.0)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=64, shuffle=True)

    model = Net().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=15, 
                                                   steps_per_epoch=len(train_loader))
    
    best_val_acc = 0
    for epoch in range(15):  # Less than 20 epochs
        # Training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_acc = 100. * correct / len(val_dataset)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Test on 10k samples
            correct = 0
            total = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
            
            test_acc = 100. * correct / total
            print(f'Validation Accuracy: {val_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')
    
    assert test_acc >= 99.4, f"Model accuracy is {test_acc:.2f}%, which is below the required 99.4%"