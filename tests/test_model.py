import torch
from torchvision import datasets, transforms
from dummy import Net, test

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
    
    # Load the test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=128, shuffle=True)

    # Load model and evaluate
    model = Net().to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    assert accuracy >= 99.4, f"Model accuracy is {accuracy:.2f}%, which is below the required 99.4%" 