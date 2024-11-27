import torch
from dummy import Net

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