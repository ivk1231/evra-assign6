import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )
        
        # CONV Block 1
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 28x28x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1)
        )
        
        # Transition Block 1
        self.trans1 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 14x14x64
            nn.Conv2d(64, 32, 1)  # 14x14x32
        )
        
        # CONV Block 2
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 14x14x64
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1)
        )
        
        # Transition Block 2
        self.trans2 = nn.Sequential(
            nn.MaxPool2d(2, 2),  # 7x7x64
            nn.Conv2d(64, 32, 1)  # 7x7x32
        )
        
        # CONV Block 3
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # 7x7x32
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1)
        )
        
        # Output Block
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=7)  # 1x1x32
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 10, 1)  # 1x1x10
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.trans1(x)
        x = self.conv3(x)
        x = self.trans2(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.conv5(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item():.4f} batch_id={batch_idx}')

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
    return 100. * correct / len(test_loader.dataset)

def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    torch.manual_seed(1)
    batch_size = 64

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                      transform=transforms.Compose([
                          transforms.RandomRotation((-7.0, 7.0)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, epochs=20, steps_per_epoch=len(train_loader))

    for epoch in range(1, 20):
        print(f'\nEpoch: {epoch}')
        train(model, device, train_loader, optimizer, epoch)
        accuracy = evaluate(model, device, test_loader)
        scheduler.step()

if __name__ == '__main__':
    main() 