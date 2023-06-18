import torch.nn as nn


class DPCNN(nn.Module):
    def __init__(self):
        super(DPCNN, self).__init__()
        self.act = nn.Tanh()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            self.act,
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        out = self.body(x).view(x.size(0), -1)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        return out
