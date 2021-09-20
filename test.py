import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.form = nn.Linear(100,16384)
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(1024,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16,8,4,2,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8,3,4,2,1),
            nn.BatchNorm2d(3),
            nn.Sigmoid(),
        )
    
    def forward(self, value):
        value = self.form(value)
        value = value.view(-1,1024,4,4)
        return self.seq(value)

model = Generator()

PATH = './model/Gnet.pth'

noise = torch.randn(1,100, device="cuda")
model = model = torch.load(PATH, map_location=torch.device('cuda'))
out = model(noise)
print(out.shape)

plt.imshow(out.view(3,1024,1024).permute(1,2,0).cpu().detach().numpy())
plt.show()