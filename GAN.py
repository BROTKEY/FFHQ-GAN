import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
batch = 32

dataset = torchvision.datasets.ImageFolder("data", transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=6,)

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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3,8,4,2,1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(8,16,4,2,1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16,32,4,2,1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512,1024,4,2,1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )
        self.linear = nn.Linear(16384,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, value):
        value = self.seq(value)
        value = value.view(-1,16384)
        value = self.linear(value)
        return self.sigmoid(value)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train(dataloader, epoch):
    static_noise = torch.randn(1,100, device=device)

    Dnet = Discriminator()
    Dnet.train()
    Dnet.apply(weights_init)
    Dnet.to(device=device)
    print(Dnet)

    Gnet = Generator()
    Gnet.train()
    Gnet.apply(weights_init)
    Gnet.to(device=device)
    print(Gnet)

    criterion = nn.BCELoss().to(device=device)
    optimD = optim.Adam(Dnet.parameters(), lr=0.01)
    optimG = optim.Adam(Gnet.parameters(), lr=0.0003)

    for i in range(epoch):
        for image, _ in tqdm(dataloader):
            image = image.to(device=device)
            label_fake = torch.full((batch, 1), 0.0 , dtype=torch.float, device=device)
            label_real = torch.full((batch, 1), 1.0 , dtype=torch.float, device=device)

            Dnet.zero_grad()
            Dout = Dnet(image)
            loss = criterion(Dout, label_real) 
            loss.backward()

            noise = torch.randn(32, 100, device=device)
            Gout = Gnet(noise)
            Dout = Dnet(Gout.detach())
            loss = criterion(Dout, label_fake)
            loss.backward()
            optimD.step()
            

            Gnet.zero_grad()
            Gout = Dnet(Gout)
            loss_fake = criterion(Gout,label_real)
            loss_fake.backward()
            optimG.step()
        plt.imshow(Gnet(static_noise)[0].permute(1,2,0).cpu().detach().numpy())
        plt.show()

            
if __name__ == "__main__":
    train(dataloader,1)