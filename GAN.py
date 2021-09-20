import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    
batch = 16

dataset = torchvision.datasets.ImageFolder("data", transforms.Compose([
    transforms.ToTensor(),
]))

dataloader = DataLoader(dataset=dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=6,drop_last=True)

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
    optimD = optim.Adam(Dnet.parameters(), lr=0.0003,betas=(0.5,0.999))
    optimG = optim.Adam(Gnet.parameters(), lr=0.0003,betas=(0.5,0.999))

    static_noise = torch.randn(batch, 100, device=device)
    writer = SummaryWriter()
    writer.add_images("TestImages/base", Gnet(static_noise).cpu(), 0, dataformats='NCHW')

    for i in range(epoch):
        print(f"epoch: {i+1}")
        for image, _ in tqdm(dataloader):
            image = image.to(device=device)

            Dout = Dnet(image).reshape(-1)
            loss_real = criterion(Dout, torch.ones_like(Dout)) 

            noise = torch.randn(batch, 100, device=device)
            Gout = Gnet(noise)
            Dout = Dnet(Gout.detach()).reshape(-1)
            loss_fake = criterion(Dout, torch.zeros_like(Dout))   
            for param in Dnet.parameters():
                param.grad = None
            loss = (loss_fake + loss_real)/2
            loss.backward(retain_graph = True)
            optimD.step()
            
            Dout = Dnet(Gout).reshape(-1)
            loss = criterion(Dout,torch.zeros_like(Dout))
            for param in Gnet.parameters():
                param.grad = None
            loss.backward()
            optimG.step()
        writer.add_images(f"TestImages/epoch-{i +1}", Gnet(static_noise).cpu(), i+1, dataformats='NCHW')
    
    writer.flush()
    writer.close()
    return Gnet

def start():
    model = train(dataloader, 3)
    PATH = './model/Gnet.pth'
    torch.save(model, PATH)
            
if __name__ == "__main__":
    start()