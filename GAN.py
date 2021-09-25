import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

PATH = './model/Gnet.pth'

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
    optimD = optim.AdamW(Dnet.parameters(), lr=0.0003,betas=(0.5,0.999))
    optimG = optim.AdamW(Gnet.parameters(), lr=0.0003,betas=(0.5,0.999))

    static_noise = torch.randn(batch, 100, device=device)
    tensorboard_step = 0
    writer = SummaryWriter("runs/GAN/test")
    with torch.no_grad():
        test = Gnet(static_noise).cpu()
        test_grid = torchvision.utils.make_grid(
            test[:batch], normalize=True
        )
        writer.add_image("TestImage", test_grid, global_step=tensorboard_step)
        tensorboard_step += 1

    for i in range(epoch):
        print(f"epoch: {i+1}")
        iter = 0
        for image, _ in tqdm(dataloader):
            image = image.to(device=device)
            noise = torch.randn(batch, 100, device=device)
            fake = Gnet(noise)

            disc_real = Dnet(image).reshape(-1)
            loss_real = criterion(disc_real, torch.ones_like(disc_real)) 
            disc_fake = Dnet(fake).reshape(-1)
            loss_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss = (loss_fake + loss_real)/2   
            for param in Dnet.parameters():
                param.grad = None
            loss.backward(retain_graph = True)
            optimD.step()
            
            Dout = Dnet(fake).reshape(-1)
            loss_Gnet = criterion(Dout,torch.ones_like(Dout))
            for param in Gnet.parameters():
                param.grad = None
            loss_Gnet.backward()
            optimG.step()
            iter += 1

            if iter%100 == 0:
                with torch.no_grad():
                    test = Gnet(static_noise).cpu()
                    test_grid = torchvision.utils.make_grid(
                        test[:batch], normalize=True
                    )
                    writer.add_image("TestImage", test_grid, global_step=tensorboard_step)
                    tensorboard_step += 1
        print("saving...")            
        torch.save(Gnet, PATH)    
        writer.flush()
        print("saved")

    with torch.no_grad():
        test = Gnet(static_noise).cpu()
        test_grid = torchvision.utils.make_grid(
            test[:batch], normalize=True
        )
        writer.add_image("TestImage", test_grid, global_step=tensorboard_step)
        tensorboard_step += 1
        
    
    writer.flush()
    writer.close()
    torch.save(Gnet, PATH)

def start():
    train(dataloader, 10)
            
if __name__ == "__main__":
    start()