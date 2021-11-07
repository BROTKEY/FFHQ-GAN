import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.transforms import Resize
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

PATH = './model/'

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BATCH = 32

IMAGESIZE = 256

DATASET = torchvision.datasets.ImageFolder("data", transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGESIZE),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]))

DATALOADER = DataLoader(dataset=DATASET, batch_size=BATCH,
                        shuffle=True, pin_memory=True, num_workers=6, drop_last=True)


class Generator(nn.Module):
    def __init__(self, image_size):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        ))

        for i in range(self.log2(image_size)-3):
            self.layers.append(self.ConvolutionalTransposeLayer(
                pow(2, 10 - i), pow(2, 10 - (i+1)), 4, 2, 1))

        self.layers.append(nn.Sequential(
            nn.ConvTranspose2d(
                pow(2, 10 - self.log2(image_size) + 3), 3, 4, 2, 1),
            nn.Tanh()
        ))

    def log2(self, x):
        return int(torch.log2(torch.tensor(x)).item())

    def ConvolutionalTransposeLayer(self, input_size, output_size, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_size, output_size, kernel_size,
                               stride, padding, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(),
        )

    def forward(self, value):
        output = value
        for layer in self.layers:
            output = layer(output)
        return output


class Critic(nn.Module):
    def __init__(self, image_size):
        super(Critic, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Sequential(
            nn.Conv2d(3, pow(2, 10 - self.log2(image_size) + 3),
                      4, 2, 1, bias=False),
            nn.BatchNorm2d(pow(2, 10 - self.log2(image_size) + 3)),
            nn.LeakyReLU(),
        ))

        for i in range(self.log2(image_size)-3):
            self.layers.append(
                self.ConvolutionalLayer(
                    pow(2, (10 - self.log2(image_size)) + i + 3), pow(2, (10 - self.log2(image_size)) + i + 4), 4, 2, 1)
            )

        self.layers.append(nn.Sequential(
            nn.Conv2d(1024, 1, 4, 1, 0),
        ))

    def log2(self, x):
        return int(torch.log2(torch.tensor(x)).item())

    def ConvolutionalLayer(self, input_size, output_size, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(),
        )

    def forward(self, value):
        output = value
        for layer in self.layers:
            output = layer(output)
        return output


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if isinstance(m, (nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


def train(dataloader, epoch, load=False):

    critic = Critic(IMAGESIZE)
    if load:
        critic = torch.load(PATH + "Cnet.pth")
    critic.train()
    critic.apply(initialize_weights)
    critic.to(device=DEVICE)
    print(critic)

    generator = Generator(IMAGESIZE)
    if load:
        generator = torch.load(PATH + "Gnet.pth")
    generator.train()
    generator.apply(initialize_weights)
    generator.to(device=DEVICE)
    print(generator)

    optim_critic = optim.RMSprop(critic.parameters(), lr=0.00005)
    optim_generator = optim.RMSprop(generator.parameters(), lr=0.00005)
    tensorboard_step = 0
    writer = SummaryWriter("runs/GAN/test")

    ld = []
    lg = []
    for i in range(epoch):
        print(f"epoch: {i+1}")
        iter = 0
        for image, _ in tqdm(dataloader):
            image = image.to(device=DEVICE)

            # train critic
            for _ in range(5):
                critic_noise = torch.randn((BATCH, 100, 1, 1), device=DEVICE)
                critic_fake = generator(critic_noise)
                critic_real = critic(image).reshape(-1)
                critic_output = critic(critic_fake).reshape(-1)

                # wasserstein loss
                loss_critic = -(torch.mean(critic_real) -
                                torch.mean(critic_output))

                ld.append(loss_critic.item())
                for param in critic.parameters():
                    param.grad = None
                loss_critic.backward(retain_graph=True)
                optim_critic.step()

                # weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train generator
            generator_noise = torch.randn((BATCH, 100, 1, 1), device=DEVICE)
            generator_fake = generator(generator_noise)
            generator_output = critic(generator_fake).reshape(-1)
            loss_generator = -torch.mean(generator_output)
            lg.append(loss_generator.item())
            for param in generator.parameters():
                param.grad = None
            loss_generator.backward()
            optim_generator.step()
            iter += 1

            if iter % 50 == 0 or iter == 1:
                # unnecessary duplicated code
                logToTensorboard(ld, lg, generator_noise, writer,
                                 tensorboard_step, generator, image)
                tensorboard_step += 1
                ld = []
                lg = []

        print("----------------\n\rsaving...")
        torch.save(generator, PATH + "Gnet.pth")
        torch.save(critic, PATH + "Cnet.pth")
        writer.flush()
        print("saved\n\r----------------")
    writer.flush()
    writer.close()


def logToTensorboard(ld, lg, static_noise, writer, step, generator, real):
    with torch.no_grad():
        writer.add_scalars('current_run', {'loss_Critic': torch.mean(torch.tensor(ld)).item(
        ), 'loss_Generator': torch.mean(torch.tensor(lg)).item()}, global_step=step)
        test = generator(static_noise).cpu()
        test_grid = torchvision.utils.make_grid(
            test[:BATCH], normalize=True
        )
        writer.add_image("TestImage", test_grid, global_step=step)
        real_grid = torchvision.utils.make_grid(
            real[:BATCH], normalize=True
        )
        writer.add_image("RealImage", real_grid, global_step=step)


def start():
    train(DATALOADER, 50, load=False)


if __name__ == "__main__":
    start()
