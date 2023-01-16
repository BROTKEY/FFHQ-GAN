import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision

PATH = './model/'

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")

BATCH = 128

IMAGESIZE = 512

LOAD = False

DATASET = torchvision.datasets.ImageFolder("../data/", transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMAGESIZE),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]))

DATALOADER = DataLoader(dataset=DATASET, batch_size=BATCH,
                        shuffle=True, pin_memory=True, num_workers=12, drop_last=True, prefetch_factor=2)


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
                    pow(2, 10 - self.log2(image_size) + i + 3), pow(2, 10 - self.log2(image_size) + i + 4), 4, 2, 1)
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


def train(dataloader, epoch):

    critic = Critic(IMAGESIZE)
    critic = initialize_model(critic, "Cnet.pth")

    generator = Generator(IMAGESIZE)
    generator = initialize_model(generator, "Gnet.pth")

    optim_critic = optim.RMSprop(critic.parameters(), lr=0.00005)
    optim_generator = optim.RMSprop(generator.parameters(), lr=0.00005)
    tensorboard_step = 0
    writer = SummaryWriter("runs/GAN/test")

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

                for param in critic.parameters():
                    param.grad = None
                loss_critic.backward()
                optim_critic.step()

                # weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-0.01, 0.01)

            # train generator
            generator_noise = torch.randn((BATCH, 100, 1, 1), device=DEVICE)
            generator_fake = generator(generator_noise)
            generator_output = critic(generator_fake).reshape(-1)
            loss_generator = -torch.mean(generator_output)
            for param in generator.parameters():
                param.grad = None
            loss_generator.backward()
            optim_generator.step()
            iter += 1
            if iter % 250 == 0:
                # unnecessary duplicated code
                logToTensorboard(generator_fake, writer,
                                 tensorboard_step)
                tensorboard_step += 1
                save(generator, critic, writer)

        save(generator, critic, writer)
    writer.flush()
    writer.close()


def initialize_model(model, file):
    model.to(device=DEVICE)
    model.train()
    model.apply(initialize_weights)
    if torch.cuda.device_count() > 1:
        print("Using: '", torch.cuda.device_count(), "' Devices")
        model = nn.DataParallel(model)
    if LOAD:
        model.load_state_dict(torch.load(PATH + file))
    print(model)
    return model


def save(generator, critic, writer):
    print("----------------\r\nsaving\r\n.")
    torch.save(generator.state_dict(), PATH + "Gnet.pth")
    print(".")
    torch.save(critic.state_dict(), PATH + "Cnet.pth")
    print(".")
    writer.flush()
    print("saved\r\n----------------")


def logToTensorboard(fake, writer, step):
    with torch.no_grad():
        fake.cpu()
        fake_grid = torchvision.utils.make_grid(
            fake[:BATCH], normalize=True
        )
        writer.add_image("TestImage", fake_grid, global_step=step)


def start():
    train(DATALOADER, 100)


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("---\r\nGPU MODE\r\n---")
    else:
        print("---\r\nCPU MODE\r\n---")
    if LOAD:
        print("---\r\nLOAD MODE\r\n---")
    start()
