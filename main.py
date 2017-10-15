# Loading the modules
import torch, torchvision
import torch.utils as tUtils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim



# Loading the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]);

trainset = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform
                        );
trainloader = tUtils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=4)

testset = datasets.MNIST(root='./data', train=True,
                            download=True, transform=transform
                        );

testloader = tUtils.data.DataLoader(testset, batch_size=4,
                                        shuffle=True, num_workers=4)

## Model
# D = GAN_D().cuda()
# G = GAN_G().cuda()
D = GAN_D()
G = GAN_G()

opt_D = optim.RMSprop(D.parameters(), lr = 1e-4)
opt_G = optim.RMSprop(G.parameters(), lr = 1e-4)

criterion = nn.BCELoss()

def train_D(discriminator, images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()
    outputs = discriminator(images)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs

    outputs = discriminator(fake_images)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    d_loss = real_loss + fake_loss
    d_loss.backward()
    opt_D.step()
    return d_loss, real_score, fake_score

def train_G(generator, discriminator_outputs, real_labels):
    generator.zero_grad()
    gen_loss = criterion(discriminator_outputs, real_labels)
    gen_loss.backward()
    opt_G.step()
    return gen_loss

def get_noise(shape, size):
    return torch.randn(shape, size)

def train(epochs, dataloader):
    for epoch in range(epochs):
        for n, (images, _) in enumerate(dataloader): #Look up for the output od dataloader
            images = Variable(images.view(-1,784))
            real_labels = Variable(torch.ones(images.size(0)))

            #Get input from generator
            noise = get_noise(images.size(0), 100)
            fake_images = G(Variable(noise))
            fake_labels = Variable(torch.zeros(images.size(0)))

            # Train the discriminator
            train_D(D, images, real_labels, fake_images, fake_labels)

            # Get some samples from generator
            noise = get_noise(images.size(0), 100)
            fake_images = G(Variable(noise))
            outputs = D(fake_images)

            # Train the generator
            train_G(G, outputs, real_labels)
