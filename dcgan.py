import torch, torchvision
import torch.utils as tUtils
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')

opt = parser.parse_args()
print(opt)


class Disriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Disriminator, self).__init__()
        self.main = nn.Sequential(
            #input size : nc * H * W
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # No BatchNorm layer for the input of the Disciminator
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: nf * H/2 * W/2
            nn.Conv2d(ndf, ndf *2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: nf*2 * H/4 * W/4
            nn.Conv2d(ndf*2, ndf *4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: nf*4 * H/8 * W/8
            nn.Conv2d(ndf*4, ndf *8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # Shape: nf*8 * H/16 * W/16
            nn.Conv2d(ndf*8, 1 , 4, 1, 0, bias=False),
            #Shape: 1 * 1 * 1 for H/W = 64
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.main(inp)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            #Input noise: batch_size * Z
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            #Shape: ngf*8 x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #Shape: ngf*4 x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #Shape: ngf*8 x 16 x 16
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #Shape: ngf*8 x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4,2,1, bias=False),
            nn.Tanh()
            #Shape: nc * 64 * 64
        )

    def forward(self,inp):
        return self.main(inp)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

nc =3; nz = 100; ngf = 64; ndf =64

netG = Generator(nz, ngf, nc)
netG.apply(weights_init)
print(netG)

netD = Disriminator(nc, ndf)
netD.apply(weights_init)
print(netD)

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

def get_noise(shape):
    return torch.randn(shape)

outf = os.getcwd() + '/results'
for epoch in range(opt.epochs):
    for i, (images,_) in enumerate(dataloader, 0):
        batchsize = images.size[0]
        inp = images.cuda()
        inpv = Variable(inp)
        real_labels = Variable(torch.ones(batchsize))

        outp = netD(inpv)
        errD_real = criterion(outp, real_labels)
        errD_real.backward()
        D_x = outp.data.mean()

        noise = get_noise((batchsize, nz, 1, 1))
        noiseV = Variable(noise)
        fake_inp = netG(noiseV)
        fake_labels = Variable(torch.zeros(batchsize))

        outp = netD(fake_inp.detach())
        errD_fake = criterion(outp, fake_labels)
        errD_fake.backward()
        D_G_z1 = outp.data.mean()

        errD = errD_real + errD_fake
        optimizerD.step()

        #Update Generator
        netG.zero_grad()
        labelsv = Variable(torch.ones(batchsize))
        outp = netD(fake_inp)
        errG = criterion(outp, labelsv)
        errG.backward()
        D_G_z2 = outp.data.mean()

        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, epochs, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(inp,
                '%s/real_samples.png' % outf,
                normalize=True)
            fake = netG(Variable(get_noise((batchsize, nz, 1, 1))))
            vutils.save_image(fake.data,
                '%s/fake_samples_epoch_%03d.png' % (outf, epoch),
                normalize=True)
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (outf, epoch))
