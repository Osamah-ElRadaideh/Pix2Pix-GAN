import torch
import torch.nn as nn

class down_block(nn.Module):
    #encoding path of the Unet
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(down_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding='same')
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding='same')
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class up_block(nn.Module):
    #decoding path of the Unet
    def __init__(self,in_channels,out_channels,kernel_size=5,stride=2,padding=2):
        super(up_block,self).__init__()
        self.trans = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.4)
    def forward(self,x):
        x = self.trans(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x 


class conv_block(nn.Module):
    #discriminator's conv block
    def __init__(self, in_channels, out_channels,act='relu',padding = 1):
        super().__init__()
        assert act.lower() in ['relu', 'none'], f'expected activation to be either relu or none got {act} instead.'
        self.act = act
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2,padding=padding)
        self.relu = nn.LeakyReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        if self.act =='relu':
            return self.relu(x)
        else:
            return x


class Generator(nn.Module):
    #the Unet model, works with any input shape 2^n x 2^n
    def __init__(self,channels=[1,32,64,128,256,512]):
        super(Generator,self).__init__()
        self.conv1= down_block(channels[0],channels[1])
        self.maxpool= nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2= down_block(channels[1],channels[2])
        self.conv3= down_block(channels[2],channels[3])
        self.conv4= down_block(channels[3],channels[4])
        self.conv5= down_block(channels[4],channels[5])
        self.up1 = up_block(channels[5],channels[4])
        self.up2 = up_block(channels[4],channels[3])
        self.up3 = up_block(channels[3],channels[2])
        self.up4 = up_block(channels[2],channels[1])
        self.up5 = up_block(channels[1],channels[0])
        self.upconv1 = down_block(channels[5],channels[4])
        self.upconv2 = down_block(channels[4],channels[3])
        self.upconv3 = down_block(channels[3],channels[2])
        self.upconv4 = down_block(channels[2],channels[1])
        self.upconv5 = nn.Conv2d(channels[1],3,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x = x[:, None, : , :]
        x = self.conv1(x)
        state1 = x
        x = self.maxpool(x)
        x = self.conv2(x)
        state2 = x 
        x = self.maxpool(x)
        x = self.conv3(x)
        state3 = x 
        x = self.maxpool(x)
        x = self.conv4(x)
        state4 = x 
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.up1(x)
        x = torch.cat([x,state4],1)
        x = self.upconv1(x)
        x = self.up2(x)
        x = torch.cat([x,state3],1)
        x = self.upconv2(x)
        x = self.up3(x)
        x = torch.cat([x,state2],1)
        x = self.upconv3(x)
        x = self.up4(x)
        x = torch.cat([x,state1],1)
        x = self.upconv4(x)
        x = self.upconv5(x)
        x = self.relu(x)
        return x
    


class Discriminator(nn.Module):
    #patch gan discriminator
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64)
        self.conv2 = conv_block(64,128)
        self.conv3 = conv_block(128,256)
        self.conv4 = conv_block(256,512)
        self.conv5 = conv_block(512,1, act='None',padding=0)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x
            



def gen_loss(fake_outs):
    loss = 0.5 * torch.mean((fake_outs - 1) ** 2)

    return loss


def disc_loss(real_outs, fake_outs):
    d_loss = 0.5 * torch.mean((real_outs - 1)**2)
    g_loss = 0.5 * torch.mean(fake_outs ** 2)

    return d_loss + g_loss
