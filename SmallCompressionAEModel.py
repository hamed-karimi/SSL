import torch
import torch.nn as nn

def get_configs(arch='vgg16'):

    if arch == 'vgg11':
        configs = [1, 1, 2, 2, 2]
    elif arch == 'vgg13':
        configs = [2, 2, 2, 2, 2]
    elif arch == 'vgg16':
        configs = [2, 2, 3, 3, 3]
    elif arch == 'vgg19':
        configs = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Undefined model")
    
    return configs

class VGGAutoEncoder(nn.Module):

    def __init__(self, configs):

        super(VGGAutoEncoder, self).__init__()

        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = VGGEncoder(configs=configs,       enable_bn=True)
        self.decoder = VGGDecoder(configs=configs[::-1], enable_bn=True)
        
    
    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x

class VGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGEncoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = EncoderBlock(input_dim=3,   output_dim=16,  hidden_dim=16,  layers=configs[0], enable_bn=enable_bn)
        self.conv2 = EncoderBlock(input_dim=16,  output_dim=24, hidden_dim=24, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = EncoderBlock(input_dim=24, output_dim=32, hidden_dim=32, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = EncoderBlock(input_dim=32, output_dim=32, hidden_dim=32, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = EncoderBlock(input_dim=32, output_dim=32, hidden_dim=32, layers=configs[4], enable_bn=enable_bn)
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class VGGDecoder(nn.Module):

    def __init__(self, configs, enable_bn=False):

        super(VGGDecoder, self).__init__()

        if len(configs) != 5:

            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = DecoderBlock(input_dim=32, output_dim=32, hidden_dim=32, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = DecoderBlock(input_dim=32, output_dim=32, hidden_dim=32, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = DecoderBlock(input_dim=32, output_dim=24, hidden_dim=32, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = DecoderBlock(input_dim=24, output_dim=16,  hidden_dim=24, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = DecoderBlock(input_dim=16,  output_dim=3,   hidden_dim=16,  layers=configs[4], enable_bn=enable_bn)
        self.gate = nn.Sigmoid()
    
    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gate(x)

        return x

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(EncoderBlock, self).__init__()

        if layers == 1:

            layer = EncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = EncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = EncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d EncoderLayer' % i, layer)
        
        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class DecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super(DecoderBlock, self).__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = DecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = DecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = DecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                
                self.add_module('%d DecoderLayer' % (i+1), layer)
    
    def forward(self, x):

        for name, layer in self.named_children():

            x = layer(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(EncoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
    
    def forward(self, x):

        return self.layer(x)

class DecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super(DecoderLayer, self).__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
    
    def forward(self, x):

        return self.layer(x)

# if __name__ == "__main__":

#     input = torch.randn((5,3,224,224))

#     configs = get_configs()

#     model = VGGAutoEncoder(configs)

#     output = model(input)

#     print(output.shape)



    