from models import VGG_AE as vgg
from models import ResNet_AE as resnet


def build_model(arch):
    if arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(arch)
        model = vgg.VGGAutoEncoder(configs)

    elif arch in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]:
        configs, bottleneck = resnet.get_configs(arch)
        model = resnet.ResNetAutoEncoder(configs, bottleneck)
    else:
        raise ValueError("Undefined model")

    return model