import torch
import torchvision
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CSRNet(nn.Module):
    
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        
        self.frontend = make_layers(self.frontend_feat, in_channels=3, dilation=False)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        if not load_weights:
            
            # Load VGG16 with a progress bar
            weights = torchvision.models.VGG16_Weights.DEFAULT  # Pretrained weights on ImageNet
            vgg16_model = torchvision.models.vgg16(weights=weights, progress=True).to(device)  # progress=True enables the download progress bar
            
            self._initialize_weights()
            # Load weights into frontend layers
            for i, (name, module) in enumerate(self.frontend.named_children()):
                if isinstance(module, nn.Conv2d):
                    module.weight.data[:] = vgg16_model.features[i].weight.data[:]
                    if module.bias is not None:
                        module.bias.data[:] = vgg16_model.features[i].bias.data[:]
                
    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
                
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(cfg, in_channels=3, dilation=False):
    layers = []
    d_rate = 2 if dilation else 1
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    return nn.Sequential(*layers)


# csrnet = CSRNet()
# print(csrnet)
