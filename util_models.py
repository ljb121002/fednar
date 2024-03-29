from util_libs import *
import torch.nn as nn

#####################################################
# Neural net architectures
####################################################
    
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(2,planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(2,self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(2,64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 3)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
def resnet18(n_c):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=n_c)


def get_model(model,n_c):
    return resnet18(n_c)



class shake_transf(nn.Module):
    def __init__(self, args):
        super(shake_transf, self).__init__()
        embedding_dim = 128
        num_layers = args['num_layer']
        hidden_size = 512
        input_length = 80
        self.n_cls = 80

        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(input_length, embedding_dim)
        self.position_encoder = nn.Embedding(input_length, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, \
                                                    dim_feedforward=hidden_size, \
                                                    dropout=args['drop_out'])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, self.n_cls)

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.embedding_dim)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).repeat(x.size(0), 1)
        x = x + self.position_encoder(positions)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)[:, -1, :]
        x = self.fc(x)

        return x
              
        