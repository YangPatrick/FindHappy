import torch.nn as nn

from src.blocks import CNN, FFNN
from src.resnet import ResNet, BasicRSB, BasicBlock, Bottleneck, BottleneckRSB


class CNNClassifier(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super(CNNClassifier, self).__init__()
        self.cnn = CNN(in_channels, 50, [3, 4, 5])
        self.fc = nn.Linear(3 * 50, num_classes)

    def forward(self, inputs):
        return self.fc(self.cnn(inputs))


class ResNetClassifier(nn.Module):
    depth_map = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, in_channels, shrink=False, depth=18, num_classes=5):
        super(ResNetClassifier, self).__init__()
        assert depth in self.depth_map
        if shrink:
            blocks = [BasicRSB, BottleneckRSB]
        else:
            blocks = [BasicBlock, Bottleneck]
        self.resnet = ResNet(blocks[int(depth >= 50)], in_channels, self.depth_map[depth], num_classes=num_classes)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        return self.resnet(inputs)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, hidden_size, batch_norm, num_classes=5):
        super(MLP, self).__init__()
        self.fc = FFNN(hidden_layers, input_size, hidden_size, num_classes, 0.2, batch_norm=batch_norm)

    def forward(self, inputs):
        return self.fc(inputs)
