import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FCN8sx2(nn.Module):
    def __init__(self, num_classes=19):
        super(FCN8sx2, self).__init__()
        # conv1x1
        self.conv1x1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1x1_1 = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1x1_2 = nn.ReLU(inplace=True)
        self.pool1x1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv1x2
        self.conv1x2_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1x2_1 = nn.ReLU(inplace=True)
        self.conv1x2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1x2_2 = nn.ReLU(inplace=True)
        self.pool1x2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # score
        self.score = nn.Conv2d(4096, num_classes, 1)

        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x1, x2, parameters=None):
        if parameters is None:
            # concatenate features for 1st and 2nd inputs
            if x1 is None:
                # 2nd input
                h = x2
                h = self.relu1x2_1(self.conv1x2_1(h))
                h = self.relu1x2_2(self.conv1x2_2(h))
                conv1x2 = self.pool1x2(h)
                conv1 = conv1x2
            elif x2 is None:
                # 1st input
                h = x1
                h = self.relu1x1_1(self.conv1x1_1(h))
                h = self.relu1x1_2(self.conv1x1_2(h))
                conv1x1 = self.pool1x1(h)
                conv1 = conv1x1
            else:
                # 1st input
                h = x1
                h = self.relu1x1_1(self.conv1x1_1(h))
                h = self.relu1x1_2(self.conv1x1_2(h))
                conv1x1 = self.pool1x1(h)
                # 2nd input
                h = x2
                h = self.relu1x2_1(self.conv1x2_1(h))
                h = self.relu1x2_2(self.conv1x2_2(h))
                conv1x2 = self.pool1x2(h)
                conv1 = torch.cat((conv1x1, conv1x2), dim=0)

            h = self.relu2_1(self.conv2_1(conv1))
            h = self.relu2_2(self.conv2_2(h))
            conv2x2 = self.pool2(h)

            h = self.relu3_1(self.conv3_1(conv2x2))
            h = self.relu3_2(self.conv3_2(h))
            h = self.relu3_3(self.conv3_3(h))
            conv3 = self.pool3(h) # 1/8

            h = self.relu4_1(self.conv4_1(conv3))
            h = self.relu4_2(self.conv4_2(h))
            h = self.relu4_3(self.conv4_3(h))
            conv4 = self.pool4(h) # 1/16

            h = self.relu5_1(self.conv5_1(conv4))
            h = self.relu5_2(self.conv5_2(h))
            h = self.relu5_3(self.conv5_3(h))
            conv5 = self.pool5(h) # [1, 512, 16, 32]

            h = self.relu6(self.fc6(conv5))
            h = self.drop6(h)
            h = self.relu7(self.fc7(h))
            h = self.drop7(h)
            score = self.score(h)

            score_pool4 = self.score_pool4(conv4)  # channels: 19
            score_pool3 = self.score_pool3(conv3)  # channels: 19

            score1 = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
            score1 += score_pool4
            score2 = F.interpolate(score1, score_pool3.size()[2:], mode='bilinear', align_corners=True)
            score2 += score_pool3
            x = x2 if x1 is None else x1
            out = F.interpolate(score2, x.size()[2:], mode='bilinear', align_corners=True)

        else:
            if x1 is None:
                # 2nd input
                h = x2
                h = self.conv_layer_ff(h, parameters, 'conv1x2_1', 1, 1)
                h = self.conv_layer_ff(h, parameters, 'conv1x2_2', 1, 1)
                conv1x2 = self.maxpool_ff(h)
                conv1 = conv1x2
            elif x2 is None:
                # 1st input
                h = x1
                h = self.conv_layer_ff(h, parameters, 'conv1x1_1', 1, 1)
                h = self.conv_layer_ff(h, parameters, 'conv1x1_2', 1, 1)
                conv1x1 = self.maxpool_ff(h)
                conv1 = conv1x1
            else:
                # 1st input
                h = x1
                h = self.conv_layer_ff(h, parameters, 'conv1x1_1', 1, 1)
                h = self.conv_layer_ff(h, parameters, 'conv1x1_2', 1, 1)
                conv1x1 = self.maxpool_ff(h)
                # 2nd input
                h = x2
                h = self.conv_layer_ff(h, parameters, 'conv1x2_1', 1, 1)
                h = self.conv_layer_ff(h, parameters, 'conv1x2_2', 1, 1)
                conv1x2 = self.maxpool_ff(h)
                conv1 = torch.cat((conv1x1, conv1x2), dim=0)

            h = self.conv_layer_ff(conv1, parameters, 'conv2_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv2_2', 1, 1)
            conv2 = self.maxpool_ff(h)

            h = self.conv_layer_ff(conv2, parameters, 'conv3_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv3_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv3_3', 1, 1)
            h = self.maxpool_ff(h)
            conv3 = h  # 1/8

            h = self.conv_layer_ff(conv3, parameters, 'conv4_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv4_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv4_3', 1, 1)
            h = self.maxpool_ff(h)
            conv4 = h  # 1/16

            h = self.conv_layer_ff(conv4, parameters, 'conv5_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv5_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv5_3', 1, 1)
            conv5 = self.maxpool_ff(h)

            h = self.conv_layer_ff(conv5, parameters, 'fc6', 1, 0)
            h = F.dropout(h)
            h = self.conv_layer_ff(h, parameters, 'fc7', 1, 0)
            h = F.dropout(h)
            score = self.conv_layer_ff(h, parameters, 'score', 1, 0)

            score_pool4 = self.conv_layer_ff(conv4, parameters, 'score_pool4', 1, 0)
            score_pool3 = self.conv_layer_ff(conv3, parameters, 'score_pool3', 1, 0)

            score1 = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
            score1 += score_pool4
            score2 = F.interpolate(score1, score_pool3.size()[2:], mode='bilinear', align_corners=True)
            score2 += score_pool3
            x = x1 if x2 is None else x2
            out = F.interpolate(score2, x.size()[2:], mode='bilinear', align_corners=True)

        return conv5, out


    def conv_layer_ff(self, x, parameters, param_name, stride, padding):
        x = F.conv2d(input=x, weight=parameters['%s.weight'%param_name],
                              bias=parameters['%s.bias'%param_name],
                              stride=stride, padding=padding)
        x = F.relu(x, inplace=True)
        return x

    def maxpool_ff(self, x, kernel_size=2, stride=2, ceil_mode=True):
        x = F.max_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=ceil_mode)
        return x

    def get_parameters(self, bias=False):
        import torch.nn as nn
        modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            FCN8sx2,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))

    def adjust_learning_rate(self, optimizer, i, cfg):
        if cfg.TRAIN.SCHEDULER == 'step':
            optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = cfg.TRAIN.LEARNING_RATE * (0.1**(int(i/50000))) * 2
        elif cfg.TRAIN.SCHEDULER == 'polynomial':
            optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE * ((1 - float(i) / cfg.TRAIN.MAX_ITERS) ** cfg.TRAIN.POWER)
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = cfg.TRAIN.LEARNING_RATE * ((1 - float(i) / cfg.TRAIN.MAX_ITERS) ** cfg.TRAIN.POWER) * 2

def init_vggfcnx2():
    import torchvision.models as models

    # initiate a FCN model
    model = VGG16_FCN8sx2(num_classes=19, init_weights=None)
    params = model.state_dict().copy()

    # load the pytorch pretrained VGG16
    vgg16 = models.vgg16(pretrained=True)
    vgg16_pretrained = vgg16.state_dict().copy()

    # copy convolution parameters
    params['conv1x1_1.weight'] = vgg16_pretrained['features.0.weight']
    params['conv1x1_1.bias'] = vgg16_pretrained['features.0.bias']
    params['conv1x1_2.weight'] = vgg16_pretrained['features.2.weight']
    params['conv1x1_2.bias'] = vgg16_pretrained['features.2.bias']

    params['conv1x2_1.weight'] = vgg16_pretrained['features.0.weight']
    params['conv1x2_1.bias'] = vgg16_pretrained['features.0.bias']
    params['conv1x2_2.weight'] = vgg16_pretrained['features.2.weight']
    params['conv1x2_2.bias'] = vgg16_pretrained['features.2.bias']

    params['conv2_1.weight'] = vgg16_pretrained['features.5.weight']
    params['conv2_1.bias'] = vgg16_pretrained['features.5.bias']
    params['conv2_2.weight'] = vgg16_pretrained['features.7.weight']
    params['conv2_2.bias'] = vgg16_pretrained['features.7.bias']
    params['conv3_1.weight'] = vgg16_pretrained['features.10.weight']
    params['conv3_1.bias'] = vgg16_pretrained['features.10.bias']
    params['conv3_2.weight'] = vgg16_pretrained['features.12.weight']
    params['conv3_2.bias'] = vgg16_pretrained['features.12.bias']
    params['conv3_3.weight'] = vgg16_pretrained['features.14.weight']
    params['conv3_3.bias'] = vgg16_pretrained['features.14.bias']
    params['conv4_1.weight'] = vgg16_pretrained['features.17.weight']
    params['conv4_1.bias'] = vgg16_pretrained['features.17.bias']
    params['conv4_2.weight'] = vgg16_pretrained['features.19.weight']
    params['conv4_2.bias'] = vgg16_pretrained['features.19.bias']
    params['conv4_3.weight'] = vgg16_pretrained['features.21.weight']
    params['conv4_3.bias'] = vgg16_pretrained['features.21.bias']
    params['conv5_1.weight'] = vgg16_pretrained['features.24.weight']
    params['conv5_1.bias'] = vgg16_pretrained['features.24.bias']
    params['conv5_2.weight'] = vgg16_pretrained['features.26.weight']
    params['conv5_2.bias'] = vgg16_pretrained['features.26.bias']
    params['conv5_3.weight'] = vgg16_pretrained['features.28.weight']
    params['conv5_3.bias'] = vgg16_pretrained['features.28.bias']

    # copy the fc parameters
    params['fc6.weight'] = vgg16_pretrained['classifier.0.weight'].view(params['fc6.weight'].size())
    params['fc6.bias'] = vgg16_pretrained['classifier.0.bias']
    params['fc7.weight'] = vgg16_pretrained['classifier.3.weight'].view(params['fc7.weight'].size())
    params['fc7.bias'] = vgg16_pretrained['classifier.3.bias']
    params['score.weight'] = vgg16_pretrained['classifier.6.weight'][:19].view(params['score.weight'].size())
    params['score.bias'] = vgg16_pretrained['classifier.6.bias'][:19]

    # load into FCN and save as init model
    model.load_state_dict(params)
    torch.save(model.state_dict(), '../../pretrained_models/vggfcnx2_init_v2.pth')


def VGG16_FCN8sx2(num_classes=21, init_weights=None, restore_from=None):
    model = FCN8sx2(num_classes=num_classes)
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location=lambda storage, loc: storage))
    if restore_from is not None:
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))
    return model


def get_fcn8sx2_vgg(num_classes=19, init_weights='../../pretrained_models/vggfcnx2_init_v2.pth', restore_from=None):
    model = FCN8sx2(num_classes=num_classes)
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location=lambda storage, loc: storage))
    if restore_from is not None:
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))
    return model


class FCN8s(nn.Module):
    def __init__(self, num_classes=19):
        super(FCN8s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # score
        self.score = nn.Conv2d(4096, num_classes, 1)

        self.score_pool4 = nn.Conv2d(512, num_classes, 1)
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Make a 2D bilinear kernel suitable for upsampling"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
        return torch.from_numpy(weight).float()

    def forward(self, x, parameters=None):
        if parameters is None:
            h = x
            h = self.relu1_1(self.conv1_1(h))
            h = self.relu1_2(self.conv1_2(h))
            conv1 = self.pool1(h)

            h = self.relu2_1(self.conv2_1(conv1))
            h = self.relu2_2(self.conv2_2(h))
            conv2 = self.pool2(h)

            h = self.relu3_1(self.conv3_1(conv2))
            h = self.relu3_2(self.conv3_2(h))
            h = self.relu3_3(self.conv3_3(h))
            conv3 = self.pool3(h) # 1/8

            h = self.relu4_1(self.conv4_1(conv3))
            h = self.relu4_2(self.conv4_2(h))
            h = self.relu4_3(self.conv4_3(h))
            conv4 = self.pool4(h) # 1/16

            h = self.relu5_1(self.conv5_1(conv4))
            h = self.relu5_2(self.conv5_2(h))
            h = self.relu5_3(self.conv5_3(h))
            conv5 = self.pool5(h) # [1, 512, 16, 32]

            h = self.relu6(self.fc6(conv5))
            h = self.drop6(h)
            h = self.relu7(self.fc7(h))
            h = self.drop7(h)
            score = self.score(h)

            score_pool4 = self.score_pool4(conv4)  # channels: 19
            score_pool3 = self.score_pool3(conv3)  # channels: 19

            score1 = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
            score1 += score_pool4
            score2 = F.interpolate(score1, score_pool3.size()[2:], mode='bilinear', align_corners=True)
            score2 += score_pool3
            out = F.interpolate(score2, x.size()[2:], mode='bilinear', align_corners=True)

        else:
            h = x
            h = self.conv_layer_ff(h, parameters, 'conv1_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv1_2', 1, 1)
            conv1 = self.maxpool_ff(h)

            h = self.conv_layer_ff(conv1, parameters, 'conv2_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv2_2', 1, 1)
            conv2 = self.maxpool_ff(h)

            h = self.conv_layer_ff(conv2, parameters, 'conv3_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv3_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv3_3', 1, 1)
            h = self.maxpool_ff(h)
            conv3 = h  # 1/8

            h = self.conv_layer_ff(conv3, parameters, 'conv4_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv4_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv4_3', 1, 1)
            h = self.maxpool_ff(h)
            conv4 = h  # 1/16

            h = self.conv_layer_ff(conv4, parameters, 'conv5_1', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv5_2', 1, 1)
            h = self.conv_layer_ff(h, parameters, 'conv5_3', 1, 1)
            conv5 = self.maxpool_ff(h)

            h = self.conv_layer_ff(conv5, parameters, 'fc6', 1, 0)
            h = F.dropout(h)
            h = self.conv_layer_ff(h, parameters, 'fc7', 1, 0)
            h = F.dropout(h)
            score = self.conv_layer_ff(h, parameters, 'score', 1, 0)

            score_pool4 = self.conv_layer_ff(conv4, parameters, 'score_pool4', 1, 0)
            score_pool3 = self.conv_layer_ff(conv3, parameters, 'score_pool3', 1, 0)

            score1 = F.interpolate(score, score_pool4.size()[2:], mode='bilinear', align_corners=True)
            score1 += score_pool4
            score2 = F.interpolate(score1, score_pool3.size()[2:], mode='bilinear', align_corners=True)
            score2 += score_pool3
            out = F.interpolate(score2, x.size()[2:], mode='bilinear', align_corners=True)

        return conv5, out

    def conv_layer_ff(self, x, parameters, param_name, stride, padding):
        x = F.conv2d(input=x, weight=parameters['%s.weight'%param_name],
                              bias=parameters['%s.bias'%param_name],
                              stride=stride, padding=padding)
        x = F.relu(x, inplace=True)
        return x

    def maxpool_ff(self, x, kernel_size=2, stride=2, ceil_mode=True):
        x = F.max_pool2d(x, kernel_size=stride, stride=stride, ceil_mode=ceil_mode)
        return x

    def get_parameters(self, bias=False):
        import torch.nn as nn
        modules_skipped = (
            nn.ReLU,
            nn.MaxPool2d,
            nn.Dropout2d,
            FCN8s,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if bias:
                    yield m.bias
                else:
                    yield m.weight
            elif isinstance(m, modules_skipped):
                continue
            else:
                raise ValueError('Unexpected module: %s' % str(m))

    def adjust_learning_rate(self, optimizer, i, cfg):
        if cfg.TRAIN.SCHEDULER == 'step':
            optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = cfg.TRAIN.LEARNING_RATE * (0.1**(int(i/50000))) * 2
        elif cfg.TRAIN.SCHEDULER == 'polynomial':
            optimizer.param_groups[0]['lr'] = cfg.TRAIN.LEARNING_RATE * ((1 - float(i) / cfg.TRAIN.MAX_ITERS) ** cfg.TRAIN.POWER)
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = cfg.TRAIN.LEARNING_RATE * ((1 - float(i) / cfg.TRAIN.MAX_ITERS) ** cfg.TRAIN.POWER) * 2


def VGG16_FCN8s(num_classes=21, init_weights=None, restore_from=None):
    model = FCN8s(num_classes=num_classes)
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location=lambda storage, loc: storage))
    if restore_from is not None:
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))
    return model

def get_fcn8s_vgg(num_classes=19, init_weights='../../pretrained_models/vggfcn_init_v2.pth', restore_from=None):
    model = FCN8s(num_classes=num_classes)
    if init_weights is not None:
        pass
        # model.load_state_dict(torch.load(init_weights, map_location=lambda storage, loc: storage))
    if restore_from is not None:
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))
    return model


if __name__ == '__main__':
    init_vggfcnx2()
