from torch import nn
import torch
from networks.resnet_TTA import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from utils.convert import *


class SaveFeatures():
    def __init__(self, m, n):
        self.features = None
        self.name = n
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        out = self.bn(F.relu(cat_p))
        return out


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, convert=True, newBN=AdaBN, warm_n=5):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet50':
            base_model = resnet50
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise Exception('The Resnet Model only accept resnet34 and resnet50!')

        self.res = base_model(pretrained=pretrained)

        self.num_classes = num_classes

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        # Convert BN layer
        self.newBN = newBN
        if convert:
            self.res = convert_encoder_to_target(self.res, newBN, start=0, end=5, verbose=False, bottleneck=bottleneck, warm_n=warm_n)
            self.up1, self.up2, self.up3, self.up4, self.bnout = convert_decoder_to_target(
                [self.up1, self.up2, self.up3, self.up4, self.bnout], newBN, start=0, end=5, verbose=False, warm_n=warm_n)

        # Save the output feature of each BN layer.
        self.feature_hooks = []
        layers = [self.res.bn1, self.res.layer1, self.res.layer2, self.res.layer3, self.res.layer4]
        for i, layer in enumerate(layers):
            if i == 0:
                self.feature_hooks.append(SaveFeatures(layer, 'first_bn'))
            else:
                for j, block in enumerate(layer):
                    self.feature_hooks.append(SaveFeatures(block.bn1, str(i)+'-bn1'))      # BasicBlock
                    self.feature_hooks.append(SaveFeatures(block.bn2, str(i)+'-bn2'))      # BasicBlock
                    if resnet == 'resnet50':
                        self.feature_hooks.append(SaveFeatures(block.bn3, str(i)+'-bn3'))  # Bottleneck
                    if block.downsample is not None:
                        self.feature_hooks.append(SaveFeatures(block.downsample[1], str(i)+'-downsample_bn'))
        self.feature_hooks.append(SaveFeatures(self.up1.bn, '1-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up2.bn, '2-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up3.bn, '3-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up4.bn, '4-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.bnout, 'last_bn'))

    def change_BN_status(self, new_sample=True):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample

    def reset_sample_num(self):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = 0

    def forward(self, x):
        x, sfs = self.res(x)
        x = F.relu(x)

        x = self.up1(x, sfs[3])
        x = self.up2(x, sfs[2])
        x = self.up3(x, sfs[1])
        x = self.up4(x, sfs[0])
        x = self.up5(x)
        head_input = F.relu(self.bnout(x))

        seg_output = self.seg_head(head_input)

        return seg_output, sfs, head_input

    def close(self):
        for sf in self.sfs:
            sf.remove()


if __name__ == "__main__":
    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False)
    print(model.res)
    model.cuda().eval()
    input = torch.rand(2, 3, 512, 512).cuda()
    seg_output, x_iw_list, iw_loss = model(input)
    print(seg_output.size())

