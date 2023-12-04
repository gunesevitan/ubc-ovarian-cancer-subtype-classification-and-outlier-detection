import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, encoder_dim=(32, 64, 128, 256), upscale=4, num_classes=1):

        super(Decoder, self).__init__()

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dim[i] + encoder_dim[i - 1], encoder_dim[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dim[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dim))
        ])

        self.logit = nn.Conv2d(encoder_dim[0], num_classes, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode='bilinear')

    def forward(self, feature):

        for i in range(len(feature) - 1, 0, -1):
            f_up = F.interpolate(feature[i], scale_factor=2, mode='bilinear')
            f = torch.cat([feature[i - 1], f_up], dim=1)
            f_down = self.conv[i - 1](f)
            feature[i - 1] = f_down

        x = self.logit(feature[0])
        out = self.up(x)

        return out


class SegModel(nn.Module):

    def __init__(self, num_classes):

        super(SegModel, self).__init__()

        self.encoder = timm.create_model(
            'maxvit_tiny_tf_512.in1k',
            features_only=True,
            pretrained=False,
            out_indices=[1, 2, 3, 4]
        )

        self.decoder = Decoder(
            self.encoder.feature_info.channels(),
            upscale=self.encoder.feature_info.reduction()[0],
            num_classes=num_classes
        )

    def forward(self, x):

        feat_maps = self.encoder(x)
        logits = self.decoder(feat_maps)
        masks = torch.sigmoid(logits)

        return masks
