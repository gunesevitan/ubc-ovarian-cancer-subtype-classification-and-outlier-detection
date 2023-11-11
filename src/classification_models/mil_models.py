import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from heads import ClassificationHead


class GeM(nn.Module):

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
            ', ' + 'eps=' + str(self.eps) + ')'


class Attention(nn.Module):

    def __init__(self, sequence_length, dimensions, bias=True):

        super(Attention, self).__init__()

        weight = torch.zeros(dimensions, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.zeros(sequence_length))

    def forward(self, x):

        input_batch_size, input_sequence_length, input_dimensions = x.shape
        eij = torch.mm(
            x.contiguous().view(-1, input_dimensions),
            self.weight
        ).view(-1, input_sequence_length)

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        output = torch.sum(weighted_input, 1)

        return output


class TimmMILClassificationModel(nn.Module):

    def __init__(self, model_name, pretrained, backbone_args, mil_pooling_type, feature_pooling_type, dropout_rate, freeze_parameters, head_args):

        super(TimmMILClassificationModel, self).__init__()

        self.backbone = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            **backbone_args
        )

        if freeze_parameters:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        self.mil_pooling_type = mil_pooling_type
        self.feature_pooling_type = feature_pooling_type
        input_features = self.backbone.get_classifier().in_features
        self.backbone.classifier = nn.Identity()

        if self.feature_pooling_type == 'gem':
            self.pooling = GeM()
        elif self.feature_pooling_type == 'attention':
            self.pooling = nn.Sequential(
                nn.LayerNorm(normalized_shape=input_features),
                Attention(sequence_length=1024, dimensions=input_features)
            )
        else:
            self.pooling = nn.Identity()

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = ClassificationHead(input_dimensions=input_features, **head_args)

    def forward(self, x):

        input_batch_size, input_channel, input_depth, input_height, input_width = x.shape
        x = x.view(input_batch_size * input_depth, input_channel, input_height, input_width)
        x = self.backbone.forward_features(x)

        feature_batch_size, feature_channel, feature_height, feature_width = x.shape

        if self.mil_pooling_type == 'avg':
            x = x.contiguous().view(input_batch_size, input_depth, feature_channel, feature_height, feature_width)
            x = torch.mean(x, dim=1)
        elif self.mil_pooling_type == 'max':
            x = x.contiguous().view(input_batch_size, input_depth, feature_channel, feature_height, feature_width)
            x = torch.max(x, dim=1)[0]
        elif self.mil_pooling_type == 'concat':
            x = x.contiguous().view(input_batch_size, input_depth * feature_channel, feature_height, feature_width)
        else:
            raise ValueError(f'Invalid MIL pooling type {self.mil_pooling_type}')

        if self.feature_pooling_type == 'avg':
            x = F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'max':
            x = F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
        elif self.feature_pooling_type == 'concat':
            x = torch.cat([
                F.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.size(0), -1),
                F.adaptive_max_pool2d(x, output_size=(1, 1)).view(x.size(0), -1)
            ], dim=-1)
        elif self.feature_pooling_type == 'gem':
            x = self.pooling(x).view(x.size(0), -1)
        elif self.feature_pooling_type == 'attention':
            input_batch_size, feature_channel = x.shape[:2]
            x = x.contiguous().view(input_batch_size, feature_channel, -1).permute(0, 2, 1)
            x = self.pooling(x)
        else:
            raise ValueError(f'Invalid feature pooling type {self.feature_pooling_type}')

        x = self.dropout(x)
        cancer_output = self.head(x)

        return cancer_output
