import torch.nn as nn


class ClassificationHead(nn.Module):

    def __init__(self, input_dimensions):

        super(ClassificationHead, self).__init__()

        self.cancer_head = nn.Linear(input_dimensions, 5, bias=True)

    def forward(self, x):

        cancer_output = self.cancer_head(x)

        return cancer_output
