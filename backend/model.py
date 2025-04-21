import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.fc = nn.Linear(4, 2)
        
        # 가중치 초기화 (시각화를 위해 특정 값으로 초기화)
        self.conv1.weight.data = torch.tensor([[[[1.0, 0.5], [0.5, 1.0]]]], requires_grad=True)
        self.fc.weight.data = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], requires_grad=True)
        self.fc.bias.data = torch.tensor([0.1, -0.1], requires_grad=True)

    def forward(self, x):
        conv_out = self.conv1(x)
        relu_out = F.relu(conv_out)
        pool_out = self.pool1(relu_out)
        flatten = pool_out.view(pool_out.size(0), -1)
        fc_out = self.fc(flatten)
        return fc_out

    def forward_with_intermediates(self, x):
        """중간 결과를 저장하면서 순전파 수행"""
        intermediates = {'input': x.clone()}
        
        # 합성곱
        conv_out = self.conv1(x)
        intermediates['conv_out'] = conv_out.clone()
        
        # ReLU
        relu_out = F.relu(conv_out)
        intermediates['relu_out'] = relu_out.clone()
        
        # 풀링
        pool_out = self.pool1(relu_out)
        intermediates['pool_out'] = pool_out.clone()
        
        # 평탄화
        flatten = pool_out.view(pool_out.size(0), -1)
        intermediates['flatten'] = flatten.clone()
        
        # 완전 연결 계층
        fc_out = self.fc(flatten)
        intermediates['fc_out'] = fc_out.clone()
        
        # 중간 텐서들에 requires_grad=True 설정하여 그래디언트 계산 보장
        for key, tensor in intermediates.items():
            if tensor.requires_grad == False and tensor.dtype.is_floating_point:
                tensor.requires_grad_(True)
        
        return fc_out, intermediates
