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
        # 역전파를 위해 각 중간 단계를 보존하여 그래디언트 계산이 가능하도록 함
        x = x.clone().detach().requires_grad_(True)
        intermediates = {'input': x}
        
        # 합성곱
        conv_out = self.conv1(x)
        conv_out.retain_grad()  # 그래디언트 보존 명시적 설정
        intermediates['conv_out'] = conv_out
        
        # ReLU
        relu_out = F.relu(conv_out)
        relu_out.retain_grad()  # 그래디언트 보존 명시적 설정
        intermediates['relu_out'] = relu_out
        
        # 풀링
        pool_out = self.pool1(relu_out)
        pool_out.retain_grad()  # 그래디언트 보존 명시적 설정
        intermediates['pool_out'] = pool_out
        
        # 평탄화
        flatten = pool_out.view(pool_out.size(0), -1)
        flatten.retain_grad()  # 그래디언트 보존 명시적 설정
        intermediates['flatten'] = flatten
        
        # 완전 연결 계층
        fc_out = self.fc(flatten)
        fc_out.retain_grad()  # 그래디언트 보존 명시적 설정
        intermediates['fc_out'] = fc_out
        
        return fc_out, intermediates
