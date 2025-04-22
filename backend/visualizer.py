import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import SimpleCNN
import copy
from collections import defaultdict

class ModelVisualizer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.iterations = []
        
    def _compute_conv2d_matrix_form(self, input_tensor, layer):
        """Conv2d 연산을 행렬로 표현하는 함수"""
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_height, kernel_width = layer.weight.shape
        
        # 출력 크기 계산
        out_height = in_height - kernel_height + 1
        out_width = in_width - kernel_width + 1
        
        # 입력 데이터를 unfolding하여 행렬로 변환
        # 실제 구현에서는 im2col 방식을 사용할 수 있습니다
        unfolded = F.unfold(input_tensor, 
                          kernel_size=(kernel_height, kernel_width),
                          padding=0)
        
        # 가중치 행렬 재구성
        weight_matrix = layer.weight.view(out_channels, -1)
        
        # 행렬 곱 연산
        out_unfolded = weight_matrix @ unfolded
        
        # 결과를 원래 형태로 되돌림
        conv_out = out_unfolded.view(batch_size, out_channels, out_height, out_width)
        
        return {
            'input_tensor': input_tensor.detach().numpy(),
            'weight_tensor': layer.weight.detach().numpy(),
            'unfolded_input': unfolded.detach().numpy(),
            'weight_matrix': weight_matrix.detach().numpy(),
            'output_tensor': conv_out.detach().numpy()
        }
    
    def _compute_fc_matrix_form(self, input_tensor, layer):
        """Linear 연산을 행렬로 표현하는 함수"""
        weight = layer.weight
        bias = layer.bias
        
        # 행렬 곱 연산
        output = input_tensor @ weight.t() + bias
        
        return {
            'input_tensor': input_tensor.detach().numpy(),
            'weight': weight.detach().numpy(),
            'bias': bias.detach().numpy(),
            'output': output.detach().numpy()
        }
    
    def _compute_relu_details(self, input_tensor):
        """ReLU 연산 상세 정보 계산"""
        output = F.relu(input_tensor)
        mask = (input_tensor > 0).float()
        
        return {
            'input_tensor': input_tensor.detach().numpy(),
            'output_tensor': output.detach().numpy(),
            'mask': mask.detach().numpy()
        }
    
    def _compute_maxpool_details(self, input_tensor, layer):
        """MaxPool 연산 상세 정보 계산"""
        batch_size, channels, height, width = input_tensor.shape
        kernel_size = layer.kernel_size
        stride = layer.stride
        
        # 출력 크기 계산
        out_height = (height - kernel_size) // stride + 1
        out_width = (width - kernel_size) // stride + 1
        
        # MaxPool 연산 수행
        output, indices = F.max_pool2d_with_indices(
            input_tensor, kernel_size=kernel_size, stride=stride, return_indices=True
        )
        
        return {
            'input_tensor': input_tensor.detach().numpy(),
            'output_tensor': output.detach().numpy(),
            'indices': indices.detach().numpy(),
            'kernel_size': kernel_size,
            'stride': stride
        }
    
    def run_iteration(self, input_data, target):
        """한 번의 반복(iteration)을 실행하고 모든 계산 과정 추적"""
        iteration_data = {
            'input_data': input_data.detach().numpy(),
            'target': target.detach().numpy(),
            'learning_rate': self.learning_rate,
            'forward': {},
            'backward': {}
        }
        
        # 모델 가중치 상태 복사
        iteration_data['initial_weights'] = {
            'conv1_weight': self.model.conv1.weight.detach().clone().numpy(),
            'fc_weight': self.model.fc.weight.detach().clone().numpy(),
            'fc_bias': self.model.fc.bias.detach().clone().numpy()
        }
        
        # 모델 초기화 및 중간 결과 저장
        self.model.zero_grad()
        
        # 모든 중간 결과를 명시적으로 추적하여 역전파를 위한 계산 그래프 유지
        x = input_data.clone()
        x.requires_grad_(True)
        
        # 합성곱
        conv_out = self.model.conv1(x)
        conv_out.retain_grad()  # 그래디언트 보존 명시적 설정
        
        # ReLU
        relu_out = F.relu(conv_out.clone())  # 클론으로 복사하여 인플레이스 문제 방지
        relu_out.retain_grad()  # 그래디언트 보존 명시적 설정
        
        # 풀링
        pool_out = self.model.pool1(relu_out)
        pool_out.retain_grad()  # 그래디언트 보존 명시적 설정
        
        # 평탄화
        flatten = pool_out.reshape(pool_out.size(0), -1)  # view 대신 reshape 사용
        flatten.retain_grad()  # 그래디언트 보존 명시적 설정
        
        # 완전 연결 계층
        fc_out = self.model.fc(flatten)
        fc_out.retain_grad()  # 그래디언트 보존 명시적 설정
        
        # 중간 텐서들을 딕셔너리에 저장
        intermediates = {
            'input': x,
            'conv_out': conv_out,
            'relu_out': relu_out,
            'pool_out': pool_out,
            'flatten': flatten,
            'fc_out': fc_out
        }
        
        # 각 레이어별 상세 계산 과정 추적
        iteration_data['forward']['conv'] = self._compute_conv2d_matrix_form(
            intermediates['input'], self.model.conv1
        )
        
        iteration_data['forward']['relu'] = self._compute_relu_details(
            intermediates['conv_out']
        )
        
        iteration_data['forward']['pool'] = self._compute_maxpool_details(
            intermediates['relu_out'], self.model.pool1
        )
        
        iteration_data['forward']['fc'] = self._compute_fc_matrix_form(
            intermediates['flatten'], self.model.fc
        )
        
        # Loss 계산
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(fc_out, target)
        iteration_data['loss'] = loss.item()
        
        # Backward pass - 모든 그래디언트 계산
        loss.backward()
        
        # FC 출력 그래디언트 계산 (소프트맥스-크로스엔트로피의 그래디언트)
        softmax_output = F.softmax(fc_out, dim=1)
        one_hot = torch.zeros_like(softmax_output)
        one_hot[0, target[0]] = 1
        fc_output_grad = (softmax_output - one_hot)
        
        # 그래디언트가 없거나 0인 경우를 위한 안전한 그래디언트 추출 함수
        def safe_grad(tensor):
            if tensor.grad is None:
                return np.zeros_like(tensor.detach().numpy())
            return tensor.grad.detach().numpy()
        
        # Gradients 저장
        iteration_data['gradients'] = {
            'conv1_weight_grad': self.model.conv1.weight.grad.detach().clone().numpy(),
            'fc_weight_grad': self.model.fc.weight.grad.detach().clone().numpy(),
            'fc_bias_grad': self.model.fc.bias.grad.detach().clone().numpy()
        }
        
        # 역전파 순서대로 데이터 저장 (FC -> Pool -> ReLU -> Conv)
        # FC 역전파
        iteration_data['backward']['fc'] = {
            'output_grad': fc_output_grad.detach().numpy(),
            'input_grad': flatten.grad.detach().numpy() if flatten.grad is not None else np.zeros_like(flatten.detach().numpy()),
            'weight_grad': self.model.fc.weight.grad.detach().numpy(),
            'bias_grad': self.model.fc.bias.grad.detach().numpy()
        }
        
        # 수동으로 예상되는 FC 입력 그래디언트 계산 (시각화 확인용)
        expected_flatten_grad = torch.matmul(fc_output_grad, self.model.fc.weight).detach().numpy()
        iteration_data['backward']['fc']['expected_input_grad'] = expected_flatten_grad
        
        # Pool 역전파 - 형태 변환으로 인한 그래디언트 흐름 확인
        if flatten.grad is not None:
            # reshape flatten 그래디언트를 pool 출력 형태로
            reshaped_flatten_grad = flatten.grad.view_as(pool_out).detach()
            iteration_data['backward']['pool'] = {
                'output_grad': reshaped_flatten_grad.numpy(),
                'input_grad': relu_out.grad.detach().numpy() if relu_out.grad is not None else np.zeros_like(relu_out.detach().numpy())
            }
        else:
            # 그래디언트가 없는 경우 대비
            iteration_data['backward']['pool'] = {
                'output_grad': np.zeros_like(pool_out.detach().numpy()),
                'input_grad': np.zeros_like(relu_out.detach().numpy())
            }
        
        # ReLU 역전파
        relu_mask = (conv_out > 0).float().detach()
        if relu_out.grad is not None:
            iteration_data['backward']['relu'] = {
                'output_grad': relu_out.grad.detach().numpy(),
                'input_grad': conv_out.grad.detach().numpy() if conv_out.grad is not None else np.zeros_like(conv_out.detach().numpy()),
                'mask': relu_mask.numpy()
            }
            # 예상되는 ReLU 입력 그래디언트 계산 (마스크 적용)
            expected_conv_out_grad = (relu_out.grad * relu_mask).detach().numpy()
            iteration_data['backward']['relu']['expected_input_grad'] = expected_conv_out_grad
        else:
            iteration_data['backward']['relu'] = {
                'output_grad': np.zeros_like(relu_out.detach().numpy()),
                'input_grad': np.zeros_like(conv_out.detach().numpy()),
                'mask': relu_mask.numpy()
            }
        
        # Conv 역전파
        iteration_data['backward']['conv'] = {
            'output_grad': safe_grad(conv_out),
            'weight_grad': self.model.conv1.weight.grad.detach().numpy()
        }
        
        # 가중치 업데이트 전 복사
        old_conv_w = self.model.conv1.weight.detach().clone()
        old_fc_w = self.model.fc.weight.detach().clone()
        old_fc_b = self.model.fc.bias.detach().clone()
        
        # 가중치 업데이트
        with torch.no_grad():
            self.model.conv1.weight -= self.learning_rate * self.model.conv1.weight.grad
            self.model.fc.weight -= self.learning_rate * self.model.fc.weight.grad
            self.model.fc.bias -= self.learning_rate * self.model.fc.bias.grad
        
        # 업데이트된 가중치 저장
        iteration_data['updated_weights'] = {
            'conv1_weight': self.model.conv1.weight.detach().clone().numpy(),
            'fc_weight': self.model.fc.weight.detach().clone().numpy(),
            'fc_bias': self.model.fc.bias.detach().clone().numpy()
        }
        
        # 가중치 변화량 저장 (검증용)
        iteration_data['weight_delta'] = {
            'conv1_weight': (self.model.conv1.weight.detach() - old_conv_w).numpy(),
            'fc_weight': (self.model.fc.weight.detach() - old_fc_w).numpy(),
            'fc_bias': (self.model.fc.bias.detach() - old_fc_b).numpy(),
            'expected_conv1_weight': (-self.learning_rate * self.model.conv1.weight.grad.detach()).numpy(),
            'expected_fc_weight': (-self.learning_rate * self.model.fc.weight.grad.detach()).numpy(),
            'expected_fc_bias': (-self.learning_rate * self.model.fc.bias.grad.detach()).numpy()
        }
        
        self.iterations.append(iteration_data)
        return iteration_data
    
    def run_epochs(self, input_data, target, num_epochs=3):
        """지정된 에포크 수만큼 학습 반복 실행"""
        for epoch in range(num_epochs):
            print(f"Running epoch {epoch+1}/{num_epochs}")
            self.run_iteration(input_data, target)
        
        return self.iterations
