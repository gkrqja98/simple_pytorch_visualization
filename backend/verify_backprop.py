import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
from model import SimpleCNN
from visualizer import ModelVisualizer
import matplotlib.pyplot as plt
import math

def verify_gradients():
    """
    역전파에서 시각화할 때 필요한 그래디언트 값들이 올바르게 계산되는지 검증하는 함수
    각 레이어별 그래디언트 계산을 수동으로 검증
    """
    print("=== 백엔드 역전파 그래디언트 검증 ===")
    
    # 모델과 시각화 도구 초기화
    model = SimpleCNN()
    lr = 0.01  # backend/visualizer.py에서 사용하는 값과 동일하게 설정
    visualizer = ModelVisualizer(model, learning_rate=lr)
    
    # 고정된 입력과 타겟 (visualizer에서 사용하는 것과 동일)
    input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                              [5.0, 6.0, 7.0, 8.0],
                              [9.0, 10.0, 11.0, 12.0],
                              [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)
    
    # 시각화 도구를 사용하여 첫 번째 반복 실행
    iteration_data = visualizer.run_iteration(input_data, target)
    
    print(f"Learning rate: {lr}")
    print(f"Loss: {iteration_data['loss']:.6f}")
    
    # 수동 계산을 위한 모델 초기화 (visualizer에서 사용한 모델과 동일한 초기 가중치)
    manual_model = SimpleCNN()
    
    # 가중치가 시각화 도구와 동일한지 확인 (필요한 경우)
    print("\n=== 초기 가중치 검증 ===")
    visualizer_conv_w = iteration_data['initial_weights']['conv1_weight']
    manual_conv_w = manual_model.conv1.weight.data.numpy()
    print(f"Convolution weights match: {np.allclose(visualizer_conv_w, manual_conv_w)}")
    
    visualizer_fc_w = iteration_data['initial_weights']['fc_weight']
    manual_fc_w = manual_model.fc.weight.data.numpy()
    print(f"FC weights match: {np.allclose(visualizer_fc_w, manual_fc_w)}")
    
    visualizer_fc_b = iteration_data['initial_weights']['fc_bias']
    manual_fc_b = manual_model.fc.bias.data.numpy()
    print(f"FC bias match: {np.allclose(visualizer_fc_b, manual_fc_b)}")
    
    # 수동 순전파 및 그래디언트 계산
    manual_model.zero_grad()
    
    # 전체 네트워크를 통과하면서 중간 결과 저장
    conv_out = manual_model.conv1(input_data)
    conv_out.retain_grad()
    
    relu_out = F.relu(conv_out)
    relu_out.retain_grad()
    
    pool_out = manual_model.pool1(relu_out)
    pool_out.retain_grad()
    
    flatten = pool_out.view(input_data.size(0), -1)
    flatten.retain_grad()
    
    fc_out = manual_model.fc(flatten)
    fc_out.retain_grad()
    
    # 손실 계산 및 역전파
    criterion = nn.CrossEntropyLoss()
    loss = criterion(fc_out, target)
    loss.backward()
    
    # 1. FC 레이어 그래디언트 검증
    print("\n=== 1. FC 레이어 그래디언트 검증 ===")
    
    # FC 출력에 대한 그래디언트
    fc_out_grad = fc_out.grad.detach().numpy()
    vis_fc_out_grad = np.array(iteration_data['backward']['fc'].get('output_grad', np.zeros_like(fc_out_grad)))
    print(f"FC 출력 그래디언트 확인: {(vis_fc_out_grad is not None)}")
    
    # FC weight 그래디언트
    fc_weight_grad = manual_model.fc.weight.grad.detach().numpy()
    vis_fc_weight_grad = np.array(iteration_data['gradients']['fc_weight_grad'])
    print(f"FC weight 그래디언트 일치: {np.allclose(fc_weight_grad, vis_fc_weight_grad)}")
    print(f"수동 계산: \n{fc_weight_grad}")
    print(f"시각화 도구: \n{vis_fc_weight_grad}")
    
    # FC bias 그래디언트
    fc_bias_grad = manual_model.fc.bias.grad.detach().numpy()
    vis_fc_bias_grad = np.array(iteration_data['gradients']['fc_bias_grad'])
    print(f"FC bias 그래디언트 일치: {np.allclose(fc_bias_grad, vis_fc_bias_grad)}")
    print(f"수동 계산: {fc_bias_grad}")
    print(f"시각화 도구: {vis_fc_bias_grad}")
    
    # flatten에 대한 그래디언트 (FC → Flatten)
    flatten_grad = flatten.grad.detach().numpy()
    vis_flatten_grad = np.array(iteration_data['backward']['fc']['input_grad'])
    print(f"Flatten 그래디언트 일치: {np.allclose(flatten_grad, vis_flatten_grad)}")
    print(f"수동 계산: {flatten_grad}")
    print(f"시각화 도구: {vis_flatten_grad}")
    
    # 수동으로 flatten_grad 계산해보기
    expected_flatten_grad = np.matmul(fc_out_grad, manual_model.fc.weight.detach().numpy())
    print(f"계산된 Flatten 그래디언트 일치: {np.allclose(flatten_grad, expected_flatten_grad)}")
    print(f"직접 계산: {expected_flatten_grad}")
    
    # 2. MaxPool 레이어 그래디언트 검증 (Flatten → MaxPool)
    print("\n=== 2. MaxPool 레이어 그래디언트 검증 ===")
    
    # pool_out에 대한 그래디언트는 flatten에 대한 그래디언트를 reshape한 것과 같아야 함
    pool_out_grad = pool_out.grad.detach().numpy()
    vis_pool_out_grad = np.array(iteration_data['backward']['pool']['output_grad'])
    
    # reshape된 flatten 그래디언트와 pool_out 그래디언트 비교
    reshaped_flatten_grad = flatten_grad.reshape(pool_out.shape)
    print(f"Pool 출력 그래디언트 일치: {np.allclose(pool_out_grad, vis_pool_out_grad)}")
    print(f"reshape된 Flatten 그래디언트와 일치: {np.allclose(pool_out_grad, reshaped_flatten_grad)}")
    print(f"수동 계산: \n{pool_out_grad}")
    print(f"시각화 도구: \n{vis_pool_out_grad}")
    
    # 3. ReLU 레이어 그래디언트 검증 (MaxPool → ReLU)
    print("\n=== 3. ReLU 레이어 그래디언트 검증 ===")
    
    # ReLU 출력에 대한 그래디언트 (MaxPool 입력 그래디언트)
    relu_out_grad = relu_out.grad.detach().numpy()
    vis_relu_out_grad = np.array(iteration_data['backward']['pool']['input_grad'])
    print(f"ReLU 출력 그래디언트 일치: {np.allclose(relu_out_grad, vis_relu_out_grad)}")
    print(f"수동 계산: \n{relu_out_grad.squeeze()}")
    print(f"시각화 도구: \n{vis_relu_out_grad.squeeze()}")
    
    # ReLU 입력에 대한 그래디언트
    conv_out_grad = conv_out.grad.detach().numpy()
    vis_conv_out_grad = np.array(iteration_data['backward']['relu']['input_grad'])
    print(f"ReLU 입력 그래디언트 일치: {np.allclose(conv_out_grad, vis_conv_out_grad)}")
    
    # ReLU 마스크 검증
    relu_mask = (conv_out.detach() > 0).float().numpy()
    vis_relu_mask = np.array(iteration_data['backward']['relu']['mask'])
    print(f"ReLU 마스크 일치: {np.allclose(relu_mask, vis_relu_mask)}")
    
    # 4. Convolution kernel 그래디언트 검증 (ReLU → Conv)
    print("\n=== 4. Convolution 커널 그래디언트 검증 ===")
    
    # Conv 가중치 그래디언트
    conv_weight_grad = manual_model.conv1.weight.grad.detach().numpy()
    vis_conv_weight_grad = np.array(iteration_data['gradients']['conv1_weight_grad'])
    print(f"Conv 가중치 그래디언트 일치: {np.allclose(conv_weight_grad, vis_conv_weight_grad)}")
    print(f"수동 계산: \n{conv_weight_grad.squeeze()}")
    print(f"시각화 도구: \n{vis_conv_weight_grad.squeeze()}")
    
    # 5. 가중치 업데이트 검증
    print("\n=== 5. 가중치 업데이트 검증 ===")
    
    # Conv 가중치 업데이트
    old_conv_w = torch.tensor(iteration_data['initial_weights']['conv1_weight'])
    new_conv_w = torch.tensor(iteration_data['updated_weights']['conv1_weight'])
    expected_delta = -lr * torch.tensor(vis_conv_weight_grad)
    actual_delta = new_conv_w - old_conv_w
    print(f"Conv 가중치 업데이트 일치: {torch.allclose(actual_delta, expected_delta)}")
    print(f"예상 변화량: \n{expected_delta.squeeze()}")
    print(f"실제 변화량: \n{actual_delta.squeeze()}")
    
    # FC 가중치 업데이트
    old_fc_w = torch.tensor(iteration_data['initial_weights']['fc_weight'])
    new_fc_w = torch.tensor(iteration_data['updated_weights']['fc_weight'])
    expected_delta = -lr * torch.tensor(vis_fc_weight_grad)
    actual_delta = new_fc_w - old_fc_w
    print(f"FC 가중치 업데이트 일치: {torch.allclose(actual_delta, expected_delta)}")
    
    # FC bias 업데이트
    old_fc_b = torch.tensor(iteration_data['initial_weights']['fc_bias'])
    new_fc_b = torch.tensor(iteration_data['updated_weights']['fc_bias'])
    expected_delta = -lr * torch.tensor(vis_fc_bias_grad)
    actual_delta = new_fc_b - old_fc_b
    print(f"FC bias 업데이트 일치: {torch.allclose(actual_delta, expected_delta)}")
    
    # 6. Visualizer가 시각화에 필요한 모든 데이터를 포함하는지 확인
    print("\n=== 6. 시각화 필요 데이터 검증 ===")
    
    # FC 레이어
    print("FC 레이어 역전파 데이터:")
    print(f"- FC 가중치 그래디언트: {'weight_grad' in iteration_data['backward']['fc']}")
    print(f"- FC bias 그래디언트: {'bias_grad' in iteration_data['backward']['fc']}")
    print(f"- FC 입력 그래디언트: {'input_grad' in iteration_data['backward']['fc']}")
    
    # MaxPool 레이어
    print("\nMaxPool 레이어 역전파 데이터:")
    print(f"- MaxPool 출력 그래디언트: {'output_grad' in iteration_data['backward']['pool']}")
    print(f"- MaxPool 입력 그래디언트: {'input_grad' in iteration_data['backward']['pool']}")
    
    # ReLU 레이어
    print("\nReLU 레이어 역전파 데이터:")
    print(f"- ReLU 출력 그래디언트: {'output_grad' in iteration_data['backward']['relu']}")
    print(f"- ReLU 입력 그래디언트: {'input_grad' in iteration_data['backward']['relu']}")
    print(f"- ReLU 마스크: {'mask' in iteration_data['backward']['relu']}")
    
    # Conv 레이어
    print("\nConv 레이어 역전파 데이터:")
    print(f"- Conv 출력 그래디언트: {'output_grad' in iteration_data['backward']['conv']}")
    print(f"- Conv 가중치 그래디언트: {'weight_grad' in iteration_data['backward']['conv']}")
    
    return True

def visualize_gradient_flow():
    """
    역전파 과정에서 그래디언트가 어떻게 흐르는지 시각화하는 함수
    그래디언트 흐름을 시각적으로 확인하여 역전파가 정상적으로 동작하는지 검증
    """
    print("\n=== 그래디언트 흐름 시각화 ===")
    
    # 모델 초기화
    model = SimpleCNN()
    
    # 입력과 타깃
    input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)
    
    # 3번의 에포크 동안 그래디언트 변화 추적
    epochs = 3
    layer_names = ['Conv2d', 'ReLU', 'MaxPool2d', 'Linear']
    gradients_history = {name: [] for name in layer_names}
    
    for epoch in range(epochs):
        # 순전파 및 역전파
        model.zero_grad()
        
        # 중간 텐서들 저장
        conv_out = model.conv1(input_data)
        conv_out.retain_grad()
        
        relu_out = F.relu(conv_out)
        relu_out.retain_grad()
        
        pool_out = model.pool1(relu_out)
        pool_out.retain_grad()
        
        flatten = pool_out.view(input_data.size(0), -1)
        flatten.retain_grad()
        
        fc_out = model.fc(flatten)
        fc_out.retain_grad()
        
        # 손실 계산 및 역전파
        criterion = nn.CrossEntropyLoss()
        loss = criterion(fc_out, target)
        loss.backward()
        
        # 각 레이어의 그래디언트 크기 저장
        gradients_history['Conv2d'].append(model.conv1.weight.grad.norm().item())
        gradients_history['ReLU'].append(conv_out.grad.norm().item())
        gradients_history['MaxPool2d'].append(relu_out.grad.norm().item())
        gradients_history['Linear'].append(model.fc.weight.grad.norm().item())
        
        # 가중치 업데이트 (SGD)
        with torch.no_grad():
            lr = 0.01
            model.conv1.weight -= lr * model.conv1.weight.grad
            model.fc.weight -= lr * model.fc.weight.grad
            model.fc.bias -= lr * model.fc.bias.grad
    
    # 그래디언트 흐름 시각화를 위한 결과 출력
    print("에포크별 그래디언트 크기 변화:")
    for name in layer_names:
        print(f"{name}: {[round(grad, 6) for grad in gradients_history[name]]}")
    
    # 여기서는 시각화만 확인하고 실제 플롯은 생성하지 않음
    return gradients_history

def calculate_expected_vs_actual_gradients():
    """
    수동으로 계산한 예상 그래디언트와 PyTorch autograd가 계산한 실제 그래디언트를 비교
    """
    print("\n=== 예상 그래디언트 vs 실제 그래디언트 ===")
    
    # 모델 초기화
    model = SimpleCNN()
    
    # 입력과 타깃
    input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                               [5.0, 6.0, 7.0, 8.0],
                               [9.0, 10.0, 11.0, 12.0],
                               [13.0, 14.0, 15.0, 16.0]]]], dtype=torch.float32)
    target = torch.tensor([0], dtype=torch.long)
    
    # 순전파 및 역전파
    model.zero_grad()
    
    # 중간 텐서들 저장
    conv_out = model.conv1(input_data)
    conv_out.retain_grad()
    
    relu_out = F.relu(conv_out)
    relu_out.retain_grad()
    
    pool_out = model.pool1(relu_out)
    pool_out.retain_grad()
    
    flatten = pool_out.view(input_data.size(0), -1)
    flatten.retain_grad()
    
    fc_out = model.fc(flatten)
    fc_out.retain_grad()
    
    # 손실 계산 및 역전파
    criterion = nn.CrossEntropyLoss()
    loss = criterion(fc_out, target)
    loss.backward()
    
    # 1. FC 레이어 그래디언트 수동 계산
    # 소프트맥스-크로스엔트로피 미분
    softmax = torch.nn.functional.softmax(fc_out, dim=1)
    onehot = torch.zeros_like(softmax)
    onehot[0, target] = 1
    delta_y = (softmax - onehot).detach()
    
    # FC 가중치 그래디언트 수동 계산
    expected_fc_weight_grad = torch.matmul(delta_y.t(), flatten).detach()
    actual_fc_weight_grad = model.fc.weight.grad.detach()
    
    print("FC 가중치 그래디언트:")
    print(f"예상: \n{expected_fc_weight_grad}")
    print(f"실제: \n{actual_fc_weight_grad}")
    print(f"일치: {torch.allclose(expected_fc_weight_grad, actual_fc_weight_grad, atol=1e-5)}")
    
    # FC bias 그래디언트 수동 계산
    expected_fc_bias_grad = delta_y.sum(dim=0).detach()
    actual_fc_bias_grad = model.fc.bias.grad.detach()
    
    print("\nFC bias 그래디언트:")
    print(f"예상: {expected_fc_bias_grad}")
    print(f"실제: {actual_fc_bias_grad}")
    print(f"일치: {torch.allclose(expected_fc_bias_grad, actual_fc_bias_grad, atol=1e-5)}")
    
    # 2. FC→Flatten 그래디언트 수동 계산
    expected_flatten_grad = torch.matmul(delta_y, model.fc.weight).detach()
    actual_flatten_grad = flatten.grad.detach()
    
    print("\nFlatten 그래디언트:")
    print(f"예상: {expected_flatten_grad}")
    print(f"실제: {actual_flatten_grad}")
    print(f"일치: {torch.allclose(expected_flatten_grad, actual_flatten_grad, atol=1e-5)}")
    
    # 3. 풀링 레이어 역전파 (Flatten→MaxPool)
    expected_pool_out_grad = flatten.grad.view_as(pool_out).detach()
    actual_pool_out_grad = pool_out.grad.detach()
    
    print("\nMaxPool 출력 그래디언트:")
    print(f"예상: \n{expected_pool_out_grad.squeeze()}")
    print(f"실제: \n{actual_pool_out_grad.squeeze()}")
    print(f"일치: {torch.allclose(expected_pool_out_grad, actual_pool_out_grad, atol=1e-5)}")
    
    # 4. ReLU 레이어 역전파
    relu_mask = (conv_out > 0).float().detach()
    expected_relu_input_grad = (pool_out.grad * relu_mask).detach()
    actual_relu_input_grad = conv_out.grad.detach()
    
    print("\nReLU 입력 그래디언트:")
    print(f"일치: {torch.allclose(expected_relu_input_grad, actual_relu_input_grad, atol=1e-5)}")
    
    # 체계적으로 누락된 그래디언트가 있는지 확인
    print("\n미확인 그래디언트:")
    print(f"Conv1 입력 그래디언트: {input_data.grad is not None}")
    
    # 각 레이어의 역전파가 제대로 연결되어 있는지 확인
    print("\n=== 역전파 연결 검증 ===")
    layers = ['FC 출력', 'Flatten', 'MaxPool 출력', 'ReLU 출력', 'Conv 출력']
    gradients = [
        fc_out.grad is not None,
        flatten.grad is not None,
        pool_out.grad is not None, 
        relu_out.grad is not None,
        conv_out.grad is not None
    ]
    
    for layer, has_grad in zip(layers, gradients):
        print(f"{layer} 그래디언트: {'있음' if has_grad else '없음'}")
    
    return True

if __name__ == "__main__":
    # 역전파 그래디언트 검증
    verify_gradients()
    
    # 그래디언트 흐름 시각화
    visualize_gradient_flow()
    
    # 예상 vs 실제 그래디언트 비교
    calculate_expected_vs_actual_gradients()
