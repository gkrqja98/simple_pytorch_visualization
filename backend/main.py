import torch
import numpy as np
import json
from flask import Flask, jsonify, request
from flask_cors import CORS
from model import SimpleCNN
from visualizer import ModelVisualizer

app = Flask(__name__)
CORS(app)  # 크로스 오리진 요청 허용

# CNN 모델 및 시각화 도구 초기화
model = SimpleCNN()
visualizer = ModelVisualizer(model, learning_rate=0.01)

# 샘플 데이터 생성
def create_sample_data():
    # 4x4 입력 이미지 (배치 크기 1, 채널 1)
    input_data = torch.tensor([[[[1.0, 2.0, 1.0, 0.0],
                               [0.0, -1.0, 0.0, 1.0],
                               [-2.0, 1.0, -2.0, -2.0],
                               [0.0, 1.0, 0.0, 1.0]]]], dtype=torch.float32)
    
    # 타겟 라벨 (클래스 0)
    target = torch.tensor([0], dtype=torch.long)
    
    return input_data, target

@app.route('/api/run_visualization', methods=['POST'])
def run_visualization():
    # 요청에서 에포크 수 가져오기 (기본값 3)
    data = request.json
    num_epochs = data.get('epochs', 3)
    
    # 샘플 데이터 생성
    input_data, target = create_sample_data()
    
    # 시각화 실행
    iterations = visualizer.run_epochs(input_data, target, num_epochs)
    
    # NumPy 배열을 리스트로 변환하여 JSON으로 직렬화 가능하게 변환
    serializable_iterations = []
    
    for iteration in iterations:
        serializable_iteration = {}
        
        # 기본 정보
        serializable_iteration['learning_rate'] = iteration['learning_rate']
        serializable_iteration['loss'] = iteration['loss']
        
        # 입력 데이터 및 타겟
        serializable_iteration['input_data'] = iteration['input_data'].tolist()
        serializable_iteration['target'] = iteration['target'].tolist()
        
        # 초기 가중치
        serializable_iteration['initial_weights'] = {
            k: v.tolist() for k, v in iteration['initial_weights'].items()
        }
        
        # 업데이트된 가중치
        serializable_iteration['updated_weights'] = {
            k: v.tolist() for k, v in iteration['updated_weights'].items()
        }
        
        # 그래디언트
        serializable_iteration['gradients'] = {
            k: v.tolist() for k, v in iteration['gradients'].items()
        }
        
        # 순전파 계산
        serializable_iteration['forward'] = {}
        for layer_name, layer_data in iteration['forward'].items():
            serializable_iteration['forward'][layer_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in layer_data.items()
            }
        
        # 역전파 계산
        serializable_iteration['backward'] = {}
        for layer_name, layer_data in iteration['backward'].items():
            serializable_iteration['backward'][layer_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in layer_data.items() if v is not None
            }
        
        serializable_iterations.append(serializable_iteration)
    
    # 반환 데이터
    response_data = {
        'iterations': serializable_iterations,
        'model_config': {
            'conv1': {
                'in_channels': 1,
                'out_channels': 1,
                'kernel_size': 2,
                'padding': 0
            },
            'pool1': {
                'kernel_size': 2,
                'stride': 1
            },
            'fc': {
                'in_features': 4,
                'out_features': 2
            }
        }
    }
    
    return jsonify(response_data)

@app.route('/api/model_info', methods=['GET'])
def get_model_info():
    return jsonify({
        'name': 'SimpleCNN',
        'layers': [
            {'name': 'Conv2d', 'params': {'in_channels': 1, 'out_channels': 1, 'kernel_size': 2, 'padding': 0}},
            {'name': 'ReLU', 'params': {}},
            {'name': 'MaxPool2d', 'params': {'kernel_size': 2, 'stride': 1}},
            {'name': 'Linear', 'params': {'in_features': 4, 'out_features': 2}}
        ],
        'total_params': sum(p.numel() for p in model.parameters())
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
