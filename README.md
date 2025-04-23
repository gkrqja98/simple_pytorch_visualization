# PyTorch CNN Visualization Tool

이 프로젝트는 PyTorch로 구현된 간단한 CNN 모델의 순전파와 역전파 과정을 시각화합니다. 모든 텐서 연산을 행렬 형태로 분해하여 수학적 공식과 함께 대화형 HTML 인터페이스에서 표현합니다.

## 주요 기능

### 현재 구현된 기능
- PyTorch를 사용한 단순 CNN 모델 구현 (Conv2d, ReLU, MaxPool2d, Linear 레이어 포함)
- 모델의 순전파와 역전파 과정을 층별로 상세히 추적하는 도구
- 모든 텐서 연산을 행렬 형태로 분해하여 시각화
- 모든 계산 과정을 수학적 공식과 함께 HTML로 표현
- 3회의 학습 반복(epoch) 동안 가중치와 그래디언트 변화 시각화
- 수식(LaTeX) 지원으로 수학적 원리 명확히 표현

### 아직 구현되지 않은 기능 (개발 예정)
- 다양한 PyTorch 모델에 적용 가능한 모듈화 확장
- 웹 브라우저에서 볼 수 있는 독자적인 HTML 파일로 내보내기 기능
- 사용자 정의 데이터셋 업로드 및 처리
- 실시간 모델 수정 및 결과 시각화
- 다양한 최적화 알고리즘 비교 도구

## 프로젝트 구조

```
deepl/
  ├── backend/               # 파이썬 백엔드
  │   ├── main.py            # API 서버
  │   ├── model.py           # CNN 모델 정의
  │   ├── visualizer.py      # 모델 계산 추적
  │   ├── verify_backprop.py # 역전파 검증 도구
  │   ├── templates/         # HTML 템플릿
  │   ├── output/            # 출력 파일
  │   ├── requirements.txt   # 필요 패키지
  │   └── data/              # 샘플 데이터
  │
  └── frontend/              # React 프론트엔드
      ├── public/            # 정적 파일
      ├── src/               # 소스 코드
      │   ├── components/    # React 컴포넌트
      │   │   ├── AnimatedCalculation.js  # 계산 애니메이션
      │   │   ├── TensorVisualizer.js     # 텐서 시각화
      │   │   ├── ConvolutionVisualizer.js # 합성곱 시각화
      │   │   ├── ReluVisualizer.js       # ReLU 시각화
      │   │   ├── MaxPoolVisualizer.js    # MaxPool 시각화
      │   │   ├── FCLayerVisualizer.js    # 완전연결층 시각화
      │   │   ├── backprop/              # 역전파 컴포넌트
      │   │   │   ├── FCLayerBackprop.js  # FC 역전파
      │   │   │   ├── MaxPoolBackprop.js  # MaxPool 역전파
      │   │   │   ├── ReluBackprop.js     # ReLU 역전파
      │   │   │   └── ConvBackprop.js     # Conv 역전파
      │   │   └── ...
      │   ├── pages/         # 페이지 컴포넌트
      │   ├── utils/         # 유틸리티 함수
      │   ├── App.js         # 메인 앱
      │   └── index.js       # 진입점
      ├── package.json       # 의존성
      └── .env               # 환경 변수
```

## 설치 및 실행

### React 18 관련 중요 사항
**주의**: 이 프로젝트는 React 18을 사용해야 합니다. React 18은 Concurrent Mode와 같은 새로운 기능을 제공하며, 이 프로젝트의 많은 시각화 컴포넌트들이 React 18의 기능에 의존합니다. 이전 버전의 React를 사용할 경우 렌더링 문제가 발생할 수 있습니다.

React 버전을 확인하려면:
```bash
cd frontend
npm list react
```

React 18을 설치하려면:
```bash
cd frontend
npm install react@18 react-dom@18 --save
```

### 백엔드 설정

```bash
cd backend
pip install -r requirements.txt
```

### 백엔드 실행

```bash
cd backend
python main.py
```

또는 제공된 스크립트 사용:

```bash
# Windows
run_backend.bat

# Linux/Mac
./run_backend.sh
```

### 프론트엔드 설정

```bash
cd frontend
npm install
```

### 프론트엔드 실행

```bash
cd frontend
npm start
```

또는 제공된 스크립트 사용:

```bash
# Windows
run_frontend.bat

# Linux/Mac
./run_frontend.sh
```

## 웹 인터페이스 구조

1. **모델 아키텍처**: 모델 구조 및 레이어 설명
   - 각 레이어 유형에 대한 자세한 수학적 설명
   - 레이어 간 연결 표시
   - 수학적 공식은 LaTeX로 렌더링

2. **반복 1, 2, 3**: 각 학습 반복에 대한 상세 정보
   - 입력 데이터, 가중치, 학습률 등 표시
   - **순전파**: 순전파의 모든 계산 단계 시각화
     - Conv2d 레이어: 합성곱 연산의 행렬 분해
     - ReLU 레이어: 활성화 함수 적용 과정
     - MaxPool2d 레이어: 풀링 연산 과정
     - Linear 레이어: 행렬 곱셈 연산
   - **역전파**: 역전파의 모든 계산 단계 시각화
     - 그래디언트 흐름
     - 가중치 업데이트 과정
     - 역전파 수식 설명

3. **이터레이션 결과**: 각 이터레이션 후 손실 및 가중치 변화 요약

## 기술 스택

- **백엔드**: Python, PyTorch, Flask
- **프론트엔드**: React 18, Bootstrap, KaTeX (수식 렌더링)
- **시각화**: CSS 애니메이션, 행렬 표현

## 작동 방식

이 애플리케이션은 다음과 같은 아키텍처를 가진 간단한 CNN 모델을 시연합니다:
- 입력: 4x4 이미지
- Conv2d (kernel_size=2, padding=0): 출력 크기 3x3
- ReLU: 크기 3x3 유지
- MaxPool2d (kernel_size=2, stride=1): 출력 크기 2x2
- Flatten: 2x2 특성 맵을 4개 요소로 변환
- Linear(4, 2): 4개 특성을 2개 출력 클래스로 매핑
- Loss: CrossEntropyLoss

각 레이어에 대해 도구는 다음을 보여줍니다:
- 연산을 지배하는 수학적 공식
- 순전파 및 역전파 중에 계산된 실제 값
- 역전파 후 가중치 업데이트
- 모든 계산 단계에 대한 상세한 분석과 시각화

## 현재 제한사항

1. 현재는 고정된 4x4 입력 이미지와 사전 정의된 가중치만 지원합니다.
2. 오직 하나의 합성곱 레이어, ReLU, MaxPool 및 하나의 완전 연결 레이어만 시각화할 수 있습니다.
3. 학습 알고리즘은 기본적인 SGD(확률적 경사 하강법)로 고정되어 있습니다.
4. 역전파 계산에서 일부 중간 단계가 간소화되어 있습니다.

## 확장 계획 (로드맵)

### 단기 개발 계획 (1-3개월)
1. 사용자 정의 입력 데이터 지원
2. 다양한 커널 크기 및 패딩 옵션 제공
3. 더 깊은 네트워크 지원 (다중 합성곱 및 완전 연결 레이어)
4. 모델 구성 저장 및 불러오기 기능

### 중장기 개발 계획 (3-6개월)
1. 더 복잡한 CNN 아키텍처 지원 (ResNet, VGG 등)
2. 맞춤형 데이터셋 업로드 기능
3. 실시간 모델 수정 및 결과 시각화
4. 다양한 최적화 알고리즘 비교 도구 (SGD, Adam, RMSprop 등)

### 장기 개발 계획 (6개월 이상)
1. 학습 과정의 시간에 따른 그래디언트 흐름 애니메이션
2. 추가 레이어 지원 (BatchNorm, Dropout 등)
3. 모델 가중치의 진화를 3D로 시각화
4. 모델 해석 도구 (Grad-CAM, Saliency Maps 등)

## 참고 문헌
<<<<<<< HEAD

=======
0. QuarkML. (2023, July). Derivation of Backpropagation in Convolutional Neural Network (CNN).
Retrieved from [https://www.quarkml.com/2023/07/backward-pass-in-convolutional-neural-network-explained.html​
quarkml.com](https://www.quarkml.com/2023/07/backward-pass-in-convolutional-neural-network-explained.html)
>>>>>>> 3792a39575877c4fa8948912196bb63a5d0860d8
1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. ICCV.
4. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
5. Chellapilla, K., Puri, S., & Simard, P. (2006). High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition.
6. Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 315-323).
7. Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071.
8. Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
9. Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 807-814).
10. Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. In Artificial Neural Networks–ICANN 2010 (pp. 92-101). Springer.
