import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from './TensorVisualizer';

const BackwardPass = ({ backward, gradients, initial_weights, updated_weights, learning_rate }) => {
  return (
    <div className="backward-pass-container">
      <Accordion defaultActiveKey="0">
        {/* Softmax & Cross Entropy 그래디언트 */}
        <Accordion.Item eventKey="0">
          <Accordion.Header>1. 소프트맥스 및 교차 엔트로피 손실 그래디언트</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>교차 엔트로피 손실 함수 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="L = -\sum_{i} y_i \log(\hat{y}_i)" />
                  <p>여기서:</p>
                  <ul>
                    <li><InlineMath math="y_i" /> = 실제 레이블 (원-핫 인코딩)</li>
                    <li><InlineMath math="\hat{y}_i" /> = 소프트맥스 출력 (예측 확률)</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">소프트맥스 수식</h6>
                <div className="equation-container">
                  <BlockMath math="\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}" />
                  <p>여기서 <InlineMath math="z_i" />는 FC 레이어의 출력입니다.</p>
                </div>
                
                <h6 className="mt-4">소프트맥스 미분</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial \hat{y}_i}{\partial z_j} = \begin{cases} 
                    \hat{y}_i(1 - \hat{y}_i) & \text{if } i = j \\
                    -\hat{y}_i\hat{y}_j & \text{if } i \neq j
                  \end{cases}" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>교차 엔트로피 손실 그래디언트</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i" />
                  <p>이는 소프트맥스와 교차 엔트로피 손실의 그래디언트를 합친 간결한 형태입니다.</p>
                </div>
                
                <div className="mt-4">
                  <h6>FC 출력에 대한 그래디언트</h6>
                  <div className="matrix-container">
                    <table className="matrix-table">
                      <tbody>
                        <tr>
                          {backward.fc.input_grad.map((row, i) => (
                            row.map((value, j) => (
                              <td key={j}>{value.toFixed(4)}</td>
                            ))
                          ))}
                        </tr>
                      </tbody>
                    </table>
                  </div>
                  <p className="text-muted small mt-2">
                    실제 레이블이 클래스 0(인덱스 0)이므로, 인덱스 0의 그래디언트는 (예측 확률 - 1)이고, 인덱스 1의 그래디언트는 (예측 확률)입니다.
                  </p>
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Bridle, J. S. (1990). Probabilistic interpretation of feedforward classification network outputs, with relationships to statistical pattern recognition. In Neurocomputing (pp. 227-236). Springer.
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* FC 레이어 역전파 */}
        <Accordion.Item eventKey="1">
          <Accordion.Header>2. 완전 연결 계층 역전파 (FC Backpropagation)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>FC 레이어 그래디언트 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
                  <BlockMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
                  <BlockMath math="\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot W_{i,j}" />
                </div>
                
                <h6 className="mt-4">가중치 그래디언트 계산</h6>
                <div className="calculation-step">
                  <InlineMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
                </div>
                <div className="calculation-step">
                  <p>예: 첫 번째 가중치 그래디언트 <InlineMath math="\frac{\partial L}{\partial W_{1,1}}" />:</p>
                  <InlineMath math="\frac{\partial L}{\partial W_{1,1}} = \frac{\partial L}{\partial y_1} \cdot x_1 = dl/dy_1 \cdot 15.5" />
                </div>
                
                <h6 className="mt-4">편향 그래디언트 계산</h6>
                <div className="calculation-step">
                  <InlineMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>FC 가중치 그래디언트</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      {backward.fc.weight_grad.map((row, i) => (
                        <tr key={i}>
                          {row.map((value, j) => (
                            <td key={j}>{value.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                <h6 className="mt-3">FC 편향 그래디언트</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {backward.fc.bias_grad.map((value, i) => (
                          <td key={i}>{value.toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <h6 className="mt-4">FC 입력에 대한 그래디언트</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {backward.fc.input_grad[0].map((value, i) => (
                          <td key={i}>{value.toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4 weight-update">
                  <h6>가중치 업데이트</h6>
                  <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
                  <p>여기서 <InlineMath math="\eta" />는 학습률 ({learning_rate})입니다.</p>
                  
                  <p className="mt-2">첫 번째 가중치 업데이트 예시:</p>
                  <InlineMath math="W_{1,1}^{new} = W_{1,1}^{old} - \eta \cdot \frac{\partial L}{\partial W_{1,1}}" />
                  <InlineMath math="W_{1,1}^{new} = 0.1 - 0.01 \cdot \frac{\partial L}{\partial W_{1,1}}" />
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* 풀링 레이어 역전파 */}
        <Accordion.Item eventKey="2">
          <Accordion.Header>3. 최대 풀링 역전파 (MaxPool Backpropagation)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>MaxPool 역전파 일반 원리</h6>
                <p>최대 풀링의 역전파는 순전파에서 선택된 최대값 위치로만 그래디언트를 전달합니다.</p>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \begin{cases} 
                    \frac{\partial L}{\partial O_{m,n}} & \text{if } I_{i,j} \text{ was the maximum in the pooling region} \\
                    0 & \text{otherwise}
                  \end{cases}" />
                  <p>여기서 <InlineMath math="O_{m,n}" />은 해당 풀링 영역의 출력값입니다.</p>
                </div>
                
                <h6 className="mt-4">풀링 출력 그래디언트</h6>
                <TensorVisualizer tensor={backward.pool.output_grad[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>풀링 입력 그래디언트 (확장된 형태)</h6>
                <TensorVisualizer tensor={backward.pool.input_grad[0][0]} />
                
                <div className="mt-4">
                  <p>그래디언트는 최대값을 가진 위치로만 전파됩니다. 다른 위치의 그래디언트는 0입니다.</p>
                  <div className="calculation-step">
                    <p>예: 풀링 지역 (0,0)에서 (1,1)의 최대값은 (1,1)이므로 그래디언트는 이 위치로만 전파됩니다.</p>
                  </div>
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* ReLU 역전파 */}
        <Accordion.Item eventKey="3">
          <Accordion.Header>4. ReLU 역전파</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>ReLU 역전파 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial ReLU(x)}{\partial x} = \begin{cases} 
                    1 & \text{if } x > 0 \\
                    0 & \text{if } x \leq 0
                  \end{cases}" />
                </div>
                
                <div className="mt-4">
                  <h6>ReLU 출력 그래디언트</h6>
                  <TensorVisualizer tensor={backward.relu.output_grad[0][0]} />
                </div>
                
                <div className="mt-4">
                  <h6>ReLU 활성화 마스크 (1: 활성, 0: 비활성)</h6>
                  <TensorVisualizer tensor={backward.relu.mask[0][0]} />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>ReLU 입력 그래디언트</h6>
                <TensorVisualizer tensor={backward.relu.input_grad[0][0]} />
                
                <div className="mt-4">
                  <p>ReLU의 역전파는 단순히 입력이 양수인 위치에만 그래디언트를 통과시킵니다.</p>
                  <div className="calculation-step">
                    <InlineMath math="\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial ReLU(x_i)} \cdot \frac{\partial ReLU(x_i)}{\partial x_i}" />
                  </div>
                  <div className="calculation-step">
                    <InlineMath math="\frac{\partial L}{\partial x_i} = \begin{cases} 
                      \frac{\partial L}{\partial ReLU(x_i)} & \text{if } x_i > 0 \\
                      0 & \text{if } x_i \leq 0
                    \end{cases}" />
                  </div>
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* Conv 역전파 */}
        <Accordion.Item eventKey="4">
          <Accordion.Header>5. 합성곱 역전파 (Convolution Backpropagation)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>합성곱 그래디언트 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot I_{i+m, j+n}" />
                  <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial O_{i-m, j-n}} \cdot W_{m,n}" />
                  <p>여기서:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = 출력 특징맵의 (i,j) 위치 값</li>
                    <li><InlineMath math="I_{i,j}" /> = 입력 텐서의 (i,j) 위치 값</li>
                    <li><InlineMath math="W_{m,n}" /> = 가중치 커널의 (m,n) 위치 값</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">합성곱 출력 그래디언트</h6>
                <TensorVisualizer tensor={backward.conv.output_grad[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>합성곱 가중치 그래디언트</h6>
                <TensorVisualizer tensor={backward.conv.weight_grad[0][0]} />
                
                <div className="mt-4 weight-update">
                  <h6>가중치 업데이트</h6>
                  <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
                  <p>여기서 <InlineMath math="\eta" />는 학습률 ({learning_rate})입니다.</p>
                  
                  <h6 className="mt-3">초기 커널 가중치</h6>
                  <TensorVisualizer tensor={initial_weights.conv1_weight[0][0]} />
                  
                  <h6 className="mt-3">업데이트된 커널 가중치</h6>
                  <TensorVisualizer tensor={updated_weights.conv1_weight[0][0]} />
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    </div>
  );
};

export default BackwardPass;
