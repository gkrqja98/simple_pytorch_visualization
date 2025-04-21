import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from './TensorVisualizer';

const ForwardPass = ({ forward }) => {
  return (
    <div className="forward-pass-container">
      <Accordion defaultActiveKey="0">
        {/* Conv 레이어 */}
        <Accordion.Item eventKey="0">
          <Accordion.Header>1. Convolution</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>합성곱 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="O_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I_{i+m, j+n} \cdot W_{m,n}" />
                  <p>여기서:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = 출력 특징맵의 (i,j) 위치 값</li>
                    <li><InlineMath math="I_{i+m, j+n}" /> = 입력 텐서의 (i+m, j+n) 위치 값</li>
                    <li><InlineMath math="W_{m,n}" /> = 가중치 커널의 (m,n) 위치 값</li>
                    <li><InlineMath math="k_h, k_w" /> = 커널의 높이와 너비</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">커널 가중치</h6>
                <TensorVisualizer tensor={forward.conv.weight_tensor[0][0]} />
                
                <h6 className="mt-4">계산 예시: 출력 특징맵의 (0,0) 위치</h6>
                <div className="calculation-step">
                  <InlineMath math="O_{0,0} = I_{0,0} \cdot W_{0,0} + I_{0,1} \cdot W_{0,1} + I_{1,0} \cdot W_{1,0} + I_{1,1} \cdot W_{1,1}" />
                </div>
                <div className="calculation-step">
                  <InlineMath math="O_{0,0} = 1.0 \cdot 1.0 + 2.0 \cdot 0.5 + 5.0 \cdot 0.5 + 6.0 \cdot 1.0" />
                </div>
                <div className="calculation-step">
                  <InlineMath math="O_{0,0} = 1.0 + 1.0 + 2.5 + 6.0 = 10.5" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>입력 텐서</h6>
                <TensorVisualizer tensor={forward.conv.input_tensor[0][0]} />
                
                <h6 className="mt-4">출력 특징맵</h6>
                <TensorVisualizer tensor={forward.conv.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>행렬 형태로의 변환</h6>
                  <p>합성곱 연산은 행렬 곱셈으로 표현할 수 있습니다:</p>
                  <div className="equation-container">
                    <BlockMath math="O = W \cdot I_{unfolded}" />
                  </div>
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Chellapilla, K., Puri, S., & Simard, P. (2006). High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition.
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* ReLU 레이어 */}
        <Accordion.Item eventKey="1">
          <Accordion.Header>2. ReLU 활성화 함수</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>ReLU 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="ReLU(x) = \max(0, x)" />
                  <p>각 요소별로 적용됩니다:</p>
                  <BlockMath math="O_{i,j} = \max(0, I_{i,j})" />
                </div>
                
                <h6 className="mt-4">입력 텐서 (합성곱 출력)</h6>
                <TensorVisualizer tensor={forward.relu.input_tensor[0][0]} />
                
                <h6 className="mt-4">계산 예시: 출력 텐서의 (0,0) 위치</h6>
                <div className="calculation-step">
                  <InlineMath math="O_{0,0} = \max(0, I_{0,0}) = \max(0, 10.5) = 10.5" />
                </div>
                
                <h6 className="mt-3">계산 예시: 음수 값이 있는 경우</h6>
                <div className="calculation-step">
                  <p>예를 들어 입력이 -2.0인 경우:</p>
                  <InlineMath math="O_{i,j} = \max(0, -2.0) = 0" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>출력 텐서</h6>
                <TensorVisualizer tensor={forward.relu.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>ReLU 활성화 마스크 (1: 활성, 0: 비활성)</h6>
                  <TensorVisualizer tensor={forward.relu.mask[0][0]} />
                </div>
                
                <div className="mt-4">
                  <p>ReLU는 간단하지만 효과적인 비선형 활성화 함수로, 네트워크가 복잡한 패턴을 학습할 수 있게 합니다.</p>
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 315-323).
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* MaxPool 레이어 */}
        <Accordion.Item eventKey="2">
          <Accordion.Header>3. 최대 풀링 (MaxPooling)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>MaxPool 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="O_{i,j} = \max_{m=0,n=0}^{k-1, k-1} I_{i \cdot s + m, j \cdot s + n}" />
                  <p>여기서:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = 출력 특징맵의 (i,j) 위치 값</li>
                    <li><InlineMath math="I_{i \cdot s + m, j \cdot s + n}" /> = 입력 텐서의 해당 위치 값</li>
                    <li><InlineMath math="k" /> = 커널 크기 (2x2)</li>
                    <li><InlineMath math="s" /> = 스트라이드 (2)</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">입력 텐서 (ReLU 출력)</h6>
                <TensorVisualizer tensor={forward.pool.input_tensor[0][0]} />
                
                <h6 className="mt-4">계산 예시: 2x2 풀링 위도우</h6>
                <div className="calculation-step">
                  <p>출력 (0,0) 위치를 위한 계산:</p>
                  <InlineMath math="O_{0,0} = \max(I_{0,0}, I_{0,1}, I_{1,0}, I_{1,1})" />
                </div>
                <div className="calculation-step">
                  <InlineMath math="O_{0,0} = \max(10.5, 11.0, 15.0, 15.5) = 15.5" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>출력 텐서</h6>
                <TensorVisualizer tensor={forward.pool.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>최대값 인덱스</h6>
                  <TensorVisualizer tensor={forward.pool.indices[0][0]} />
                  <p className="text-muted small">
                    인덱스는 입력 텐서의 평탄화된 인덱스를 나타냅니다.
                  </p>
                </div>
                
                <div className="mt-4">
                  <p>최대 풀링은 특징맵의 공간적 크기를 줄이고 위치 불변성을 부여합니다.</p>
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071.
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* Flatten 연산 */}
        <Accordion.Item eventKey="3">
          <Accordion.Header>4. 평탄화 (Flatten)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>평탄화 연산</h6>
                <p>평탄화는 다차원 텐서를 1차원 벡터로 변환합니다.</p>
                <div className="equation-container">
                  <BlockMath math="f: \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{C \cdot H \cdot W}" />
                  <p>여기서 C, H, W는 각각 채널, 높이, 너비입니다.</p>
                </div>
                
                <h6 className="mt-4">입력 텐서 (MaxPool 출력)</h6>
                <p>형태: [1, 1, 2, 2]</p>
                <TensorVisualizer tensor={forward.pool.output_tensor[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>출력 벡터</h6>
                <p>형태: [1, 4]</p>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {forward.fc.input_tensor[0].map((value, i) => (
                          <td key={i}>{value.toFixed(2)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4">
                  <p>평탄화는 단순히 데이터를 재구성하는 연산으로, 학습 파라미터는 없습니다.</p>
                  <div className="calculation-step">
                    <p>2x2 텐서를 1차원 벡터로 변환:</p>
                    <InlineMath math="\begin{bmatrix} 15.5 & 17.5 \\ 23.5 & 25.5 \end{bmatrix} \rightarrow \begin{bmatrix} 15.5 & 17.5 & 23.5 & 25.5 \end{bmatrix}" />
                  </div>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* FC 레이어 */}
        <Accordion.Item eventKey="4">
          <Accordion.Header>5. 완전 연결 계층 (Fully Connected)</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>FC 레이어 일반 수식</h6>
                <div className="equation-container">
                  <BlockMath math="y = Wx + b" />
                  <p>여기서:</p>
                  <ul>
                    <li><InlineMath math="W" /> = 가중치 행렬 (크기: 출력 차원 x 입력 차원)</li>
                    <li><InlineMath math="x" /> = 입력 벡터</li>
                    <li><InlineMath math="b" /> = 편향 벡터</li>
                    <li><InlineMath math="y" /> = 출력 벡터</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">가중치 행렬 W (2x4)</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      {forward.fc.weight.map((row, i) => (
                        <tr key={i}>
                          {row.map((value, j) => (
                            <td key={j}>{value.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                <h6 className="mt-3">편향 벡터 b</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {forward.fc.bias.map((value, i) => (
                          <td key={i}>{value.toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </Col>
              
              <Col md={6}>
                <h6>입력 벡터 x</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {forward.fc.input_tensor[0].map((value, i) => (
                          <td key={i}>{value.toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <h6 className="mt-3">출력 벡터 y</h6>
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      <tr>
                        {forward.fc.output[0].map((value, i) => (
                          <td key={i}>{value.toFixed(4)}</td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
                
                <h6 className="mt-4">계산 과정</h6>
                <div className="calculation-step">
                  <InlineMath math="y_1 = W_{1,1} \cdot x_1 + W_{1,2} \cdot x_2 + W_{1,3} \cdot x_3 + W_{1,4} \cdot x_4 + b_1" />
                </div>
                <div className="calculation-step">
                  <InlineMath math="y_1 = 0.1 \cdot 15.5 + 0.2 \cdot 17.5 + 0.3 \cdot 23.5 + 0.4 \cdot 25.5 + 0.1" />
                </div>
                <div className="calculation-step">
                  <InlineMath math="y_1 = 1.55 + 3.5 + 7.05 + 10.2 + 0.1 = 22.4" />
                </div>
                
                <div className="mt-3">
                  <p className="text-muted small">
                    <strong>참고문헌:</strong> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
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

export default ForwardPass;
