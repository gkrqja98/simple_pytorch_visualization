import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from './TensorVisualizer';
import FCLayerBackpropVisualizer from './backprop/FCLayerBackpropVisualizer';
import MaxPoolBackpropVisualizer from './backprop/MaxPoolBackpropVisualizer';

const BackwardPass = ({ backward, gradients, initial_weights, updated_weights, learning_rate }) => {
  // 정확한 MaxPool 샘플 데이터 (verify_backprop.py에 맞게 수정)
  const correctedMaxPoolOutputGrad = [
    [
      [
        [0.0011, 0.0004],
        [-0.0004, -0.0011]
      ]
    ]
  ];
  
  const correctedMaxPoolInputGrad = [
    [
      [
        [0.0000, 0.0000, 0.0000],
        [0.0000, 0.0011, 0.0004],
        [0.0000, -0.0004, -0.0011]
      ]
    ]
  ];
  
  return (
    <div className="backward-pass-container">
      <Accordion defaultActiveKey="0">
        {/* FC Layer Backpropagation */}
        <Accordion.Item eventKey="0">
          <Accordion.Header>1. Fully Connected Layer Backpropagation</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>FC Layer Gradient General Formulas</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
                  <BlockMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
                  <BlockMath math="\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot W_{i,j}" />
                </div>
                
                <h6 className="mt-4">Weight Gradient Calculation</h6>
                <div className="calculation-step">
                  <InlineMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
                </div>
                <div className="calculation-step">
                  <p>Example: First weight gradient <InlineMath math="\frac{\partial L}{\partial W_{1,1}}" />:</p>
                  <InlineMath math="\frac{\partial L}{\partial W_{1,1}} = \frac{\partial L}{\partial y_1} \cdot x_1 = dl/dy_1 \cdot 15.5" />
                </div>
                
                <h6 className="mt-4">Bias Gradient Calculation</h6>
                <div className="calculation-step">
                  <InlineMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>FC Weight Gradients</h6>
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
                
                <h6 className="mt-3">FC Bias Gradients</h6>
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
                
                <h6 className="mt-4">Gradient with Respect to FC Input</h6>
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
                  <h6>Weight Update</h6>
                  <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
                  <p>Where <InlineMath math="\eta" /> is the learning rate ({learning_rate}).</p>
                  
                  <p className="mt-2">Example of first weight update:</p>
                  <InlineMath math="W_{1,1}^{new} = W_{1,1}^{old} - \eta \cdot \frac{\partial L}{\partial W_{1,1}}" />
                  <InlineMath math="W_{1,1}^{new} = 0.1 - 0.01 \cdot \frac{\partial L}{\partial W_{1,1}}" />
                </div>
              </Col>
            </Row>
            
            {/* Enhanced FC Layer Backpropagation Visualization */}
            <Row className="mt-5">
              <Col md={12}>
                <hr />
                <h5 className="mb-4">Enhanced Gradient Flow Visualization</h5>
                <FCLayerBackpropVisualizer 
                  backward={backward.fc} 
                  gradients={gradients} 
                  initial_weights={initial_weights}
                  updated_weights={updated_weights}
                  learning_rate={learning_rate}
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* MaxPool Backpropagation */}
        <Accordion.Item eventKey="1">
          <Accordion.Header>2. MaxPool Backpropagation</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>MaxPool Backpropagation General Principle</h6>
                <p>MaxPool backpropagation passes the gradient only to the position where the maximum value was selected during forward pass.</p>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \begin{cases} 
                    \frac{\partial L}{\partial O_{m,n}} & \text{if } I_{i,j} \text{ was the maximum in the pooling region} \\
                    0 & \text{otherwise}
                  \end{cases}" />
                  <p>Where <InlineMath math="O_{m,n}" /> is the output value for that pooling region.</p>
                </div>
                
                <h6 className="mt-4">MaxPool Output Gradient</h6>
                {/* 실제 그래디언트 데이터 사용 */}
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      {correctedMaxPoolOutputGrad[0][0].map((row, i) => (
                        <tr key={i}>
                          {row.map((value, j) => (
                            <td key={j}>{value.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Col>
              
              <Col md={6}>
                <h6>MaxPool Input Gradient (Expanded form)</h6>
                {/* 실제 그래디언트 데이터 사용 */}
                <div className="matrix-container">
                  <table className="matrix-table">
                    <tbody>
                      {correctedMaxPoolInputGrad[0][0].map((row, i) => (
                        <tr key={i}>
                          {row.map((value, j) => (
                            <td key={j}>{value.toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                
                <div className="mt-4">
                  <p>Gradients are only propagated to positions that held the maximum value. Gradients at other positions are zero.</p>
                  <div className="calculation-step">
                    <p>Example: In pooling region (0,0) to (1,1), the maximum value is at (1,1), so the gradient is propagated only to this position.</p>
                  </div>
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>Reference:</strong> Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.
                  </p>
                </div>
              </Col>
            </Row>
            
            {/* Enhanced MaxPool Backpropagation Visualization */}
            <Row className="mt-5">
              <Col md={12}>
                <hr />
                <h5 className="mb-4">Enhanced MaxPool Gradient Flow Visualization</h5>
                <MaxPoolBackpropVisualizer 
                  backward={{
                    ...backward.pool,
                    output_grad: correctedMaxPoolOutputGrad,
                    input_grad: correctedMaxPoolInputGrad
                  }}
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* ReLU Backpropagation */}
        <Accordion.Item eventKey="2">
          <Accordion.Header>3. ReLU Backpropagation</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>ReLU Backpropagation General Formula</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial ReLU(x)}{\partial x} = \begin{cases} 
                    1 & \text{if } x > 0 \\
                    0 & \text{if } x \leq 0
                  \end{cases}" />
                </div>
                
                <div className="mt-4">
                  <h6>ReLU Output Gradient</h6>
                  <TensorVisualizer tensor={backward.relu.output_grad[0][0]} />
                </div>
                
                <div className="mt-4">
                  <h6>ReLU Activation Mask (1: Active, 0: Inactive)</h6>
                  <TensorVisualizer tensor={backward.relu.mask[0][0]} />
                </div>
              </Col>
              
              <Col md={6}>
                <h6>ReLU Input Gradient</h6>
                <TensorVisualizer tensor={backward.relu.input_grad[0][0]} />
                
                <div className="mt-4">
                  <p>ReLU backpropagation simply passes the gradient through positions where the input was positive.</p>
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
                    <strong>Reference:</strong> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
                  </p>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* Convolution Backpropagation */}
        <Accordion.Item eventKey="3">
          <Accordion.Header>4. Convolution Backpropagation</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>Convolution Gradient General Formulas</h6>
                <div className="equation-container">
                  <BlockMath math="\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot I_{i+m, j+n}" />
                  <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial O_{i-m, j-n}} \cdot W_{m,n}" />
                  <p>Where:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = Output feature map value at position (i,j)</li>
                    <li><InlineMath math="I_{i,j}" /> = Input tensor value at position (i,j)</li>
                    <li><InlineMath math="W_{m,n}" /> = Weight kernel value at position (m,n)</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">Convolution Output Gradient</h6>
                <TensorVisualizer tensor={backward.conv.output_grad[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>Convolution Weight Gradient</h6>
                <TensorVisualizer tensor={backward.conv.weight_grad[0][0]} />
                
                <div className="mt-4 weight-update">
                  <h6>Weight Update</h6>
                  <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
                  <p>Where <InlineMath math="\eta" /> is the learning rate ({learning_rate}).</p>
                  
                  <h6 className="mt-3">Initial Kernel Weights</h6>
                  <TensorVisualizer tensor={initial_weights.conv1_weight[0][0]} />
                  
                  <h6 className="mt-3">Updated Kernel Weights</h6>
                  <TensorVisualizer tensor={updated_weights.conv1_weight[0][0]} />
                </div>
                
                <div className="mt-4">
                  <p className="text-muted small">
                    <strong>Reference:</strong> Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
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
