import React from 'react';
import { Row, Col, Accordion } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from './TensorVisualizer';
import AnimatedCalculation from './AnimatedCalculation';
import ConvolutionVisualizer from './ConvolutionVisualizer';
import ReluVisualizer from './ReluVisualizer';
import MaxPoolVisualizer from './MaxPoolVisualizer';
import FCLayerVisualizer from './FCLayerVisualizer';

const ForwardPass = ({ forward }) => {
  // Define the ReLU calculation steps
  const reluCalculationSteps = [
    {
      description: "ReLU activation function",
      equation: "ReLU(x) = \\max(0, x)",
    },
    {
      description: "For the input value at (0,0)",
      equation: "ReLU(10.5) = \\max(0, 10.5)",
    },
    {
      description: "Since 10.5 > 0, the output is 10.5",
      equation: "ReLU(10.5) = 10.5",
      result: "10.5"
    }
  ];
  
  // Define maxpool calculation steps
  const maxpoolCalculationSteps = [
    {
      description: "MaxPooling operation for a 2x2 window",
      equation: "O_{0,0} = \\max_{window}(I_{i,j})",
    },
    {
      description: "For the top-left 2x2 window",
      equation: "O_{0,0} = \\max(I_{0,0}, I_{0,1}, I_{1,0}, I_{1,1})",
    },
    {
      description: "Substituting the values",
      equation: "O_{0,0} = \\max(10.5, 11.0, 15.0, 15.5)",
    },
    {
      description: "The maximum value is 15.5",
      equation: "O_{0,0} = 15.5",
      result: "15.5"
    }
  ];
  
  return (
    <div className="forward-pass-container">
      <Accordion defaultActiveKey="0">
        {/* Conv layer */}
        <Accordion.Item eventKey="0">
          <Accordion.Header>1. Convolution</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>General Convolution Formula</h6>
                <div className="equation-container">
                  <BlockMath math="O_{i,j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} I_{i+m, j+n} \cdot W_{m,n}" />
                  <p>Where:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = Output feature map value at position (i,j)</li>
                    <li><InlineMath math="I_{i+m, j+n}" /> = Input tensor value at position (i+m, j+n)</li>
                    <li><InlineMath math="W_{m,n}" /> = Weight kernel value at position (m,n)</li>
                    <li><InlineMath math="k_h, k_w" /> = Kernel height and width</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">Kernel Weights</h6>
                <TensorVisualizer tensor={forward.conv.weight_tensor[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>Input Tensor</h6>
                <TensorVisualizer tensor={forward.conv.input_tensor[0][0]} />
                
                <h6 className="mt-4">Output Feature Map</h6>
                <TensorVisualizer tensor={forward.conv.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>Matrix Representation</h6>
                  <p>Convolution operation can be represented as matrix multiplication:</p>
                  <div className="equation-container">
                    <BlockMath math="O = W \cdot I_{unfolded}" />
                  </div>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Chellapilla, K., Puri, S., & Simard, P. (2006). High performance convolutional neural networks for document processing. In Tenth International Workshop on Frontiers in Handwriting Recognition.
                  </p>
                </div>
              </Col>
            </Row>

            {/* Convolution calculation visualization */}
            <Row className="mt-4">
              <Col md={12}>
                <ConvolutionVisualizer 
                  inputTensor={forward.conv.input_tensor[0][0]} 
                  kernel={forward.conv.weight_tensor[0][0]} 
                  outputTensor={forward.conv.output_tensor[0][0]} 
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* ReLU layer */}
        <Accordion.Item eventKey="1">
          <Accordion.Header>2. ReLU Activation</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>ReLU General Formula</h6>
                <div className="equation-container">
                  <BlockMath math="ReLU(x) = \max(0, x)" />
                  <p>Applied element-wise:</p>
                  <BlockMath math="O_{i,j} = \max(0, I_{i,j})" />
                </div>
                
                <h6 className="mt-4">Input Tensor (Convolution Output)</h6>
                <TensorVisualizer tensor={forward.relu.input_tensor[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>Output Tensor</h6>
                <TensorVisualizer tensor={forward.relu.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>ReLU Activation Mask (1: Active, 0: Inactive)</h6>
                  <TensorVisualizer tensor={forward.relu.mask[0][0]} />
                </div>
                
                <div className="mt-4">
                  <p>ReLU is a simple but effective non-linear activation function that enables the network to learn complex patterns.</p>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 315-323).
                  </p>
                </div>
              </Col>
            </Row>

            {/* ReLU calculation visualization */}
            <Row className="mt-4">
              <Col md={12}>
                <ReluVisualizer 
                  inputTensor={forward.relu.input_tensor[0][0]} 
                  outputTensor={forward.relu.output_tensor[0][0]} 
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* MaxPool layer */}
        <Accordion.Item eventKey="2">
          <Accordion.Header>3. MaxPooling</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>MaxPool General Formula</h6>
                <div className="equation-container">
                  <BlockMath math="O_{i,j} = \max_{m=0,n=0}^{k-1, k-1} I_{i \cdot s + m, j \cdot s + n}" />
                  <p>Where:</p>
                  <ul>
                    <li><InlineMath math="O_{i,j}" /> = Output feature map value at position (i,j)</li>
                    <li><InlineMath math="I_{i \cdot s + m, j \cdot s + n}" /> = Input tensor values at the corresponding positions</li>
                    <li><InlineMath math="k" /> = Kernel size (2x2)</li>
                    <li><InlineMath math="s" /> = Stride (1)</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">Input Tensor (ReLU Output)</h6>
                <TensorVisualizer tensor={forward.pool.input_tensor[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>Output Tensor</h6>
                <TensorVisualizer tensor={forward.pool.output_tensor[0][0]} />
                
                <div className="mt-4">
                  <h6>Maximum Value Indices</h6>
                  <TensorVisualizer tensor={forward.pool.indices[0][0]} />
                  <p className="text-muted small">
                    Indices represent the flattened index in the input tensor.
                  </p>
                </div>
                
                <div className="mt-4">
                  <p>Max pooling reduces the spatial size of feature maps and provides translation invariance.</p>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071.
                  </p>
                </div>
              </Col>
            </Row>
            
            {/* MaxPool calculation visualization */}
            <Row className="mt-4">
              <Col md={12}>
                <MaxPoolVisualizer 
                  inputTensor={forward.pool.input_tensor[0][0]} 
                  outputTensor={forward.pool.output_tensor[0][0]} 
                  kernelSize={2}
                  stride={1}
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* Flatten operation */}
        <Accordion.Item eventKey="3">
          <Accordion.Header>4. Flatten</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>Flatten Operation</h6>
                <p>Flatten transforms a multi-dimensional tensor into a 1D vector.</p>
                <div className="equation-container">
                  <BlockMath math="f: \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{C \cdot H \cdot W}" />
                  <p>Where C, H, W are channels, height, and width respectively.</p>
                </div>
                
                <h6 className="mt-4">Input Tensor (MaxPool Output)</h6>
                <p>Shape: [1, 1, 2, 2]</p>
                <TensorVisualizer tensor={forward.pool.output_tensor[0][0]} />
              </Col>
              
              <Col md={6}>
                <h6>Output Vector</h6>
                <p>Shape: [1, 4]</p>
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
                  <p>Flatten is simply a reshaping operation with no learnable parameters.</p>
                  <div className="calculation-step">
                    <p>Converting a 2x2 tensor to a 1D vector:</p>
                    <InlineMath math="\begin{bmatrix} 15.5 & 17.5 \\ 23.5 & 25.5 \end{bmatrix} \rightarrow \begin{bmatrix} 15.5 & 17.5 & 23.5 & 25.5 \end{bmatrix}" />
                  </div>
                </div>
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
        
        {/* FC layer */}
        <Accordion.Item eventKey="4">
          <Accordion.Header>5. Fully Connected Layer</Accordion.Header>
          <Accordion.Body>
            <Row>
              <Col md={6}>
                <h6>FC Layer General Formula</h6>
                <div className="equation-container">
                  <BlockMath math="y = Wx + b" />
                  <p>Where:</p>
                  <ul>
                    <li><InlineMath math="W" /> = Weight matrix (size: output dimension x input dimension)</li>
                    <li><InlineMath math="x" /> = Input vector</li>
                    <li><InlineMath math="b" /> = Bias vector</li>
                    <li><InlineMath math="y" /> = Output vector</li>
                  </ul>
                </div>
                
                <h6 className="mt-4">Weight Matrix W (2x4)</h6>
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
                
                <h6 className="mt-3">Bias Vector b</h6>
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
                <h6>Input Vector x</h6>
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
                
                <h6 className="mt-3">Output Vector y</h6>
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
                
                <div className="mt-4">
                  <p>The fully connected layer transforms the flattened features into class scores.</p>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Bishop, C. M. (2006). Pattern recognition and machine learning. Springer.
                  </p>
                </div>
              </Col>
            </Row>
            
            {/* FC Layer calculation visualization - newly added */}
            <Row className="mt-4">
              <Col md={12}>
                <FCLayerVisualizer 
                  inputVector={forward.fc.input_tensor[0]} 
                  weights={forward.fc.weight} 
                  bias={forward.fc.bias}
                  outputVector={forward.fc.output[0]} 
                />
              </Col>
            </Row>
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    </div>
  );
};

export default ForwardPass;
