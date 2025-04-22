import React, { useState } from 'react';
import { Row, Col, Form } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

const ReluBackpropVisualizer = ({ backward }) => {
  const [decimalPlaces, setDecimalPlaces] = useState(6); // 소수점 자리 수 증가
  
  // ReLU 역전파 설명을 위한 계산 단계들
  const reluBackpropSteps = [
    {
      description: "ReLU derivative calculation",
      equation: "\\frac{\\partial ReLU(x)}{\\partial x} = \\begin{cases} 1 & \\text{if } x > 0 \\\\ 0 & \\text{if } x \\leq 0 \\end{cases}",
    },
    {
      description: "Applying the chain rule",
      equation: "\\frac{\\partial L}{\\partial x} = \\frac{\\partial L}{\\partial ReLU(x)} \\cdot \\frac{\\partial ReLU(x)}{\\partial x}",
    },
    {
      description: "Input gradient calculation",
      equation: "\\frac{\\partial L}{\\partial x} = \\begin{cases} \\frac{\\partial L}{\\partial ReLU(x)} & \\text{if } x > 0 \\\\ 0 & \\text{if } x \\leq 0 \\end{cases}",
    }
  ];

  // 마스크 적용 설명
  const maskApplicationSteps = [
    {
      description: "Mask application to gradient",
      equation: "\\text{input\\_grad} = \\text{output\\_grad} \\odot \\text{mask}",
    },
    {
      description: "Where mask is 1 for active neurons and 0 for inactive",
      equation: "\\text{mask}_{i,j} = \\begin{cases} 1 & \\text{if input}_{i,j} > 0 \\\\ 0 & \\text{otherwise} \\end{cases}",
    },
    {
      description: "Element-wise multiplication",
      equation: "\\text{input\\_grad}_{i,j} = \\text{output\\_grad}_{i,j} \\cdot \\text{mask}_{i,j}",
    }
  ];

  // Format values with scientific notation for very small numbers
  const formatValue = (value) => {
    if (Math.abs(value) < 0.000001) {
      return value.toExponential(decimalPlaces - 1);
    }
    return value.toFixed(decimalPlaces);
  };

  // Check if we have all necessary data from backend
  if (!backward || !backward.output_grad || !backward.input_grad || !backward.mask) {
    return (
      <div className="alert alert-warning">
        Missing ReLU gradient data from backend. Please ensure the backend is properly calculating ReLU gradients.
      </div>
    );
  }

  return (
    <div className="relu-backprop-visualizer">
      {/* Decimal Places Control */}
      <Row className="mb-3">
        <Col md={6}>
          <Form.Group>
            <Form.Label>Decimal Places Precision:</Form.Label>
            <Form.Select 
              value={decimalPlaces}
              onChange={(e) => setDecimalPlaces(Number(e.target.value))}
            >
              <option value="4">4 digits (0.0000)</option>
              <option value="6">6 digits (0.000000)</option>
              <option value="8">8 digits (0.00000000)</option>
              <option value="10">10 digits (0.0000000000)</option>
            </Form.Select>
            <Form.Text className="text-muted">
              Increase precision to see very small gradient values.
            </Form.Text>
          </Form.Group>
        </Col>
      </Row>

      <Row>
        <Col md={6}>
          <h6>ReLU Activation Mask Visualization</h6>
          <div className="tensor-comparison">
            <div className="tensor-item">
              <p>ReLU Activation Mask (1: Active, 0: Inactive)</p>
              <table className="tensor-table mx-auto">
                <tbody>
                  {backward.mask[0][0].map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          style={{ 
                            backgroundColor: value > 0 ? 'rgba(40, 167, 69, 0.2)' : 'rgba(220, 53, 69, 0.2)'
                          }}
                        >
                          {value}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="tensor-item mt-4">
              <p>ReLU Input Values (From Forward Pass)</p>
              {backward.input_tensor ? (
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.input_tensor[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: value > 0 ? 'rgba(40, 167, 69, 0.2)' : 'rgba(220, 53, 69, 0.2)'
                            }}
                          >
                            {formatValue(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-muted">Input tensor data not available</p>
              )}
            </div>
          </div>
          
          <div className="mt-4">
            <AnimatedCalculation steps={reluBackpropSteps} />
          </div>
        </Col>
        
        <Col md={6}>
          <h6>Gradient Flow Through ReLU</h6>
          <div className="tensor-comparison">
            <div className="tensor-item">
              <p>ReLU Output Gradient</p>
              <table className="tensor-table mx-auto">
                <tbody>
                  {backward.output_grad[0][0].map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          style={{ 
                            backgroundColor: `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})`
                          }}
                        >
                          {formatValue(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            <div className="tensor-item mt-4">
              <p>ReLU Input Gradient</p>
              <table className="tensor-table mx-auto">
                <tbody>
                  {backward.input_grad[0][0].map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => {
                        // Get mask value at this position
                        const maskValue = backward.mask[0][0][rowIdx][colIdx];
                        
                        return (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: Math.abs(value) > 0.0000001 
                                ? `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})` 
                                : (maskValue > 0 ? 'rgba(173, 181, 189, 0.2)' : '')
                            }}
                          >
                            {formatValue(value)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          <div className="mt-4">
            <AnimatedCalculation steps={maskApplicationSteps} />
          </div>
          
          <div className="mt-4">
            <h6>Gradient Flow Explanation</h6>
            <p>During backpropagation, ReLU only passes gradients for neurons that were active during the forward pass. 
              This is why ReLU helps with the vanishing gradient problem - it provides a clear gradient pathway for active neurons.</p>
            
            <div className="gradient-blocking-example">
              <h6 className="mt-3">Example: Gradient Blocking</h6>
              <p>For a negative input value during forward pass:</p>
              <InlineMath math="\\text{If } x \\leq 0 \\text{ then } \\frac{\\partial L}{\\partial x} = 0" />
              <p>This means the gradient is completely blocked for this neuron.</p>
            </div>
          </div>
        </Col>
      </Row>

      {/* Gradient Flow Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <div className="gradient-flow-visualization p-3 bg-light rounded">
            <h6 className="mb-3">Gradient × Mask Visualization</h6>
            <Row>
              <Col md={4}>
                <h6 className="text-center">Output Gradient</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.output_grad[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})`
                            }}
                          >
                            {formatValue(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-center mt-2">∂L/∂ReLU(x)</p>
              </Col>
              
              <Col md={4} className="text-center d-flex flex-column justify-content-center">
                <div className="operation-symbol mb-2">×</div>
                <h6>Element-wise Multiplication</h6>
                <div className="operation-symbol mt-2">⇓</div>
              </Col>
              
              <Col md={4}>
                <h6 className="text-center">Activation Mask</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.mask[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: value > 0 ? 'rgba(40, 167, 69, 0.2)' : 'rgba(220, 53, 69, 0.2)'
                            }}
                          >
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-center mt-2">∂ReLU(x)/∂x</p>
              </Col>
            </Row>
            
            <Row className="mt-4">
              <Col md={{ span: 4, offset: 4 }}>
                <h6 className="text-center">Input Gradient</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.input_grad[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => {
                          // Get mask and output gradient values
                          const maskValue = backward.mask[0][0][rowIdx][colIdx];
                          const outputGradValue = backward.output_grad[0][0][rowIdx][colIdx];
                          
                          // Check if this is a blocked gradient (mask is 0)
                          const isBlocked = maskValue === 0;
                          
                          return (
                            <td 
                              key={colIdx}
                              className={isBlocked ? "blocked-gradient" : ""}
                              style={{ 
                                backgroundColor: Math.abs(value) > 0.0000001 
                                  ? `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})` 
                                  : (isBlocked ? 'rgba(220, 53, 69, 0.2)' : 'rgba(173, 181, 189, 0.2)')
                              }}
                            >
                              {formatValue(value)}
                              {isBlocked && (
                                <div className="blocked-label">BLOCKED</div>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-center mt-2">∂L/∂x = ∂L/∂ReLU(x) × ∂ReLU(x)/∂x</p>
              </Col>
            </Row>
            
            <div className="gradient-flow-explanation mt-4">
              <p>
                <strong>Explanation:</strong> ReLU's derivative is 1 for positive inputs and 0 for negative inputs.
                During backpropagation, this creates a binary mask where gradients only flow through neurons that were activated 
                (had positive inputs) during the forward pass.
              </p>
              <p>
                This visualization shows how the output gradient is element-wise multiplied by the activation mask 
                to produce the input gradient. Notice how gradients are completely blocked (set to zero) where 
                the activation mask is zero, regardless of the output gradient value.
              </p>
            </div>
          </div>
        </Col>
      </Row>
      
      {/* Academic References */}
      <Row className="mt-4">
        <Col md={12}>
          <div className="academic-references">
            <h6>Academic References</h6>
            <ul className="text-muted small">
              <li>Glorot, X., Bordes, A., & Bengio, Y. (2011). Deep sparse rectifier neural networks. In Proceedings of the fourteenth international conference on artificial intelligence and statistics (pp. 315-323).</li>
              <li>He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).</li>
              <li>Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th international conference on machine learning (ICML-10) (pp. 807-814).</li>
            </ul>
          </div>
        </Col>
      </Row>
      
      <style jsx>{`
        .tensor-table {
          border-collapse: collapse;
        }
        
        .tensor-table td {
          border: 1px solid #dee2e6;
          padding: 8px;
          text-align: center;
          width: 40px;
          height: 40px;
          transition: all 0.3s;
          font-size: 0.8rem;
          position: relative;
        }
        
        .tensor-comparison {
          margin-bottom: 20px;
        }
        
        .tensor-item {
          margin-bottom: 15px;
        }
        
        .operation-symbol {
          font-size: 2rem;
          font-weight: bold;
          color: #6c757d;
        }
        
        .blocked-gradient {
          position: relative;
        }
        
        .blocked-label {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          font-size: 0.6rem;
          background-color: rgba(220, 53, 69, 0.7);
          color: white;
          padding: 1px 3px;
          border-radius: 2px;
          pointer-events: none;
        }
        
        .gradient-flow-visualization {
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
};

export default ReluBackpropVisualizer;