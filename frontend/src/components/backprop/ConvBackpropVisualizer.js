import React, { useState } from 'react';
import { Row, Col, Form } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

const ConvBackpropVisualizer = ({ backward, initial_weights, updated_weights, learning_rate }) => {
  const [decimalPlaces, setDecimalPlaces] = useState(6); // 소수점 자리 수 증가

  // 컨볼루션 역전파 설명을 위한 계산 단계들
  const convGradientSteps = [
    {
      description: "Convolution weight gradient calculation",
      equation: "\\frac{\\partial L}{\\partial W_{m,n}} = \\sum_{i,j} \\frac{\\partial L}{\\partial O_{i,j}} \\cdot I_{i+m, j+n}",
    },
    {
      description: "For each position in the output gradient, we multiply with the corresponding input region",
      equation: "\\frac{\\partial L}{\\partial W_{0,0}} = \\sum_{i,j} \\frac{\\partial L}{\\partial O_{i,j}} \\cdot I_{i, j}",
    },
    {
      description: "Using actual gradient values from the computation",
      equation: "\\frac{\\partial L}{\\partial W_{0,0}} = (\\text{output\\_grad}_{0,0} \\cdot \\text{input}_{0,0}) + ... + (\\text{output\\_grad}_{m,n} \\cdot \\text{input}_{m,n})",
    }
  ];

  // 가중치 업데이트 계산 단계
  const weightUpdateSteps = [
    {
      description: "Weight update using gradient descent",
      equation: "W_{\\text{new}} = W_{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial W}",
    },
    {
      description: `Using the learning rate (η = ${learning_rate})`,
      equation: `W_{\\text{new}} = W_{\\text{old}} - ${learning_rate} \\cdot \\frac{\\partial L}{\\partial W}`,
    },
    {
      description: "Applying the update to each weight value",
      equation: `W_{0,0}^{\\text{new}} = W_{0,0}^{\\text{old}} - ${learning_rate} \\cdot \\frac{\\partial L}{\\partial W_{0,0}}`,
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
  if (!backward || !backward.output_grad || !backward.weight_grad) {
    return (
      <div className="alert alert-warning">
        Missing Convolution gradient data from backend. Please ensure the backend is properly calculating convolution gradients.
      </div>
    );
  }

  return (
    <div className="conv-backprop-visualizer">
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
          <h6>Convolution Gradient Formation</h6>
          <div className="tensor-comparison">
            <div className="tensor-item">
              <p>Output Gradient</p>
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
              <p className="text-center mt-2">∂L/∂O - Gradient from ReLU layer</p>
            </div>
            
            <div className="tensor-item mt-4">
              <p>Input Feature Map</p>
              {backward.input_tensor ? (
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.input_tensor[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value/16), 0.7)})`
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
              <p className="text-center mt-2">Input tensor from forward pass</p>
            </div>
          </div>
          
          <div className="mt-4">
            <AnimatedCalculation steps={convGradientSteps} />
          </div>
        </Col>
        
        <Col md={6}>
          <h6>Weight Gradient and Update</h6>
          <div className="tensor-comparison">
            <div className="tensor-item">
              <p>Weight Gradient</p>
              <table className="tensor-table mx-auto">
                <tbody>
                  {backward.weight_grad[0][0].map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          style={{ 
                            backgroundColor: `rgba(220, 53, 69, ${Math.min(Math.abs(value), 0.7)})`
                          }}
                        >
                          {formatValue(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="text-center mt-2">∂L/∂W - Gradient for weights</p>
            </div>
            
            <div className="tensor-item mt-4">
              <p>Initial Weights → Updated Weights</p>
              <div className="weight-update-comparison">
                <div>
                  <p className="text-center">Initial Weights</p>
                  <table className="tensor-table mx-auto">
                    <tbody>
                      {initial_weights[0][0].map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {row.map((value, colIdx) => (
                            <td 
                              key={colIdx}
                              style={{ 
                                backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value), 0.7)})`
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
                
                <div className="update-arrow">→</div>
                
                <div>
                  <p className="text-center">Updated Weights</p>
                  <table className="tensor-table mx-auto">
                    <tbody>
                      {updated_weights[0][0].map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {row.map((value, colIdx) => {
                            // Calculate the delta for color intensity
                            const initialValue = initial_weights[0][0][rowIdx][colIdx];
                            const delta = value - initialValue;
                            
                            return (
                              <td 
                                key={colIdx}
                                style={{ 
                                  backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value), 0.7)})`,
                                  border: Math.abs(delta) > 0.00001 ? '2px solid #dc3545' : '1px solid #dee2e6'
                                }}
                              >
                                {formatValue(value)}
                                {Math.abs(delta) > 0.00001 && (
                                  <div className="delta-indicator">
                                    {delta > 0 ? '+' : ''}{formatValue(delta)}
                                  </div>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
          
          <div className="mt-4">
            <AnimatedCalculation steps={weightUpdateSteps} />
          </div>
          
          <div className="mt-4">
            <h6>Gradient Flow Visualization</h6>
            <p>The gradient flows from the ReLU layer to the convolution weights and inputs.</p>
            <div className="gradient-flow-diagram">
              <BlockMath math="\\frac{\\partial L}{\\partial ReLU} \\rightarrow \\frac{\\partial L}{\\partial Conv\\_out} \\rightarrow \\begin{cases} 
                \\frac{\\partial L}{\\partial W} \\\\ 
                \\frac{\\partial L}{\\partial Input}
              \\end{cases}" />
            </div>
          </div>
        </Col>
      </Row>

      {/* Weight Update Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <div className="weight-update-visualization p-3 bg-light rounded">
            <h6 className="mb-3">Weight Update Visualization</h6>
            
            <div className="weight-update-formula text-center mb-4">
              <BlockMath math={`W_{\\text{new}} = W_{\\text{old}} - ${learning_rate} \\cdot \\frac{\\partial L}{\\partial W}`} />
            </div>
            
            <Row>
              <Col md={4}>
                <h6 className="text-center">Initial Weights</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {initial_weights[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => (
                          <td 
                            key={colIdx}
                            style={{ 
                              backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value), 0.7)})`
                            }}
                          >
                            {formatValue(value)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-center mt-2">W<sub>old</sub></p>
              </Col>
              
              <Col md={4} className="text-center">
                <h6 className="text-center">Weight Gradient × Learning Rate</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {backward.weight_grad[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => {
                          const scaledGrad = value * learning_rate;
                          
                          return (
                            <td 
                              key={colIdx}
                              style={{ 
                                backgroundColor: `rgba(220, 53, 69, ${Math.min(Math.abs(scaledGrad*10), 0.7)})`
                              }}
                            >
                              {formatValue(scaledGrad)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="text-center mt-2">η · ∂L/∂W = {learning_rate} · ∂L/∂W</p>
              </Col>
              
              <Col md={4}>
                <h6 className="text-center">Updated Weights</h6>
                <table className="tensor-table mx-auto">
                  <tbody>
                    {updated_weights[0][0].map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => {
                          const initialValue = initial_weights[0][0][rowIdx][colIdx];
                          const expectedUpdate = initialValue - (backward.weight_grad[0][0][rowIdx][colIdx] * learning_rate);
                          const isCorrect = Math.abs(value - expectedUpdate) < 0.00001;
                          
                          return (
                            <td 
                              key={colIdx}
                              style={{ 
                                backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value), 0.7)})`,
                                border: isCorrect ? '1px solid #dee2e6' : '2px solid #ffc107'
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
                <p className="text-center mt-2">W<sub>new</sub> = W<sub>old</sub> - η · ∂L/∂W</p>
              </Col>
            </Row>
            
            <div className="update-explanation mt-4">
              <p>
                <strong>Explanation:</strong> The weight update process uses gradient descent to minimize the loss function.
                The gradients indicate the direction of steepest ascent, so we move in the opposite direction (subtract).
                The learning rate (η = {learning_rate}) controls the step size of this update.
              </p>
              <p>
                For each weight element W<sub>i,j</sub>, the update formula is:
                W<sub>i,j</sub><sup>new</sup> = W<sub>i,j</sub><sup>old</sup> - η · ∂L/∂W<sub>i,j</sub>
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
              <li>Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.</li>
              <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.</li>
              <li>Long, J., Shelhamer, E., & Darrell, T. (2015). Fully convolutional networks for semantic segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3431-3440).</li>
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
        
        .weight-update-comparison {
          display: flex;
          align-items: center;
          justify-content: space-between;
        }
        
        .update-arrow {
          font-size: 2rem;
          padding: 0 15px;
          color: #6c757d;
        }
        
        .delta-indicator {
          position: absolute;
          bottom: 2px;
          right: 2px;
          font-size: 0.6rem;
          color: #dc3545;
          font-weight: bold;
        }
        
        .gradient-flow-diagram {
          padding: 15px;
          background-color: rgba(0, 123, 255, 0.1);
          border-radius: 5px;
          margin-top: 10px;
        }
        
        .weight-update-visualization {
          overflow-x: auto;
        }
      `}</style>
    </div>
  );
};

export default ConvBackpropVisualizer;