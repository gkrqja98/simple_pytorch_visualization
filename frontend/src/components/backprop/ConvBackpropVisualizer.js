import React, { useState, useEffect } from 'react';
import { Row, Col, Form, Nav, Tab, Alert, Card } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

const ConvBackpropVisualizer = ({ backward, initial_weights, updated_weights, learning_rate }) => {
  const [decimalPlaces, setDecimalPlaces] = useState(6);
  const [activeTab, setActiveTab] = useState('basic');
  const [highlightCell, setHighlightCell] = useState({ row: -1, col: -1 });
  const [calculationExample, setCalculationExample] = useState('weightGrad');

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
    if (value === undefined || value === null) return "N/A";
    if (Math.abs(value) < 0.000001) {
      return value.toExponential(decimalPlaces - 1);
    }
    return value.toFixed(decimalPlaces);
  };

  // Calculate flipped kernel for transposed convolution visualization
  const getFlippedKernel = () => {
    if (!initial_weights || !Array.isArray(initial_weights)) 
      return [[]];
    
    // Safely access the kernel with proper type checking
    try {
      const kernel = initial_weights[0][0];
      if (Array.isArray(kernel) && kernel.length > 0 && Array.isArray(kernel[0])) {
        return kernel.map(row => [...row].reverse()).reverse();
      }
    } catch (error) {
      console.error("Error flipping kernel:", error);
    }
    
    return [[]];
  };

  // Safely access tensor data with error handling
  const getTensorData = (tensor, defaultValue = []) => {
    if (!tensor) return defaultValue;
    
    try {
      // Check for specific structure
      if (tensor[0] && tensor[0][0]) {
        return tensor[0][0];
      }
    } catch (error) {
      console.error("Error accessing tensor data:", error);
    }
    
    return defaultValue;
  };

  // Check if we have all necessary data from backend before attempting to render
  if (!backward || !backward.output_grad || !backward.weight_grad) {
    return (
      <Alert variant="warning">
        Missing Convolution gradient data from backend. Please ensure the backend is properly calculating convolution gradients.
      </Alert>
    );
  }

  // Safely get the output gradient and weight gradient data
  const outputGradData = getTensorData(backward.output_grad, [[]]);
  const weightGradData = getTensorData(backward.weight_grad, [[]]);
  const inputTensorData = backward.input_tensor ? getTensorData(backward.input_tensor, [[]]) : [[]];
  const initialWeightsData = Array.isArray(initial_weights) ? getTensorData(initial_weights, [[]]) : [[]];
  const updatedWeightsData = Array.isArray(updated_weights) ? getTensorData(updated_weights, [[]]) : [[]];

  // Example values for detailed calculation (normally these would come from the actual data)
  const detailedCalculationExample = {
    weightGrad: {
      position: "0,0",
      expandedFormula: "\\frac{\\partial L}{\\partial W_{0,0}} = \\frac{\\partial L}{\\partial O_{0,0}} \\cdot I_{0,0} + \\frac{\\partial L}{\\partial O_{0,1}} \\cdot I_{0,1} + \\frac{\\partial L}{\\partial O_{1,0}} \\cdot I_{1,0} + \\frac{\\partial L}{\\partial O_{1,1}} \\cdot I_{1,1}",
      substitution: "\\frac{\\partial L}{\\partial W_{0,0}} = 0.23 \\cdot 1.0 + 0.15 \\cdot 2.0 + 0.18 \\cdot 3.0 + 0.12 \\cdot 4.0",
      result: "1.47",
      steps: [
        {
          title: "1. Expanded formula for the gradient at position (0,0)",
          formula: "\\frac{\\partial L}{\\partial W_{0,0}} = \\frac{\\partial L}{\\partial O_{0,0}} \\cdot I_{0,0} + \\frac{\\partial L}{\\partial O_{0,1}} \\cdot I_{0,1} + \\frac{\\partial L}{\\partial O_{1,0}} \\cdot I_{1,0} + \\frac{\\partial L}{\\partial O_{1,1}} \\cdot I_{1,1}"
        },
        {
          title: "2. Substituting the actual values",
          formula: "\\frac{\\partial L}{\\partial W_{0,0}} = 0.23 \\cdot 1.0 + 0.15 \\cdot 2.0 + 0.18 \\cdot 3.0 + 0.12 \\cdot 4.0"
        },
        {
          title: "3. Final result",
          formula: "\\frac{\\partial L}{\\partial W_{0,0}} = 0.23 + 0.30 + 0.54 + 0.48 = 1.55"
        }
      ]
    },
    weightUpdate: {
      position: "0,0",
      expandedFormula: "W_{0,0}^{\\text{new}} = W_{0,0}^{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial W_{0,0}}",
      substitution: `W_{0,0}^{\\text{new}} = 1.0 - ${learning_rate} \\cdot 1.55`,
      result: `${formatValue(1.0 - learning_rate * 1.55)}`,
      steps: [
        {
          title: "1. Weight update formula for position (0,0)",
          formula: "W_{0,0}^{\\text{new}} = W_{0,0}^{\\text{old}} - \\eta \\cdot \\frac{\\partial L}{\\partial W_{0,0}}"
        },
        {
          title: "2. Substituting the actual values",
          formula: `W_{0,0}^{\\text{new}} = 1.0 - ${learning_rate} \\cdot 1.55`
        },
        {
          title: "3. Final result",
          formula: `W_{0,0}^{\\text{new}} = 1.0 - ${formatValue(learning_rate * 1.55)} = ${formatValue(1.0 - learning_rate * 1.55)}`
        }
      ]
    },
    forwardConv: {
      position: "0,0",
      expandedFormula: "O_{0,0} = I_{0,0} \\cdot W_{0,0} + I_{0,1} \\cdot W_{0,1} + I_{1,0} \\cdot W_{1,0} + I_{1,1} \\cdot W_{1,1}",
      substitution: "O_{0,0} = 1.0 \\cdot 1.0 + 2.0 \\cdot 0.5 + 5.0 \\cdot 0.5 + 6.0 \\cdot 1.0",
      result: "10.5",
      steps: [
        {
          title: "1. Expanded formula for the output at position (0,0)",
          formula: "O_{0,0} = I_{0,0} \\cdot W_{0,0} + I_{0,1} \\cdot W_{0,1} + I_{1,0} \\cdot W_{1,0} + I_{1,1} \\cdot W_{1,1}"
        },
        {
          title: "2. Substituting the actual values",
          formula: "O_{0,0} = 1.0 \\cdot 1.0 + 2.0 \\cdot 0.5 + 5.0 \\cdot 0.5 + 6.0 \\cdot 1.0"
        },
        {
          title: "3. Final result",
          formula: "O_{0,0} = 1.0 + 1.0 + 2.5 + 6.0 = 10.5"
        }
      ]
    }
  };

  return (
    <div className="conv-backprop-visualizer">
      {/* Controls */}
      <Row className="mb-3">
        <Col md={4}>
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
          </Form.Group>
        </Col>
      </Row>

      {/* Main Visualization Tabs */}
      <Tab.Container id="conv-backprop-tabs" activeKey={activeTab} onSelect={(k) => setActiveTab(k)}>
        <Nav variant="tabs" className="mb-3">
          <Nav.Item>
            <Nav.Link eventKey="basic">Basic Visualization</Nav.Link>
          </Nav.Item>
          <Nav.Item>
            <Nav.Link eventKey="detailed">Calculation & Learning Rate</Nav.Link>
          </Nav.Item>
        </Nav>

        <Tab.Content>
          {/* Basic Visualization Tab */}
          <Tab.Pane eventKey="basic">
            <Row>
              <Col md={6}>
                <h6>Convolution Gradient Formation</h6>
                <div className="tensor-comparison">
                  <div className="tensor-item">
                    <p>Output Gradient</p>
                    {outputGradData.length > 0 ? (
                      <table className="tensor-table mx-auto">
                        <tbody>
                          {outputGradData.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {Array.isArray(row) && row.map((value, colIdx) => (
                                <td 
                                  key={colIdx}
                                  style={{ 
                                    backgroundColor: `rgba(0, 123, 255, ${Math.min(Math.abs(value || 0), 0.7)})` 
                                  }}
                                  onMouseEnter={() => setHighlightCell({ row: rowIdx, col: colIdx })}
                                  onMouseLeave={() => setHighlightCell({ row: -1, col: -1 })}
                                  className={highlightCell.row === rowIdx && highlightCell.col === colIdx ? 'highlighted' : ''}
                                >
                                  {formatValue(value)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-muted">Output gradient data not available</p>
                    )}
                    <p className="text-center mt-2">∂L/∂O - Gradient from ReLU layer</p>
                  </div>
                  
                  <div className="tensor-item mt-4">
                    <p>Input Feature Map</p>
                    {inputTensorData.length > 0 ? (
                      <table className="tensor-table mx-auto">
                        <tbody>
                          {inputTensorData.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {Array.isArray(row) && row.map((value, colIdx) => (
                                <td 
                                  key={colIdx}
                                  style={{ 
                                    backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs((value || 0)/16), 0.7)})` 
                                  }}
                                  onMouseEnter={() => setHighlightCell({ row: rowIdx, col: colIdx })}
                                  onMouseLeave={() => setHighlightCell({ row: -1, col: -1 })}
                                  className={highlightCell.row === rowIdx && highlightCell.col === colIdx ? 'highlighted' : ''}
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
                    {weightGradData.length > 0 ? (
                      <table className="tensor-table mx-auto">
                        <tbody>
                          {weightGradData.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {Array.isArray(row) && row.map((value, colIdx) => (
                                <td 
                                  key={colIdx}
                                  style={{ 
                                    backgroundColor: `rgba(220, 53, 69, ${Math.min(Math.abs(value || 0), 0.7)})` 
                                  }}
                                  onMouseEnter={() => setHighlightCell({ row: rowIdx, col: colIdx })}
                                  onMouseLeave={() => setHighlightCell({ row: -1, col: -1 })}
                                  className={highlightCell.row === rowIdx && highlightCell.col === colIdx ? 'highlighted' : ''}
                                >
                                  {formatValue(value)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <p className="text-muted">Weight gradient data not available</p>
                    )}
                    <p className="text-center mt-2">∂L/∂W - Gradient for weights</p>
                  </div>
                  
                  <div className="tensor-item mt-4">
                    <p>Initial Weights → Updated Weights</p>
                    <div className="weight-update-comparison">
                      <div>
                        <p className="text-center">Initial Weights</p>
                        {initialWeightsData.length > 0 ? (
                          <table className="tensor-table mx-auto">
                            <tbody>
                              {initialWeightsData.map((row, rowIdx) => (
                                <tr key={rowIdx}>
                                  {Array.isArray(row) && row.map((value, colIdx) => (
                                    <td 
                                      key={colIdx}
                                      style={{ 
                                        backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value || 0), 0.7)})` 
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
                          <p className="text-muted">Initial weights data not available</p>
                        )}
                      </div>
                      
                      <div className="update-arrow">→</div>
                      
                      <div>
                        <p className="text-center">Updated Weights</p>
                        {updatedWeightsData.length > 0 ? (
                          <table className="tensor-table mx-auto">
                            <tbody>
                              {updatedWeightsData.map((row, rowIdx) => (
                                <tr key={rowIdx}>
                                  {Array.isArray(row) && row.map((value, colIdx) => {
                                    // Calculate the delta for color intensity
                                    const initialValue = initialWeightsData[rowIdx] && initialWeightsData[rowIdx][colIdx] || 0;
                                    const delta = (value || 0) - initialValue;
                                    
                                    return (
                                      <td 
                                        key={colIdx}
                                        style={{ 
                                          backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value || 0), 0.7)})`,
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
                        ) : (
                          <p className="text-muted">Updated weights data not available</p>
                        )}
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
          </Tab.Pane>

          {/* Detailed Calculation & Learning Rate Tab */}
          <Tab.Pane eventKey="detailed">
            <div className="detailed-gradient-calculation p-3 bg-light rounded">
              <h5 className="mb-3">Step-by-Step Convolution Gradient Calculation</h5>
              
              <div className="calculation-overview mb-4">
                <p>The gradient of the loss with respect to the convolution weights is calculated as:</p>
                <BlockMath math="\\frac{\\partial L}{\\partial W_{i,j}} = \\sum_{m=0}^{M-1} \\sum_{n=0}^{N-1} \\frac{\\partial L}{\\partial O_{m,n}} \\cdot X_{m+i, n+j}" />
                <p>Where:</p>
                <ul>
                  <li><InlineMath math="O_{m,n}" /> is the output feature map at position (m,n)</li>
                  <li><InlineMath math="X_{m+i, n+j}" /> is the corresponding input feature value</li>
                  <li><InlineMath math="W_{i,j}" /> is the weight at position (i,j) in the kernel</li>
                </ul>
              </div>
              
              {/* Calculation Type Selector */}
              <div className="calculation-selector mb-4">
                <Form.Group>
                  <Form.Label><strong>Select calculation example:</strong></Form.Label>
                  <Form.Select 
                    value={calculationExample}
                    onChange={(e) => setCalculationExample(e.target.value)}
                    className="mb-3"
                  >
                    <option value="weightGrad">Weight Gradient Calculation</option>
                    <option value="weightUpdate">Weight Update Calculation</option>
                    <option value="forwardConv">Forward Convolution Calculation</option>
                  </Form.Select>
                </Form.Group>
              </div>
              
              {/* Detailed Step-by-Step Calculation Cards */}
              <div className="calculation-steps-display">
                <Card className="mb-4">
                  <Card.Header className="bg-primary text-white">
                    <h5 className="mb-0">Calculation Steps</h5>
                  </Card.Header>
                  <Card.Body>
                    {detailedCalculationExample[calculationExample].steps.map((step, index) => (
                      <div key={index} className="calculation-step">
                        <div className="step-container">
                          <div className="step-indicator">
                            <div className="step-number">{index + 1}</div>
                          </div>
                          <div className="step-content">
                            <h6>{step.title}</h6>
                            <div className="step-formula">
                              <BlockMath math={step.formula} />
                            </div>
                          </div>
                        </div>
                        {index < detailedCalculationExample[calculationExample].steps.length - 1 && (
                          <div className="step-divider"></div>
                        )}
                      </div>
                    ))}
                    <div className="calculation-result">
                      <strong>Result:</strong> {calculationExample === 'forwardConv' ? '10.5' : 
                                              calculationExample === 'weightGrad' ? '1.55' : 
                                              formatValue(1.0 - learning_rate * 1.55)}
                    </div>
                  </Card.Body>
                </Card>
              </div>
              
              {/* Visualization of the calculation process */}
              <div className="calculation-visualization mt-4">
                <h5 className="mb-3">Visualization of the Calculation</h5>
                
                {calculationExample === 'weightGrad' && (
                  <Row>
                    <Col md={5}>
                      <div className="matrix-visualization">
                        <h6>Output Gradient Matrix</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [0.23, 0.15],
                              [0.18, 0.12]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(2)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">∂L/∂O</p>
                      </div>
                    </Col>
                    <Col md={2} className="d-flex align-items-center justify-content-center">
                      <div className="operation-symbol">⊗</div>
                    </Col>
                    <Col md={5}>
                      <div className="matrix-visualization">
                        <h6>Input Matrix</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [1.0, 2.0],
                              [3.0, 4.0]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value/4), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(1)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">Input X</p>
                      </div>
                    </Col>
                  </Row>
                )}
                
                {calculationExample === 'forwardConv' && (
                  <Row>
                    <Col md={4}>
                      <div className="matrix-visualization">
                        <h6>Input Matrix</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [1.0, 2.0],
                              [5.0, 6.0]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value/6), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(1)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">Input I</p>
                      </div>
                    </Col>
                    <Col md={1} className="d-flex align-items-center justify-content-center">
                      <div className="operation-symbol">*</div>
                    </Col>
                    <Col md={4}>
                      <div className="matrix-visualization">
                        <h6>Weight Kernel</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [1.0, 0.5],
                              [0.5, 1.0]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(255, 193, 7, ${Math.min(Math.abs(value), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(1)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">Weights W</p>
                      </div>
                    </Col>
                    <Col md={1} className="d-flex align-items-center justify-content-center">
                      <div className="operation-symbol">=</div>
                    </Col>
                    <Col md={2}>
                      <div className="matrix-visualization">
                        <h6>Output</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [10.5]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(0, 123, 255, ${Math.min(Math.abs(value/10), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(1)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">Output O</p>
                      </div>
                    </Col>
                  </Row>
                )}
                
                {calculationExample === 'weightUpdate' && (
                  <Row>
                    <Col md={4}>
                      <div className="matrix-visualization">
                        <h6>Initial Weight</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [1.0]
                            ].map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {row.map((value, colIdx) => (
                                  <td 
                                    key={colIdx}
                                    style={{ 
                                      backgroundColor: `rgba(40, 167, 69, ${Math.min(Math.abs(value), 0.7)})`
                                    }}
                                  >
                                    {value.toFixed(1)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        <p className="text-center mt-2">W<sub>old</sub></p>
                      </div>
                    </Col>
                    <Col md={1} className="d-flex align-items-center justify-content-center">
                      <div className="operation-symbol">-</div>
                    </Col>
                    <Col md={4}>
                      <div className="matrix-visualization">
                        <h6>Learning Rate × Gradient</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [learning_rate * 1.55]
                            ].map((row, rowIdx) => (
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
                        <p className="text-center mt-2">η · ∂L/∂W</p>
                      </div>
                    </Col>
                    <Col md={1} className="d-flex align-items-center justify-content-center">
                      <div className="operation-symbol">=</div>
                    </Col>
                    <Col md={2}>
                      <div className="matrix-visualization">
                        <h6>Updated Weight</h6>
                        <table className="tensor-table mx-auto">
                          <tbody>
                            {[
                              [1.0 - learning_rate * 1.55]
                            ].map((row, rowIdx) => (
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
                        <p className="text-center mt-2">W<sub>new</sub></p>
                      </div>
                    </Col>
                  </Row>
                )}
              </div>
              
              {/* Learning Rate Impact Section */}
              <div className="learning-rate-impact mt-5 p-3 bg-light rounded">
                <h5 className="mb-3">Learning Rate Impact on Weight Updates</h5>
                
                <div className="learning-rate-explanation mb-4">
                  <p>
                    The learning rate (η) controls the step size during gradient descent. 
                    It significantly impacts the training process - too small causes slow convergence, 
                    too large can cause divergence or oscillation.
                  </p>
                  <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
                </div>
                
                {weightGradData.length > 0 && initialWeightsData.length > 0 ? (
                  <Row>
                    <Col md={4}>
                      <div className="learning-rate-scenario p-3 border rounded">
                        <h6 className="text-center">Small Learning Rate (η = 0.001)</h6>
                        <div className="weight-scenario">
                          <p><strong>Initial Weight:</strong> {formatValue(initialWeightsData[0] && initialWeightsData[0][0])}</p>
                          <p><strong>Weight Gradient:</strong> {formatValue(weightGradData[0] && weightGradData[0][0])}</p>
                          <p><strong>Update Amount:</strong> {formatValue((weightGradData[0] && weightGradData[0][0] || 0) * 0.001)}</p>
                          <div className="updated-weight">
                            <p><strong>New Weight:</strong> {formatValue((initialWeightsData[0] && initialWeightsData[0][0] || 0) - (weightGradData[0] && weightGradData[0][0] || 0) * 0.001)}</p>
                          </div>
                        </div>
                        <div className="scenario-comment mt-3">
                          <p className="text-muted">
                            Small learning rate results in minimal weight changes, potentially slow convergence but stable training.
                          </p>
                        </div>
                      </div>
                    </Col>
                    
                    <Col md={4}>
                      <div className="learning-rate-scenario p-3 border rounded border-primary">
                        <h6 className="text-center">Current Learning Rate (η = {learning_rate})</h6>
                        <div className="weight-scenario">
                          <p><strong>Initial Weight:</strong> {formatValue(initialWeightsData[0] && initialWeightsData[0][0])}</p>
                          <p><strong>Weight Gradient:</strong> {formatValue(weightGradData[0] && weightGradData[0][0])}</p>
                          <p><strong>Update Amount:</strong> {formatValue((weightGradData[0] && weightGradData[0][0] || 0) * (learning_rate || 0))}</p>
                          <div className="updated-weight bg-light p-2 rounded">
                            <p><strong>New Weight:</strong> {formatValue(updatedWeightsData[0] && updatedWeightsData[0][0])}</p>
                          </div>
                        </div>
                        <div className="scenario-comment mt-3">
                          <p className="text-muted">
                            Current learning rate provides a balanced update - significant enough for learning but controlled for stability.
                          </p>
                        </div>
                      </div>
                    </Col>
                    
                    <Col md={4}>
                      <div className="learning-rate-scenario p-3 border rounded">
                        <h6 className="text-center">Large Learning Rate (η = 0.1)</h6>
                        <div className="weight-scenario">
                          <p><strong>Initial Weight:</strong> {formatValue(initialWeightsData[0] && initialWeightsData[0][0])}</p>
                          <p><strong>Weight Gradient:</strong> {formatValue(weightGradData[0] && weightGradData[0][0])}</p>
                          <p><strong>Update Amount:</strong> {formatValue((weightGradData[0] && weightGradData[0][0] || 0) * 0.1)}</p>
                          <div className="updated-weight">
                            <p><strong>New Weight:</strong> {formatValue((initialWeightsData[0] && initialWeightsData[0][0] || 0) - (weightGradData[0] && weightGradData[0][0] || 0) * 0.1)}</p>
                          </div>
                        </div>
                        <div className="scenario-comment mt-3">
                          <p className="text-muted">
                            Large learning rate causes dramatic weight changes, potentially faster convergence but risk of overshooting or divergence.
                          </p>
                        </div>
                      </div>
                    </Col>
                  </Row>
                ) : (
                  <Alert variant="info">
                    Learning rate comparison requires weight gradient and initial weights data.
                  </Alert>
                )}
              </div>
            </div>
          </Tab.Pane>
        </Tab.Content>
      </Tab.Container>

      {/* Research References Section */}
      <Row className="mt-5">
        <Col md={12}>
          <div className="research-references p-3 bg-light rounded">
            <h5>Advanced Research References</h5>
            <div className="references-list">
              <ol>
                <li>
                  <strong>Backpropagation in Convolutional Neural Networks</strong><br/>
                  LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
                  <p className="text-muted">Foundational paper that introduced modern CNNs and their training via backpropagation.</p>
                </li>
                
                <li>
                  <strong>Detailed Convolution Arithmetic</strong><br/>
                  Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
                  <p className="text-muted">Comprehensive guide on convolution operations, transpose convolutions, and their mathematical properties.</p>
                </li>
                
                <li>
                  <strong>Understanding Backpropagation Algorithm</strong><br/>
                  Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.
                  <p className="text-muted">The original paper on backpropagation explaining the fundamental concepts.</p>
                </li>
                
                <li>
                  <strong>Visualization of CNNs</strong><br/>
                  Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. European conference on computer vision (pp. 818-833).
                  <p className="text-muted">Pioneering work on visualizing what CNNs learn and how they operate.</p>
                </li>
              </ol>
            </div>
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
          width: 60px;
          height: 40px;
          transition: all 0.3s;
          font-size: 0.8rem;
          position: relative;
        }
        
        .tensor-table td.highlighted {
          box-shadow: 0 0 10px rgba(0, 123, 255, 0.8);
          z-index: 10;
          transform: scale(1.1);
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
        
        /* Step-by-step calculation styles */
        .calculation-step {
          position: relative;
          padding: 15px 0;
        }
        
        .step-container {
          display: flex;
          align-items: flex-start;
        }
        
        .step-indicator {
          flex-shrink: 0;
          margin-right: 15px;
        }
        
        .step-number {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 30px;
          height: 30px;
          border-radius: 50%;
          background-color: #28a745;
          color: white;
          font-weight: bold;
        }
        
        .step-content {
          flex-grow: 1;
        }
        
        .step-divider {
          position: absolute;
          left: 15px;
          bottom: 0;
          width: 1px;
          height: 15px;
          background-color: #dee2e6;
        }
        
        .calculation-result {
          margin-top: 20px;
          padding: 10px;
          background-color: #e9f7ef;
          border-radius: 5px;
          font-size: 1.1rem;
          text-align: center;
        }
        
        /* Matrix visualization styles */
        .matrix-visualization {
          text-align: center;
          margin-bottom: 20px;
        }
        
        .operation-symbol {
          font-size: 2rem;
          font-weight: bold;
          color: #6c757d;
          text-align: center;
        }
      `}</style>
    </div>
  );
};

export default ConvBackpropVisualizer;