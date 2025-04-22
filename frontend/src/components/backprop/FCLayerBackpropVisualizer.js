import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

/**
 * Component for visualizing backpropagation in a fully connected layer
 */
const FCLayerBackpropVisualizer = ({ backward, gradients, initial_weights, updated_weights, learning_rate }) => {
  const [selectedOutput, setSelectedOutput] = useState(0);
  const [selectedWeight, setSelectedWeight] = useState({ i: 0, j: 0 });
  const [decimalPlaces, setDecimalPlaces] = useState(6); // 소수점 자리 수 증가
  
  // Weight dimensions
  const outputSize = backward.weight_grad.length;
  const inputSize = backward.weight_grad[0].length;
  
  // Format values with scientific notation for very small numbers
  const formatValue = (value) => {
    if (Math.abs(value) < 0.000001) {
      return value.toExponential(decimalPlaces - 1);
    }
    return value.toFixed(decimalPlaces);
  };
  
  // Handle output neuron selection change
  const handleOutputChange = (event) => {
    setSelectedOutput(parseInt(event.target.value));
  };
  
  // Next output handler
  const handleNextOutput = () => {
    setSelectedOutput((prev) => (prev + 1) % outputSize);
  };
  
  // Previous output handler
  const handlePrevOutput = () => {
    setSelectedOutput((prev) => (prev - 1 + outputSize) % outputSize);
  };
  
  // Handle weight selection change
  const handleWeightChange = (event) => {
    const [i, j] = event.target.value.split(',').map(Number);
    setSelectedWeight({ i, j });
  };
  
  // Calculate weight gradient for the selected position
  const calculateWeightGradient = () => {
    const { i, j } = selectedWeight;
    const dldy = backward.output_grad[0][i];
    const x = backward.input_grad[0][j]; // Using input for demonstration
    const weightGrad = backward.weight_grad[i][j];
    
    const steps = [
      {
        description: `Formula for weight gradient at position (${i},${j})`,
        equation: `\\frac{\\partial L}{\\partial W_{${i},${j}}} = \\frac{\\partial L}{\\partial y_{${i}}} \\cdot x_{${j}}`,
      },
      {
        description: "Substituting values",
        equation: `\\frac{\\partial L}{\\partial W_{${i},${j}}} = ${formatValue(dldy)} \\cdot ${formatValue(x)}`,
      },
      {
        description: "Result",
        equation: `\\frac{\\partial L}{\\partial W_{${i},${j}}} = ${formatValue(weightGrad)}`,
        result: formatValue(weightGrad)
      }
    ];
    
    return steps;
  };
  
  // Calculate weight update for the selected position
  const calculateWeightUpdate = () => {
    const { i, j } = selectedWeight;
    const weightGrad = backward.weight_grad[i][j];
    const oldWeight = initial_weights.fc_weight[i][j];
    const newWeight = updated_weights.fc_weight[i][j];
    
    const steps = [
      {
        description: `Formula for weight update at position (${i},${j})`,
        equation: `W_{${i},${j}}^{new} = W_{${i},${j}}^{old} - \\eta \\cdot \\frac{\\partial L}{\\partial W_{${i},${j}}}`,
      },
      {
        description: "Substituting values",
        equation: `W_{${i},${j}}^{new} = ${formatValue(oldWeight)} - ${learning_rate} \\cdot ${formatValue(weightGrad)}`,
      },
      {
        description: "Result",
        equation: `W_{${i},${j}}^{new} = ${formatValue(newWeight)}`,
        result: formatValue(newWeight)
      }
    ];
    
    return steps;
  };
  
  // Generate steps for input gradient calculation
  const calculateInputGradient = () => {
    const selectedInputIndex = selectedWeight.j;
    
    // For the selected input, calculate the gradient contribution from all outputs
    const steps = [
      {
        description: `Formula for input gradient at position ${selectedInputIndex}`,
        equation: `\\frac{\\partial L}{\\partial x_{${selectedInputIndex}}} = \\sum_{i} \\frac{\\partial L}{\\partial y_{i}} \\cdot W_{i,${selectedInputIndex}}`,
      }
    ];
    
    // Build equation with actual values
    let equation = `\\frac{\\partial L}{\\partial x_{${selectedInputIndex}}} = `;
    let terms = [];
    
    for (let i = 0; i < outputSize; i++) {
      const dldy = backward.output_grad[0][i];
      const weight = initial_weights.fc_weight[i][selectedInputIndex];
      terms.push(`${formatValue(dldy)} \\cdot ${formatValue(weight)}`);
    }
    
    steps.push({
      description: "Substituting values",
      equation: equation + terms.join(" + "),
    });
    
    // Calculate the result
    const inputGrad = backward.input_grad[0][selectedInputIndex];
    
    steps.push({
      description: "Result",
      equation: `\\frac{\\partial L}{\\partial x_{${selectedInputIndex}}} = ${formatValue(inputGrad)}`,
      result: formatValue(inputGrad)
    });
    
    return steps;
  };
  
  const weightGradientSteps = calculateWeightGradient();
  const weightUpdateSteps = calculateWeightUpdate();
  const inputGradientSteps = calculateInputGradient();
  
  return (
    <div className="fc-backprop-visualizer">
      <h5 className="mb-4">Fully Connected Layer Backpropagation Visualization</h5>
      
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
      
      {/* Layer Gradient Flow Visualization */}
      <div className="gradient-flow-visualization mb-4">
        <h6>Gradient Flow Visualization</h6>
        <Row>
          <Col md={12}>
            <div className="network-visualization p-3 bg-light rounded">
              <div className="network-container">
                {/* Output Gradient Layer */}
                <div className="layer output-layer">
                  <div className="layer-label">Output Gradients</div>
                  {Array.from({ length: outputSize }).map((_, idx) => (
                    <div 
                      key={idx} 
                      className={`neuron ${idx === selectedOutput ? 'selected' : ''}`}
                      style={{
                        backgroundColor: `rgba(220, 53, 69, ${Math.abs(backward.output_grad[0][idx]) * 5})`
                      }}
                      onClick={() => setSelectedOutput(idx)}
                    >
                      <div className="neuron-value">{formatValue(backward.output_grad[0][idx])}</div>
                      <div className="neuron-label">∂L/∂y<sub>{idx}</sub></div>
                    </div>
                  ))}
                </div>
                
                {/* Weight Gradient Connections */}
                <div className="connections">
                  <svg width="100%" height="100%" className="connection-svg">
                    {/* Draw lines from each output to each input */}
                    {Array.from({ length: outputSize }).map((_, i) => (
                      Array.from({ length: inputSize }).map((_, j) => (
                        <line 
                          key={`${i}-${j}`}
                          x1="0%" 
                          y1={`${(100 / outputSize) * (i + 0.5)}%`}
                          x2="100%" 
                          y2={`${(100 / inputSize) * (j + 0.5)}%`}
                          stroke={
                            i === selectedWeight.i && j === selectedWeight.j 
                              ? "#ffc107" 
                              : (i === selectedOutput ? "#dc3545" : "#adb5bd")
                          }
                          strokeWidth={
                            i === selectedWeight.i && j === selectedWeight.j 
                              ? 3
                              : (i === selectedOutput ? 2 : 1)
                          }
                          strokeOpacity="0.6"
                          onClick={() => setSelectedWeight({ i, j })}
                          style={{ cursor: 'pointer' }}
                        />
                      ))
                    ))}
                  </svg>
                  
                  {/* Weight Gradient Labels */}
                  {Array.from({ length: outputSize }).map((_, i) => (
                    Array.from({ length: inputSize }).map((_, j) => (
                      i === selectedOutput ? (
                        <div 
                          key={`label-${i}-${j}`}
                          className="weight-gradient-label"
                          style={{
                            top: `${(100 / outputSize) * (i + 0.5)}%`,
                            left: `${50 + (j - inputSize/2 + 0.5) * 15}%`,
                            transform: 'translate(-50%, -50%)',
                            backgroundColor: i === selectedWeight.i && j === selectedWeight.j ? '#ffc107' : 'rgba(220, 53, 69, 0.1)',
                            border: i === selectedWeight.i && j === selectedWeight.j ? '2px solid #ffc107' : 'none'
                          }}
                        >
                          {formatValue(backward.weight_grad[i][j])}
                        </div>
                      ) : null
                    ))
                  ))}
                </div>
                
                {/* Input Gradient Layer */}
                <div className="layer input-layer">
                  <div className="layer-label">Input Gradients</div>
                  {Array.from({ length: inputSize }).map((_, idx) => (
                    <div 
                      key={idx} 
                      className={`neuron ${idx === selectedWeight.j ? 'selected' : ''}`}
                      style={{
                        backgroundColor: `rgba(0, 123, 255, ${Math.abs(backward.input_grad[0][idx]) * 10})`
                      }}
                      onClick={() => setSelectedWeight({ i: selectedWeight.i, j: idx })}
                    >
                      <div className="neuron-value">{formatValue(backward.input_grad[0][idx])}</div>
                      <div className="neuron-label">∂L/∂x<sub>{idx}</sub></div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Col>
        </Row>
      </div>
      
      {/* Weight Selection Controls */}
      <Row className="mb-3 align-items-center">
        <Col md={6}>
          <Form.Group>
            <Form.Label>Select output neuron:</Form.Label>
            <Form.Select 
              value={selectedOutput}
              onChange={handleOutputChange}
            >
              {Array.from({ length: outputSize }).map((_, idx) => (
                <option key={idx} value={idx}>
                  Output Neuron {idx}
                </option>
              ))}
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={6}>
          <Form.Group>
            <Form.Label>Select weight:</Form.Label>
            <Form.Select 
              value={`${selectedWeight.i},${selectedWeight.j}`}
              onChange={handleWeightChange}
            >
              {Array.from({ length: outputSize }).map((_, i) => (
                Array.from({ length: inputSize }).map((_, j) => (
                  <option key={`${i}-${j}`} value={`${i},${j}`}>
                    Weight W[{i},{j}]
                  </option>
                ))
              )).flat()}
            </Form.Select>
          </Form.Group>
        </Col>
      </Row>
      
      {/* Calculation Visualizations */}
      <Row className="mb-4">
        <Col md={6}>
          <div className="p-3 bg-light rounded">
            <h6>Weight Gradient Calculation</h6>
            <div className="calculation-steps">
              {weightGradientSteps.map((step, index) => (
                <div key={index} className="calculation-step mb-3">
                  <p><strong>{index + 1}. {step.description}</strong></p>
                  <div className="equation py-2">
                    <BlockMath math={step.equation} />
                  </div>
                  {step.result && (
                    <div className="result">
                      <p><strong>Result:</strong> {step.result}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Col>
        <Col md={6}>
          <div className="p-3 bg-light rounded">
            <h6>Weight Update Calculation</h6>
            <div className="calculation-steps">
              {weightUpdateSteps.map((step, index) => (
                <div key={index} className="calculation-step mb-3">
                  <p><strong>{index + 1}. {step.description}</strong></p>
                  <div className="equation py-2">
                    <BlockMath math={step.equation} />
                  </div>
                  {step.result && (
                    <div className="result">
                      <p><strong>Result:</strong> {step.result}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Col>
      </Row>
      
      {/* Input Gradient Calculation */}
      <Row className="mb-4">
        <Col md={12}>
          <div className="p-3 bg-light rounded">
            <h6>Input Gradient Calculation</h6>
            <div className="calculation-steps">
              {inputGradientSteps.map((step, index) => (
                <div key={index} className="calculation-step mb-3">
                  <p><strong>{index + 1}. {step.description}</strong></p>
                  <div className="equation py-2">
                    <BlockMath math={step.equation} />
                  </div>
                  {step.result && (
                    <div className="result">
                      <p><strong>Result:</strong> {step.result}</p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </Col>
      </Row>
      
      {/* Weight Matrices Visualization */}
      <Row className="mb-4">
        <Col md={4}>
          <div className="p-3 bg-light rounded h-100">
            <h6>Initial Weights</h6>
            <div className="matrix-container">
              <table className="matrix-table mx-auto">
                <tbody>
                  {initial_weights.fc_weight.map((row, i) => (
                    <tr key={i}>
                      {row.map((value, j) => (
                        <td 
                          key={j}
                          className={i === selectedWeight.i && j === selectedWeight.j ? 'selected' : ''}
                        >
                          {formatValue(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Col>
        <Col md={4}>
          <div className="p-3 bg-light rounded h-100">
            <h6>Weight Gradients</h6>
            <div className="matrix-container">
              <table className="matrix-table mx-auto">
                <tbody>
                  {backward.weight_grad.map((row, i) => (
                    <tr key={i}>
                      {row.map((value, j) => (
                        <td 
                          key={j}
                          className={i === selectedWeight.i && j === selectedWeight.j ? 'selected' : ''}
                          style={{ 
                            backgroundColor: `rgba(220, 53, 69, ${Math.min(Math.abs(value), 0.3)})`
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
          </div>
        </Col>
        <Col md={4}>
          <div className="p-3 bg-light rounded h-100">
            <h6>Updated Weights</h6>
            <div className="matrix-container">
              <table className="matrix-table mx-auto">
                <tbody>
                  {updated_weights.fc_weight.map((row, i) => (
                    <tr key={i}>
                      {row.map((value, j) => (
                        <td 
                          key={j}
                          className={i === selectedWeight.i && j === selectedWeight.j ? 'selected' : ''}
                        >
                          {formatValue(value)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </Col>
      </Row>
      
      {/* Weight Update Animation */}
      <Row>
        <Col md={12}>
          <div className="p-3 bg-light rounded">
            <h6>Weight Update Animation</h6>
            <div className="weight-update-animation">
              <div className="animation-container text-center p-4">
                <div className="d-flex justify-content-between align-items-center">
                  <div className="old-weight px-4 py-2 rounded" style={{ backgroundColor: '#f8f9fa', border: '1px solid #dee2e6' }}>
                    <div><strong>Old Weight</strong></div>
                    <div>W<sub>{selectedWeight.i},{selectedWeight.j}</sub><sup>old</sup> = {formatValue(initial_weights.fc_weight[selectedWeight.i][selectedWeight.j])}</div>
                  </div>
                  
                  <div className="gradient-part px-4 py-2 mx-3">
                    <div><strong>Learning Rate × Gradient</strong></div>
                    <div>η · ∂L/∂W<sub>{selectedWeight.i},{selectedWeight.j}</sub> = {learning_rate} · {formatValue(backward.weight_grad[selectedWeight.i][selectedWeight.j])} = {formatValue(learning_rate * backward.weight_grad[selectedWeight.i][selectedWeight.j])}</div>
                  </div>
                  
                  <div className="new-weight px-4 py-2 rounded" style={{ backgroundColor: '#e2f0d9', border: '1px solid #c5e0b4' }}>
                    <div><strong>New Weight</strong></div>
                    <div>W<sub>{selectedWeight.i},{selectedWeight.j}</sub><sup>new</sup> = {formatValue(updated_weights.fc_weight[selectedWeight.i][selectedWeight.j])}</div>
                  </div>
                </div>
                
                <div className="update-equation mt-3">
                  <BlockMath math={`W_{${selectedWeight.i},${selectedWeight.j}}^{new} = W_{${selectedWeight.i},${selectedWeight.j}}^{old} - \\eta \\cdot \\frac{\\partial L}{\\partial W_{${selectedWeight.i},${selectedWeight.j}}} = ${formatValue(initial_weights.fc_weight[selectedWeight.i][selectedWeight.j])} - ${learning_rate} \\cdot ${formatValue(backward.weight_grad[selectedWeight.i][selectedWeight.j])} = ${formatValue(updated_weights.fc_weight[selectedWeight.i][selectedWeight.j])}`} />
                </div>
              </div>
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
              <li>Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. Nature, 323(6088), 533-536.</li>
              <li>LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.</li>
              <li>Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.</li>
            </ul>
          </div>
        </Col>
      </Row>
      
      <style jsx>{`
        .network-container {
          display: flex;
          height: 300px;
          position: relative;
        }
        
        .layer {
          display: flex;
          flex-direction: column;
          justify-content: space-around;
          width: 20%;
          position: relative;
        }
        
        .layer-label {
          position: absolute;
          top: -30px;
          width: 100%;
          text-align: center;
          font-weight: bold;
        }
        
        .output-layer {
          margin-right: 10%;
        }
        
        .input-layer {
          margin-left: 10%;
        }
        
        .connections {
          flex: 1;
          position: relative;
        }
        
        .connection-svg {
          position: absolute;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
        }
        
        .neuron {
          width: 70px;
          height: 70px;
          border-radius: 50%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          margin: 5px auto;
          border: 1px solid #dee2e6;
          background-color: #f8f9fa;
          position: relative;
          transition: all 0.3s;
          cursor: pointer;
        }
        
        .neuron.selected {
          border: 3px solid #ffc107;
          box-shadow: 0 0 10px rgba(255, 193, 7, 0.5);
        }
        
        .neuron-value {
          font-weight: bold;
          font-size: 0.8rem;
        }
        
        .neuron-label {
          font-size: 0.8rem;
          color: #6c757d;
        }
        
        .weight-gradient-label {
          position: absolute;
          font-size: 0.75rem;
          background-color: rgba(255, 255, 255, 0.8);
          padding: 2px 5px;
          border-radius: 3px;
          z-index: 10;
          transition: all 0.3s;
        }
        
        .calculation-step {
          border-bottom: 1px dashed #dee2e6;
          padding-bottom: 10px;
        }
        
        .calculation-step:last-child {
          border-bottom: none;
        }
        
        .matrix-table {
          border-collapse: collapse;
        }
        
        .matrix-table td {
          border: 1px solid #dee2e6;
          padding: 5px 8px;
          text-align: center;
          transition: all 0.3s;
          font-size: 0.8rem;
        }
        
        .matrix-table td.selected {
          border: 2px solid #ffc107;
          background-color: rgba(255, 193, 7, 0.2);
          font-weight: bold;
        }
      `}</style>
    </div>
  );
};

export default FCLayerBackpropVisualizer;