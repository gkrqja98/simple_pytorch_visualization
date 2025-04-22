import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';

/**
 * Component for visualizing backpropagation in a MaxPool layer
 */
const MaxPoolBackpropVisualizer = ({ backward }) => {
  const [selectedPosition, setSelectedPosition] = useState({ i: 0, j: 0 });
  const [decimalPlaces, setDecimalPlaces] = useState(6); // 소수점 자리 수 증가
  
  // Get tensor dimensions - safely access properties
  const outputGradHeight = backward?.output_grad?.[0]?.[0]?.length || 0;
  const outputGradWidth = backward?.output_grad?.[0]?.[0]?.[0]?.length || 0;
  const inputGradHeight = backward?.input_grad?.[0]?.[0]?.length || 0;
  const inputGradWidth = backward?.input_grad?.[0]?.[0]?.[0]?.length || 0;
  
  // Create array of valid output positions
  const validPositions = [];
  for (let i = 0; i < outputGradHeight; i++) {
    for (let j = 0; j < outputGradWidth; j++) {
      validPositions.push({ i, j });
    }
  }
  
  // Handle position selection change
  const handlePositionChange = (event) => {
    const [i, j] = event.target.value.split(',').map(Number);
    setSelectedPosition({ i, j });
  };
  
  // Next position handler
  const handleNextPosition = () => {
    const currentIndex = validPositions.findIndex(
      pos => pos.i === selectedPosition.i && pos.j === selectedPosition.j
    );
    
    if (currentIndex < validPositions.length - 1) {
      setSelectedPosition(validPositions[currentIndex + 1]);
    } else {
      setSelectedPosition(validPositions[0]); // Wrap to beginning
    }
  };
  
  // Previous position handler
  const handlePrevPosition = () => {
    const currentIndex = validPositions.findIndex(
      pos => pos.i === selectedPosition.i && pos.j === selectedPosition.j
    );
    
    if (currentIndex > 0) {
      setSelectedPosition(validPositions[currentIndex - 1]);
    } else {
      setSelectedPosition(validPositions[validPositions.length - 1]); // Wrap to end
    }
  };
  
  // Find corresponding pooling window for the selected position
  const getPoolingWindow = () => {
    const { i, j } = selectedPosition;
    // For a 2x2 pooling with stride 1, the window starts at (i,j) and extends 2x2
    return {
      startRow: i,
      startCol: j,
      endRow: i + 2, // Assuming kernel size is 2
      endCol: j + 2  // Assuming kernel size is 2
    };
  };
  
  const { startRow, startCol, endRow, endCol } = getPoolingWindow();
  
  // Find the max value position in the input window using indices from forward pass
  const findMaxValuePosition = () => {
    // Using indices from forward pass which were saved during forward pass
    if (!backward.indices) {
      return { maxRow: -1, maxCol: -1, flatIndex: -1 };
    }
    
    const { i, j } = selectedPosition;
    const poolIndices = backward.indices[0][0];
    
    // If we have the indices data, extract the relevant index
    if (i < poolIndices.length && j < poolIndices[0].length) {
      const flatIndex = poolIndices[i][j];
      
      // Convert the flat index back to 2D coordinates
      // For a 2x2 window with stride 1
      const windowWidth = 2;
      const maxRow = Math.floor(flatIndex / windowWidth) + i;
      const maxCol = (flatIndex % windowWidth) + j;
      
      return { maxRow, maxCol, flatIndex };
    }
    
    return { maxRow: -1, maxCol: -1, flatIndex: -1 };
  };
  
  const { maxRow, maxCol, flatIndex } = findMaxValuePosition();
  
  // Get the gradient values (safely)
  const outputGradValue = backward?.output_grad?.[0]?.[0]?.[selectedPosition.i]?.[selectedPosition.j] || 0;
  const inputGradValue = maxRow >= 0 && maxCol >= 0 && backward?.input_grad?.[0]?.[0]?.[maxRow]?.[maxCol] || 0;
  
  // Format values with scientific notation for very small numbers
  const formatValue = (value) => {
    if (Math.abs(value) < 0.000001) {
      return value.toExponential(decimalPlaces - 1);
    }
    return value.toFixed(decimalPlaces);
  };
  
  // Generate steps for gradient backpropagation explanation
  const generateGradientSteps = () => {
    const { i, j } = selectedPosition;
    
    const steps = [
      {
        description: `MaxPool gradient backpropagation for position (${i},${j})`,
        equation: `\\frac{\\partial L}{\\partial O_{${i},${j}}} = ${formatValue(outputGradValue)}`
      },
      {
        description: "MaxPool backpropagation rule",
        equation: `\\frac{\\partial L}{\\partial I_{r,c}} = \\begin{cases} 
          \\frac{\\partial L}{\\partial O_{i,j}} & \\text{if } I_{r,c} \\text{ was the maximum in the pooling region} \\\\
          0 & \\text{otherwise}
        \\end{cases}`
      }
    ];
    
    if (maxRow >= 0 && maxCol >= 0) {
      steps.push({
        description: `For the pooling window at (${startRow},${startCol}) to (${endRow-1},${endCol-1})`,
        equation: `\\text{Maximum value was at position } (${maxRow},${maxCol}) \\text{ (index: ${flatIndex})}`
      });
      
      // Show which position receives the gradient
      steps.push({
        description: `Gradient propagation to position (${maxRow},${maxCol})`,
        equation: `\\frac{\\partial L}{\\partial I_{${maxRow},${maxCol}}} = \\frac{\\partial L}{\\partial O_{${i},${j}}} = ${formatValue(outputGradValue)}`
      });
      
      steps.push({
        description: "Result",
        equation: `\\frac{\\partial L}{\\partial I_{${maxRow},${maxCol}}} = ${formatValue(inputGradValue)}`,
        result: formatValue(inputGradValue)
      });
    } else {
      steps.push({
        description: "Could not determine the max value position in this window",
        equation: "\\text{Gradient flow cannot be visualized}"
      });
    }
    
    return steps;
  };
  
  const gradientSteps = generateGradientSteps();
  
  // Check if we have all necessary data from backend
  if (!backward || !backward.output_grad || !backward.input_grad) {
    return (
      <div className="alert alert-warning">
        Missing MaxPool gradient data from backend. Please ensure the backend is properly calculating MaxPool gradients.
      </div>
    );
  }
  
  return (
    <div className="maxpool-backprop-visualizer">
      <h5 className="mb-4">MaxPool Backpropagation Visualization</h5>
      
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
      
      {/* Position Selection Controls */}
      <Row className="mb-3 align-items-center">
        <Col md={6}>
          <Form.Group>
            <Form.Label>Select output position:</Form.Label>
            <Form.Select 
              value={`${selectedPosition.i},${selectedPosition.j}`}
              onChange={handlePositionChange}
            >
              {validPositions.map((pos, idx) => (
                <option key={idx} value={`${pos.i},${pos.j}`}>
                  Position ({pos.i}, {pos.j})
                </option>
              ))}
            </Form.Select>
          </Form.Group>
        </Col>
        <Col md={6} className="d-flex justify-content-end">
          <Button 
            variant="outline-secondary" 
            size="sm" 
            onClick={handlePrevPosition}
            className="me-2"
          >
            &laquo; Previous Position
          </Button>
          <Button 
            variant="outline-secondary" 
            size="sm" 
            onClick={handleNextPosition}
          >
            Next Position &raquo;
          </Button>
        </Col>
      </Row>
      
      {/* Current Position Info */}
      <div className="position-indicator mb-3">
        <div className="d-flex align-items-center">
          <div className="me-3">
            <strong>Current Output Position:</strong> ({selectedPosition.i}, {selectedPosition.j})
          </div>
          <div>
            <strong>Output Gradient Value:</strong> {formatValue(outputGradValue)}
          </div>
        </div>
      </div>
      
      {/* Gradient Flow Visualization */}
      <Row className="mb-4">
        <Col md={12}>
          <div className="gradient-flow-visualization p-3 bg-light rounded">
            <h6 className="mb-3">Gradient Flow Visualization</h6>
            <Row>
              <Col md={6}>
                <h6 className="text-center">Output Gradient</h6>
                <div className="tensor-container position-relative">
                  <table className="tensor-table mx-auto">
                    <tbody>
                      {backward.output_grad[0][0].map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {row.map((value, colIdx) => (
                            <td 
                              key={colIdx}
                              className={(rowIdx === selectedPosition.i && colIdx === selectedPosition.j) ? "selected" : ""}
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
                  
                  {/* Highlight selected position */}
                  <div className="position-label">
                    <strong>∂L/∂O<sub>{selectedPosition.i},{selectedPosition.j}</sub></strong>
                  </div>
                </div>
              </Col>
              
              <Col md={6}>
                <h6 className="text-center">Input Gradient</h6>
                <div className="tensor-container position-relative">
                  <table className="tensor-table mx-auto">
                    <tbody>
                      {backward.input_grad[0][0].map((row, rowIdx) => (
                        <tr key={rowIdx}>
                          {row.map((value, colIdx) => {
                            // Determine if this cell is in the current pooling window
                            const isInWindow = (
                              rowIdx >= startRow && rowIdx < endRow &&
                              colIdx >= startCol && colIdx < endCol
                            );
                            
                            // Determine if this was the max value position
                            const isMaxPosition = (rowIdx === maxRow && colIdx === maxCol);
                            
                            return (
                              <td 
                                key={colIdx}
                                className={`
                                  ${isInWindow ? "in-window" : ""}
                                  ${isMaxPosition ? "max-position" : ""}
                                `}
                                style={{ 
                                  backgroundColor: Math.abs(value) > 0.0000001 
                                    ? `rgba(0, 123, 255, ${Math.min(Math.abs(value), 0.7)})` 
                                    : (isInWindow ? 'rgba(173, 181, 189, 0.2)' : '')
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
                  
                  {/* Pooling window indicator */}
                  <div className="window-indicator" style={{
                    top: `${(startRow * 40) + 40}px`,
                    left: `${(startCol * 40) + 40}px`,
                    width: `${(endCol - startCol) * 40}px`,
                    height: `${(endRow - startRow) * 40}px`
                  }}>
                  </div>
                  
                  {/* Arrow from output to max position */}
                  {maxRow >= 0 && maxCol >= 0 && (
                    <svg className="gradient-flow-arrow" width="100%" height="100%" style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
                      <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                          <polygon points="0 0, 10 3.5, 0 7" fill="#dc3545" />
                        </marker>
                      </defs>
                      <line 
                        x1="-50" 
                        y1={(selectedPosition.i + 0.5) * 40}
                        x2={(maxCol + 0.5) * 40} 
                        y2={(maxRow + 0.5) * 40}
                        stroke="#dc3545" 
                        strokeWidth="2" 
                        markerEnd="url(#arrowhead)"
                        strokeDasharray="5,3"
                      />
                    </svg>
                  )}
                </div>
              </Col>
            </Row>
            
            {/* Explanation */}
            <div className="gradient-flow-explanation mt-4">
              <p>
                <strong>Explanation:</strong> In MaxPool backpropagation, the gradient from the output flows only to the position 
                that contained the maximum value during the forward pass. All other positions receive zero gradient.
              </p>
              {maxRow >= 0 && maxCol >= 0 && (
                <p>
                  For the selected output position ({selectedPosition.i}, {selectedPosition.j}), 
                  the maximum value in the corresponding input window was at position ({maxRow}, {maxCol}). 
                  Therefore, this position receives the full gradient of {formatValue(outputGradValue)}, 
                  while all other positions in the window receive zero gradient.
                </p>
              )}
              <p className="small text-muted mt-2">
                Note: The maximum position is determined by the indices saved during the forward pass.
                Index {flatIndex} corresponds to position ({maxRow}, {maxCol}) in the input tensor.
              </p>
            </div>
          </div>
        </Col>
      </Row>
      
      {/* Calculation Steps */}
      <Row className="mb-4">
        <Col md={12}>
          <div className="calculation-steps p-3 bg-light rounded">
            <h6 className="mb-3">Gradient Backpropagation Steps</h6>
            {gradientSteps.map((step, index) => (
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
        </Col>
      </Row>
      
      {/* Animated Visualization */}
      <Row className="mb-4">
        <Col md={12}>
          <div className="animated-visualization p-3 bg-light rounded">
            <h6 className="mb-3">Animated Gradient Flow</h6>
            <div className="animation-container text-center p-4">
              <div className="gradient-animation">
                <div className="animation-step">
                  <h6>Step 1: Output Layer Gradient</h6>
                  <div className="output-gradient">
                    <div className="neuron">
                      <div className="value">
                        {formatValue(outputGradValue)}
                      </div>
                      <div className="label">
                        ∂L/∂O<sub>{selectedPosition.i},{selectedPosition.j}</sub>
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="animation-arrow">
                  <svg width="100" height="40">
                    <defs>
                      <marker id="arrow" markerWidth="10" markerHeight="7" refX="0" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#dc3545" />
                      </marker>
                    </defs>
                    <line x1="10" y1="20" x2="90" y2="20" stroke="#dc3545" strokeWidth="2" markerEnd="url(#arrow)" className="flow-arrow" />
                  </svg>
                </div>
                
                <div className="animation-step">
                  <h6>Step 2: MaxPool Window</h6>
                  <div className="maxpool-window">
                    <table className="window-table">
                      <tbody>
                        {Array.from({ length: 2 }).map((_, rowOffset) => (
                          <tr key={rowOffset}>
                            {Array.from({ length: 2 }).map((_, colOffset) => {
                              const r = startRow + rowOffset;
                              const c = startCol + colOffset;
                              const isMaxPosition = r === maxRow && c === maxCol;
                              const value = r < inputGradHeight && c < inputGradWidth 
                                ? backward.input_grad[0][0][r][c] 
                                : 0;
                              
                              return (
                                <td 
                                  key={colOffset}
                                  className={isMaxPosition ? "max-position" : ""}
                                >
                                  <div className="cell-content">
                                    <div className="position">({r},{c})</div>
                                    <div className="grad-value">{formatValue(value)}</div>
                                    {isMaxPosition && (
                                      <div className="max-label">MAX</div>
                                    )}
                                  </div>
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                
                <div className="animation-arrow">
                  <svg width="100" height="40">
                    <line x1="10" y1="20" x2="90" y2="20" stroke="#28a745" strokeWidth="2" markerEnd="url(#arrow)" className="result-arrow" />
                  </svg>
                </div>
                
                <div className="animation-step">
                  <h6>Step 3: Result</h6>
                  <div className="final-result">
                    {maxRow >= 0 && maxCol >= 0 ? (
                      <div>
                        <p>Only position ({maxRow},{maxCol}) receives gradient:</p>
                        <div className="result-value">
                          ∂L/∂I<sub>{maxRow},{maxCol}</sub> = {formatValue(inputGradValue)}
                        </div>
                        <p>All other positions receive zero gradient</p>
                      </div>
                    ) : (
                      <p>Could not determine max position in this window</p>
                    )}
                  </div>
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
              <li>Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. In International conference on artificial neural networks (pp. 92-101).</li>
              <li>Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.</li>
              <li>Graham, B. (2014). Fractional max-pooling. arXiv preprint arXiv:1412.6071.</li>
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
        }
        
        .tensor-table td.selected {
          border: 2px solid #ffc107;
          font-weight: bold;
        }
        
        .tensor-table td.in-window {
          border: 1px dashed #adb5bd;
        }
        
        .tensor-table td.max-position {
          border: 2px solid #28a745;
          font-weight: bold;
        }
        
        .tensor-container {
          position: relative;
          margin-bottom: 20px;
        }
        
        .window-indicator {
          position: absolute;
          border: 2px dashed #dc3545;
          pointer-events: none;
          z-index: 5;
          box-sizing: content-box;
          margin-top: -40px;
          margin-left: -40px;
        }
        
        .position-label {
          position: absolute;
          top: -30px;
          left: 50%;
          transform: translateX(-50%);
          background-color: rgba(255, 255, 255, 0.8);
          padding: 2px 5px;
          border-radius: 3px;
          font-size: 0.9rem;
        }
        
        .calculation-step {
          border-bottom: 1px dashed #dee2e6;
          padding-bottom: 10px;
        }
        
        .calculation-step:last-child {
          border-bottom: none;
        }
        
        .gradient-animation {
          display: flex;
          flex-direction: row;
          align-items: center;
          justify-content: space-around;
          flex-wrap: wrap;
        }
        
        .animation-step {
          flex: 1;
          min-width: 180px;
          padding: 10px;
          text-align: center;
        }
        
        .animation-arrow {
          width: 100px;
          flex-shrink: 0;
        }
        
        .neuron {
          width: 80px;
          height: 80px;
          border-radius: 50%;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          margin: 0 auto;
          border: 1px solid #dee2e6;
          background-color: rgba(220, 53, 69, 0.2);
          transition: all 0.3s;
        }
        
        .neuron .value {
          font-weight: bold;
          font-size: 0.8rem;
        }
        
        .neuron .label {
          font-size: 0.8rem;
          color: #6c757d;
        }
        
        .maxpool-window {
          display: inline-block;
        }
        
        .window-table {
          border-collapse: collapse;
          margin: 0 auto;
        }
        
        .window-table td {
          border: 1px solid #dee2e6;
          width: 80px;
          height: 80px;
          text-align: center;
          padding: 5px;
        }
        
        .window-table td.max-position {
          background-color: rgba(40, 167, 69, 0.2);
          border: 2px solid #28a745;
        }
        
        .cell-content {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100%;
        }
        
        .position {
          font-size: 0.8rem;
          color: #6c757d;
        }
        
        .grad-value {
          font-weight: bold;
          font-size: 0.8rem;
        }
        
        .max-label {
          font-size: 0.7rem;
          background-color: #28a745;
          color: white;
          padding: 1px 4px;
          border-radius: 3px;
          margin-top: 3px;
        }
        
        .final-result {
          padding: 10px;
          background-color: rgba(40, 167, 69, 0.1);
          border-radius: 5px;
        }
        
        .result-value {
          font-weight: bold;
          font-size: 1.1rem;
          margin: 10px 0;
        }
        
        .flow-arrow {
          animation: flowPulse 2s infinite;
        }
        
        .result-arrow {
          animation: resultPulse 2s infinite;
        }
        
        @keyframes flowPulse {
          0% { opacity: 0.3; }
          50% { opacity: 1; }
          100% { opacity: 0.3; }
        }
        
        @keyframes resultPulse {
          0% { opacity: 0.3; }
          50% { opacity: 1; }
          100% { opacity: 0.3; }
        }
      `}</style>
    </div>
  );
};

export default MaxPoolBackpropVisualizer;