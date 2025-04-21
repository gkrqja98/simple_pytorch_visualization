import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';

/**
 * Component for visualizing MaxPool operations at different positions
 */
const MaxPoolVisualizer = ({ inputTensor, outputTensor, kernelSize = 2, stride = 1 }) => {
  const [selectedPosition, setSelectedPosition] = useState({ i: 0, j: 0 });
  
  // Get tensor dimensions
  const inputHeight = inputTensor.length;
  const inputWidth = inputTensor[0].length;
  const outputHeight = outputTensor.length;
  const outputWidth = outputTensor[0].length;
  
  // Create array of valid output positions
  const validPositions = [];
  for (let i = 0; i < outputHeight; i++) {
    for (let j = 0; j < outputWidth; j++) {
      validPositions.push({ i, j });
    }
  }
  
  // Generate calculation steps for selected position
  const calculateSteps = () => {
    const { i, j } = selectedPosition;
    
    // Get input region for this output position
    const startRow = i * stride;
    const startCol = j * stride;
    const endRow = startRow + kernelSize;
    const endCol = startCol + kernelSize;
    
    // Collect values from the pooling region
    const regionValues = [];
    for (let r = startRow; r < endRow; r++) {
      for (let c = startCol; c < endCol; c++) {
        regionValues.push(inputTensor[r][c]);
      }
    }
    
    // Find maximum value
    const maxValue = Math.max(...regionValues);
    
    // Steps for calculation
    const steps = [
      {
        description: `MaxPool formula for position (${i},${j})`,
        equation: `O_{${i},${j}} = \\max_{window}(I_{r,c})`,
      },
      {
        description: `For the ${kernelSize}x${kernelSize} window at position (${startRow},${startCol})`,
        equation: `O_{${i},${j}} = \\max\\{${regionValues.map(v => v.toFixed(1)).join(", ")}\\}`,
      },
      {
        description: "Final result",
        equation: `O_{${i},${j}} = ${maxValue.toFixed(1)}`,
        result: maxValue.toFixed(1)
      }
    ];
    
    return {
      steps,
      startRow,
      startCol,
      endRow,
      endCol,
      maxValue
    };
  };
  
  const { steps, startRow, startCol, endRow, endCol, maxValue } = calculateSteps();
  
  // Position change handler
  const handlePositionChange = (event) => {
    const position = event.target.value.split(',');
    setSelectedPosition({ i: parseInt(position[0]), j: parseInt(position[1]) });
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
  
  return (
    <div className="maxpool-visualizer">
      <h6 className="mb-3">MaxPool Calculation Visualization</h6>
      
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
      
      <div className="position-indicator mb-3">
        <div className="d-flex align-items-center">
          <div className="me-3">
            <strong>Current Output Position:</strong> ({selectedPosition.i}, {selectedPosition.j})
          </div>
          <div>
            <strong>Input Window:</strong> ({startRow}:{endRow-1}, {startCol}:{endCol-1})
            <span className="mx-2">→</span>
            <strong>Output Value:</strong> {outputTensor[selectedPosition.i][selectedPosition.j].toFixed(1)}
          </div>
        </div>
      </div>
      
      <div className="tensor-visualization mb-4">
        <Row>
          <Col md={7}>
            <div className="tensor-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Input Tensor with Pooling Window</h6>
              <table className="tensor-table mx-auto">
                <tbody>
                  {inputTensor.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => {
                        // Check if this cell is in the current pooling window
                        const isInWindow = (
                          rowIdx >= startRow && rowIdx < endRow &&
                          colIdx >= startCol && colIdx < endCol
                        );
                        
                        // Check if this cell has the maximum value
                        const isMaxValue = isInWindow && Math.abs(value - maxValue) < 0.001;
                        
                        return (
                          <td 
                            key={colIdx}
                            className={`
                              ${isInWindow ? "in-window" : ""}
                              ${isMaxValue ? "max-value" : ""}
                            `}
                          >
                            {value.toFixed(1)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Col>
          
          <Col md={5}>
            <div className="tensor-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Output Tensor (After MaxPool)</h6>
              <table className="tensor-table mx-auto">
                <tbody>
                  {outputTensor.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          className={(rowIdx === selectedPosition.i && colIdx === selectedPosition.j) ? "selected" : ""}
                        >
                          {value.toFixed(1)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Col>
        </Row>
      </div>
      
      <div className="calculation-steps p-3 bg-light rounded">
        <h6 className="mb-3">Calculation Steps</h6>
        {steps.map((step, index) => (
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
        }
        
        .tensor-table td.in-window {
          background-color: rgba(0, 123, 255, 0.1);
        }
        
        .tensor-table td.max-value {
          background-color: rgba(40, 167, 69, 0.3);
          font-weight: bold;
        }
        
        .tensor-table td.selected {
          border: 2px solid #212529;
          font-weight: bold;
        }
        
        .calculation-step {
          border-bottom: 1px dashed #dee2e6;
          padding-bottom: 10px;
        }
        
        .calculation-step:last-child {
          border-bottom: none;
        }
      `}</style>
    </div>
  );
};

export default MaxPoolVisualizer;
