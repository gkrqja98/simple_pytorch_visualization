import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';

/**
 * Component for visualizing ReLU operations at different positions
 */
const ReluVisualizer = ({ inputTensor, outputTensor }) => {
  const [selectedPosition, setSelectedPosition] = useState({ i: 0, j: 0 });
  
  // Get tensor dimensions
  const height = inputTensor.length;
  const width = inputTensor[0].length;
  
  // Create array of valid positions
  const validPositions = [];
  for (let i = 0; i < height; i++) {
    for (let j = 0; j < width; j++) {
      validPositions.push({ i, j });
    }
  }
  
  // Generate calculation steps for selected position
  const calculateSteps = () => {
    const { i, j } = selectedPosition;
    const inputValue = inputTensor[i][j];
    const outputValue = outputTensor[i][j];
    
    const steps = [
      {
        description: `ReLU formula for position (${i},${j})`,
        equation: `ReLU(I_{${i},${j}}) = \\max(0, I_{${i},${j}})`,
      },
      {
        description: "Substituting the actual value",
        equation: `ReLU(${inputValue.toFixed(1)}) = \\max(0, ${inputValue.toFixed(1)})`,
      },
      {
        description: "Final result",
        equation: inputValue > 0 
          ? `ReLU(${inputValue.toFixed(1)}) = ${inputValue.toFixed(1)}` 
          : `ReLU(${inputValue.toFixed(1)}) = 0`,
        result: outputValue.toFixed(1)
      }
    ];
    
    return steps;
  };
  
  const steps = calculateSteps();
  
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
    <div className="relu-visualizer">
      <h6 className="mb-3">ReLU Calculation Visualization</h6>
      
      <Row className="mb-3 align-items-center">
        <Col md={6}>
          <Form.Group>
            <Form.Label>Select position:</Form.Label>
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
            <strong>Current Position:</strong> ({selectedPosition.i}, {selectedPosition.j})
          </div>
          <div>
            <strong>Input Value:</strong> {inputTensor[selectedPosition.i][selectedPosition.j].toFixed(1)}
            <span className="mx-2">→</span>
            <strong>Output Value:</strong> {outputTensor[selectedPosition.i][selectedPosition.j].toFixed(1)}
          </div>
        </div>
      </div>
      
      <div className="tensor-visualization mb-4">
        <Row>
          <Col md={6}>
            <div className="tensor-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Input Tensor</h6>
              <table className="tensor-table mx-auto">
                <tbody>
                  {inputTensor.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          className={(rowIdx === selectedPosition.i && colIdx === selectedPosition.j) ? "selected" : ""}
                          style={{ 
                            backgroundColor: value > 0 ? `rgba(0, 123, 255, ${Math.min(value / 10, 0.7)})` : 'rgba(220, 53, 69, 0.2)' 
                          }}
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
          
          <Col md={6}>
            <div className="tensor-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Output Tensor (After ReLU)</h6>
              <table className="tensor-table mx-auto">
                <tbody>
                  {outputTensor.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td 
                          key={colIdx}
                          className={(rowIdx === selectedPosition.i && colIdx === selectedPosition.j) ? "selected" : ""}
                          style={{ 
                            backgroundColor: value > 0 ? `rgba(40, 167, 69, ${Math.min(value / 10, 0.7)})` : 'rgba(239, 239, 239, 0.5)' 
                          }}
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

export default ReluVisualizer;
