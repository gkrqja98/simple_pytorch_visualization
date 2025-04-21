import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';

/**
 * Component for visualizing convolution operations at different positions
 */
const ConvolutionVisualizer = ({ inputTensor, kernel, outputTensor }) => {
  const [selectedPosition, setSelectedPosition] = useState({ i: 0, j: 0 });
  
  // Calculate input tensor and kernel dimensions
  const inputHeight = inputTensor.length;
  const inputWidth = inputTensor[0].length;
  const kernelHeight = kernel.length;
  const kernelWidth = kernel[0].length;
  
  // Calculate output tensor dimensions
  const outputHeight = inputHeight - kernelHeight + 1;
  const outputWidth = inputWidth - kernelWidth + 1;
  
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
    
    // Steps array (removed the general formula step)
    const steps = [];
    
    // Expanded formula for specific position
    let expandedFormula = `O_{${i},${j}} = `;
    let terms = [];
    
    for (let m = 0; m < kernelHeight; m++) {
      for (let n = 0; n < kernelWidth; n++) {
        terms.push(`I_{${i+m},${j+n}} \\cdot W_{${m},${n}}`);
      }
    }
    
    expandedFormula += terms.join(" + ");
    steps.push({
      description: `Expanded formula for the output at position (${i},${j})`,
      equation: expandedFormula,
    });
    
    // Substituting actual values
    let valuesFormula = `O_{${i},${j}} = `;
    let valueTerms = [];
    let sum = 0;
    
    for (let m = 0; m < kernelHeight; m++) {
      for (let n = 0; n < kernelWidth; n++) {
        const inputValue = inputTensor[i+m][j+n];
        const kernelValue = kernel[m][n];
        valueTerms.push(`${inputValue.toFixed(1)} \\cdot ${kernelValue.toFixed(1)}`);
        sum += inputValue * kernelValue;
      }
    }
    
    valuesFormula += valueTerms.join(" + ");
    steps.push({
      description: "Substituting the actual values",
      equation: valuesFormula,
    });
    
    // Final result (removed computing each term step)
    steps.push({
      description: "Final result",
      equation: `O_{${i},${j}} = ${sum.toFixed(1)}`,
      result: sum.toFixed(1)
    });
    
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
    <div className="convolution-visualizer">
      <h6 className="mb-3">Calculation Visualization</h6>
      
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
            <strong>Current Calculation Position:</strong> ({selectedPosition.i}, {selectedPosition.j})
          </div>
          <div>
            <strong>Output Value:</strong> {outputTensor[selectedPosition.i][selectedPosition.j].toFixed(1)}
          </div>
        </div>
      </div>
      
      <div className="calculation-visualization mb-4">
        <Row>
          <Col md={4}>
            <div className="tensor-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Input Tensor</h6>
              <div className="position-relative">
                <table className="conv-table mx-auto">
                  <tbody>
                    {inputTensor.map((row, rowIdx) => (
                      <tr key={rowIdx}>
                        {row.map((value, colIdx) => {
                          // Highlight convolution region for current selected position
                          const isInKernelRegion = (
                            rowIdx >= selectedPosition.i && 
                            rowIdx < selectedPosition.i + kernelHeight && 
                            colIdx >= selectedPosition.j && 
                            colIdx < selectedPosition.j + kernelWidth
                          );
                          
                          return (
                            <td 
                              key={colIdx}
                              className={isInKernelRegion ? "highlight" : ""}
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
            </div>
          </Col>
          
          <Col md={3}>
            <div className="kernel-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Kernel</h6>
              <table className="conv-table mx-auto">
                <tbody>
                  {kernel.map((row, rowIdx) => (
                    <tr key={rowIdx}>
                      {row.map((value, colIdx) => (
                        <td key={colIdx}>{value.toFixed(1)}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Col>
          
          <Col md={5}>
            <div className="output-container p-3 bg-light rounded">
              <h6 className="text-center mb-2">Output Tensor</h6>
              <table className="conv-table mx-auto">
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
        .conv-table {
          border-collapse: collapse;
        }
        
        .conv-table td {
          border: 1px solid #dee2e6;
          padding: 8px;
          text-align: center;
          width: 40px;
          height: 40px;
          transition: all 0.3s;
        }
        
        .conv-table td.highlight {
          background-color: rgba(0, 123, 255, 0.2);
        }
        
        .conv-table td.selected {
          background-color: rgba(40, 167, 69, 0.3);
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

export default ConvolutionVisualizer;
