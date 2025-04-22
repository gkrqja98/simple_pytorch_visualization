import React, { useState } from 'react';
import { Row, Col, Form, Button, OverlayTrigger, Tooltip } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

const ConvGradientVisualizer = ({ outputGrad, inputTensor, weightGrad }) => {
  // 디버깅을 위한 콘솔 로그 추가
  console.log('Output Gradient:', outputGrad);
  console.log('Input Tensor:', inputTensor);
  console.log('Weight Gradient:', weightGrad);
  
  // 값의 포맷팅 함수 - 소수점 자릿수 조절 및 표시 최적화
  const formatValue = (value) => {
    // 전체 값은 정확한 값
    const fullPrecision = value.toFixed(6);
    
    // 화면에 표시될 값 - 크기에 따라 다르게 처리
    let displayValue;
    
    if (Math.abs(value) < 0.0001) {
      // 매우 작은 값은 과학적 표기법 사용
      displayValue = value.toExponential(2);
    } else if (Math.abs(value) < 0.01) {
      // 작은 값은 최대 4자리까지 표시
      displayValue = value.toFixed(4);
    } else {
      // 일반적인 값은 2자리까지 표시
      displayValue = value.toFixed(2);
    }
    
    return { displayValue, fullPrecision };
  };
  
  // State for the selected kernel gradient position
  const [selectedPosition, setSelectedPosition] = useState({ row: 0, col: 0 });
  
  // If essential data is missing, use sample data for demonstration
  const sampleInputTensor = [
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [9.0, 10.0, 11.0, 12.0],
    [13.0, 14.0, 15.0, 16.0]
  ];
  
  const sampleOutputGrad = [
    [0.2, 0.3, 0.1],
    [0.4, 0.5, 0.2],
    [0.1, 0.3, 0.2]
  ];
  
  const sampleWeightGrad = [
    [0.15, 0.25],
    [0.35, 0.45]
  ];
  
  // Use real data if available, otherwise use sample data
  const displayInputTensor = inputTensor || sampleInputTensor;
  const displayOutputGrad = outputGrad || sampleOutputGrad;
  const displayWeightGrad = weightGrad || sampleWeightGrad;
  
  // Get dimensions from display tensors
  const kernelHeight = displayWeightGrad.length;
  const kernelWidth = displayWeightGrad[0].length;
  
  // Create position options for the gradient weight tensor
  const positionOptions = [];
  for (let i = 0; i < kernelHeight; i++) {
    for (let j = 0; j < kernelWidth; j++) {
      positionOptions.push(`Position (${i}, ${j})`);
    }
  }
  
  // Handle position selection
  const handlePositionChange = (e) => {
    const value = e.target.value;
    const match = value.match(/Position \((\d+), (\d+)\)/);
    if (match) {
      setSelectedPosition({ row: parseInt(match[1]), col: parseInt(match[2]) });
    }
  };

  // Go to previous position
  const handlePreviousPosition = () => {
    let newRow = selectedPosition.row;
    let newCol = selectedPosition.col - 1;
    
    if (newCol < 0) {
      newCol = kernelWidth - 1;
      newRow = newRow - 1;
      if (newRow < 0) {
        newRow = kernelHeight - 1;
      }
    }
    
    setSelectedPosition({ row: newRow, col: newCol });
  };

  // Go to next position
  const handleNextPosition = () => {
    let newRow = selectedPosition.row;
    let newCol = selectedPosition.col + 1;
    
    if (newCol >= kernelWidth) {
      newCol = 0;
      newRow = newRow + 1;
      if (newRow >= kernelHeight) {
        newRow = 0;
      }
    }
    
    setSelectedPosition({ row: newRow, col: newCol });
  };

  // Calculate gradient for the selected position
  const calculateGradientSteps = () => {
    const { row, col } = selectedPosition;
    const steps = [];
    
    // Step 1: Expanded formula
    steps.push({
      description: `Expanded formula for the gradient at position (${row},${col})`,
      equation: "\\frac{\\partial L}{\\partial W_{m,n}} = \\sum_{i,j} \\frac{\\partial L}{\\partial O_{i,j}} \\cdot I_{i+m, j+n}"
    });
    
    // Step 2: Substituting values
    let equationWithValues = "\\frac{\\partial L}{\\partial W_{" + row + "," + col + "}} = ";
    let terms = [];
    let sum = 0;
    
    // For each position in the output gradient
    for (let i = 0; i < displayOutputGrad.length; i++) {
      for (let j = 0; j < displayOutputGrad[0].length; j++) {
        // Calculate the corresponding input position
        const inputRow = i + row;
        const inputCol = j + col;
        
        // Get input value if the position is valid
        let inputValue = 0;
        if (inputRow < displayInputTensor.length && inputCol < displayInputTensor[0].length) {
          inputValue = displayInputTensor[inputRow][inputCol];
        }
        
        const outputGradValue = displayOutputGrad[i][j];
        const term = outputGradValue * inputValue;
        
        // Format values for display in equation
        const outputGradFormatted = formatValue(outputGradValue).displayValue;
        const inputFormatted = formatValue(inputValue).displayValue;
        
        terms.push(`(${outputGradFormatted} \\cdot ${inputFormatted})`);
        sum += term;
      }
    }
    
    equationWithValues += terms.join(" + ");
    
    steps.push({
      description: "Substituting the actual values",
      equation: equationWithValues
    });
    
    // Step 3: Final result
    const gradientValue = displayWeightGrad[row][col];
    const { displayValue, fullPrecision } = formatValue(gradientValue);
    
    steps.push({
      description: "Final result",
      equation: `\\frac{\\partial L}{\\partial W_{${row},${col}}} = ${displayValue}`,
      result: fullPrecision
    });
    
    return steps;
  };

  const gradientSteps = calculateGradientSteps();
  const gradientValue = displayWeightGrad[selectedPosition.row][selectedPosition.col];
  const { displayValue, fullPrecision } = formatValue(gradientValue);
  
  return (
    <div className="conv-gradient-visualizer">
      <div className="calculation-visualization">
        <h5>Calculation Visualization</h5>
        <div className="position-selector mb-3">
          <Row>
            <Col xs={8}>
              <Form.Select 
                value={`Position (${selectedPosition.row}, ${selectedPosition.col})`}
                onChange={handlePositionChange}
              >
                {positionOptions.map((option) => (
                  <option key={option} value={option}>{option}</option>
                ))}
              </Form.Select>
            </Col>
            <Col xs={4} className="d-flex justify-content-end">
              <Button variant="outline-secondary" onClick={handlePreviousPosition} className="me-2">
                « Previous Position
              </Button>
              <Button variant="outline-secondary" onClick={handleNextPosition}>
                Next Position »
              </Button>
            </Col>
          </Row>
        </div>
        
        <div className="current-calculation mb-4">
          <h6>Current Calculation Position: ({selectedPosition.row}, {selectedPosition.col})</h6>
          <OverlayTrigger
            placement="top"
            overlay={<Tooltip>{fullPrecision}</Tooltip>}
          >
            <h6>Gradient Value: {displayValue}</h6>
          </OverlayTrigger>
        </div>
        
        <Row>
          <Col md={4}>
            <h6>Input Tensor</h6>
            <TensorVisualizer tensor={displayInputTensor} />
          </Col>
          <Col md={4}>
            <h6>Output Gradient</h6>
            <TensorVisualizer tensor={displayOutputGrad} />
          </Col>
          <Col md={4}>
            <h6>Weight Gradient</h6>
            <TensorVisualizer tensor={displayWeightGrad} />
          </Col>
        </Row>
        
        <div className="calculation-steps mt-4">
          <h6>Calculation Steps</h6>
          {gradientSteps.map((step, index) => (
            <div key={index} className="step">
              <div className="step-number">{index + 1}</div>
              <div className="step-content">
                <div className="step-description">{step.description}</div>
                <div className="equation-container">
                  <BlockMath math={step.equation} />
                </div>
                {step.result && (
                  <div className="result">
                    <strong>Result: </strong>
                    <OverlayTrigger
                      placement="top"
                      overlay={<Tooltip>{step.result}</Tooltip>}
                    >
                      <span>{step.result}</span>
                    </OverlayTrigger>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default ConvGradientVisualizer;