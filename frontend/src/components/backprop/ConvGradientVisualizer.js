import React, { useState } from 'react';
import { Row, Col, Form, Button, OverlayTrigger, Tooltip, Card } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import AnimatedCalculation from '../AnimatedCalculation';

const ConvGradientVisualizer = ({ outputGrad, inputTensor, weightGrad, initialWeights, updatedWeights, learningRate = 0.01 }) => {
  // 디버깅을 위한 콘솔 로그 추가
  console.log('Output Gradient:', outputGrad);
  console.log('Input Tensor:', inputTensor);
  console.log('Weight Gradient:', weightGrad);
  console.log('Initial Weights:', initialWeights);
  console.log('Updated Weights:', updatedWeights);
  console.log('Learning Rate:', learningRate);
  
  // 값의 포맷팅 함수 - 소수점 자릿수 조절 및 표시 최적화
  const formatValue = (value) => {
    // 전체 값은 정확한 값
    const fullPrecision = value.toFixed(6);
    
    // 화면에 표시될 값 - 크기에 따라 다르게 처리
    let displayValue;
    
    if (Math.abs(value) < 0.0001) {
      // 매우 작은 값은 과학적 표기법 사용
      displayValue = value.toExponential(3);
    } else if (Math.abs(value) < 0.01) {
      // 작은 값은 최대 5자리까지 표시
      displayValue = value.toFixed(5);
    } else if (Math.abs(value) < 0.1) {
      // 중간 크기 값은 4자리까지 표시
      displayValue = value.toFixed(4);
    } else {
      // 일반적인 값은 3자리까지 표시
      displayValue = value.toFixed(3);
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
  
  // 샘플 가중치 데이터 (실제 데이터가 없을 경우 사용)
  const sampleInitialWeights = [
    [1.0, 0.5],
    [0.5, 1.0]
  ];
  
  const sampleUpdatedWeights = [
    [0.985, 0.4875],
    [0.4825, 0.9775]
  ];
  
  // 실제 데이터 또는 샘플 데이터 사용
  const displayInitialWeights = initialWeights || sampleInitialWeights;
  const displayUpdatedWeights = updatedWeights || sampleUpdatedWeights;
  const displayLearningRate = learningRate || 0.01;
  
  // 선택된 가중치 위치에 대한 업데이트 계산 단계
  const calculateWeightUpdateSteps = () => {
    const { row, col } = selectedPosition;
    const steps = [];
    
    // 현재 선택된 위치의 값들
    const initialWeight = displayInitialWeights[row][col];
    const weightGradVal = displayWeightGrad[row][col];
    const updatedWeight = displayUpdatedWeights[row][col];
    
    // 가중치 업데이트 공식
    steps.push({
      description: `Weight update formula for position (${row},${col})`,
      equation: "W_{new} = W_{old} - \\eta \\cdot \\frac{\\partial L}{\\partial W}"
    });
    
    // 값 대입
    const initialWeightFormatted = initialWeight.toFixed(6); // 6자리까지 정확히 표시
    const gradientFormatted = weightGradVal.toFixed(6); // 6자리까지 정확히 표시
    const lrFormatted = displayLearningRate.toFixed(6); // 6자리까지 정확히 표시
    
    steps.push({
      description: "Substituting the values",
      equation: `W_{new} = ${initialWeightFormatted} - ${lrFormatted} \\cdot (${gradientFormatted})`
    });
    
    // 계산 과정 표시
    const gradientTerm = displayLearningRate * weightGradVal;
    const gradientTermFormatted = gradientTerm.toFixed(6); // 6자리까지 정확히 표시
    
    steps.push({
      description: "Calculating the gradient term",
      equation: `${lrFormatted} \\cdot (${gradientFormatted}) = ${gradientTermFormatted}`
    });
    
    // 계산 결과
    const expectedUpdatedWeight = initialWeight - displayLearningRate * weightGradVal;
    const expectedFormatted = expectedUpdatedWeight.toFixed(6); // 6자리까지 정확히 표시
    
    steps.push({
      description: "Final calculation",
      equation: `W_{new} = ${initialWeightFormatted} - (${gradientTermFormatted}) = ${expectedFormatted}`,
      result: expectedFormatted
    });
    
    // 실제 업데이트된 가중치가 계산값과 조금 다를 수 있음
    const updatedFormatted = updatedWeight.toFixed(6); // 6자리까지 정확히 표시
    
    if (Math.abs(expectedUpdatedWeight - updatedWeight) > 0.0001) {
      steps.push({
        description: "Actual updated weight (may differ slightly due to precision)",
        equation: `W_{new} = ${updatedFormatted}`,
        result: updatedFormatted
      });
    }
    // 계산 결과와 실제 값이 정확히 동일한 경우에도 정확한 값을 보여주기 위해 추가
    else {
      steps.push({
        description: "Final updated weight",
        equation: `W_{new} = ${updatedFormatted}`,
        result: updatedFormatted
      });
    }
    
    return steps;
  };
  
  const weightUpdateSteps = calculateWeightUpdateSteps();
  
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
        
        {/* Weight Update Visualization Section */}
        <div className="weight-update-visualization mt-5">
          <h5>Weight Update Visualization</h5>
          <p className="mb-4">
            After calculating the gradient, we update the weights using gradient descent with learning rate {displayLearningRate}.
          </p>
          
          <Row>
            <Col md={4}>
              <Card className="mb-3">
                <Card.Header>Initial Weight</Card.Header>
                <Card.Body>
                  <TensorVisualizer tensor={displayInitialWeights} highlightPosition={selectedPosition} />
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={4}>
              <Card className="mb-3">
                <Card.Header>Weight Gradient</Card.Header>
                <Card.Body>
                  <TensorVisualizer tensor={displayWeightGrad} highlightPosition={selectedPosition} />
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={4}>
              <Card className="mb-3">
                <Card.Header>Updated Weight</Card.Header>
                <Card.Body>
                  <TensorVisualizer tensor={displayUpdatedWeights} highlightPosition={selectedPosition} />
                </Card.Body>
              </Card>
            </Col>
          </Row>
          
          <div className="calculation-steps mt-4">
            <h6>Weight Update Steps for Position ({selectedPosition.row}, {selectedPosition.col})</h6>
            {weightUpdateSteps.map((step, index) => (
              <div key={`update-${index}`} className="step">
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
          
          <div className="references mt-4">
            <p className="text-muted small">
              <strong>Reference:</strong> Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConvGradientVisualizer;