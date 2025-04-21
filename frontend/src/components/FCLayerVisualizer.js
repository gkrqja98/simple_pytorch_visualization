import React, { useState } from 'react';
import { Row, Col, Form, Button } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';

/**
 * Component for visualizing Fully Connected Layer operations
 */
const FCLayerVisualizer = ({ inputVector, weights, bias, outputVector }) => {
  const [selectedOutput, setSelectedOutput] = useState(0);
  
  // Get dimensions
  const inputSize = inputVector.length;
  const outputSize = outputVector.length;
  
  // Generate calculation steps for selected output neuron
  const calculateSteps = () => {
    // Steps for calculation
    const steps = [
      {
        description: `Formula for output neuron ${selectedOutput}`,
        equation: `y_{${selectedOutput}} = \\sum_{i=0}^{${inputSize-1}} W_{${selectedOutput},i} \\cdot x_i + b_{${selectedOutput}}`,
      },
      {
        description: "Expanded formula with weights",
        equation: `y_{${selectedOutput}} = ${weights[selectedOutput].map((w, i) => `W_{${selectedOutput},${i}} \\cdot x_{${i}}`).join(' + ')} + b_{${selectedOutput}}`,
      },
      {
        description: "Substituting actual values",
        equation: `y_{${selectedOutput}} = ${weights[selectedOutput].map((w, i) => `${w.toFixed(4)} \\cdot ${inputVector[i].toFixed(2)}`).join(' + ')} + ${bias[selectedOutput].toFixed(4)}`,
      }
    ];
    
    // Calculate weighted sums for the selected output
    const weightedSums = weights[selectedOutput].map((w, i) => w * inputVector[i]);
    
    steps.push({
      description: "Computing each term",
      equation: `y_{${selectedOutput}} = ${weightedSums.map(sum => sum.toFixed(4)).join(' + ')} + ${bias[selectedOutput].toFixed(4)}`,
    });
    
    // Calculate final result
    const result = weightedSums.reduce((a, b) => a + b, 0) + bias[selectedOutput];
    
    steps.push({
      description: "Final result",
      equation: `y_{${selectedOutput}} = ${result.toFixed(4)}`,
      result: result.toFixed(4)
    });
    
    return { steps, weightedSums };
  };
  
  const { steps, weightedSums } = calculateSteps();
  
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
  
  return (
    <div className="fc-layer-visualizer">
      <h6 className="mb-3">Fully Connected Layer Calculation Visualization</h6>
      
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
        <Col md={6} className="d-flex justify-content-end">
          <Button 
            variant="outline-secondary" 
            size="sm" 
            onClick={handlePrevOutput}
            className="me-2"
          >
            &laquo; Previous Neuron
          </Button>
          <Button 
            variant="outline-secondary" 
            size="sm" 
            onClick={handleNextOutput}
          >
            Next Neuron &raquo;
          </Button>
        </Col>
      </Row>
      
      <div className="position-indicator mb-3">
        <div className="d-flex align-items-center">
          <div className="me-4">
            <strong>Selected Output Neuron:</strong> {selectedOutput}
          </div>
          <div>
            <strong>Output Value:</strong> {outputVector[selectedOutput].toFixed(4)}
          </div>
        </div>
      </div>
      
      <div className="network-visualization mb-4">
        <Row>
          <Col md={12}>
            <div className="fc-visualization p-3 bg-light rounded">
              <h6 className="text-center mb-3">Network Visualization</h6>
              <div className="network-container">
                {/* Input Layer */}
                <div className="layer input-layer">
                  {inputVector.map((val, idx) => (
                    <div 
                      key={idx} 
                      className="neuron"
                      style={{backgroundColor: `rgba(0, 123, 255, ${Math.abs(val) / 30})`}}
                    >
                      <div className="neuron-value">{val.toFixed(2)}</div>
                      <div className="neuron-label">x<sub>{idx}</sub></div>
                    </div>
                  ))}
                </div>
                
                {/* Connection Lines */}
                <div className="connections">
                  <svg width="100%" height="100%" className="connection-svg">
                    {inputVector.map((_, inputIdx) => (
                      <line 
                        key={inputIdx}
                        x1="0%" 
                        y1={`${(100 / inputSize) * (inputIdx + 0.5)}%`}
                        x2="100%" 
                        y2={`${(100 / outputSize) * (selectedOutput + 0.5)}%`}
                        stroke={weightedSums[inputIdx] > 0 ? "#28a745" : "#dc3545"}
                        strokeWidth={Math.abs(weightedSums[inputIdx]) * 2 + 1}
                        strokeOpacity="0.6"
                      />
                    ))}
                  </svg>
                  
                  {/* Weight Labels */}
                  {inputVector.map((_, inputIdx) => (
                    <div 
                      key={inputIdx}
                      className="weight-label"
                      style={{
                        top: `${(100 / inputSize) * (inputIdx + 0.5)}%`,
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        color: weightedSums[inputIdx] > 0 ? '#28a745' : '#dc3545'
                      }}
                    >
                      {weights[selectedOutput][inputIdx].toFixed(2)}
                    </div>
                  ))}
                </div>
                
                {/* Output Layer */}
                <div className="layer output-layer">
                  {outputVector.map((val, idx) => (
                    <div 
                      key={idx} 
                      className={`neuron ${idx === selectedOutput ? 'selected' : ''}`}
                      style={{
                        backgroundColor: idx === selectedOutput ? 
                          `rgba(40, 167, 69, ${Math.abs(val) / 30})` : 
                          `rgba(108, 117, 125, ${Math.abs(val) / 30})`
                      }}
                    >
                      <div className="neuron-value">{val.toFixed(2)}</div>
                      <div className="neuron-label">y<sub>{idx}</sub></div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Bias indicator */}
              <div className="bias-indicator mt-3 text-center">
                <strong>Bias for neuron {selectedOutput}:</strong> {bias[selectedOutput].toFixed(4)}
              </div>
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
        .network-container {
          display: flex;
          height: 300px;
          position: relative;
        }
        
        .layer {
          display: flex;
          flex-direction: column;
          justify-content: space-around;
          width: 25%;
        }
        
        .input-layer {
          margin-right: 10%;
        }
        
        .output-layer {
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
          width: 60px;
          height: 60px;
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
        }
        
        .neuron.selected {
          border: 3px solid #28a745;
        }
        
        .neuron-value {
          font-weight: bold;
        }
        
        .neuron-label {
          font-size: 0.8rem;
          color: #6c757d;
        }
        
        .weight-label {
          position: absolute;
          font-size: 0.7rem;
          background-color: rgba(255, 255, 255, 0.8);
          padding: 1px 4px;
          border-radius: 3px;
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

export default FCLayerVisualizer;
