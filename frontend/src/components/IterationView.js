import React from 'react';
import { Card, Row, Col, Accordion } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from './TensorVisualizer';
import ForwardPass from './ForwardPass';
import BackwardPass from './BackwardPass';

const IterationView = ({ iteration, iterationIndex }) => {
  return (
    <div className="iteration-container">
      <Row className="mb-4">
        <Col md={6}>
          <Card>
            <Card.Body>
              <Card.Title>Input Data and Settings</Card.Title>
              <div className="mb-3">
                <h6>Input Tensor (4x4)</h6>
                <TensorVisualizer tensor={iteration.input_data[0][0]} />
                <p className="mt-2 text-muted">Shape: [1, 1, 4, 4] (batch size, channels, height, width)</p>
              </div>
              
              <div className="mb-3">
                <h6>Target Label</h6>
                <p>{iteration.target}</p>
              </div>
              
              <div>
                <h6>Training Settings</h6>
                <ul>
                  <li>Learning Rate: {iteration.learning_rate}</li>
                  <li>Optimization Algorithm: SGD (Stochastic Gradient Descent)</li>
                  <li>Loss Function: CrossEntropyLoss</li>
                </ul>
              </div>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          <Card>
            <Card.Body>
              <Card.Title>Weights and Biases</Card.Title>
              
              <Accordion defaultActiveKey="0">
                <Accordion.Item eventKey="0">
                  <Accordion.Header>Convolutional Layer Weights</Accordion.Header>
                  <Accordion.Body>
                    <h6>Initial Weights</h6>
                    <TensorVisualizer tensor={iteration.initial_weights.conv1_weight[0][0]} />
                    
                    <h6 className="mt-3">Updated Weights</h6>
                    <TensorVisualizer tensor={iteration.updated_weights.conv1_weight[0][0]} />
                    
                    {iterationIndex > 0 && (
                      <div className="mt-2 text-muted">
                        Updated weights from the previous iteration are used as initial weights for this iteration.
                      </div>
                    )}
                  </Accordion.Body>
                </Accordion.Item>
                
                <Accordion.Item eventKey="1">
                  <Accordion.Header>Fully Connected Layer Weights and Biases</Accordion.Header>
                  <Accordion.Body>
                    <h6>Initial Weights</h6>
                    <div className="matrix-container">
                      <table className="matrix-table">
                        <tbody>
                          {iteration.initial_weights.fc_weight.map((row, i) => (
                            <tr key={i}>
                              {row.map((value, j) => (
                                <td key={j}>{value.toFixed(4)}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    <h6 className="mt-3">Initial Bias</h6>
                    <div className="matrix-container">
                      <table className="matrix-table">
                        <tbody>
                          <tr>
                            {iteration.initial_weights.fc_bias.map((value, i) => (
                              <td key={i}>{value.toFixed(4)}</td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                    
                    <h6 className="mt-3">Updated Weights</h6>
                    <div className="matrix-container">
                      <table className="matrix-table">
                        <tbody>
                          {iteration.updated_weights.fc_weight.map((row, i) => (
                            <tr key={i}>
                              {row.map((value, j) => (
                                <td key={j}>{value.toFixed(4)}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    
                    <h6 className="mt-3">Updated Bias</h6>
                    <div className="matrix-container">
                      <table className="matrix-table">
                        <tbody>
                          <tr>
                            {iteration.updated_weights.fc_bias.map((value, i) => (
                              <td key={i}>{value.toFixed(4)}</td>
                            ))}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </Accordion.Body>
                </Accordion.Item>
              </Accordion>
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col md={12}>
          <Card className="mb-4">
            <Card.Header>
              <h5 className="mb-0">Forward Pass</h5>
            </Card.Header>
            <Card.Body>
              <ForwardPass forward={iteration.forward} />
            </Card.Body>
          </Card>
          
          <Card>
            <Card.Header>
              <h5 className="mb-0">Backward Pass</h5>
            </Card.Header>
            <Card.Body>
              <BackwardPass 
                backward={iteration.backward}
                gradients={iteration.gradients}
                initial_weights={iteration.initial_weights}
                updated_weights={iteration.updated_weights}
                learning_rate={iteration.learning_rate}
              />
            </Card.Body>
          </Card>
        </Col>
      </Row>
      
      <Row className="mt-4">
        <Col>
          <Card>
            <Card.Body>
              <Card.Title>Iteration Results</Card.Title>
              <div className="mt-3">
                <h6>Loss</h6>
                <p>
                  <InlineMath math={`L = ${iteration.loss.toFixed(4)}`} />
                </p>
                
                <h6 className="mt-3">Weight Change Summary</h6>
                <p>
                  Conv1 Weight Mean Change: 
                  <InlineMath math={` ${calculateMeanChange(
                    iteration.initial_weights.conv1_weight,
                    iteration.updated_weights.conv1_weight
                  ).toFixed(6)}`} />
                </p>
                <p>
                  FC Weight Mean Change: 
                  <InlineMath math={` ${calculateMeanChange(
                    iteration.initial_weights.fc_weight,
                    iteration.updated_weights.fc_weight
                  ).toFixed(6)}`} />
                </p>
                <p>
                  FC Bias Mean Change: 
                  <InlineMath math={` ${calculateMeanChange(
                    iteration.initial_weights.fc_bias,
                    iteration.updated_weights.fc_bias
                  ).toFixed(6)}`} />
                </p>
                
                <div className="mt-4 text-muted">
                  <strong>Training Progress:</strong> Iteration {iterationIndex + 1} / 3
                </div>
              </div>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

// 가중치 변화량 계산 함수
const calculateMeanChange = (initial, updated) => {
  // 평탄화 함수
  const flatten = (arr) => {
    return arr.flat(Infinity);
  };
  
  const flatInitial = flatten(initial);
  const flatUpdated = flatten(updated);
  
  // 평균 변화량 계산
  let totalChange = 0;
  for (let i = 0; i < flatInitial.length; i++) {
    totalChange += Math.abs(flatUpdated[i] - flatInitial[i]);
  }
  
  return totalChange / flatInitial.length;
};

export default IterationView;
