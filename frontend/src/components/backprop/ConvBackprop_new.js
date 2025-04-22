import React from 'react';
import { Row, Col, Alert } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import ConvGradientVisualizer from './ConvGradientVisualizer';

const ConvBackprop = ({ backward, initial_weights, updated_weights, learning_rate, forward }) => {
  // 디버깅을 위한 콘솔 로그 추가
  console.log('Backward Data:', backward);
  console.log('Initial Weights:', initial_weights);
  console.log('Updated Weights:', updated_weights);
  console.log('Forward Data:', forward);
  // Check if all required data is present
  if (!backward || !backward.conv) {
    return (
      <Alert variant="warning">
        Missing data required for convolution backpropagation visualization. Please ensure the backend is correctly computing and providing all gradient information.
      </Alert>
    );
  }

  return (
    <>
      <Row>
        <Col md={12}>
          <div className="conv-backprop-header mb-4">
            <h5>Convolution Backpropagation</h5>
            <p>
              The backpropagation through a convolutional layer involves calculating gradients with respect to both 
              the weights and the inputs. These gradients are essential for the weight update in the current layer 
              and for propagating the error signal to earlier layers.
            </p>
          </div>
        </Col>
      </Row>

      <Row>
        <Col md={6}>
          <h6>Convolution Gradient General Formulas</h6>
          <div className="equation-container p-3 border rounded bg-light">
            <p><strong>Gradient with respect to weights:</strong></p>
            <BlockMath math="\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot I_{i+m, j+n}" />
            
            <p className="mt-3"><strong>Gradient with respect to inputs:</strong></p>
            <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial O_{i-m, j-n}} \cdot W_{m,n}" />
            
            <p>Where:</p>
            <ul>
              <li><InlineMath math="O_{i,j}" /> = Output feature map value at position (i,j)</li>
              <li><InlineMath math="I_{i,j}" /> = Input tensor value at position (i,j)</li>
              <li><InlineMath math="W_{m,n}" /> = Weight kernel value at position (m,n)</li>
            </ul>
          </div>
          
          <h6 className="mt-4">Convolution Output Gradient</h6>
          {backward.conv.output_grad ? (
            <div className="tensor-container p-2 border rounded">
              <p className="text-center">Gradient flowing from the ReLU layer (∂L/∂O)</p>
              <TensorVisualizer tensor={backward.conv.output_grad[0][0]} />
              <p className="text-muted text-center mt-2 small">
                This gradient represents how the loss changes with respect to each element in the convolution output.
              </p>
            </div>
          ) : (
            <Alert variant="info">Output gradient data not available</Alert>
          )}
        </Col>
        
        <Col md={6}>
          <h6>Convolution Weight Gradient</h6>
          {backward.conv.weight_grad ? (
            <div className="tensor-container p-2 border rounded">
              <p className="text-center">Computed gradient for convolution weights (∂L/∂W)</p>
              <TensorVisualizer tensor={backward.conv.weight_grad[0][0]} />
              <p className="text-muted text-center mt-2 small">
                This gradient is used to update the weights during optimization.
              </p>
            </div>
          ) : (
            <Alert variant="info">Weight gradient data not available</Alert>
          )}
          
          <div className="mt-4 weight-update">
            <h6>Weight Update Process</h6>
            <div className="p-3 border rounded bg-light">
              <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
              <p>Where <InlineMath math="\eta" /> is the learning rate ({learning_rate}).</p>
              
              {initial_weights && initial_weights.conv1_weight && updated_weights && updated_weights.conv1_weight ? (
                <div className="weight-comparison mt-3">
                  <Row>
                    <Col md={6}>
                      <h6 className="text-center">Initial Weights</h6>
                      <TensorVisualizer tensor={initial_weights.conv1_weight[0][0]} />
                    </Col>
                    <Col md={6}>
                      <h6 className="text-center">Updated Weights</h6>
                      <TensorVisualizer tensor={updated_weights.conv1_weight[0][0]} />
                    </Col>
                  </Row>
                </div>
              ) : (
                <Alert variant="info">Weight update data not available</Alert>
              )}
            </div>
          </div>
          
          <div className="mt-4">
            <p className="text-muted small">
              <strong>Reference:</strong> Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
            </p>
          </div>
        </Col>
      </Row>
      
      {/* Convolution Gradient Calculation Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <hr />
          <h5 className="mb-4">Convolution Gradient Calculation Visualization</h5>
          {backward.conv && backward.conv.output_grad && backward.conv.weight_grad ? (
            <ConvGradientVisualizer 
              outputGrad={backward.conv.output_grad[0][0]}
              inputTensor={forward && forward.conv && forward.conv.input_tensor ? forward.conv.input_tensor[0][0] : null}
              weightGrad={backward.conv.weight_grad[0][0]}
              initialWeights={initial_weights && initial_weights.conv1_weight ? initial_weights.conv1_weight[0][0] : null}
              updatedWeights={updated_weights && updated_weights.conv1_weight ? updated_weights.conv1_weight[0][0] : null}
              learningRate={learning_rate}
            />
          ) : (
            <Alert variant="warning">
              Missing data required for convolution gradient visualization.
            </Alert>
          )}
        </Col>
      </Row>
    </>
  );
};

export default ConvBackprop;