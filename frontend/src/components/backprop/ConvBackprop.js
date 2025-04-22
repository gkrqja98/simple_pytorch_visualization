import React from 'react';
import { Row, Col } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import ConvBackpropVisualizer from './ConvBackpropVisualizer';

const ConvBackprop = ({ backward, initial_weights, updated_weights, learning_rate }) => {
  return (
    <>
      <Row>
        <Col md={6}>
          <h6>Convolution Gradient General Formulas</h6>
          <div className="equation-container">
            <BlockMath math="\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial O_{i,j}} \cdot I_{i+m, j+n}" />
            <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial O_{i-m, j-n}} \cdot W_{m,n}" />
            <p>Where:</p>
            <ul>
              <li><InlineMath math="O_{i,j}" /> = Output feature map value at position (i,j)</li>
              <li><InlineMath math="I_{i,j}" /> = Input tensor value at position (i,j)</li>
              <li><InlineMath math="W_{m,n}" /> = Weight kernel value at position (m,n)</li>
            </ul>
          </div>
          
          <h6 className="mt-4">Convolution Output Gradient</h6>
          <TensorVisualizer tensor={backward.conv.output_grad[0][0]} />
        </Col>
        
        <Col md={6}>
          <h6>Convolution Weight Gradient</h6>
          <TensorVisualizer tensor={backward.conv.weight_grad[0][0]} />
          
          <div className="mt-4 weight-update">
            <h6>Weight Update</h6>
            <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
            <p>Where <InlineMath math="\eta" /> is the learning rate ({learning_rate}).</p>
            
            <h6 className="mt-3">Initial Kernel Weights</h6>
            <TensorVisualizer tensor={initial_weights.conv1_weight[0][0]} />
            
            <h6 className="mt-3">Updated Kernel Weights</h6>
            <TensorVisualizer tensor={updated_weights.conv1_weight[0][0]} />
          </div>
          
          <div className="mt-4">
            <p className="text-muted small">
              <strong>Reference:</strong> Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
            </p>
          </div>
        </Col>
      </Row>
      
      {/* Enhanced Convolution Backpropagation Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <hr />
          <h5 className="mb-4">Enhanced Convolution Gradient Flow Visualization</h5>
          <ConvBackpropVisualizer 
            backward={backward.conv}
            initial_weights={initial_weights.conv1_weight}
            updated_weights={updated_weights.conv1_weight}
            learning_rate={learning_rate}
          />
        </Col>
      </Row>
    </>
  );
};

export default ConvBackprop;