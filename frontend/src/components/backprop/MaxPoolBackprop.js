import React from 'react';
import { Row, Col } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import MaxPoolBackpropVisualizer from './MaxPoolBackpropVisualizer';

const MaxPoolBackprop = ({ backward }) => {
  return (
    <>
      <Row>
        <Col md={6}>
          <h6>MaxPool Backpropagation General Principle</h6>
          <p>MaxPool backpropagation passes the gradient only to the position where the maximum value was selected during forward pass.</p>
          <div className="equation-container">
            <BlockMath math="\frac{\partial L}{\partial I_{i,j}} = \begin{cases} 
              \frac{\partial L}{\partial O_{m,n}} & \text{if } I_{i,j} \text{ was the maximum in the pooling region} \\
              0 & \text{otherwise}
            \end{cases}" />
            <p>Where <InlineMath math="O_{m,n}" /> is the output value for that pooling region.</p>
          </div>
          
          <h6 className="mt-4">MaxPool Output Gradient</h6>
          <TensorVisualizer tensor={backward.pool.output_grad[0][0]} />
        </Col>
        
        <Col md={6}>
          <h6>MaxPool Input Gradient (Expanded form)</h6>
          <TensorVisualizer tensor={backward.pool.input_grad[0][0]} />
          
          <div className="mt-4">
            <p>Gradients are only propagated to positions that held the maximum value. Gradients at other positions are zero.</p>
            <div className="calculation-step">
              <p>Example: In pooling region (0,0) to (1,1), the maximum value is at (1,1), so the gradient is propagated only to this position.</p>
            </div>
          </div>
          
          <div className="mt-4">
            <p className="text-muted small">
              <strong>Reference:</strong> Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. arXiv preprint arXiv:1412.6806.
            </p>
          </div>
        </Col>
      </Row>
      
      {/* Enhanced MaxPool Backpropagation Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <hr />
          <h5 className="mb-4">Enhanced MaxPool Gradient Flow Visualization</h5>
          <MaxPoolBackpropVisualizer backward={backward.pool} />
        </Col>
      </Row>
    </>
  );
};

export default MaxPoolBackprop;