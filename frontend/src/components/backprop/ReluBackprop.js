import React from 'react';
import { Row, Col } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import TensorVisualizer from '../TensorVisualizer';
import ReluBackpropVisualizer from './ReluBackpropVisualizer';

const ReluBackprop = ({ backward }) => {
  return (
    <>
      <Row>
        <Col md={6}>
          <h6>ReLU Backpropagation General Formula</h6>
          <div className="equation-container">
            <BlockMath math="\frac{\partial ReLU(x)}{\partial x} = \begin{cases} 
              1 & \text{if } x > 0 \\
              0 & \text{if } x \leq 0
            \end{cases}" />
          </div>
          
          <div className="mt-4">
            <h6>ReLU Output Gradient</h6>
            <TensorVisualizer tensor={backward.relu.output_grad[0][0]} />
          </div>
          
          <div className="mt-4">
            <h6>ReLU Activation Mask (1: Active, 0: Inactive)</h6>
            <TensorVisualizer tensor={backward.relu.mask[0][0]} />
          </div>
        </Col>
        
        <Col md={6}>
          <h6>ReLU Input Gradient</h6>
          <TensorVisualizer tensor={backward.relu.input_grad[0][0]} />
          
          <div className="mt-4">
            <p>ReLU backpropagation simply passes the gradient through positions where the input was positive.</p>
            <div className="calculation-step">
              <InlineMath math="\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial ReLU(x_i)} \cdot \frac{\partial ReLU(x_i)}{\partial x_i}" />
            </div>
            <div className="calculation-step">
              <InlineMath math="\frac{\partial L}{\partial x_i} = \begin{cases} 
                \frac{\partial L}{\partial ReLU(x_i)} & \text{if } x_i > 0 \\
                0 & \text{if } x_i \leq 0
              \end{cases}" />
            </div>
          </div>
          
          <div className="mt-4">
            <p className="text-muted small">
              <strong>Reference:</strong> He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision (pp. 1026-1034).
            </p>
          </div>
        </Col>
      </Row>
      
      {/* Enhanced ReLU Backpropagation Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <hr />
          <h5 className="mb-4">Enhanced ReLU Gradient Flow Visualization</h5>
          <ReluBackpropVisualizer backward={backward.relu} />
        </Col>
      </Row>
    </>
  );
};

export default ReluBackprop;