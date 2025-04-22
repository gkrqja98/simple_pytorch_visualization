import React from 'react';
import { Row, Col } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';
import FCLayerBackpropVisualizer from './FCLayerBackpropVisualizer';

const FCLayerBackprop = ({ backward, gradients, initial_weights, updated_weights, learning_rate }) => {
  return (
    <>
      <Row>
        <Col md={6}>
          <h6>FC Layer Gradient General Formulas</h6>
          <div className="equation-container">
            <BlockMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
            <BlockMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
            <BlockMath math="\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot W_{i,j}" />
          </div>
          
          <h6 className="mt-4">Weight Gradient Calculation</h6>
          <div className="calculation-step">
            <InlineMath math="\frac{\partial L}{\partial W_{i,j}} = \frac{\partial L}{\partial y_i} \cdot x_j" />
          </div>
          <div className="calculation-step">
            <p>Example: First weight gradient <InlineMath math="\frac{\partial L}{\partial W_{1,1}}" />:</p>
            <InlineMath math="\frac{\partial L}{\partial W_{1,1}} = \frac{\partial L}{\partial y_1} \cdot x_1 = dl/dy_1 \cdot 15.5" />
          </div>
          
          <h6 className="mt-4">Bias Gradient Calculation</h6>
          <div className="calculation-step">
            <InlineMath math="\frac{\partial L}{\partial b_i} = \frac{\partial L}{\partial y_i}" />
          </div>
        </Col>
        
        <Col md={6}>
          <h6>FC Weight Gradients</h6>
          <div className="matrix-container">
            <table className="matrix-table">
              <tbody>
                {backward.fc.weight_grad.map((row, i) => (
                  <tr key={i}>
                    {row.map((value, j) => (
                      <td key={j}>{value.toFixed(4)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <h6 className="mt-3">FC Bias Gradients</h6>
          <div className="matrix-container">
            <table className="matrix-table">
              <tbody>
                <tr>
                  {backward.fc.bias_grad.map((value, i) => (
                    <td key={i}>{value.toFixed(4)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          
          <h6 className="mt-4">Gradient with Respect to FC Input</h6>
          <div className="matrix-container">
            <table className="matrix-table">
              <tbody>
                <tr>
                  {backward.fc.input_grad[0].map((value, i) => (
                    <td key={i}>{value.toFixed(4)}</td>
                  ))}
                </tr>
              </tbody>
            </table>
          </div>
          
          <div className="mt-4 weight-update">
            <h6>Weight Update</h6>
            <BlockMath math="W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}" />
            <p>Where <InlineMath math="\eta" /> is the learning rate ({learning_rate}).</p>
            
            <p className="mt-2">Example of first weight update:</p>
            <InlineMath math="W_{1,1}^{new} = W_{1,1}^{old} - \eta \cdot \frac{\partial L}{\partial W_{1,1}}" />
            <InlineMath math="W_{1,1}^{new} = 0.1 - 0.01 \cdot \frac{\partial L}{\partial W_{1,1}}" />
          </div>
        </Col>
      </Row>
      
      {/* Enhanced FC Layer Backpropagation Visualization */}
      <Row className="mt-5">
        <Col md={12}>
          <hr />
          <h5 className="mb-4">Enhanced Gradient Flow Visualization</h5>
          <FCLayerBackpropVisualizer 
            backward={backward.fc} 
            gradients={gradients} 
            initial_weights={initial_weights}
            updated_weights={updated_weights}
            learning_rate={learning_rate}
          />
        </Col>
      </Row>
    </>
  );
};

export default FCLayerBackprop;