import React from 'react';
import { Accordion } from 'react-bootstrap';
import FCLayerBackprop from './backprop/FCLayerBackprop';
import MaxPoolBackprop from './backprop/MaxPoolBackprop';
import ReluBackprop from './backprop/ReluBackprop';
import ConvBackprop from './backprop/ConvBackprop';

const BackwardPass = ({ backward, gradients, initial_weights, updated_weights, learning_rate, forward }) => {
  return (
    <div className="backward-pass-container">
      <Accordion defaultActiveKey="0">
        {/* FC Layer Backpropagation */}
        <Accordion.Item eventKey="0">
          <Accordion.Header>1. Fully Connected Layer Backpropagation</Accordion.Header>
          <Accordion.Body>
            <FCLayerBackprop 
              backward={backward} 
              gradients={gradients} 
              initial_weights={initial_weights}
              updated_weights={updated_weights}
              learning_rate={learning_rate}
            />
          </Accordion.Body>
        </Accordion.Item>
        
        {/* MaxPool Backpropagation */}
        <Accordion.Item eventKey="1">
          <Accordion.Header>2. MaxPool Backpropagation</Accordion.Header>
          <Accordion.Body>
            <MaxPoolBackprop backward={backward} />
          </Accordion.Body>
        </Accordion.Item>
        
        {/* ReLU Backpropagation */}
        <Accordion.Item eventKey="2">
          <Accordion.Header>3. ReLU Backpropagation</Accordion.Header>
          <Accordion.Body>
            <ReluBackprop backward={backward} />
          </Accordion.Body>
        </Accordion.Item>
        
        {/* Convolution Backpropagation */}
        <Accordion.Item eventKey="3">
          <Accordion.Header>4. Convolution Backpropagation</Accordion.Header>
          <Accordion.Body>
            <ConvBackprop 
              backward={backward} 
              initial_weights={initial_weights}
              updated_weights={updated_weights}
              learning_rate={learning_rate}
              forward={forward}
            />
          </Accordion.Body>
        </Accordion.Item>
      </Accordion>
    </div>
  );
};

export default BackwardPass;