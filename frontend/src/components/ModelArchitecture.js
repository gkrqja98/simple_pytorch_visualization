import React from 'react';
import { Card, Row, Col } from 'react-bootstrap';
import { InlineMath, BlockMath } from 'react-katex';

const ModelArchitecture = ({ modelData }) => {
  return (
    <div className="model-architecture">
      <Card>
        <Card.Body>
          <Card.Title>{modelData.name} Architecture</Card.Title>
          <Card.Text>
            Total parameters: {modelData.total_params}
          </Card.Text>
          
          <div className="architecture-diagram text-center my-4">
            {modelData.layers.map((layer, index) => (
              <React.Fragment key={index}>
                <div className="layer-box">
                  <div><strong>{layer.name}</strong></div>
                  <div className="small">
                    {Object.entries(layer.params).map(([key, value]) => (
                      <div key={key}>{key}: {value}</div>
                    ))}
                  </div>
                </div>
                {index < modelData.layers.length - 1 && (
                  <span className="arrow">→</span>
                )}
              </React.Fragment>
            ))}
          </div>
          
          <h5 className="mt-4">Layer Details</h5>
          
          <Row className="mt-3">
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h6>Conv2d</h6>
                  <p>Convolutional layer extracts features from images.</p>
                  <div className="equation-container">
                    <BlockMath math="(f * g)(x, y) = \sum_{m}\sum_{n} f(m, n) \cdot g(x-m, y-n)" />
                    <p>Where:</p>
                    <ul>
                      <li><InlineMath math="f" />: Input image</li>
                      <li><InlineMath math="g" />: Kernel (weights)</li>
                      <li><InlineMath math="*" />: Convolution operation</li>
                    </ul>
                  </div>
                  <p className="text-muted small">
                    <strong>Reference:</strong> LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
                  </p>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h6>ReLU (Rectified Linear Unit)</h6>
                  <p>Activation function that adds non-linearity.</p>
                  <div className="equation-container">
                    <BlockMath math="ReLU(x) = \max(0, x)" />
                  </div>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Nair, V., & Hinton, G. E. (2010). Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML-10) (pp. 807-814).
                  </p>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h6>MaxPool2d</h6>
                  <p>Downsampling to reduce the size of feature maps.</p>
                  <div className="equation-container">
                    <BlockMath math="MaxPool(X)_{i,j} = \max_{m,n \in R_{i,j}} X_{m,n}" />
                    <p>where <InlineMath math="R_{i,j}" /> is the pooling region at position (i,j).</p>
                  </div>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Scherer, D., Müller, A., & Behnke, S. (2010). Evaluation of pooling operations in convolutional architectures for object recognition. In Artificial Neural Networks–ICANN 2010 (pp. 92-101). Springer.
                  </p>
                </Card.Body>
              </Card>
            </Col>
            
            <Col md={6}>
              <Card className="mb-3">
                <Card.Body>
                  <h6>Linear (Fully Connected Layer)</h6>
                  <p>Transforms features into class scores.</p>
                  <div className="equation-container">
                    <BlockMath math="y = Wx + b" />
                    <p>where:</p>
                    <ul>
                      <li><InlineMath math="W" />: Weight matrix</li>
                      <li><InlineMath math="x" />: Input vector</li>
                      <li><InlineMath math="b" />: Bias vector</li>
                    </ul>
                  </div>
                  <p className="text-muted small">
                    <strong>Reference:</strong> Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
                  </p>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </Card.Body>
      </Card>
    </div>
  );
};

export default ModelArchitecture;
