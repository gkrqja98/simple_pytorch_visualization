# PyTorch CNN Visualization Tool

This project visualizes the forward and backward propagation process of a simple CNN model implemented in PyTorch. It breaks down all tensor operations into matrix form and presents them with mathematical formulas in an interactive HTML interface.

## Key Features

- Simple CNN model implementation (including Conv2d, ReLU, MaxPool2d, Linear layers)
- Layer-by-layer detailed tracing of the model's forward and backward propagation
- Visualization of all tensor operations in matrix form
- Presentation of all calculation processes with mathematical formulas in HTML
- Visualization of weight and gradient changes during 3 training iterations (epochs)
- Support for mathematical expressions (LaTeX) for clear explanation of principles
- Modular code applicable to various PyTorch models

## Project Structure

```
deepl/
  ├── backend/               # Python backend
  │   ├── main.py            # API server
  │   ├── model.py           # CNN model definition
  │   ├── visualizer.py      # Model calculation tracing
  │   ├── requirements.txt   # Required packages
  │   └── data/              # Sample data
  │
  └── frontend/              # React frontend
      ├── public/            # Static files
      ├── src/               # Source code
      │   ├── components/    # React components
      │   ├── pages/         # Page components
      │   ├── utils/         # Utility functions
      │   ├── App.js         # Main app
      │   └── index.js       # Entry point
      ├── package.json       # Dependencies
      └── .env               # Environment variables
```

## Installation and Execution

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Running the Backend

```bash
cd backend
python main.py
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running the Frontend

```bash
cd frontend
npm start
```

## Web Interface Structure

1. **Model Architecture**: Model structure and layer descriptions
2. **Iterations 1, 2, 3**: Detailed information for each training iteration
   - Display of input data, weights, learning rate, etc.
   - Forward Pass: Visualization of all computational steps in the forward propagation
   - Backward Pass: Visualization of all computational steps in the backward propagation

## Tech Stack

- **Backend**: Python, PyTorch, Flask
- **Frontend**: React, Bootstrap, KaTeX (formula rendering)

## How It Works

The application demonstrates a simple CNN model with the following architecture:
- Input: 4x4 image
- Conv2d (kernel_size=2, padding=0): Output size 3x3
- ReLU: Maintains size 3x3
- MaxPool2d (kernel_size=2, stride=1): Output size 2x2
- Flatten: Converts 2x2 feature map to 4 elements
- Linear(4, 2): Maps 4 features to 2 output classes
- Loss: CrossEntropyLoss

For each layer, the tool shows:
- Mathematical formulas that govern the operations
- Actual values computed during forward and backward passes
- Weight updates after backpropagation

## References

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. ICCV.
4. Dumoulin, V., & Visin, F. (2016). A guide to convolution arithmetic for deep learning. arXiv preprint arXiv:1603.07285.
