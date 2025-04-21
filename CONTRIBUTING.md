# Contributing to PyTorch CNN Visualization Tool

Thank you for considering contributing to this project! Here are some guidelines to help you get started.

## How to Contribute

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Run tests and ensure all changes work properly
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to the branch (`git push origin feature/your-feature-name`)
7. Create a new Pull Request

## Development Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Code Style

- For Python code, follow PEP 8 guidelines
- For JavaScript/React code, use ESLint with the provided configuration

## Adding New Features

- For new layer types, update both `model.py` and the corresponding visualization components
- For UI improvements, ensure they work across different browsers

## Reporting Issues

If you find a bug or have a suggestion, please submit an issue with:
- A clear description of the problem
- Steps to reproduce
- Expected behavior
- Screenshots if applicable
- Environment details (OS, browser, etc.)

Thank you for your contributions!
