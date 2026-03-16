# Contributing to Stock Prediction CLI

Thank you for your interest in contributing to Stock Prediction CLI! This document outlines the process for contributing to this project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive.

## How to Contribute

### Reporting Bugs

1. **Check existing issues** - Search the issue tracker to see if the bug has already been reported
2. **Create a new issue** - Use the bug report template and provide:
   - Clear title and description
   - Steps to reproduce the bug
   - Expected vs actual behavior
   - Environment details (Python version, OS, etc.)
   - Any relevant logs or error messages

### Requesting Features

1. **Search existing requests** - Check if the feature has already been requested
2. **Open a feature request** - Use the feature request template with:
   - Clear description of the feature
   - Use cases for the feature
   - Potential implementation approaches
   - Any alternatives considered

### Pull Requests

#### Prerequisites

- Python 3.9+
- All tests passing (28/28 required)
- Code follows PEP 8 style guidelines
- No linting errors

#### Development Workflow

1. **Fork the repository**

   Click the "Fork" button on the repository page, then clone your fork:

   ```bash
   git clone https://github.com/your-username/pythia-stock-predictor.git
   cd pythia-stock-predictor
   ```

2. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

3. **Set up development environment**

   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Install dev dependencies
   pip install pytest pytest-cov pytest-mock
   ```

4. **Make your changes**

   - Write clean, readable code
   - Add tests for new functionality
   - Update documentation as needed
   - Follow the existing code style

5. **Run the test suite**

   ```bash
   # Run all tests
   pytest
   
   # Run with coverage
   pytest --cov=src --cov-report=html
   
   # Run specific test file
   pytest tests/test_predictor.py -v
   ```

   **All 28 tests must pass** before submitting a pull request.

6. **Run linting checks**

   ```bash
   # Using ruff (configured in project)
   ruff check src/ tests/
   
   # Or using flake8
   flake8 src/ tests/
   ```

7. **Commit your changes**

   ```bash
   git add .
   git commit -m "Add feature: description of changes"
   ```

   Follow conventional commit messages:
   - `feat: Add new LSTM model`
   - `fix: Resolve ensemble prediction bug`
   - `docs: Update README with new examples`
   - `test: Add tests for predictor`

8. **Push to your fork**

   ```bash
   git push origin feature/your-feature-name
   ```

9. **Create a Pull Request**

   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template
   - Link any related issues

#### PR Requirements

- [ ] All tests pass (28/28 required)
- [ ] Code coverage maintained or improved
- [ ] No linting errors
- [ ] Documentation updated (if applicable)
- [ ] PR description explains the changes
- [ ] Related issue linked (if applicable)

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints where applicable
- Write docstrings for all public functions
- Keep lines under 100 characters

### Testing Standards

- Use `pytest` for testing
- Aim for 100% code coverage
- Write descriptive test names
- Include docstrings for test functions

### Documentation

- Keep README.md updated
- Document new CLI commands
- Update config.yaml defaults if needed
- Add docstrings to new functions

## Project Structure

```
pythia-stock-predictor/
├── src/
│   ├── api/          # REST API server
│   ├── data/         # Data collection & preprocessing
│   ├── models/       # ML models
│   ├── sentiment/    # Sentiment analysis
│   └── visualization/  # Dashboard & plots
├── tests/            # Test suite
├── config.yaml       # Configuration
└── main.py           # CLI entry point
```

## Getting Help

- **Issues**: Use the issue tracker for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the wiki for detailed guides

## Recognition

Contributors will be acknowledged in the README.md file.

Thank you for contributing to Stock Prediction CLI!
