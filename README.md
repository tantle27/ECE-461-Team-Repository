
# ECE-461 Team Repository

## Team Members
- Trevor
- Jackson  
- Jain Iftesam
- William Ilkanic

## 📁 Project Structure

```
ECE-461-Team-Repository/
├── src/                           # Main source code
│   ├── api/                       # API-related modules
│   ├── git/                       # Git integration utilities
│   ├── metrics/                   # Metric evaluation classes
│   │   ├── BaseMetric.py         # Legacy base metric (deprecated)
│   │   ├── base_metric.py        # Modern base metric abstract class
│   │   ├── bus_factor_metric.py  # Bus factor evaluation
│   │   ├── code_quality_metric.py # Code quality assessment
│   │   ├── community_rating_metric.py # Community engagement metrics
│   │   ├── dataset_availability_metric.py # Dataset accessibility
│   │   ├── dataset_quality_metric.py # Dataset quality evaluation
│   │   ├── license_metric.py     # License compatibility check
│   │   ├── performance_claims_metric.py # Performance validation
│   │   ├── ramp_up_time_metric.py # Learning curve assessment
│   │   ├── size_metric.py        # Codebase size metrics
│   │   └── MetricEval.py         # Legacy metric evaluator (deprecated)
│   ├── app.py                    # Main application entry point
│   ├── handlers.py               # URL and request handlers
│   ├── metric_eval.py            # Modern metric evaluation orchestrator
│   ├── net_scorer.py             # Weighted scoring system
│   ├── repo_context.py           # Repository context data structure
│   ├── score_result.py           # Score result data structure
│   └── url_router.py             # URL routing and parsing
├── tests/                        # Comprehensive test suite
│   ├── test_base_metric.py       # BaseMetric abstract class tests
│   ├── test_cli_layer.py         # CLI interface tests
│   ├── test_community_rating_metric.py # Community metric tests
│   ├── test_error_handling.py    # Error handling tests
│   ├── test_extended_metrics.py  # Extended metric functionality tests
│   ├── test_integration.py       # Integration tests
│   ├── test_metric_eval.py       # MetricEval orchestrator tests
│   ├── test_net_scorer.py        # NetScorer weighted scoring tests
│   ├── test_repo_context.py      # RepoContext data structure tests
│   ├── test_routing_layer.py     # URL routing tests
│   ├── test_run_install.py       # Installation script tests
│   ├── test_run_test.py          # Test runner tests
│   └── test_url_handlers.py      # URL handler tests
├── .flake8                       # Flake8 linting configuration
├── pytest.ini                   # Pytest configuration
├── requirements.txt              # Python dependencies
├── run                          # Main executable script
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Git

### Installation

#### Method 1: Using the run script (Recommended)
```bash
# Make the run script executable (Unix/Linux/Mac)
chmod +x run

# If you get "cannot execute: required file not found", convert line endings
dos2unix run

# Install dependencies
./run install
```

#### Method 2: Manual installation
```bash
# Create and activate virtual environment
python3 -m venv venv

# On Windows:
venv\Scripts\activate

# On Unix/Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Alternative Installation (if ./run install doesn't work)
```bash
python3 -m venv venv
cd venv
source bin/activate  # On Windows: Scripts\activate
cd ..
./run install
```

## 🧪 Running Tests

### Quick Test Run
```bash
# Run all tests
./run test

# Or manually:
python -m pytest tests/ -v
```

### Comprehensive Testing
```bash
# Run all tests with coverage
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

# Run specific test categories
python -m pytest tests/test_integration.py -v          # Integration tests
python -m pytest tests/test_error_handling.py -v      # Error handling tests
python -m pytest tests/test_metric_eval.py -v         # Metric evaluation tests
```

### Test Coverage Report
After running tests with coverage, open `htmlcov/index.html` in your browser to view detailed coverage reports.

## 📊 Code Quality

### Linting
The project uses flake8 for code style enforcement:
```bash
# Check all source files
flake8 src/

# Check all test files
flake8 tests/

# Check specific file
flake8 src/app.py
```

### Style Standards
- PEP 8 compliant
- Maximum line length: 88 characters
- Proper import ordering
- No unused variables or imports

## 📚 Usage

### Running the Application
```bash
# Execute the main application
./run

# Or manually:
python src/app.py
```

### Evaluating Models
The system evaluates AI/ML models across multiple metrics:
- **Bus Factor**: Team knowledge distribution
- **Code Quality**: Code maintainability and best practices
- **Community Rating**: User engagement and feedback
- **Dataset Availability**: Data accessibility and documentation
- **Dataset Quality**: Data completeness and accuracy
- **License Compatibility**: Legal compliance assessment
- **Performance Claims**: Benchmark validation
- **Ramp-up Time**: Learning curve estimation
- **Size Metrics**: Codebase complexity assessment

## 🤝 Contributing

1. Follow the existing code style (flake8 compliant)
2. Add comprehensive tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting

## 📄 License

This project is part of the ECE-461 coursework.