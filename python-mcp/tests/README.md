# Running Tests

This directory contains the test suite for the Papr Memory MCP Server. The tests are written using pytest and include both synchronous and asynchronous tests.

## Setup

Before running the tests, make sure you have installed all the required dependencies. You can install them using one of the following methods:

1. Install with all test dependencies:
```bash
pip install ".[test]"
```

2. Or using hatch (recommended):
```bash
hatch env create
```

## Running Tests

### Basic Test Run

To run all tests:
```bash
pytest
```

### Run with Coverage Report

To run tests with coverage reporting:
```bash
pytest --cov=. --cov-report=term-missing
```

### Run Specific Test Files

To run tests from a specific file:
```bash
pytest tests/test_specific_file.py
```

### Run Tests with Verbose Output

For detailed test output:
```bash
pytest -v
```

## Test Configuration

The test configuration is defined in `pyproject.toml` with the following settings:

- `asyncio_mode = "auto"`: Automatically handles async tests
- `testpaths = ["tests"]`: Tests are located in the `tests` directory
- `python_files = ["test_*.py"]`: Test files must start with `test_`

## Environment Variables

Some tests may require environment variables to be set. Create a `.env` file in the root directory with the following variables:

```env
MEMERY_SERVER_URL=your_server_url
PAPR_API_KEY=your_api_key
```

## Writing Tests

When writing new tests:

1. Create test files with the prefix `test_`
2. Use async functions for testing async code:
```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    # Your async test code here
    pass
```

3. Use fixtures when needed:
```python
import pytest

@pytest.fixture
def sample_data():
    return {"key": "value"}

def test_with_fixture(sample_data):
    assert sample_data["key"] == "value"
```

## Debugging Tests

For debugging tests, you can use:

1. Print statements with `-s` flag:
```bash
pytest -s
```

2. Debug on errors:
```bash
pytest --pdb
```

3. Increase verbosity:
```bash
pytest -vv
```

## CI/CD Integration

The test suite is designed to be run in CI/CD pipelines. Make sure all tests pass before submitting pull requests:

```bash
pytest --cov=. --cov-report=xml
```

This will generate a coverage report that can be used by CI tools. 