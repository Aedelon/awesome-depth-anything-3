# Contributing to awesome-depth-anything-3

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Important Note

This is an **optimized fork** of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance.

- **Model/architecture changes** should be proposed to the [upstream repository](https://github.com/ByteDance-Seed/Depth-Anything-3)
- **Optimization/deployment improvements** are welcome here

## Development Setup

```bash
# Clone the repository
git clone https://github.com/Aedelon/awesome-depth-anything-3.git
cd awesome-depth-anything-3

# Install with development dependencies (using uv)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_adaptive_batching.py -v

# Run with coverage
uv run pytest tests/ --cov=src/depth_anything_3
```

## Code Style

We use `ruff` for linting and formatting:

```bash
# Check for issues
uv run ruff check src/

# Auto-fix issues
uv run ruff check src/ --fix

# Format code
uv run ruff format src/
```

## Pre-commit Hooks

We recommend using pre-commit hooks:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature (`git checkout -b feature/amazing-feature`)
3. **Make your changes** with clear, descriptive commits
4. **Run tests** and linting
5. **Update documentation** if needed
6. **Push** to your fork and **open a Pull Request**

### PR Guidelines

- Keep PRs focused on a single change
- Include tests for new functionality
- Update CHANGELOG.md for user-facing changes
- Ensure CI passes before requesting review

## Types of Contributions Welcome

### Highly Welcome

- Performance optimizations
- Bug fixes
- Documentation improvements
- Test coverage improvements
- CI/CD improvements
- Device compatibility (CUDA, MPS, CPU)

### Discuss First

- New CLI commands
- API changes
- New dependencies

### Redirect to Upstream

- Model architecture changes
- Training code changes
- New model variants

## Reporting Issues

When reporting bugs, please include:

- Python version
- PyTorch version
- Device type (CUDA/MPS/CPU)
- Minimal reproduction code
- Full error traceback

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
