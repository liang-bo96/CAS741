# Contributing to McMaster EEG Visualization Project

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Coding Standards

- Follow PEP 8 style guide for Python code
- Use type hints when appropriate
- Write docstrings for all functions, classes, and modules

### Testing Requirements

All contributions should maintain or improve the current code coverage. The project uses pytest for testing.

Before submitting a pull request:

1. Add appropriate tests for your changes
2. Verify all tests pass: `python -m pytest src`
3. Check coverage: `python -m pytest src --cov=src`
4. Fix any coverage regressions by adding more tests

## Continuous Integration

The project uses GitHub Actions for CI/CD. Every pull request will automatically trigger the test suite to run. The PR cannot be merged if tests are failing.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.