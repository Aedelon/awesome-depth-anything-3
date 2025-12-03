# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-03

### Added

- **Model Caching**: ~200x faster model loading after first use via `ModelCache` singleton
- **Adaptive Batching**: Automatic batch size optimization based on available GPU memory
  - `batch_inference()` method with `batch_size="auto"` option
  - `get_optimal_batch_size()` for memory-aware batch sizing
- **CLI Batching Options**: `--batch-size`, `--max-batch-size`, `--target-memory-utilization`
- **Apple Silicon Optimizations**: Smart CPU/GPU preprocessing selection for MPS
- **GPU Preprocessing**: Kornia-based GPU preprocessing with NVJPEG support on CUDA
- **Comprehensive Benchmarks**: Performance comparison scripts and documentation
- **PyPI Package**: Published as `awesome-depth-anything-3`
- **CI/CD**: GitHub Actions for testing, linting, and PyPI publishing
- **HF Spaces Demo**: Interactive Gradio demo on Hugging Face
- **Colab Tutorial**: Interactive notebook with examples

### Changed

- Package renamed from `depth-anything-3` to `awesome-depth-anything-3`
- Improved error handling in CLI commands
- Better logging with configurable levels

### Credits

This package is an optimized fork of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
by ByteDance. All model architecture and weights are their work. See README for full attribution.
