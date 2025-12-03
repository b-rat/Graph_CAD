# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Graph_CAD combines Graph Autoencoders with Generative AI for Computer-Aided Design (CAD). The project aims to leverage graph-based representations of CAD models for machine learning tasks.

## Project Structure

```
graph_cad/          # Main package
├── models/         # Graph autoencoder and generative model definitions
├── data/           # Data loaders, preprocessors, CAD parsers
├── utils/          # Graph operations, visualization, helpers
└── training/       # Training loops, loss functions, optimizers

tests/              # Test suite
├── unit/           # Unit tests for individual components
└── integration/    # End-to-end workflow tests

data/               # Data directory (gitignored)
├── raw/            # Original CAD files
├── processed/      # Preprocessed graph representations
└── models/         # Saved model checkpoints

notebooks/          # Jupyter notebooks for experimentation
scripts/            # CLI scripts for training, evaluation, preprocessing
docs/               # Documentation and architecture diagrams
```

## Development Commands

**Setup environment:**
```bash
# Install dependencies
pip install -e .                    # Install package in development mode
pip install -r requirements-dev.txt # Install development tools

# Or install everything at once
pip install -e ".[dev]"
```

**Testing:**
```bash
pytest                              # Run all tests
pytest tests/unit                   # Run unit tests only
pytest tests/integration            # Run integration tests only
pytest -v -k "test_specific"        # Run specific test
pytest --cov=graph_cad              # Run with coverage report
```

**Code quality:**
```bash
black graph_cad tests               # Format code
ruff check graph_cad tests          # Lint code
ruff check --fix graph_cad tests    # Auto-fix linting issues
mypy graph_cad                      # Type checking
```

**Run from scripts directory:**
```bash
# Example commands (create scripts as needed)
python scripts/train.py --config config.yaml
python scripts/evaluate.py --checkpoint path/to/model.pt
python scripts/preprocess_cad.py --input data/raw --output data/processed
```

## Key Considerations

**Graph Representation**: CAD models will likely need conversion to/from graph structures. Consider how to represent geometric primitives, constraints, and topological relationships as nodes and edges.

**ML Framework**: When choosing between PyTorch, TensorFlow, or JAX, document the decision rationale as this affects the entire codebase architecture.

**Data Pipeline**: CAD data formats (STEP, IGES, STL, etc.) will need parsers and converters. Keep data loading separate from model code.

**Model Architecture**: Graph autoencoders typically use Graph Neural Networks (GNNs). Common libraries include PyTorch Geometric, DGL, or Spektral. Document which library is chosen and why.
