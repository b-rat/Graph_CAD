# Data Directory

This directory contains datasets and trained models.

## Structure

- `raw/` - Original CAD files and datasets (not processed)
- `processed/` - Preprocessed data ready for training (graph representations)
- `models/` - Saved model checkpoints and weights

## Notes

- Add data files to `.gitignore` to avoid committing large files
- Consider using Git LFS for tracking model checkpoints
- Document data sources and preprocessing steps for reproducibility
