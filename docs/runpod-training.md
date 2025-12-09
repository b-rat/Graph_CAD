# Training LLM Latent Editor on RunPod

## Prerequisites

- RunPod account with credits
- Trained VAE checkpoint (`outputs/vae_16d/best_model.pt`)
- This repository cloned locally

## 1. Create RunPod Instance

1. Go to [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**

2. **Recommended GPU options:**
   | GPU | VRAM | Cost | Training Time |
   |-----|------|------|---------------|
   | A10G (24GB) | 24GB | ~$0.35/hr | ~20 hrs |
   | A100 (40GB) | 40GB | ~$1.00/hr | ~8 hrs |
   | A100 (80GB) | 80GB | ~$1.50/hr | ~6 hrs |

3. **Template:** Select `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

4. **Volume:** Add 50GB persistent volume (for model weights + data)

5. Click **Deploy**

## 2. Connect to Pod

```bash
# Option A: Web terminal
# Click "Connect" → "Start Web Terminal"

# Option B: SSH (if configured)
ssh root@<pod-ip> -p <port>
```

## 3. Upload Code

**Option A: Git clone (recommended)**
```bash
cd /workspace
git clone https://github.com/b-rat/Graph_CAD.git
cd Graph_CAD
```

**Option B: Upload via runpodctl**
```bash
# On local machine
brew install runpod/runpodctl/runpodctl
runpodctl send graph_cad/ scripts/ requirements-cloud-gpu.txt outputs/

# On RunPod, receive files
runpodctl receive <code-from-send>
```

## 4. Upload VAE Checkpoint

The trained VAE is required to generate training data.

```bash
# Create output directory on RunPod
mkdir -p /workspace/Graph_CAD/outputs/vae_16d

# Option A: Use runpodctl from local machine
runpodctl send outputs/vae_16d/best_model.pt

# Option B: Download from cloud storage (if uploaded)
# wget <your-cloud-url> -O /workspace/Graph_CAD/outputs/vae_16d/best_model.pt
```

## 5. Install Dependencies

```bash
cd /workspace/Graph_CAD
pip install -r requirements-cloud-gpu.txt
pip install -e .
```

**Verify CUDA:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

## 6. Generate Training Data

```bash
python scripts/generate_edit_data.py \
    --vae-checkpoint outputs/vae_16d/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data
```

This creates:
- `data/edit_data/train.json` - Training samples (40k)
- `data/edit_data/val.json` - Validation samples (5k)
- `data/edit_data/test.json` - Test samples (5k)
- `data/edit_data/metadata.json` - Generation config

**Dataset composition:**
- 70% single-parameter edits ("make leg1 20mm longer")
- 20% compound edits ("make it bigger")
- 10% no-ops ("keep it the same")

**Time:** ~2 hours (CPU-bound, generates CAD geometry for each sample)
**Cost:** ~$0.80 on A10G

## 7. Train Latent Editor

```bash
python scripts/train_latent_editor.py \
    --data-dir data/edit_data \
    --output-dir outputs/latent_editor \
    --epochs 10 \
    --batch-size 4 \
    --gradient-accumulation 8 \
    --learning-rate 2e-4 \
    --use-4bit
```

```bash
python scripts/train_latent_editor.py \
      --data-dir data/edit_data \
      --epochs 10 \
      --batch-size 8 \
      --gradient-accumulation 4 \
      --use-4bit
```

**Monitor training:**
```bash
# In separate terminal or use tmux
tail -f outputs/latent_editor/training.log
```

**Resume from checkpoint (if interrupted):**
```bash
python scripts/train_latent_editor.py \
    --data-dir data/edit_data \
    --output-dir outputs/latent_editor \
    --resume outputs/latent_editor/checkpoint-latest
```

## 8. Download Results

After training completes:

```bash
# On RunPod - compress outputs
cd /workspace/Graph_CAD
tar -czvf latent_editor_trained.tar.gz outputs/latent_editor/

# Option A: runpodctl receive on local machine
runpodctl send latent_editor_trained.tar.gz
# Then on local: runpodctl receive <code>

# Option B: Upload to cloud storage
# Example with rclone or aws cli
```

**Key files to download:**
- `outputs/latent_editor/best_model.pt` - Best checkpoint
- `outputs/latent_editor/config.json` - Model config
- `outputs/latent_editor/training_log.json` - Metrics

## 9. Stop Pod

**Important:** Stop or terminate pod when done to avoid charges.

- **Stop:** Preserves volume, stops billing for GPU
- **Terminate:** Deletes everything, stops all billing

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch-size 2 --gradient-accumulation 16

# Or use 8-bit instead of 4-bit (slightly more VRAM but sometimes more stable)
--use-8bit
```

### bitsandbytes errors
```bash
# Ensure CUDA paths are set
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reinstall bitsandbytes
pip uninstall bitsandbytes && pip install bitsandbytes --no-cache-dir
```

### Slow data loading
```bash
# Increase dataloader workers
--num-workers 4
```

### Connection dropped mid-training
Use `tmux` or `screen` to persist sessions:
```bash
tmux new -s training
# Run training command
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t training
```

## Estimated Costs

| Step | GPU | Time | Cost |
|------|-----|------|------|
| Data generation | A10G | ~2 hrs | ~$0.80 |
| Training | A10G (24GB) | ~20 hrs | ~$7 |
| Training | A100 (40GB) | ~8 hrs | ~$8 |
| Training | A100 (80GB) | ~6 hrs | ~$9 |

**Total (A10G):** ~22 hours, ~$8-9

*Costs are approximate and may vary.*
