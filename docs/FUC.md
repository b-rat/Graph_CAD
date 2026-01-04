```bash
python scripts/generate_edit_data.py \
      --vae-checkpoint outputs/vae_16d_lowbeta/best_model.pt \
      --num-samples 50000 \
      --output data/edit_data && \
python scripts/train_latent_editor.py \
      --data-dir data/edit_data \
      --output-dir outputs/latent_editor_vae16d_lowbeta \
      --epochs 10 \
      --batch-size 8 \
      --gradient-accumulation 4
```

```bash
cd /workspace/Graph_CAD
GIT_TOKEN={{ RUNPOD_SECRET_GIT_TOKEN }}
git remote set-url origin https://GIT_TOKEN@github.com/b-rat/Graph_CAD.git
git config user.email "brian.ratliff@mechnlengr.com"
git config user.name "b-rat"
git pull
apt-get update && apt-get install -y git-lfs && git lfs install && git lfs pull
pip install -r requirements-cloud-gpu.txt && pip install -e . && pip install hf_transfer
echo 'export HF_HOME=/workspace/.cache/huggingface/' >> ~/.bashrc # changes default download for hugging face to network volume
source ~/.bashrc            # re-read bash file
git add -f outputs/latent_editor/best_model.pt
git add -f outputs/latent_editor/training_results.json
git commit -m "Add trained latent editor checkpoint"
git push
git reset                   #unstage everything add file name to unstage single file
git reset --soft HEAD~1     #keep changes staged remove commit one commit back
git reset HEAD~1            #remove commit and unstage
apt update && apt install tmux -y
apt-get update && apt-get install -y libxrender1 libxext6
```

```bash
python scripts/train_latent_editor.py \
      --data-dir data/edit_data \
      --epochs 10 \
      --batch-size 8 \
      --gradient-accumulation 4 \
      --resume outputs/latent_editor/checkpoint_epoch_9.pt
```

```bash
apt update && apt install tmux -y

# Start a new named session
tmux new -s training

# You're now "inside" tmux - run your stuff
python finetune.py

# Detach (leaves it running in background)
Ctrl+B, then D

# List running sessions
tmux ls

# Reattach to your session
tmux attach -s training
tmux attach # if only one session
```

```bash
python scripts/infer_latent_editor.py \
      --random-bracket \
      --instruction "make leg1 20mm longer" \
      --verbose
```

```bash
python scripts/infer_latent_editor.py --random-bracket --instruction "make it wider" --verbose
python scripts/infer_latent_editor.py --random-bracket --instruction "increase hole1 diameter by 3mm" --verbose
python scripts/infer_latent_editor.py --random-bracket --instruction "make leg2 shorter" --verbose
```

```bash
python scripts/infer_latent_editor.py \
    --random-bracket \
    --instruction "make leg1 20mm longer" \
    --seed $RANDOM \
    --verbose
```

```bash
#training sequence end-to-end
rm -rf data/feature_regressor_cache && \
python scripts/train_vae.py \
    --epochs 100 \
    --latent-dim 16 \
    --target-beta 0.01 \
    --free-bits 2.0 \
    --output-dir outputs/vae_16d_lowbeta && \
python scripts/train_feature_regressor.py \
    --vae-checkpoint outputs/vae_16d_lowbeta/best_model.pt \
    --train-size 10000 \
    --epochs 100 \
    --output-dir outputs/feature_regressor && \
python scripts/generate_edit_data.py \
    --vae-checkpoint outputs/vae_16d_lowbeta/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data && \
python scripts/train_latent_editor.py \
    --data-dir data/edit_data \
    --epochs 10 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --output-dir outputs/latent_editor
```

```bash
# On runpod, run:
# Quick test (4 trials)
python scripts/explore_instruction_domain.py --num-brackets 1 --quick
# Full exploration (480 trials with 10 brackets)
python scripts/explore_instruction_domain.py --num-brackets 10
# Larger study
python scripts/explore_instruction_domain.py --num-brackets 50 --output outputs/exploration/full_study_251213.json
```

```bash
python scripts/train_vae.py \
    --latent-dim 16 --target-beta 0.01 --free-bits 2.0 \
    --aux-weight 1.0 \
    --output-dir outputs/vae_aux
```

```bash
python scripts/train_feature_regressor.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --train-size 10000 --epochs 100 \
    --output-dir outputs/feature_regressor_aux && \
python scripts/generate_edit_data.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 --output data/edit_data_aux && \
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_aux --epochs 10 \
    --batch-size 8 --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_aux

# After training, test with:
python scripts/infer_latent_editor.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --regressor-checkpoint outputs/feature_regressor_aux/best_model.pt \
    --editor-checkpoint outputs/latent_editor_aux/best_model.pt \
    --random-bracket \
    --instruction "make leg1 20mm longer"
```

```bash
# Option 3: Separate sessions
tmux new -s job1
# run job 1, then Ctrl+b d to detach
tmux new -s job2
# run job 2, then Ctrl+b d to detach
tmux new -s monitor
watch -n 5 nvidia-smi # monitors gpu performance every 5 seconds
# Switch: 
tmux attach -t job1

# Quick reference:
Ctrl+b d # - detach (jobs keep running)
tmux ls # - list sessions
tmux attach -t <name> # - reattach
```


Ablation to be done in parallel
```bash
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_aux \
    --epochs 20 \
    --learning-rate 2e-4 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_aux_ep20_lr2e4

python scripts/train_latent_editor.py \
    --data-dir data/edit_data_aux \
    --epochs 20 \
    --learning-rate 1e-4 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_aux_ep20_lr1e4
```

```bash
python scripts/infer_latent_editor.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --regressor-checkpoint outputs/feature_regressor_aux/best_model.pt \
    --editor-checkpoint outputs/latent_editor_direction/best_model.pt \
    --random-bracket \
    --instruction "make leg1 20mm longer" \
    --seed $RANDOM \
    --verbose
```

```bash
python scripts/explore_instruction_domain.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --regressor-checkpoint outputs/feature_regressor_aux/best_model.pt \
    --editor-checkpoint outputs/latent_editor_aux_ep20_lr1e4/best_model.pt \
    --num-brackets 50 \
    --output outputs/exploration/full_study_251214_best_lr1e4.json
```

```bash
python3 -c "
import json
import numpy as np
data = json.load(open('data/edit_data_aux/train.json'))
leg1_inc = [s['delta_z'] for s in data if 'leg1' in s['instruction'].lower() and 'longer' in s['instruction'].lower()]
leg2_inc = [s['delta_z'] for s in data if 'leg2' in s['instruction'].lower() and 'longer' in s['instruction'].lower()]
m1, m2 = np.mean(leg1_inc, axis=0), np.mean(leg2_inc, axis=0)
cos = np.dot(m1,m2)/(np.linalg.norm(m1)*np.linalg.norm(m2))
print(f'Cosine(leg1_inc, leg2_inc): {cos:.3f}')
print('Should be low/negative if leg1 and leg2 are distinguishable')
"
```

```bash
python scripts/explore_instruction_domain.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --regressor-checkpoint outputs/feature_regressor_aux/best_model.pt \
    --editor-checkpoint outputs/latent_editor_contrastive_w0.1/best_model.pt \
    --num-brackets 50 \
    --output outputs/exploration/full_study_contrastive_251217.json
```

```bash
python scripts/generate_edit_data.py \
    --paired \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_paired && \
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_paired \
    --contrastive-weight 0.5 \
    --epochs 20 \
    --output-dir outputs/latent_editor_contrastive
```

```bash
python scripts/generate_edit_data.py \
    --vae-checkpoint outputs/vae_aux/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_direction && \
python scripts/train_latent_editor.py \
    --data-dir data/edit_data_direction \
    --direction-weight 0.5 \
    --epochs 20 \
    --output-dir outputs/latent_editor_direction
```

```bash
echo $HF_HOME
To reclaim container disk space:
# Set HF_HOME first to load huggingface model into network volume
export HF_HOME=/workspace/.cache/huggingface/
# Delete the container disk cache
rm -rf ~/.cache/huggingface/
# Now container disk drops from 83% to ~10-15%
```

```bash
export TOKENIZERS_PARALLELISM=false
# OR
TOKENIZERS_PARALLELISM=false python your_script.py
```

```bash
python scripts/analyze_vae_asymmetry.py --n-samples 200 --output outputs/vae_asymmetry.json
```

```bash
# New training script for variable geometry
python scripts/train_variable_vae.py \
    --train-size 5000 \
    --epochs 100 \
    --latent-dim 32 \
    --output-dir outputs/vae_variable
```

```bash
python scripts/train_variable_vae.py \
    --train-size 5000 --val-size 500 --test-size 500 \
    --epochs 100 \
    --latent-dim 32 \
    --output-dir outputs/vae_variable
```

```bash
python scripts/generate_variable_edit_data.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --num-samples 50000 \
    --output data/edit_data_variable \
    --paired \
    --device cuda && \
TOKENIZERS_PARALLELISM=false python scripts/train_latent_editor.py \
    --data-dir data/edit_data_variable \
    --latent-dim 32 \
    --direction-weight 0.5 \
    --epochs 20 \
    --batch-size 8 \
    --gradient-accumulation 4 \
    --output-dir outputs/latent_editor_variable
```

```bash
python scripts/train_variable_feature_regressor.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --train-size 10000 \
    --epochs 100 \
    --cache-dir data/feature_regressor_variable_cache \
    --output-dir outputs/feature_regressor_variable \
    --device cuda
```

```bash
# for training a feature regressor directly from latent, skipping decoder
python scripts/train_latent_regressor.py \
    --vae-checkpoint outputs/vae_variable/best_model.pt \
    --train-size 10000 \
    --epochs 100 \
    --cache-dir data/latent_regressor_cache \
    --output-dir outputs/latent_regressor \
    --device cuda
```