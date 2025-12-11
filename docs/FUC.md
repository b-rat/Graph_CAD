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
git remote set-url origin https://YOUR_TOKEN@github.com/b-rat/Graph_CAD.git
git config user.email "brian.ratliff@mechnlengr.com"
git config user.name "b-rat"
git pull
apt-get update && apt-get install -y git-lfs
git lfs install
git lfs pull
pip install -r requirements-cloud-gpu.txt
pip install -e .
pip install hf_transfer
git add -f outputs/latent_editor/best_model.pt
git add -f outputs/latent_editor/training_results.json
git commit -m "Add trained latent editor checkpoint"
git push
git reset                   #unstage everything add file name to unstage single file
git reset --soft HEAD~1     #keep changes staged remove commit one commit back
git reset HEAD~1            #remove commit and unstage
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

