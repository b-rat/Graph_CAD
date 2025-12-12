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
python scripts/train_feature_regressor.py \
      --vae-checkpoint outputs/vae_16d_lowbeta/best_model.pt \
      --cache-dir data/feature_regressor_cache \
      --train-size 10000 --epochs 100
```

```bash
rm -rf outputs/inference && python scripts/infer_latent_editor.py \
      --random-bracket \
      --instruction "make leg1 20mm longer" \
      --seed 42 \
      --verbose
```
```bash
python3 -c "
  import numpy as np
  from graph_cad.data import LBracket

  rng = np.random.default_rng(42)
  bracket = LBracket.random(rng)
  print('Ground truth params:')
  for k, v in bracket.to_dict().items():
      print(f'  {k}: {v:.4f}')
  "
```

```bash
import numpy as np
from graph_cad.data import LBracket

rng = np.random.default_rng(42)
bracket = LBracket.random(rng)
print('Ground truth params:')
for k, v in bracket.to_dict().items():
    print(f'  {k}: {v:.4f}')
```
```bash
python3 << 'EOF'
import numpy as np
import torch
from graph_cad.data import LBracket, extract_graph_from_solid
from graph_cad.training.vae_trainer import load_checkpoint

# Generate same bracket
rng = np.random.default_rng(42)
bracket = LBracket.random(rng)
print("GT leg1_length:", bracket.leg1_length)

# Extract graph
solid = bracket.to_solid()
graph = extract_graph_from_solid(solid)
print("Graph: nodes=%d, edges=%d" % (graph.num_faces, graph.num_edges))

# Load VAE and encode
vae, _ = load_checkpoint("outputs/vae_16d_lowbeta/best_model.pt", device="cuda")
vae.eval()

x = torch.tensor(graph.node_features, dtype=torch.float32, device="cuda")
edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device="cuda")
edge_attr = torch.tensor(graph.edge_features, dtype=torch.float32, device="cuda")

with torch.no_grad():
    mu, logvar = vae.encode(x, edge_index, edge_attr, batch=None)
    z = mu  # deterministic in eval

print("z_src:", [round(x, 3) for x in z.cpu().numpy().flatten()])
EOF
```
```bash
python3 << 'EOF'
import numpy as np
from graph_cad.data import LBracket, extract_graph_from_solid

rng = np.random.default_rng(42)
bracket = LBracket.random(rng)
solid = bracket.to_solid()
graph = extract_graph_from_solid(solid)

print("Node features (first 3 faces):")
print(graph.node_features[:3])
print("\nEdge features (first 3 edges):")
print(graph.edge_features[:3])
EOF
```