import torch
import numpy as np
from scipy.stats import pearsonr
from graph_cad.data.dataset import create_variable_data_loaders
from graph_cad.models.graph_vae import VariableGraphVAEConfig, VariableGraphVAEEncoder
from graph_cad.models.transformer_decoder import TransformerDecoderConfig, TransformerGraphVAE
# Load model
ckpt = torch.load("outputs/vae_transformer/best_model.pt", map_location="cpu")
encoder = VariableGraphVAEEncoder(VariableGraphVAEConfig(**ckpt["encoder_config"]))
model = TransformerGraphVAE(encoder, TransformerDecoderConfig(**ckpt["decoder_config"]))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# Get test data
_, _, test_loader = create_variable_data_loaders(train_size=100, val_size=100, test_size=500, batch_size=32)
# Collect latents and parameters
all_z, all_params = [], []
with torch.no_grad():
    for batch in test_loader:
        mu, _ = model.encode(batch.x, batch.face_types, batch.edge_index, batch.edge_attr, batch.batch, batch.node_mask)
        all_z.append(mu)
        all_params.append(batch.y)
z = torch.cat(all_z).numpy()
params = torch.cat(all_params).numpy()
# Correlations
param_names = ["leg1", "leg2", "width", "thickness"]
print("Parameter correlations with latent dimensions:")
print("-" * 50)
for i, name in enumerate(param_names):
    correlations = [abs(pearsonr(z[:, j], params[:, i])[0]) for j in range(z.shape[1])]
    max_corr = max(correlations)
    max_dim = correlations.index(max_corr)
    print(f"{name:12s}: r={max_corr:.3f} (dim {max_dim})")
