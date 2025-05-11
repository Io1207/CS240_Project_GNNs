import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data

from gat import GAT, DatasetType, load_graph_data, build_edge_index, get_training_state, device
import os
from torch_geometric.utils import to_networkx

# === Load GAT model from file ===
@st.cache_resource
def load_trained_model(path="models/binaries/gat_000000.pth"):
    model_state = torch.load(path, map_location='cpu')

    model = GAT(
        num_of_layers=model_state['num_of_layers'],
        num_heads_per_layer=model_state['num_heads_per_layer'],
        num_features_per_layer=model_state['num_features_per_layer'],
        add_skip_connection=model_state['add_skip_connection'],
        bias=model_state['bias'],
        dropout=model_state['dropout'],
        log_attention_weights=False
    )

    model.load_state_dict(model_state['state_dict'], strict=True)
    model.eval()
    return model

# === Load Cora graph data ===
@st.cache_data
def load_data():
    config = {
        'dataset_name': DatasetType.CORA.name,
        'should_visualize': False
    }
    return load_graph_data(config, device=torch.device('cpu'))

# === Build subgraph for visualization ===
def get_subgraph(G, node_id, depth=2):
    nodes = nx.single_source_shortest_path_length(G, node_id, cutoff=depth).keys()
    return G.subgraph(nodes)

def draw_subgraph(G_sub, node_labels):
    fig, ax = plt.subplots(figsize=(6, 6))
    pos = nx.spring_layout(G_sub, seed=42)
    node_colors = [int(node_labels[n]) for n in G_sub.nodes()]
    nx.draw(G_sub, pos, node_color=node_colors, cmap=plt.cm.tab10,
            with_labels=True, node_size=300, font_size=8)
    return fig

# === Streamlit App ===
st.title("Trained GAT on Cora: Node Classification Demo")

# Load data & model
node_features, node_labels, edge_index, train_idx, val_idx, test_idx = load_data()
model = load_trained_model()

# Predict
with torch.no_grad():
    out = model((node_features, edge_index))[0]
    pred = out.argmax(dim=1)

# Node selection
node_id = st.slider("Select a node ID", 0, node_features.shape[0] - 1, 0)

st.write(f"**True Label:** {node_labels[node_id].item()}")
st.write(f"**Predicted Label:** {pred[node_id].item()}")

probs = F.softmax(out[node_id], dim=0)
st.write("**Class Probabilities:**")
st.bar_chart(probs.detach().numpy())

# Subgraph Visualization
data_obj = Data(x=node_features, edge_index=edge_index)
G_nx = to_networkx(data_obj, to_undirected=True)
subG = get_subgraph(G_nx, node_id, depth=2)

st.subheader("2-Hop Neighborhood Graph")
fig = draw_subgraph(subG, node_labels)
st.pyplot(fig)
