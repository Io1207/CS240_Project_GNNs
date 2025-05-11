# import streamlit as st
# import torch
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.datasets import Planetoid
# from torch_geometric.utils import to_networkx
# import torch_geometric.transforms as T
# from networkx.algorithms.link_analysis.pagerank_alg import pagerank
# from networkx.algorithms.centrality import eigenvector_centrality

# # Load the Cora dataset
# dataset = Planetoid(root='data', name='Cora', transform=T.NormalizeFeatures())
# data = dataset[0]

# # Build the NetworkX graph
# G = to_networkx(data, to_undirected=True)
# labels = data.y.numpy()

# # Centrality measures
# pr = pagerank(G)
# ec = eigenvector_centrality(G, max_iter=1000)

# # Title
# st.title("Cora GCN Demo")

# # Node selection
# node_id = st.slider("Select a node (paper)", 0, data.num_nodes - 1, 0)

# st.write(f"**Node ID:** {node_id}")
# st.write(f"**True label:** {labels[node_id]}")
# st.write("**Feature vector (truncated):**", data.x[node_id][:10])

# # Centrality display
# centrality_type = st.selectbox("Vertex Size by:", ["Degree", "PageRank", "Eigenvector Centrality"])

# if centrality_type == "Degree":
#     sizes = [G.degree(n)*20 for n in G.nodes()]
# elif centrality_type == "PageRank":
#     sizes = [pr[n]*3000 for n in G.nodes()]
# else:
#     sizes = [ec[n]*3000 for n in G.nodes()]

# # Plot graph
# st.subheader("Graph View (Zoomed Around Node)")
# sub_nodes = list(nx.single_source_shortest_path_length(G, node_id, cutoff=2).keys())
# subgraph = G.subgraph(sub_nodes)

# fig, ax = plt.subplots(figsize=(8, 6))
# pos = nx.spring_layout(subgraph)
# nx.draw(subgraph, pos,
#         node_size=[sizes[n] for n in subgraph.nodes()],
#         node_color=[labels[n] for n in subgraph.nodes()],
#         with_labels=False, ax=ax, cmap=plt.cm.rainbow)

# st.pyplot(fig)

import streamlit as st
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx
import torch_geometric.transforms as T
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality import eigenvector_centrality

# ----------------------------
# GCN Model (same as in your CoraGCN.ipynb)
# ----------------------------
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# ----------------------------
# Load data and model
# ----------------------------
@st.cache_data
def load_data():
    dataset = Planetoid(root='data', name='Cora', transform=T.NormalizeFeatures())
    return dataset

@st.cache_resource
def load_model():
    dataset = load_data()
    data = dataset[0]
    model = GCN(dataset.num_node_features, 64, dataset.num_classes)
    model.load_state_dict(torch.load("gcnCora.pt", map_location=torch.device('cpu')))
    model.eval()
    return model, data, dataset

@st.cache_data
def build_graph(_data):
    return to_networkx(data, to_undirected=True)

# ----------------------------
# UI + Prediction + Visualization
# ----------------------------
st.title("Cora GCN Demo (Pretrained)")

model, data, dataset = load_model()
G = build_graph(data)
labels = data.y.numpy()

# Compute prediction
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim=1)

# Node selection
node_id = st.slider("Select a node (paper)", 0, data.num_nodes - 1, 0)

st.write(f"**Node ID:** {node_id}")
st.write(f"**True Label:** {labels[node_id]}")
st.write(f"**Predicted Label:** {pred[node_id].item()}")
st.write("**Feature Vector (Truncated):**", data.x[node_id][:10])

# Probabilities
probs = F.softmax(logits[node_id], dim=0)
st.write("**Class Probabilities:**")
st.bar_chart(probs.detach().numpy())

# Centrality
pr = pagerank(G)
ec = eigenvector_centrality(G, max_iter=1000)
centrality_type = st.selectbox("Vertex Size by:", ["Degree", "PageRank", "Eigenvector Centrality"])

if centrality_type == "Degree":
    sizes = [G.degree(n)*20 for n in G.nodes()]
elif centrality_type == "PageRank":
    sizes = [pr[n]*3000 for n in G.nodes()]
else:
    sizes = [ec[n]*3000 for n in G.nodes()]

# Plot subgraph
st.subheader("2-Hop Neighborhood Graph")
sub_nodes = list(nx.single_source_shortest_path_length(G, node_id, cutoff=2).keys())
subgraph = G.subgraph(sub_nodes)

fig, ax = plt.subplots(figsize=(8, 6))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw(subgraph, pos,
        node_size=[sizes[n] for n in subgraph.nodes()],
        node_color=[labels[n] for n in subgraph.nodes()],
        with_labels=False, ax=ax, cmap=plt.cm.rainbow)

st.pyplot(fig)
