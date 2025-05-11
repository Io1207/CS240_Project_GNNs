# CS240_Project_GNNs

<b>Graph Neural Networks</b> (GNNs) are a class of neural networks designed to operate on graph-structured data. Unlike traditional neural networks that excel at grid-like data (images, sequences), GNNs leverage the relationships and dependencies between entities represented as nodes and edges in a graph.
<br>
In this project, we focussed on implementing a GAT and a GCN on the Cora research paper dataset to perform field classification tasks.
<ol>
  <li>
    <b>Graph Convolutional Networks</b>- Introduced in 2017, GCNs adapt the concept of convolution to graph data. In a simplified view, a GCN layer updates a node's representation by taking a weighted average of its own features and the features of its neighbors.
  </li>
  <li>
    <b>Graph Attention Networks</b>- Introduced in 2018, GATs introduce an attention mechanism to the neighborhood aggregation process. Instead of treating all neighbors equally, GAT layers learn to assign different importance weights to the neighbors of a node.
  </li>
  </ol>
