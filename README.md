# GAT-PSO Community Detection Streamlit App

This is the first working Streamlit version of the GAT/DGI + PSO community detection model.
It allows users to run benchmark datasets or upload their own graph edge list, then view metrics,
training time, visualizations, number of detected communities and representative-node explanations.

## Main features

- Benchmark dataset support: **Cora**, **PubMed** and **Citeseer** through PyTorch Geometric Planetoid.
- Uploaded dataset support through CSV/TXT/edgelist files.
- Structural features for uploaded graphs without node attributes:
  - node degree
  - log degree
  - normalized degree
  - constant bias feature
- Two-layer GAT encoder trained with Deep Graph Infomax.
- Baseline comparison: **GAT/DGI + K-means**.
- Proposed method: **GAT/DGI + PSO**.
- Metrics dashboard:
  - selected number of communities
  - modularity
  - separation / silhouette score
  - conductance
  - NMI and ARI where ground-truth labels exist
  - training time
  - PSO time
  - total runtime
  - memory usage
- Visualizations:
  - model performance comparison
  - DGI-GAT training loss
  - PSO convergence
  - community size distribution
  - PCA embedding plot
  - sampled network graph
- Representative-node explanation based on:
  - centroid proximity
  - intra-community degree

## Installation

Create a virtual environment first, then install the requirements.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Note: `torch` and `torch-geometric` installation can depend on your Python version, CUDA version and operating system.
> If the generic installation fails, install PyTorch and PyTorch Geometric using the official installation instructions for your environment.

## Run the app

```bash
streamlit run app.py
```

The app will open in your browser.

## Uploaded edge-list format

The simplest upload format is a CSV with two columns:

```csv
source,target
A,B
A,C
B,D
C,D
```

The app uses the first two columns by default. You can also specify the source and target column names in the sidebar.

## Optional labels format

Labels are only required if you want NMI and ARI for your uploaded data.

```csv
node,label
A,0
B,0
C,1
D,1
```

## Optional node features format

If you upload node features, the first column should identify the node and the other columns should be numeric features.

```csv
node,feature_1,feature_2,feature_3
A,0.2,1.5,3
B,0.5,1.1,2
C,0.1,0.8,1
```

If no features are uploaded, the app generates structural features from the graph.

## Recommended first test

Use **Demo mode** with **Cora** first. After confirming that the app runs, switch to PubMed, Citeseer or your own uploaded dataset.

## Notes for large datasets

For very large social networks, start with a node limit such as 5,000 or 10,000. Full GAT/DGI training and PSO optimization can be computationally expensive.
