"""
Core GAT/DGI + PSO community detection utilities for the Streamlit web app.

The implementation follows the uploaded notebook logic:
1. Load benchmark or uploaded graph data.
2. Preprocess graph and create/align node features.
3. Train a two-layer GAT encoder using Deep Graph Infomax.
4. Generate embeddings.
5. Compare GAT/DGI + K-means against GAT/DGI + PSO.
6. Produce metrics, community labels, representative nodes and runtime values.
"""

from __future__ import annotations

import gc
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    from torch_geometric.datasets import Planetoid
    from torch_geometric.nn import DeepGraphInfomax, GATConv
    from torch_geometric.utils import from_networkx

    PYG_AVAILABLE = True
    PYG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on installation environment
    PYG_AVAILABLE = False
    PYG_IMPORT_ERROR = exc
    Planetoid = None
    DeepGraphInfomax = None
    GATConv = None
    from_networkx = None


@dataclass
class ModelConfig:
    dataset_name: str = "Cora"
    use_planetoid: bool = True
    largest_component_only: bool = True
    max_nodes: Optional[int] = None
    seed: int = 42

    hidden_dim: int = 64
    embedding_dim: int = 64
    heads: int = 4
    dropout: float = 0.2
    epochs: int = 30
    lr: float = 0.001
    weight_decay: float = 5e-4

    k_min: int = 2
    k_max: int = 10
    known_k: Optional[int] = None
    swarm_size: int = 20
    max_iter: int = 80
    alpha_modularity: float = 0.45
    beta_separation: float = 0.35
    gamma_balance: float = 0.20
    modularity_edge_sample: Optional[int] = 20_000

    top_representatives: int = 5
    pso_enabled: bool = True

# -----------------------------------------------------------------------------
# Scalability sampling and repeated-trial evaluation
# -----------------------------------------------------------------------------

@dataclass
class ScalabilityConfig:

    node_sizes: Tuple[int, ...] = (500, 1_000, 2_500, 5_000, 10_000)
    repeats: int = 3
    sampling_method: str = "snowball"  # options: "snowball", "random"
    seed: int = 42
    largest_component_only: bool = True
    skip_sizes_larger_than_graph: bool = True
    use_sample_label_count_for_k: bool = True

    
    epochs: Optional[int] = None
    swarm_size: Optional[int] = None
    max_iter: Optional[int] = None
    modularity_edge_sample: Optional[int] = 20_000
    pso_enabled: Optional[bool] = None
def _largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a copied graph."""
    if graph.number_of_nodes() == 0:
        return graph.copy()
    if nx.is_connected(graph):
        return graph.copy()
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def _random_node_sample(graph: nx.Graph, target_nodes: int, seed: int) -> List[int]:
    """Uniformly sample nodes from the graph."""
    rng = np.random.default_rng(seed)
    nodes = np.array(list(graph.nodes()))
    sample_size = min(target_nodes, len(nodes))
    return sorted(rng.choice(nodes, size=sample_size, replace=False).tolist())


def _snowball_node_sample(graph: nx.Graph, target_nodes: int, seed: int) -> List[int]:
    """
    Sample nodes using a BFS/snowball expansion.

    This is preferred for scalability testing because it preserves local graph
    structure better than pure random node sampling, which can create many
    disconnected fragments.
    """
    rng = np.random.default_rng(seed)
    all_nodes = list(graph.nodes())

    if target_nodes >= len(all_nodes):
        return sorted(all_nodes)

    start_node = rng.choice(all_nodes)
    visited = {start_node}
    frontier = [start_node]

    while frontier and len(visited) < target_nodes:
        current = frontier.pop(0)
        neighbours = list(graph.neighbors(current))
        rng.shuffle(neighbours)

        for neighbour in neighbours:
            if neighbour not in visited:
                visited.add(neighbour)
                frontier.append(neighbour)

                if len(visited) >= target_nodes:
                    break

    # If the start area was too small, fill the remaining nodes randomly.
    if len(visited) < target_nodes:
        remaining = [node for node in all_nodes if node not in visited]
        extra_count = min(target_nodes - len(visited), len(remaining))

        if extra_count > 0:
            visited.update(
                rng.choice(remaining, size=extra_count, replace=False).tolist()
            )

    return sorted(visited)
def sample_loaded_graph_for_scalability(
    loaded: LoadedGraph,
    target_nodes: int,
    seed: int,
    method: str = "snowball",
    largest_component_only: bool = True,
    use_sample_label_count_for_k: bool = True,
) -> LoadedGraph:
    """
    Create a sampled LoadedGraph while keeping features and labels aligned.

    Important alignment rule:
    The original LoadedGraph is already remapped to integer node IDs. Therefore,
    sampled node IDs can be used directly to slice loaded.features and loaded.labels.
    """
    graph = loaded.graph

    if graph.number_of_nodes() < 2:
        raise GATPSOError("Scalability sampling requires at least two graph nodes.")

    if method.lower() == "snowball":
        sampled_old_nodes = _snowball_node_sample(graph, target_nodes, seed)
    elif method.lower() == "random":
        sampled_old_nodes = _random_node_sample(graph, target_nodes, seed)
    else:
        raise GATPSOError("Unsupported sampling method. Use 'snowball' or 'random'.")

    sampled_graph = graph.subgraph(sampled_old_nodes).copy()
    sampled_graph.remove_edges_from(nx.selfloop_edges(sampled_graph))
    sampled_graph.remove_nodes_from(list(nx.isolates(sampled_graph)))

    if sampled_graph.number_of_nodes() < 2:
        raise GATPSOError(
            f"Sample with target_nodes={target_nodes} became too small after cleaning. "
            "Try a larger node size or use snowball sampling."
        )

    if largest_component_only:
        sampled_graph = _largest_component_subgraph(sampled_graph)

    # Keep this sorted order so feature/label slicing matches the relabelled graph.
    old_nodes = sorted(sampled_graph.nodes())
    mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
    relabelled_graph = nx.relabel_nodes(sampled_graph, mapping, copy=True)

    # Preserve the original ID and the parent node ID for traceability.
    for old_node, new_node in mapping.items():
        relabelled_graph.nodes[new_node]["original_id"] = graph.nodes[old_node].get(
            "original_id", old_node
        )
        relabelled_graph.nodes[new_node]["sample_parent_node"] = int(old_node)

    old_nodes_array = np.array(old_nodes, dtype=int)
    sampled_features = loaded.features[old_nodes_array].astype(np.float32)

    sampled_labels = None
    if loaded.labels is not None:
        sampled_labels = loaded.labels[old_nodes_array].astype(int)

    if sampled_labels is not None and use_sample_label_count_for_k:
        known_k = int(len(np.unique(sampled_labels)))
    else:
        known_k = loaded.known_k

    return LoadedGraph(
        graph=relabelled_graph,
        features=sampled_features,
        labels=sampled_labels,
        known_k=known_k,
        node_table=make_node_table(relabelled_graph),
        source_description=(
            f"Scalability sample from {loaded.source_description}; "
            f"method={method}; requested_nodes={target_nodes}; "
            f"actual_nodes={relabelled_graph.number_of_nodes()}"
        ),
    )
def sample_loaded_graph_for_scalability(
    loaded: LoadedGraph,
    target_nodes: int,
    seed: int,
    method: str = "snowball",
    largest_component_only: bool = True,
    use_sample_label_count_for_k: bool = True,
) -> LoadedGraph:
    """
    Create a sampled LoadedGraph while keeping features and labels aligned.

    Important alignment rule:
    The original LoadedGraph is already remapped to integer node IDs. Therefore,
    sampled node IDs can be used directly to slice loaded.features and loaded.labels.
    """
    graph = loaded.graph

    if graph.number_of_nodes() < 2:
        raise GATPSOError("Scalability sampling requires at least two graph nodes.")

    if method.lower() == "snowball":
        sampled_old_nodes = _snowball_node_sample(graph, target_nodes, seed)
    elif method.lower() == "random":
        sampled_old_nodes = _random_node_sample(graph, target_nodes, seed)
    else:
        raise GATPSOError("Unsupported sampling method. Use 'snowball' or 'random'.")

    sampled_graph = graph.subgraph(sampled_old_nodes).copy()
    sampled_graph.remove_edges_from(nx.selfloop_edges(sampled_graph))
    sampled_graph.remove_nodes_from(list(nx.isolates(sampled_graph)))

    if sampled_graph.number_of_nodes() < 2:
        raise GATPSOError(
            f"Sample with target_nodes={target_nodes} became too small after cleaning. "
            "Try a larger node size or use snowball sampling."
        )

    if largest_component_only:
        sampled_graph = _largest_component_subgraph(sampled_graph)

    # Keep this sorted order so feature/label slicing matches the relabelled graph.
    old_nodes = sorted(sampled_graph.nodes())
    mapping = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
    relabelled_graph = nx.relabel_nodes(sampled_graph, mapping, copy=True)

    # Preserve the original ID and the parent node ID for traceability.
    for old_node, new_node in mapping.items():
        relabelled_graph.nodes[new_node]["original_id"] = graph.nodes[old_node].get(
            "original_id", old_node
        )
        relabelled_graph.nodes[new_node]["sample_parent_node"] = int(old_node)

    old_nodes_array = np.array(old_nodes, dtype=int)
    sampled_features = loaded.features[old_nodes_array].astype(np.float32)

    sampled_labels = None
    if loaded.labels is not None:
        sampled_labels = loaded.labels[old_nodes_array].astype(int)

    if sampled_labels is not None and use_sample_label_count_for_k:
        known_k = int(len(np.unique(sampled_labels)))
    else:
        known_k = loaded.known_k

    return LoadedGraph(
        graph=relabelled_graph,
        features=sampled_features,
        labels=sampled_labels,
        known_k=known_k,
        node_table=make_node_table(relabelled_graph),
        source_description=(
            f"Scalability sample from {loaded.source_description}; "
            f"method={method}; requested_nodes={target_nodes}; "
            f"actual_nodes={relabelled_graph.number_of_nodes()}"
        ),
    )
def _build_scalability_trial_config(
    base_cfg: ModelConfig,
    scale_cfg: ScalabilityConfig,
    seed: int,
) -> ModelConfig:
    """Copy the base model configuration and apply scalability-specific overrides."""
    trial_cfg = ModelConfig(**asdict(base_cfg))
    trial_cfg.seed = int(seed)

    # Sampling is handled explicitly by the scalability layer.
    trial_cfg.max_nodes = None

    if scale_cfg.epochs is not None:
        trial_cfg.epochs = int(scale_cfg.epochs)

    if scale_cfg.swarm_size is not None:
        trial_cfg.swarm_size = int(scale_cfg.swarm_size)

    if scale_cfg.max_iter is not None:
        trial_cfg.max_iter = int(scale_cfg.max_iter)

    if scale_cfg.modularity_edge_sample is not None:
        trial_cfg.modularity_edge_sample = int(scale_cfg.modularity_edge_sample)

    if scale_cfg.pso_enabled is not None:
        trial_cfg.pso_enabled = bool(scale_cfg.pso_enabled)

    return trial_cfg


def _extract_final_model_row(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Return the PSO row when available; otherwise return the final available row."""
    if results_df.empty:
        return {}

    pso_rows = results_df[
        results_df["model"].astype(str).str.contains("PSO", case=False, na=False)
    ]

    if not pso_rows.empty:
        return pso_rows.iloc[-1].to_dict()

    return results_df.iloc[-1].to_dict()
def summarize_scalability_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize repeated scalability results by requested node size.

    Produces mean and standard deviation for the main scalability and quality
    measures. Use this table for dissertation/report writing, not one-off runs.
    """
    if results.empty:
        return pd.DataFrame()

    ok = results[results["status"] == "ok"].copy()

    if ok.empty:
        return pd.DataFrame()

    numeric_cols = [
        "actual_nodes",
        "edges",
        "communities",
        "selected_k",
        "sampling_seconds",
        "train_seconds",
        "pso_seconds",
        "total_seconds",
        "memory_mb",
        "memory_delta_mb",
        "score",
        "modularity",
        "separation",
        "conductance",
        "NMI",
        "ARI",
    ]

    numeric_cols = [col for col in numeric_cols if col in ok.columns]

    summary = ok.groupby("requested_nodes")[numeric_cols].agg(
        ["mean", "std", "min", "max"]
    )

    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    summary.insert(
        1,
        "trials_completed",
        ok.groupby("requested_nodes").size().values,
    )

    return summary
@dataclass
class LoadedGraph:
    graph: nx.Graph
    features: np.ndarray
    labels: Optional[np.ndarray]
    known_k: Optional[int]
    node_table: pd.DataFrame
    source_description: str


class GATPSOError(RuntimeError):
    """Raised when the model cannot continue because of missing/invalid inputs."""


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_pyg() -> None:
    if not PYG_AVAILABLE:
        raise GATPSOError(
            "PyTorch Geometric is required for the GAT/DGI model. "
            "Install dependencies from requirements.txt. Original import error: "
            f"{PYG_IMPORT_ERROR}"
        )


def preprocess_graph_nx(
    graph: nx.Graph,
    largest_component_only: bool = True,
    max_nodes: Optional[int] = None,
    seed: int = 42,
) -> nx.Graph:
    """Clean the graph and remap node IDs to contiguous integers."""
    graph = nx.Graph(graph)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    graph.remove_nodes_from(list(nx.isolates(graph)))

    if graph.number_of_nodes() == 0:
        raise GATPSOError("The graph is empty after removing self-loops and isolated nodes.")

    if largest_component_only and graph.number_of_nodes() > 0:
        largest_nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest_nodes).copy()

    if max_nodes is not None and graph.number_of_nodes() > max_nodes:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(list(graph.nodes()), size=max_nodes, replace=False)
        graph = graph.subgraph(chosen).copy()
        if largest_component_only and graph.number_of_nodes() > 0:
            largest_nodes = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_nodes).copy()

    graph = nx.convert_node_labels_to_integers(graph, label_attribute="original_id")
    return graph


def build_structural_features(graph: nx.Graph) -> np.ndarray:
    """Create notebook-style structural features for nodes without explicit attributes."""
    n_nodes = graph.number_of_nodes()
    degree = np.array([graph.degree(node) for node in range(n_nodes)], dtype=np.float32)
    log_degree = np.log1p(degree)
    max_degree = degree.max() if degree.size and degree.max() > 0 else 1.0
    norm_degree = degree / max_degree
    bias = np.ones(n_nodes, dtype=np.float32)
    return np.vstack([degree, log_degree, norm_degree, bias]).T.astype(np.float32)


def make_node_table(graph: nx.Graph) -> pd.DataFrame:
    rows = []
    for node in graph.nodes():
        rows.append(
            {
                "node": int(node),
                "original_id": graph.nodes[node].get("original_id", node),
                "degree": int(graph.degree(node)),
            }
        )
    return pd.DataFrame(rows).sort_values("node").reset_index(drop=True)


def load_benchmark_dataset(
    name: str,
    largest_component_only: bool = True,
    max_nodes: Optional[int] = None,
    seed: int = 42,
    root: str = "./benchmark_data",
) -> LoadedGraph:
    """Load Cora, PubMed or Citeseer from PyTorch Geometric Planetoid."""
    _require_pyg()
    canonical = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed"}
    dataset_name = canonical.get(name.lower(), name)
    if dataset_name not in {"Cora", "Citeseer", "PubMed"}:
        raise GATPSOError("Supported benchmark datasets are Cora, Citeseer and PubMed.")

    dataset = Planetoid(root=os.path.join(root, dataset_name), name=dataset_name)
    data = dataset[0]

    edge_index = data.edge_index.detach().cpu().numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))
    graph.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    graph.remove_edges_from(nx.selfloop_edges(graph))

    graph = preprocess_graph_nx(graph, largest_component_only, max_nodes, seed)
    original_ids = [graph.nodes[node].get("original_id", node) for node in range(graph.number_of_nodes())]

    features = data.x.detach().cpu().numpy()[original_ids].astype(np.float32)
    labels = data.y.detach().cpu().numpy()[original_ids].astype(int)
    node_table = make_node_table(graph)

    return LoadedGraph(
        graph=graph,
        features=features,
        labels=labels,
        known_k=int(dataset.num_classes),
        node_table=node_table,
        source_description=f"Benchmark dataset: {dataset_name}",
    )


def read_edges_from_dataframe(df: pd.DataFrame, source_col: str, target_col: str) -> nx.Graph:
    if source_col not in df.columns or target_col not in df.columns:
        raise GATPSOError("Source and target columns must exist in the uploaded edge file.")

    edges_df = df[[source_col, target_col]].dropna()
    graph = nx.Graph()
    for source, target in edges_df.itertuples(index=False):
        graph.add_edge(str(source), str(target))
    return graph


def read_edges_auto(path: str, source_col: Optional[str] = None, target_col: Optional[str] = None) -> nx.Graph:
    """Read a CSV/TXT edge list. Defaults to the first two columns where possible."""
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] >= 2:
            src = source_col or df.columns[0]
            dst = target_col or df.columns[1]
            return read_edges_from_dataframe(df, src, dst)
    except Exception:
        pass

    try:
        graph = nx.read_edgelist(path, nodetype=str)
        return nx.Graph(graph)
    except Exception as exc:
        raise GATPSOError(
            "Could not read the uploaded file as a CSV edge list or whitespace-delimited edge list. "
            "Use at least two columns representing source and target nodes."
        ) from exc


def align_feature_file(
    graph: nx.Graph,
    feature_df: pd.DataFrame,
    node_id_col: Optional[str] = None,
) -> np.ndarray:
    """Align uploaded node features to graph nodes using original IDs."""
    if feature_df.empty:
        return build_structural_features(graph)

    node_col = node_id_col or feature_df.columns[0]
    if node_col not in feature_df.columns:
        raise GATPSOError("The selected node ID column was not found in the feature file.")

    feature_cols = [col for col in feature_df.columns if col != node_col]
    if not feature_cols:
        return build_structural_features(graph)

    tmp = feature_df.copy()
    tmp[node_col] = tmp[node_col].astype(str)
    numeric = tmp.set_index(node_col)[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    rows = []
    missing = 0
    fallback = build_structural_features(graph)
    for node in range(graph.number_of_nodes()):
        original_id = str(graph.nodes[node].get("original_id", node))
        if original_id in numeric.index:
            rows.append(numeric.loc[original_id].to_numpy(dtype=np.float32))
        else:
            missing += 1
            rows.append(np.zeros(len(feature_cols), dtype=np.float32))

    features = np.vstack(rows).astype(np.float32)
    if missing == graph.number_of_nodes() or features.shape[1] == 0:
        return fallback
    return features


def align_label_file(
    graph: nx.Graph,
    label_df: pd.DataFrame,
    node_id_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> Optional[np.ndarray]:
    """Align uploaded ground-truth labels to graph nodes, if supplied."""
    if label_df is None or label_df.empty:
        return None

    node_col = node_id_col or label_df.columns[0]
    lbl_col = label_col or (label_df.columns[1] if len(label_df.columns) > 1 else None)
    if lbl_col is None or node_col not in label_df.columns or lbl_col not in label_df.columns:
        return None

    tmp = label_df[[node_col, lbl_col]].dropna().copy()
    tmp[node_col] = tmp[node_col].astype(str)
    label_map = tmp.set_index(node_col)[lbl_col].to_dict()

    labels = []
    for node in range(graph.number_of_nodes()):
        original_id = str(graph.nodes[node].get("original_id", node))
        labels.append(label_map.get(original_id, None))

    if any(label is None for label in labels):
        return None

    return pd.factorize(pd.Series(labels))[0].astype(int)


def load_uploaded_dataset(
    edge_path: str,
    largest_component_only: bool = True,
    max_nodes: Optional[int] = None,
    seed: int = 42,
    source_col: Optional[str] = None,
    target_col: Optional[str] = None,
    features_df: Optional[pd.DataFrame] = None,
    feature_node_col: Optional[str] = None,
    labels_df: Optional[pd.DataFrame] = None,
    label_node_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> LoadedGraph:
    graph = read_edges_auto(edge_path, source_col, target_col)
    graph = preprocess_graph_nx(graph, largest_component_only, max_nodes, seed)

    if features_df is not None:
        features = align_feature_file(graph, features_df, feature_node_col)
    else:
        features = build_structural_features(graph)

    labels = align_label_file(graph, labels_df, label_node_col, label_col) if labels_df is not None else None
    known_k = int(len(np.unique(labels))) if labels is not None else None
    node_table = make_node_table(graph)

    return LoadedGraph(
        graph=graph,
        features=features.astype(np.float32),
        labels=labels,
        known_k=known_k,
        node_table=node_table,
        source_description="Uploaded edge-list dataset",
    )


def nx_to_pyg_data(graph: nx.Graph, features: np.ndarray):
    _require_pyg()
    data = from_networkx(graph)
    data.x = torch.tensor(features, dtype=torch.float32)
    return data


class GATEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 2, dropout: float = 0.35):
        super().__init__()
        if GATConv is None:
            raise GATPSOError("GATConv is unavailable. Install torch-geometric.")
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.gat2(x, edge_index)


def corruption(x, edge_index):
    perm = torch.randperm(x.size(0), device=x.device)
    return x[perm], edge_index


def train_dgi_gat(
    data,
    cfg: ModelConfig,
    device: torch.device,
    progress_callback: Optional[Callable[[int, float], None]] = None,
):
    _require_pyg()
    model = DeepGraphInfomax(
        hidden_channels=cfg.embedding_dim,
        encoder=GATEncoder(data.num_features, cfg.hidden_dim, cfg.embedding_dim, cfg.heads, cfg.dropout),
        summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
        corruption=corruption,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history: List[float] = []
    start = time.time()
    model.train()

    for epoch in range(1, cfg.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        pos_z, neg_z, summary = model(data.x, data.edge_index)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if progress_callback is not None and (epoch == 1 or epoch == cfg.epochs or epoch % 5 == 0):
            progress_callback(epoch, loss_value)

    return model, history, time.time() - start


@torch.no_grad()
def extract_embeddings(model, data) -> np.ndarray:
    model.eval()
    z, _, _ = model(data.x, data.edge_index)
    return z.detach().cpu().numpy()


def labels_to_communities(labels: np.ndarray) -> List[set]:
    return [set(np.where(labels == c)[0].tolist()) for c in np.unique(labels)]


def full_modularity(graph: nx.Graph, labels: np.ndarray) -> float:
    communities = labels_to_communities(labels)
    if len(communities) <= 1:
        return -1.0
    return float(nx.algorithms.community.quality.modularity(graph, communities))


def sampled_modularity(graph: nx.Graph, labels: np.ndarray, edge_sample: Optional[int] = None, seed: int = 42) -> float:
    if edge_sample is None or graph.number_of_edges() <= edge_sample:
        return full_modularity(graph, labels)

    rng = np.random.default_rng(seed)
    edges = list(graph.edges())
    idx = rng.choice(len(edges), size=edge_sample, replace=False)
    sampled_graph = nx.Graph()
    sampled_graph.add_nodes_from(graph.nodes())
    sampled_graph.add_edges_from([edges[i] for i in idx])
    return full_modularity(sampled_graph, labels)


def safe_silhouette(X: np.ndarray, labels: np.ndarray, seed: int = 42) -> float:
    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < 2 or len(unique) >= len(labels):
        return -1.0
    sample_size = min(3000, len(labels))
    try:
        return float(silhouette_score(X, labels, sample_size=sample_size, random_state=seed))
    except ValueError:
        return -1.0


def estimate_k_by_silhouette(X: np.ndarray, k_min: int = 2, k_max: int = 10, seed: int = 42) -> Tuple[int, pd.DataFrame]:
    max_allowed = max(2, min(k_max, len(X) - 1))
    rows: List[Dict[str, Any]] = []
    best = {"k": k_min, "score": -1.0}

    for k in range(k_min, max_allowed + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        score = safe_silhouette(X, labels, seed)
        rows.append({"k": k, "silhouette": score})
        if score > best["score"]:
            best = {"k": k, "score": score}

    return int(best["k"]), pd.DataFrame(rows)


def assign_by_centroids(X: np.ndarray, centroids: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
    labels = np.empty(X.shape[0], dtype=np.int32)
    for start in range(0, X.shape[0], chunk_size):
        end = min(start + chunk_size, X.shape[0])
        distances = np.linalg.norm(X[start:end, None, :] - centroids[None, :, :], axis=2)
        labels[start:end] = np.argmin(distances, axis=1)
    return labels


def objective_score(graph: nx.Graph, X: np.ndarray, labels: np.ndarray, cfg: ModelConfig, final: bool = False) -> Dict[str, float]:
    modularity = full_modularity(graph, labels) if final else sampled_modularity(
        graph, labels, cfg.modularity_edge_sample, cfg.seed
    )
    separation = safe_silhouette(X, labels, cfg.seed)
    return {
        "score": float(cfg.alpha_modularity * modularity + cfg.beta_separation * separation),
        "modularity": float(modularity),
        "separation": float(separation),
    }


class Particle:
    def __init__(self, centroids: np.ndarray):
        self.position = centroids.copy()
        self.velocity = np.random.randn(*centroids.shape) * 0.02
        self.best_position = self.position.copy()
        self.best_score = -np.inf
        self.best_metrics: Optional[Dict[str, float]] = None


class LightweightPSO:
    def __init__(self, X: np.ndarray, graph: nx.Graph, k: int, cfg: ModelConfig, init_centroids: Optional[np.ndarray] = None):
        self.X = X
        self.graph = graph
        self.k = k
        self.cfg = cfg
        self.dim = X.shape[1]
        self.swarm: List[Particle] = []
        rng = np.random.default_rng(cfg.seed)

        for particle_idx in range(cfg.swarm_size):
            if particle_idx == 0 and init_centroids is not None:
                centroids = init_centroids.copy()
            else:
                centroids = X[rng.choice(len(X), size=k, replace=False)]
            self.swarm.append(Particle(centroids))

        self.gbest_position: Optional[np.ndarray] = None
        self.gbest_score = -np.inf
        self.gbest_labels: Optional[np.ndarray] = None
        self.gbest_metrics: Optional[Dict[str, float]] = None
        self.history: List[Dict[str, Any]] = []

    def optimize(self, progress_callback: Optional[Callable[[int, Dict[str, float]], None]] = None) -> Dict[str, Any]:
        for iteration in range(1, self.cfg.max_iter + 1):
            for particle in self.swarm:
                labels = assign_by_centroids(self.X, particle.position)
                metrics = objective_score(self.graph, self.X, labels, self.cfg, final=False)

                if metrics["score"] > particle.best_score:
                    particle.best_score = metrics["score"]
                    particle.best_position = particle.position.copy()
                    particle.best_metrics = metrics

                if metrics["score"] > self.gbest_score:
                    self.gbest_score = metrics["score"]
                    self.gbest_position = particle.position.copy()
                    self.gbest_labels = labels.copy()
                    self.gbest_metrics = metrics

            if self.gbest_position is None or self.gbest_metrics is None:
                raise GATPSOError("PSO could not find a valid candidate solution.")

            for particle in self.swarm:
                r1 = np.random.rand(self.k, self.dim)
                r2 = np.random.rand(self.k, self.dim)
                cognitive = 1.49 * r1 * (particle.best_position - particle.position)
                social = 1.49 * r2 * (self.gbest_position - particle.position)
                particle.velocity = 0.72 * particle.velocity + cognitive + social
                particle.position = particle.position + particle.velocity

            row = {"iteration": iteration, **self.gbest_metrics}
            self.history.append(row)
            if progress_callback is not None and (iteration == 1 or iteration == self.cfg.max_iter or iteration % 5 == 0):
                progress_callback(iteration, self.gbest_metrics)

        final_labels = assign_by_centroids(self.X, self.gbest_position)
        final_metrics = objective_score(self.graph, self.X, final_labels, self.cfg, final=True)
        return {
            "labels": final_labels,
            "centroids": self.gbest_position,
            "metrics": final_metrics,
            "history": pd.DataFrame(self.history),
        }


def average_conductance(graph: nx.Graph, labels: np.ndarray) -> Optional[float]:
    values: List[float] = []
    for community_id in np.unique(labels):
        nodes = set(np.where(labels == community_id)[0])
        if len(nodes) == 0 or len(nodes) == graph.number_of_nodes():
            continue
        try:
            values.append(float(nx.algorithms.cuts.conductance(graph, nodes)))
        except Exception:
            continue
    return float(np.mean(values)) if values else None


def external_metrics(y_true: Optional[np.ndarray], labels: np.ndarray) -> Dict[str, Optional[float]]:
    if y_true is None or len(y_true) != len(labels):
        return {"NMI": None, "ARI": None}
    return {
        "NMI": float(normalized_mutual_info_score(y_true, labels)),
        "ARI": float(adjusted_rand_score(y_true, labels)),
    }


def representative_nodes(
    graph: nx.Graph,
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    top_n: int = 5,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Select representative nodes per community.

    Score = 0.5 * centroid proximity + 0.5 * normalized intra-community degree.
    This means a representative node is both structurally central in its community and
    close to the learned embedding centre of that community.
    """
    output: Dict[int, List[Dict[str, Any]]] = {}
    for community_id in np.unique(labels):
        nodes = np.where(labels == community_id)[0]
        if len(nodes) == 0:
            continue

        subgraph = graph.subgraph(nodes)
        intra_degree = np.array([subgraph.degree(int(node)) for node in nodes], dtype=np.float32)
        if intra_degree.max() > 0:
            intra_degree = intra_degree / intra_degree.max()

        distances = np.linalg.norm(X[nodes] - centroids[int(community_id)], axis=1)
        proximity = 1 / (1 + distances)
        score = 0.5 * proximity + 0.5 * intra_degree
        best_idx = np.argsort(-score)[:top_n]

        output[int(community_id)] = []
        for idx in best_idx:
            node = int(nodes[idx])
            original_id = graph.nodes[node].get("original_id", node)
            output[int(community_id)].append(
                {
                    "community": int(community_id),
                    "node": node,
                    "original_id": original_id,
                    "score": float(score[idx]),
                    "intra_degree": float(intra_degree[idx]),
                    "proximity": float(proximity[idx]),
                    "reason": (
                        "High representative score because it is close to the community centroid "
                        "and has strong internal connectivity within the same community."
                    ),
                }
            )
    return output


def representatives_to_dataframe(representatives: Dict[int, List[Dict[str, Any]]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for items in representatives.values():
        rows.extend(items)
    if not rows:
        return pd.DataFrame(columns=["community", "node", "original_id", "score", "proximity", "intra_degree", "reason"])
    return pd.DataFrame(rows).sort_values(["community", "score"], ascending=[True, False]).reset_index(drop=True)


def community_sizes(labels: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().sort_index()
    return pd.DataFrame({"community": counts.index.astype(int), "nodes": counts.values.astype(int)})


def pca_projection(X: np.ndarray, labels: np.ndarray, sample_size: int = 5_000, seed: int = 42) -> pd.DataFrame:
    n = len(X)
    rng = np.random.default_rng(seed)
    if n > sample_size:
        idx = np.sort(rng.choice(n, size=sample_size, replace=False))
    else:
        idx = np.arange(n)

    if X.shape[1] < 2:
        projected = np.column_stack([X[idx, 0], np.zeros(len(idx))])
    else:
        projected = PCA(n_components=2, random_state=seed).fit_transform(X[idx])

    return pd.DataFrame(
        {
            "node": idx.astype(int),
            "pc1": projected[:, 0],
            "pc2": projected[:, 1],
            "community": labels[idx].astype(str),
        }
    )


def run_gatpso_pipeline(
    loaded: LoadedGraph,
    cfg: ModelConfig,
    device_preference: str = "auto",
    train_callback: Optional[Callable[[int, float], None]] = None,
    pso_callback: Optional[Callable[[int, Dict[str, float]], None]] = None,
) -> Dict[str, Any]:
    if loaded.graph.number_of_nodes() < 2:
        raise GATPSOError("The graph must have at least two connected nodes after preprocessing.")

    set_seed(cfg.seed)
    device = get_device(device_preference)
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / (1024**2)
    total_start = time.time()

    data = nx_to_pyg_data(loaded.graph, loaded.features).to(device)
    model, loss_history, train_seconds = train_dgi_gat(data, cfg, device, train_callback)
    embeddings = extract_embeddings(model, data)
    embeddings_std = StandardScaler().fit_transform(embeddings)

    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if cfg.known_k is not None:
        k = int(cfg.known_k)
        k_search = pd.DataFrame([{"k": k, "silhouette": None, "note": "known/manual k used"}])
    else:
        k, k_search = estimate_k_by_silhouette(embeddings_std, cfg.k_min, cfg.k_max, cfg.seed)

    k = max(2, min(k, loaded.graph.number_of_nodes() - 1))
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=cfg.seed)
    kmeans_labels = kmeans.fit_predict(embeddings_std)
    kmeans_centroids = kmeans.cluster_centers_

    kmeans_final = objective_score(loaded.graph, embeddings_std, kmeans_labels, cfg, final=True)
    kmeans_row: Dict[str, Any] = {
        "model": "GAT/DGI + K-means",
        **kmeans_final,
        "conductance": average_conductance(loaded.graph, kmeans_labels),
        **external_metrics(loaded.labels, kmeans_labels),
    }

    pso_seconds = 0.0
    if cfg.pso_enabled:
        pso_start = time.time()
        pso = LightweightPSO(embeddings_std, loaded.graph, k, cfg, init_centroids=kmeans_centroids)
        pso_results = pso.optimize(pso_callback)
        pso_seconds = time.time() - pso_start
        final_labels = pso_results["labels"]
        final_centroids = pso_results["centroids"]
        pso_history = pso_results["history"]
        pso_final = pso_results["metrics"]
    else:
        final_labels = kmeans_labels
        final_centroids = kmeans_centroids
        pso_history = pd.DataFrame(columns=["iteration", "score", "modularity", "separation"])
        pso_final = kmeans_final

    pso_row: Dict[str, Any] = {
        "model": "GAT/DGI + PSO" if cfg.pso_enabled else "GAT/DGI + K-means only",
        **pso_final,
        "conductance": average_conductance(loaded.graph, final_labels),
        **external_metrics(loaded.labels, final_labels),
    }

    results_df = pd.DataFrame([kmeans_row, pso_row])
    representatives = representative_nodes(loaded.graph, embeddings_std, final_labels, final_centroids, cfg.top_representatives)

    total_seconds = time.time() - total_start
    end_memory = process.memory_info().rss / (1024**2)
    runtime = {
        "train_seconds": float(train_seconds),
        "pso_seconds": float(pso_seconds),
        "total_seconds": float(total_seconds),
        "memory_mb": float(end_memory),
        "memory_delta_mb": float(end_memory - start_memory),
        "device": str(device),
    }

    label_table = loaded.node_table.copy()
    label_table["community"] = final_labels.astype(int)
    if loaded.labels is not None:
        label_table["ground_truth_label"] = loaded.labels.astype(int)

    return {
        "config": asdict(cfg),
        "source_description": loaded.source_description,
        "graph": loaded.graph,
        "features_shape": tuple(loaded.features.shape),
        "node_table": loaded.node_table,
        "labels": final_labels,
        "label_table": label_table,
        "kmeans_labels": kmeans_labels,
        "embeddings": embeddings_std,
        "selected_k": int(k),
        "k_search": k_search,
        "results_df": results_df,
        "pso_history": pso_history,
        "loss_history": pd.DataFrame({"epoch": range(1, len(loss_history) + 1), "dgi_loss": loss_history}),
        "representatives": representatives,
        "representatives_df": representatives_to_dataframe(representatives),
        "community_sizes": community_sizes(final_labels),
        "runtime": runtime,
        "summary": {
            "nodes": int(loaded.graph.number_of_nodes()),
            "edges": int(loaded.graph.number_of_edges()),
            "communities": int(len(np.unique(final_labels))),
            "selected_k": int(k),
            **runtime,
        },
    }


def run_scalability_analysis(
    loaded: LoadedGraph,
    base_cfg: ModelConfig,
    scale_cfg: ScalabilityConfig,
    device_preference: str = "auto",
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pd.DataFrame:
    """
    Run repeated scalability tests over fixed sample sizes.

    The function returns one row per trial. Failed or skipped trials are retained
    with a status and error message so that scalability testing remains auditable.
    """
    rows: List[Dict[str, Any]] = []
    total_nodes = loaded.graph.number_of_nodes()

    for requested_nodes in scale_cfg.node_sizes:
        if requested_nodes > total_nodes and scale_cfg.skip_sizes_larger_than_graph:
            row = {
                "requested_nodes": int(requested_nodes),
                "actual_nodes": None,
                "edges": None,
                "trial": None,
                "seed": None,
                "sampling_method": scale_cfg.sampling_method,
                "status": "skipped",
                "error": "Requested sample size is larger than the source graph.",
            }
            rows.append(row)
            if progress_callback is not None:
                progress_callback(row)
            continue

        for trial in range(1, scale_cfg.repeats + 1):
            trial_seed = int(scale_cfg.seed + requested_nodes * 100 + trial)
            row_base: Dict[str, Any] = {
                "requested_nodes": int(requested_nodes),
                "trial": int(trial),
                "seed": int(trial_seed),
                "sampling_method": scale_cfg.sampling_method,
            }

            try:
                sampling_start = time.time()
                sampled_loaded = sample_loaded_graph_for_scalability(
                    loaded=loaded,
                    target_nodes=int(requested_nodes),
                    seed=trial_seed,
                    method=scale_cfg.sampling_method,
                    largest_component_only=scale_cfg.largest_component_only,
                    use_sample_label_count_for_k=scale_cfg.use_sample_label_count_for_k,
                )
                sampling_seconds = time.time() - sampling_start

                trial_cfg = _build_scalability_trial_config(
                    base_cfg=base_cfg,
                    scale_cfg=scale_cfg,
                    seed=trial_seed,
                )

                result = run_gatpso_pipeline(
                    loaded=sampled_loaded,
                    cfg=trial_cfg,
                    device_preference=device_preference,
                )

                final_row = _extract_final_model_row(result["results_df"])
                summary = result["summary"]
                runtime = result["runtime"]

                row = {
                    **row_base,
                    "actual_nodes": int(summary.get("nodes", sampled_loaded.graph.number_of_nodes())),
                    "edges": int(summary.get("edges", sampled_loaded.graph.number_of_edges())),
                    "features": int(sampled_loaded.features.shape[1]),
                    "selected_k": int(summary.get("selected_k", result.get("selected_k", 0))),
                    "communities": int(summary.get("communities", 0)),
                    "sampling_seconds": float(sampling_seconds),
                    "train_seconds": float(runtime.get("train_seconds", np.nan)),
                    "pso_seconds": float(runtime.get("pso_seconds", np.nan)),
                    "total_seconds": float(runtime.get("total_seconds", np.nan)),
                    "memory_mb": float(runtime.get("memory_mb", np.nan)),
                    "memory_delta_mb": float(runtime.get("memory_delta_mb", np.nan)),
                    "model": final_row.get("model"),
                    "score": final_row.get("score"),
                    "modularity": final_row.get("modularity"),
                    "separation": final_row.get("separation"),
                    "conductance": final_row.get("conductance"),
                    "NMI": final_row.get("NMI"),
                    "ARI": final_row.get("ARI"),
                    "status": "ok",
                    "error": None,
                }

            except Exception as exc:
                row = {
                    **row_base,
                    "actual_nodes": None,
                    "edges": None,
                    "features": None,
                    "selected_k": None,
                    "communities": None,
                    "sampling_seconds": None,
                    "train_seconds": None,
                    "pso_seconds": None,
                    "total_seconds": None,
                    "memory_mb": None,
                    "memory_delta_mb": None,
                    "model": None,
                    "score": None,
                    "modularity": None,
                    "separation": None,
                    "conductance": None,
                    "NMI": None,
                    "ARI": None,
                    "status": "failed",
                    "error": str(exc),
                }

            rows.append(row)
            if progress_callback is not None:
                progress_callback(row)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return pd.DataFrame(rows)

