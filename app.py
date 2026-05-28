"""Streamlit interface for the GAT/DGI + PSO community detection model."""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from typing import Dict, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gatpso_model import (
    GATPSOError,
    ModelConfig,
    load_benchmark_dataset,
    load_uploaded_dataset,
    pca_projection,
    run_gatpso_pipeline,
)


st.set_page_config(
    page_title="GAT-PSO Community Detection",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background: #050505;
        color: #f7f7f7;
    }
    [data-testid="stSidebar"] {
        background: #111111;
        border-right: 1px solid #333333;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4, h5, h6, p, li, label, span, div {
        color: #f7f7f7;
    }
    .hero-card {
        background: linear-gradient(135deg, #111111 0%, #1f1f1f 60%, #2c2c2c 100%);
        padding: 1.4rem 1.6rem;
        border: 1px solid #3a3a3a;
        border-radius: 20px;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 40px rgba(0,0,0,0.35);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        color: #cfcfcf;
        font-size: 1rem;
        max-width: 1100px;
    }
    .info-card {
        background: #151515;
        border: 1px solid #343434;
        border-radius: 16px;
        padding: 1rem;
        height: 100%;
    }
    .small-muted {
        color: #bcbcbc;
        font-size: 0.92rem;
    }
    div[data-testid="stMetric"] {
        background: #151515;
        border: 1px solid #333333;
        border-radius: 16px;
        padding: 1rem;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    div[data-testid="stMetricLabel"] {
        color: #cfcfcf;
    }
    .stButton>button {
        background: #ffffff;
        color: #000000;
        border-radius: 999px;
        border: 1px solid #ffffff;
        font-weight: 700;
        padding: 0.65rem 1.2rem;
    }
    .stButton>button:hover {
        background: #d8d8d8;
        color: #000000;
        border: 1px solid #d8d8d8;
    }
    .stDataFrame, .stTable {
        background: #111111;
    }
    hr {
        border-color: #303030;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"


def write_uploaded_file(uploaded_file) -> Optional[str]:
    if uploaded_file is None:
        return None
    suffix = os.path.splitext(uploaded_file.name)[1] or ".csv"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp.write(uploaded_file.getvalue())
    temp.flush()
    temp.close()
    return temp.name


def read_uploaded_csv(uploaded_file) -> Optional[pd.DataFrame]:
    if uploaded_file is None:
        return None
    try:
        return pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)


def plot_metric_comparison(results_df: pd.DataFrame):
    metrics = [col for col in ["score", "modularity", "separation", "conductance", "NMI", "ARI"] if col in results_df.columns]
    long_df = results_df.melt(id_vars="model", value_vars=metrics, var_name="metric", value_name="value")
    long_df = long_df.dropna(subset=["value"])
    fig = px.bar(
        long_df,
        x="metric",
        y="value",
        color="model",
        barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Model Performance Comparison",
        text_auto=".3f",
    )
    fig.update_layout(paper_bgcolor="#050505", plot_bgcolor="#111111", font_color="#ffffff")
    return fig


def plot_pso_convergence(history_df: pd.DataFrame):
    fig = go.Figure()
    if not history_df.empty:
        fig.add_trace(go.Scatter(x=history_df["iteration"], y=history_df["score"], mode="lines+markers", name="Objective score"))
        if "modularity" in history_df.columns:
            fig.add_trace(go.Scatter(x=history_df["iteration"], y=history_df["modularity"], mode="lines+markers", name="Modularity"))
        if "separation" in history_df.columns:
            fig.add_trace(go.Scatter(x=history_df["iteration"], y=history_df["separation"], mode="lines+markers", name="Separation"))
    fig.update_layout(
        title="PSO Optimization Convergence",
        xaxis_title="Iteration",
        yaxis_title="Value",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#050505",
        plot_bgcolor="#111111",
        font_color="#ffffff",
    )
    return fig


def plot_loss(loss_df: pd.DataFrame):
    fig = px.line(loss_df, x="epoch", y="dgi_loss", markers=True, template=PLOTLY_TEMPLATE, title="DGI-GAT Training Loss")
    fig.update_layout(paper_bgcolor="#050505", plot_bgcolor="#111111", font_color="#ffffff")
    return fig


def plot_community_sizes(size_df: pd.DataFrame):
    fig = px.bar(size_df, x="community", y="nodes", template=PLOTLY_TEMPLATE, title="Community Size Distribution", text_auto=True)
    fig.update_layout(paper_bgcolor="#050505", plot_bgcolor="#111111", font_color="#ffffff")
    return fig


def plot_embeddings(embeddings: np.ndarray, labels: np.ndarray, seed: int):
    projection = pca_projection(embeddings, labels, sample_size=5000, seed=seed)
    fig = px.scatter(
        projection,
        x="pc1",
        y="pc2",
        color="community",
        hover_data=["node"],
        template=PLOTLY_TEMPLATE,
        title="Detected Communities in Embedding Space",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.85))
    fig.update_layout(paper_bgcolor="#050505", plot_bgcolor="#111111", font_color="#ffffff")
    return fig


def plot_network_sample(graph: nx.Graph, labels: np.ndarray, sample_nodes: int = 250, seed: int = 42):
    if graph.number_of_nodes() == 0:
        return go.Figure()

    degree_sorted = sorted(graph.degree, key=lambda item: item[1], reverse=True)
    chosen_nodes = [node for node, _ in degree_sorted[: min(sample_nodes, graph.number_of_nodes())]]
    subgraph = graph.subgraph(chosen_nodes).copy()
    pos = nx.spring_layout(subgraph, seed=seed, iterations=40)

    edge_x = []
    edge_y = []
    for source, target in subgraph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x = []
    node_y = []
    node_color = []
    hover = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(int(labels[node]))
        hover.append(f"Node: {node}<br>Community: {int(labels[node])}<br>Degree: {graph.degree(node)}")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.45, color="#6e6e6e"),
            hoverinfo="none",
            mode="lines",
            name="Edges",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=hover,
            marker=dict(size=7, color=node_color, colorscale="Viridis", line_width=0.5, showscale=True),
            name="Nodes",
        )
    )
    fig.update_layout(
        title=f"Network Sample Visualization - Top {len(chosen_nodes)} Nodes by Degree",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#050505",
        plot_bgcolor="#111111",
        font_color="#ffffff",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


def build_result_zip(result: Dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("metrics.csv", result["results_df"].to_csv(index=False))
        zf.writestr("communities.csv", result["label_table"].to_csv(index=False))
        zf.writestr("representative_nodes.csv", result["representatives_df"].to_csv(index=False))
        zf.writestr("community_sizes.csv", result["community_sizes"].to_csv(index=False))
        zf.writestr("training_loss.csv", result["loss_history"].to_csv(index=False))
        zf.writestr("pso_history.csv", result["pso_history"].to_csv(index=False))
    return buffer.getvalue()


st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">GAT-PSO Community Detection Web App</div>
        <div class="hero-subtitle">
            Upload your own graph dataset or run benchmark datasets such as Cora, PubMed and Citeseer.
            The app trains a GAT/DGI embedding model, applies K-means and PSO, then explains the detected
            communities, metrics and representative nodes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Dataset")
    dataset_mode = st.radio("Choose data source", ["Benchmark dataset", "Upload edge list"], index=0)

    selected_benchmark = "Cora"
    uploaded_edge_file = None
    uploaded_features_file = None
    uploaded_labels_file = None
    source_col = None
    target_col = None
    feature_node_col = None
    label_node_col = None
    label_col = None

    if dataset_mode == "Benchmark dataset":
        selected_benchmark = st.selectbox("Benchmark", ["Cora", "PubMed", "Citeseer"], index=0)
        st.caption("Benchmark datasets use their built-in node features and ground-truth labels.")
    else:
        uploaded_edge_file = st.file_uploader("Upload edge list CSV/TXT", type=["csv", "txt", "edgelist"])
        st.caption("The first two columns are used as source and target unless you specify names below.")
        source_col = st.text_input("Source column name (optional)", value="") or None
        target_col = st.text_input("Target column name (optional)", value="") or None
        with st.expander("Optional: upload node features and labels"):
            uploaded_features_file = st.file_uploader("Node features CSV", type=["csv"], key="features")
            feature_node_col = st.text_input("Feature node ID column (optional)", value="") or None
            uploaded_labels_file = st.file_uploader("Ground-truth labels CSV", type=["csv"], key="labels")
            label_node_col = st.text_input("Label node ID column (optional)", value="") or None
            label_col = st.text_input("Label column name (optional)", value="") or None

    st.header("Preprocessing")
    largest_component_only = st.checkbox("Use largest connected component", value=True)
    max_nodes = st.number_input("Maximum nodes to process", min_value=0, max_value=100_000, value=5_000, step=500)
    max_nodes = None if max_nodes == 0 else int(max_nodes)

    st.header("Execution Mode")
    mode = st.radio("Mode", ["Demo mode", "Custom mode"], index=0)
    if mode == "Demo mode":
        epochs = 20
        swarm_size = 5
        pso_iterations = 10
        hidden_dim = 16
        embedding_dim = 16
        heads = 2
        dropout = 0.35
    else:
        epochs = st.slider("GAT/DGI epochs", 5, 200, 50, 5)
        swarm_size = st.slider("PSO particles", 2, 30, 8, 1)
        pso_iterations = st.slider("PSO iterations", 1, 100, 30, 1)
        hidden_dim = st.slider("Hidden dimension", 4, 128, 16, 4)
        embedding_dim = st.slider("Embedding dimension", 4, 128, 16, 4)
        heads = st.slider("Attention heads", 1, 8, 2, 1)
        dropout = st.slider("Dropout", 0.0, 0.8, 0.35, 0.05)

    st.header("Community Settings")
    k_strategy = st.radio("Number of communities", ["Use benchmark/label classes", "Estimate automatically", "Manual"], index=0)
    manual_k = st.number_input("Manual k", min_value=2, max_value=100, value=6, step=1)
    k_min = st.slider("Auto k minimum", 2, 20, 2, 1)
    k_max = st.slider("Auto k maximum", 3, 50, 10, 1)

    st.header("PSO Objective")
    alpha = st.slider("Modularity weight", 0.0, 1.0, 0.75, 0.05)
    beta = round(1.0 - alpha, 2)
    st.caption(f"Separation weight is automatically set to {beta:.2f}.")
    top_reps = st.slider("Representative nodes per community", 1, 20, 5, 1)
    seed = st.number_input("Random seed", min_value=1, max_value=9999, value=42, step=1)

    run_button = st.button("Run Model", use_container_width=True)

col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(
        """
        <div class="info-card">
            <h4>1. Representation Learning</h4>
            <p class="small-muted">A two-layer Graph Attention Network is trained through Deep Graph Infomax to produce node embeddings.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_b:
    st.markdown(
        """
        <div class="info-card">
            <h4>2. Community Search</h4>
            <p class="small-muted">K-means provides a baseline while PSO searches improved centroid positions using modularity and separation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col_c:
    st.markdown(
        """
        <div class="info-card">
            <h4>3. Explainability</h4>
            <p class="small-muted">Representative nodes are selected using centroid proximity and intra-community degree.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

if run_button:
    try:
        with st.status("Preparing dataset and running the model...", expanded=True) as status:
            st.write("Loading and preprocessing graph data...")

            if dataset_mode == "Benchmark dataset":
                loaded = load_benchmark_dataset(
                    selected_benchmark,
                    largest_component_only=largest_component_only,
                    max_nodes=max_nodes,
                    seed=int(seed),
                )
                default_known_k = loaded.known_k
                dataset_name = selected_benchmark
            else:
                if uploaded_edge_file is None:
                    raise GATPSOError("Please upload an edge list file before running the model.")
                edge_path = write_uploaded_file(uploaded_edge_file)
                features_df = read_uploaded_csv(uploaded_features_file) if uploaded_features_file is not None else None
                labels_df = read_uploaded_csv(uploaded_labels_file) if uploaded_labels_file is not None else None
                loaded = load_uploaded_dataset(
                    edge_path,
                    largest_component_only=largest_component_only,
                    max_nodes=max_nodes,
                    seed=int(seed),
                    source_col=source_col,
                    target_col=target_col,
                    features_df=features_df,
                    feature_node_col=feature_node_col,
                    labels_df=labels_df,
                    label_node_col=label_node_col,
                    label_col=label_col,
                )
                default_known_k = loaded.known_k
                dataset_name = "Uploaded dataset"

            if k_strategy == "Manual":
                known_k = int(manual_k)
            elif k_strategy == "Estimate automatically":
                known_k = None
            else:
                known_k = default_known_k

            cfg = ModelConfig(
                dataset_name=dataset_name,
                use_planetoid=(dataset_mode == "Benchmark dataset"),
                largest_component_only=largest_component_only,
                max_nodes=max_nodes,
                seed=int(seed),
                hidden_dim=int(hidden_dim),
                embedding_dim=int(embedding_dim),
                heads=int(heads),
                dropout=float(dropout),
                epochs=int(epochs),
                k_min=int(k_min),
                k_max=int(k_max),
                known_k=known_k,
                swarm_size=int(swarm_size),
                max_iter=int(pso_iterations),
                alpha_modularity=float(alpha),
                beta_separation=float(beta),
                top_representatives=int(top_reps),
            )

            train_log = st.empty()
            pso_log = st.empty()

            def train_callback(epoch: int, loss_value: float):
                train_log.write(f"Training GAT/DGI: epoch {epoch}/{cfg.epochs}, loss={loss_value:.4f}")

            def pso_callback(iteration: int, metrics: Dict[str, float]):
                pso_log.write(
                    f"Optimizing PSO: iteration {iteration}/{cfg.max_iter}, "
                    f"score={metrics['score']:.4f}, modularity={metrics['modularity']:.4f}, separation={metrics['separation']:.4f}"
                )

            result = run_gatpso_pipeline(loaded, cfg, train_callback=train_callback, pso_callback=pso_callback)
            st.session_state["gatpso_result"] = result
            status.update(label="Model run completed successfully.", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"Model run failed: {exc}")

result = st.session_state.get("gatpso_result")

if result is None:
    st.info("Select a dataset, adjust the settings, then click **Run Model** in the sidebar.")
else:
    summary = result["summary"]
    st.subheader("Results Dashboard")
    metric_cols = st.columns(7)
    metric_cols[0].metric("Nodes", f"{summary['nodes']:,}")
    metric_cols[1].metric("Edges", f"{summary['edges']:,}")
    metric_cols[2].metric("Communities", f"{summary['communities']:,}")
    metric_cols[3].metric("Training Time", f"{summary['train_seconds']:.2f}s")
    metric_cols[4].metric("PSO Time", f"{summary['pso_seconds']:.2f}s")
    metric_cols[5].metric("Total Time", f"{summary['total_seconds']:.2f}s")
    metric_cols[6].metric("Memory", f"{summary['memory_mb']:.1f} MB")

    st.caption(
        f"Source: {result['source_description']} | Feature matrix: {result['features_shape']} | "
        f"Selected k: {result['selected_k']} | Device: {summary['device']}"
    )

    tab_metrics, tab_training, tab_communities, tab_graph, tab_reps, tab_download = st.tabs(
        ["Metrics", "Training & PSO", "Community Visuals", "Graph Sample", "Representative Nodes", "Export"]
    )

    with tab_metrics:
        st.plotly_chart(plot_metric_comparison(result["results_df"]), use_container_width=True)
        st.dataframe(result["results_df"], use_container_width=True)
        if not result["k_search"].empty:
            with st.expander("k-selection details"):
                st.dataframe(result["k_search"], use_container_width=True)

    with tab_training:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_loss(result["loss_history"]), use_container_width=True)
        with col2:
            st.plotly_chart(plot_pso_convergence(result["pso_history"]), use_container_width=True)
        st.write("Runtime details")
        st.json(result["runtime"])

    with tab_communities:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_community_sizes(result["community_sizes"]), use_container_width=True)
        with col2:
            st.plotly_chart(plot_embeddings(result["embeddings"], result["labels"], seed=result["config"]["seed"]), use_container_width=True)

    with tab_graph:
        st.plotly_chart(
            plot_network_sample(result["graph"], result["labels"], sample_nodes=250, seed=result["config"]["seed"]),
            use_container_width=True,
        )
        st.caption("For readability, the graph view samples the highest-degree nodes rather than drawing the full graph.")

    with tab_reps:
        st.markdown(
            """
            ### How representative nodes are selected
            The app uses the same reasoning as the notebook: a representative node should be both **central in the embedding space** and **well connected inside its own community**.

            **Representative score = 0.5 × centroid proximity + 0.5 × normalized intra-community degree**

            - **Centroid proximity**: how close the node is to the learned community centre in GAT/DGI embedding space.
            - **Intra-community degree**: how strongly the node connects to other nodes in the same detected community.
            - A high score means the node is a good structural example of that community.
            """
        )
        st.dataframe(result["representatives_df"], use_container_width=True)

    with tab_download:
        zip_bytes = build_result_zip(result)
        st.download_button(
            "Download all result tables as ZIP",
            data=zip_bytes,
            file_name="gatpso_results.zip",
            mime="application/zip",
            use_container_width=True,
        )
        st.download_button(
            "Download community assignments CSV",
            data=result["label_table"].to_csv(index=False),
            file_name="community_assignments.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download metrics CSV",
            data=result["results_df"].to_csv(index=False),
            file_name="metrics.csv",
            mime="text/csv",
            use_container_width=True,
        )
