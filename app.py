import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from sklearn.ensemble import RandomForestRegressor

# --- Page Setup ---
st.set_page_config(page_title="Necrobiome PMI Predictor", layout="wide")
st.title("🔬 Necrobiome-Based Post-Mortem Interval Estimation")
st.markdown("AI-powered prototype for analyzing early forensic microbial drivers.")

# --- 1. Background Training Data ---
@st.cache_data
def load_training_data():
    df = pd.read_csv("master_pmi_matrix.csv", index_col=0)
    y = df['pmi_hours'].astype(float)
    X = df.drop(columns=['pmi_hours', 'timepoint', 'treatment', 'organ'], errors='ignore')
    return X, y

X_train, y_train = load_training_data()

# Baseline Biomarker Extraction
variances = X_train.var()
top_20_microbes = variances.nlargest(20).index
X_top_train = X_train[top_20_microbes]

# --- 2. Train the AI Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_top_train, y_train)

# --- Network Graph Generator Function ---
def build_network_graph(data, key_suffix="main"):
    corr_matrix = data.corr(method='spearman').fillna(0)
    G = nx.Graph()
    threshold = 0.65  # Strict threshold for clean, authentic biological traces
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            microbe_a = corr_matrix.columns[i]
            microbe_b = corr_matrix.columns[j]
            weight = corr_matrix.iloc[i, j]
            
            if abs(weight) > threshold and weight != 0:
                G.add_node(microbe_a, title=microbe_a, color="#2E86AB")
                G.add_node(microbe_b, title=microbe_b, color="#2E86AB")
                edge_color = "#4CAF50" if weight > 0 else "#F44336"
                G.add_edge(microbe_a, microbe_b, value=abs(weight), color=edge_color)

    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=150, spring_length=200)
    
    path = f"tmp_graph_{key_suffix}.html"
    net.save_graph(path)
    return path

# --- 3. Sidebar & User Interface ---
st.sidebar.header("📥 Upload Sample Data")
st.sidebar.markdown("Upload a CSV matrix containing RNA-Seq TPM counts.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Default active dataset for graph (Baseline)
active_graph_data = X_top_train
graph_title = "🕸️ Baseline Microbial Co-Expression Network"
graph_subtitle = "Showing baseline reference interactions from background model dataset."

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file, index_col=0)
    
    # 1. Align features for AI model prediction
    X_new = new_data.reindex(columns=top_20_microbes, fill_value=0)
    predictions = rf_model.predict(X_new)
    
    st.success("✅ Analysis Complete!")
    st.subheader("⏱️ Estimated Post-Mortem Interval (PMI)")
    
    results_df = pd.DataFrame({
        "Sample ID": new_data.index,
        "Predicted PMI (Hours)": np.round(predictions, 2)
    })
    st.dataframe(results_df, use_container_width=True)
    
    # 2. Dynamic Graph USP Logic
    numeric_df = new_data.select_dtypes(include=[np.number])
    if len(numeric_df) >= 3:
        # Extract top dynamic biomarkers from uploaded set
        up_vars = numeric_df.var().fillna(0)
        top_up_cols = up_vars.nlargest(min(20, len(numeric_df.columns))).index
        active_graph_data = numeric_df[top_up_cols]
        graph_title = "🕸️ Live Dynamic Microbial Co-Expression Network"
        graph_subtitle = "Re-calculated Spearman co-occurrence directly from your uploaded case file!"
    else:
        st.warning("ℹ️ Uploaded dataset has fewer than 3 samples. Displaying baseline interaction map for reference.")

else:
    st.info("👈 Please upload a sample CSV matrix in the sidebar to estimate PMI.")

# --- 4. Render Interactive Network ---
st.divider()
st.subheader(graph_title)
st.write(graph_subtitle)

graph_path = build_network_graph(active_graph_data, key_suffix="live")
HtmlFile = open(graph_path, 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=510)
if os.path.exists(graph_path):
    os.remove(graph_path)
