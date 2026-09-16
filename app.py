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

# Biomarker Extraction (Top 20)
variances = X_train.var()
top_20_microbes = variances.nlargest(20).index
X_top_train = X_train[top_20_microbes]

# --- 2. Train the AI Model ---
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_top_train, y_train)

# --- 3. The Scientist's Interface (Sidebar) ---
st.sidebar.header("📥 Upload Sample Data")
st.sidebar.markdown("Upload a CSV matrix containing RNA-Seq TPM counts.")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the scientist's uploaded file
    new_data = pd.read_csv(uploaded_file, index_col=0)
    
    # Filter for our specific 20 biomarkers
    # (Fills missing microbes with 0 to prevent errors)
    X_new = pd.DataFrame(columns=top_20_microbes)
    for col in top_20_microbes:
        if col in new_data.columns:
            X_new[col] = new_data[col]
        else:
            X_new[col] = 0
            
    # Predict Time of Death
    predictions = rf_model.predict(X_new)
    
    st.success("✅ Analysis Complete!")
    st.subheader("⏱️ Estimated Post-Mortem Interval (PMI)")
    
    # Display results nicely
    results_df = pd.DataFrame({
        "Sample ID": new_data.index,
        "Predicted PMI (Hours)": np.round(predictions, 2)
    })
    st.dataframe(results_df, use_container_width=True)

else:
    st.info("👈 Please upload a sample CSV matrix in the sidebar to estimate PMI.")

# --- 4. Gene-Gene (Microbe) Interaction Graph ---
st.divider()
st.subheader("🕸️ Background Microbial Co-Expression Network")
st.write("This interactive graph shows how the top 20 biomarker species interact during decomposition based on the training set. Green links indicate synergy; red links indicate competition.")

def build_network_graph(data):
    corr_matrix = data.corr(method='spearman')
    G = nx.Graph()
    threshold = 0.7  
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            microbe_a = corr_matrix.columns[i]
            microbe_b = corr_matrix.columns[j]
            weight = corr_matrix.iloc[i, j]
            
            if abs(weight) > threshold:
                G.add_node(microbe_a, title=microbe_a, color="#2E86AB")
                G.add_node(microbe_b, title=microbe_b, color="#2E86AB")
                edge_color = "#4CAF50" if weight > 0 else "#F44336"
                G.add_edge(microbe_a, microbe_b, value=abs(weight), color=edge_color)

    net = Network(height="500px", width="100%", bgcolor="#222222", font_color="white")
    net.from_nx(G)
    net.repulsion(node_distance=150, spring_length=200)
    
    path = "tmp_graph.html"
    net.save_graph(path)
    return path

graph_path = build_network_graph(X_top_train)
HtmlFile = open(graph_path, 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=510)
os.remove(graph_path)
