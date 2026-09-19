import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
import os
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# --- Page Setup ---
st.set_page_config(page_title="Necrobiome PMI Predictor", layout="wide")

# --- 1. Forensic Authentication Portal ---
def check_password():
    """Returns True if the user enters the correct access key."""
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if not st.session_state["password_correct"]:
        st.title("🔒 Forensic Gateway")
        st.write("Please authenticate to access the PMI prediction platform.")
        st.info("🔑 **Reviewer Access:** Please enter the evaluation key `demo2026` to unlock the dashboard.")
        
        access_key = st.text_input("Enter Access Key:", type="password")
        
        if st.button("Login"):
            if access_key == "demo2026":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("😕 Access Denied. Incorrect Key.")
        return False
    return True

# --- Main Application ---
if check_password():
    st.title("🦠 Necrobiome-Based Post-Mortem Interval Estimation")
    st.markdown("AI-powered prototype for analyzing early forensic microbial drivers.")

    # --- 2. Background Training Data ---
    @st.cache_data
    def load_training_data():
        df = pd.read_csv("master_pmi_matrix.csv", index_col=0)
        y = df['pmi_hours'].astype(float)
        X = df.drop(columns=['pmi_hours', 'timepoint', 'treatment', 'organ'], errors='ignore')
        return X, y

    X_train, y_train = load_training_data()
    
    variances = X_train.var()
    top_20_microbes = variances.nlargest(20).index
    X_top_train = X_train[top_20_microbes]

    # --- 3. Train the AI Model ---
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_top_train, y_train)

    # --- 4. Network Graph Generator Function ---
    def build_network_graph(data, key_suffix="main"):
        corr_matrix = data.corr(method='spearman').fillna(0)
        G = nx.Graph()
        threshold = 0.65 
        
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

    # --- 5. Sidebar & User Interface ---
    st.sidebar.header("📁 Upload Sample Data")
    st.sidebar.markdown("Upload a CSV matrix containing RNA-Seq TPM counts.")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])
    
    st.sidebar.divider()
    
    if st.sidebar.button("Load Demo Forensic Case"):
        st.session_state['use_demo'] = True

    st.sidebar.divider()
    
    # Taphonomic Sliders
    st.sidebar.subheader("🌡️ Taphonomic Metadata")
    scene_temp = st.sidebar.slider("Ambient Temperature (°C)", -10.0, 50.0, 22.0, step=0.5)
    scene_humidity = st.sidebar.slider("Relative Humidity (%)", 0, 100, 50)

    st.sidebar.divider()
    
    # Wet-to-Dry Methodology Expander
    with st.sidebar.expander("🔬 Upstream Pipeline Methodology"):
        st.write("""
        **Wet Lab Extraction:** Total RNA extracted from casework samples.  
        **Sequencing:** Metatranscriptomic library preparation (e.g., Illumina RNA-Seq).  
        **Dry Lab Pre-Processing:** Reads quantified and normalized (TPM/DESeq2) prior to model ingestion.
        """)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("👨‍🔬 **Developer:** [aafreenaazam01](https://github.com/aafreenaazam01)")
        
    # --- 6. Main Execution (Conditional Rendering) ---
    new_data = None
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file, index_col=0)
        st.session_state['use_demo'] = False
    elif st.session_state.get('use_demo', False):
        # Generate messy, realistic demo data safely
        new_data = X_top_train.sample(5, random_state=42, replace=True).copy()
        new_data.index = [f"Forensic_Case_48h_Rep{i}" for i in range(1, 6)]
        
        np.random.seed(42)
        for col in new_data.columns:
            if np.random.rand() > 0.6:
                drop_idx = np.random.choice(new_data.index, 2, replace=False)
                new_data.loc[drop_idx, col] = 0.0
            new_data[col] = new_data[col].apply(lambda x: max(0, x + np.random.normal(0, 3)) if x > 0 else 0)
            
        st.sidebar.success("✅ Loaded Messy Demo Casework")

    if new_data is not None:
        with st.spinner("Quantifying transcript abundances and calculating microbial networks..."):
            # Model prediction
            X_new = new_data.reindex(columns=top_20_microbes, fill_value=0)
            predictions = rf_model.predict(X_new)
            
            st.success("✅ Analysis Complete!")
            st.subheader("⏱️ Estimated Post-Mortem Interval (PMI)")
            
            results_df = pd.DataFrame({
                "Sample ID": new_data.index,
                "Predicted PMI (Hours)": np.round(predictions, 2)
            })
            st.dataframe(results_df, use_container_width=True)

            # Authentic Taphonomic Context Alert
            if scene_temp > 28.0 or scene_temp < 15.0:
                st.warning(f"⚠️ **Taphonomic Variance Alert:** Recorded ambient temperature (**{scene_temp}°C**) deviates from the mesophilic baseline. Microbial RNA transcription and degradation kinetics are highly temperature-dependent. Investigators must apply standard Accumulated Degree Day (ADD) corrections to the baseline PMI prediction.")
            else:
                st.info(f"ℹ️ **Environmental Context:** Scene conditions logged at **{scene_temp}°C**, **{scene_humidity}%** humidity. Model confidence is highest within this standard mesophilic range.")

            # Export Forensic Report feature
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Forensic Case Report",
                data=csv,
                file_name='forensic_PMI_report.csv',
                mime='text/csv',
            )
            
            st.divider()

            numeric_df = new_data.select_dtypes(include=[np.number])
            
            if len(numeric_df) >= 3:
                col1, col2 = st.columns(2)
                
                # Interactive Heatmap
                with col1:
                    st.subheader("🔥 Transcriptomic Heatmap")
                    st.write("Spearman correlations across active microbial biomarkers.")
                    corr_matrix = numeric_df.corr(method='spearman').fillna(0)
                    fig = px.imshow(corr_matrix, 
                                    text_auto=False, 
                                    aspect="auto",
                                    color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig, use_container_width=True)

                # Co-Expression Network
                with col2:
                    st.subheader("🕸️ Dynamic Co-Expression Network")
                    st.write("Re-calculated Spearman interactions from case file.")
                    graph_path = build_network_graph(numeric_df, key_suffix="live")
                    HtmlFile = open(graph_path, 'r', encoding='utf-8')
                    components.html(HtmlFile.read(), height=510)
                    if os.path.exists(graph_path):
                        os.remove(graph_path)
            else:
                st.warning("⚠️ A minimum of 3 samples is required to calculate correlation networks.")
    else:
        # Keeps the app completely empty until data is uploaded
        st.info("👈 Please upload a case file in the sidebar or load the Demo Case to initiate the pipeline.")
  
               
