import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

def load_data():
    """Load reconciled data"""
    data_dir = Path("data")
    pos = pd.read_csv(data_dir / "purchase_orders_extracted.csv")
    invoices = pd.read_csv(data_dir / "invoices_extracted.csv")
    grns = pd.read_csv(data_dir / "grns_extracted.csv")
    return pos, invoices, grns

def create_dashboard():
    st.title("Document Management Analytics Dashboard")
    
    try:
        pos, invoices, grns = load_data()
        
        # Key Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total POs", len(pos))
        with col2:
            st.metric("Total Invoices", len(invoices))
        with col3:
            st.metric("Total GRNs", len(grns))
            
        # Time Series Analysis
        st.subheader("Document Volume Over Time")
        daily_volumes = pd.concat([
            pos.groupby('issue_date').size(),
            invoices.groupby('invoice_date').size(),
            grns.groupby('received_date').size()
        ], axis=1)
        fig = px.line(daily_volumes, title="Daily Document Volumes")
        st.plotly_chart(fig)
        
        # Reconciliation Status
        st.subheader("Reconciliation Status")
        matched = len(pos[pos['po_number'].isin(invoices['po_reference'])])
        unmatched = len(pos) - matched
        fig = go.Figure(data=[go.Pie(labels=['Matched', 'Unmatched'], 
                                   values=[matched, unmatched])])
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error loading dashboard: {str(e)}")

if __name__ == "__main__":
    create_dashboard()