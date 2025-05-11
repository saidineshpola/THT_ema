import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_processing_dashboard(stats: dict) -> None:
    """Creates an interactive dashboard with multiple plots showing document processing statistics."""
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Generated PDFs by Type", 
                       "Document Classification Results",
                       "Processing Overview",
                       "Error Analysis"),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )

    # 1. Pie Chart for Generated PDFs
    pdf_labels = list(stats["generated_pdfs"].keys())
    pdf_values = list(stats["generated_pdfs"].values())
    fig.add_trace(
        go.Pie(labels=pdf_labels, 
               values=pdf_values,
               name="Generated PDFs"),
        row=1, col=1
    )

    # 2. Bar Chart for Document Classification
    fig.add_trace(
        go.Bar(
            x=list(stats["classified_docs"].keys()),
            y=list(stats["classified_docs"].values()),
            name="Classified Documents",
            marker_color='rgb(55, 83, 109)'
        ),
        row=1, col=2
    )

    # 3. Processing Overview
    fig.add_trace(
        go.Bar(
            x=['Total Processed', 'Successfully Extracted'],
            y=[stats["processed_files"], stats["extracted_docs"]],
            name="Processing Overview",
            marker_color=['rgb(26, 118, 255)', 'rgb(58, 171, 115)']
        ),
        row=2, col=1
    )

    # 4. Error Analysis
    fig.add_trace(
        go.Bar(
            x=['Extraction Errors', 'Reconciliation Errors'],
            y=[stats["extraction_errors"], stats["reconciliation_errors"]],
            name="Errors",
            marker_color='rgb(246, 78, 139)'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title_text="Document Processing Analytics Dashboard",
        showlegend=True,
        height=800,
        width=1200,
        template="plotly_white"
    )

    # Update axes labels
    fig.update_xaxes(title_text="Document Type", row=1, col=2)
    fig.update_xaxes(title_text="Process Type", row=2, col=1)
    fig.update_xaxes(title_text="Error Type", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    # Show the plot
    fig.show()
    # Save it to a file if needed
    fig.write_html("document_processing_dashboard.html")

# Example usage:
if __name__ == "__main__":
    sample_stats = {
        "generated_pdfs": {
            "po": 5,
            "invoice": 4,
            "grn": 3
        },
        "processed_files": 12,
        "classified_docs": {
            "purchase_order": 5,
            "invoice": 4,
            "goods_received_note": 3,
            "unknown": 0
        },
        "extracted_docs": 12,
        "extraction_errors": 0,
        "reconciliation_errors": 0
    }
    create_processing_dashboard(sample_stats)