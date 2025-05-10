# Installation


# Folder Structure

doc_management_system/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI main application
│   ├── document_processor.py    # Core document processing functionality
│   ├── data_ingestion.py        # Enhanced data ingestion framework
│   ├── document_extraction.py   # Document classification & data extraction
│   ├── reconciliation.py        # Advanced reconciliation engine
│   ├── dashboard.py             # Analytics and dashboard
│   └── ml_processing.py         # ML for document processing
│
├── data/                        # Data storage directories
│   ├── purchase_orders_extracted.csv
│   ├── invoices_extracted.csv
│   └── grns_extracted.csv
│
├── data_lake_input/             # Directory for document ingestion
│
├── output_documents/            # Directory for reconciliation outputs
│
├── static/                      # Static files for the dashboard
│
├── requirements.txt             # Project dependencies
│
└── run.py                       # Script to run the application

Document Management System Flow and Demo Structure
Overall Architecture Flow
Data Ingestion Layer
Documents arrive in data_lake_input/ directory
Watchdog monitors for new files
Files are processed based on type (PDF/CSV/XLSX)
Processing Layer
Document classification using LayoutLMv2
Data extraction using OCR and NLP
Storage in structured CSV files
Reconciliation Layer
Matching POs with Invoices and GRNs
Identifying discrepancies
Generating reports
Analytics Layer
Real-time dashboard using Streamlit
Interactive visualizations with Plotly
KPI tracking and alerts
Adding Airflow for Automation
Let's add an Airflow DAG for automation. Create a new directory dags/ and add:

Demo Flow
Setup Demo (5 mins)
Show folder structure
Explain key components
Start the FastAPI server and Airflow
Document Ingestion Demo (10 mins)
Drop sample documents in data_lake_input/
Show real-time processing
Demonstrate API endpoints
Document Processing Demo (10 mins)
Show document classification
Display extracted data
Demonstrate error handling
Reconciliation Demo (10 mins)
Show matching logic
Display discrepancies
Generate reconciliation report
Dashboard Demo (10 mins)
Show real-time metrics
Demonstrate interactive filters
Display trend analysis
Automation Demo (5 mins)
Show Airflow DAG
Demonstrate scheduled processing
Show error handling and retries
Key Points to Emphasize
Scalability
Async processing with FastAPI
Distributed processing capability
Queue-based architecture
Reliability
Error handling at each layer
Automated retries
Comprehensive logging
Monitoring
Real-time dashboard
Automated alerts
Performance metrics
Flexibility
Support for multiple file formats
Configurable processing rules
Extensible architecture
To run the demo locally:

The system is designed to handle high volumes while maintaining reliability and providing clear visibility into the document processing pipeline.

generate_pdf_grn
