# Document Management System

A FastAPI-based/PySpark system for processing and reconciling procurement documents (Purchase Orders, Invoices, and Goods Received Notes).

## Installation

```bash
# Create and activate virtual environment using uv
uv venv
.\venv\Scripts\activate

# Install dependencies using uv
uv pip install -r requirements.txt
```

## Folder Structure

```
doc_management_system/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── document_processor.py   # Core document processing logic
│   ├── document_extraction.py  # Text extraction utilities
│   ├── ml_models.py           # Document classification models
│   ├── data_ingestion.py      # Data ingestion handlers
│   └── dashboard.py           # Processing statistics dashboard
├── simulated_data_lake/
│   ├── input_documents/       # Place PDFs here for processing
│   ├── processed_data/        # Extracted data and reports
│   ├── generated_sample_pdfs/ # Demo document samples
│   └── archived_documents/    # Processed files archive
├── logs/
│   └── document_management.log
└── requirements.txt
```

## System Architecture

### Current Implementation (FastAPI)

1. **Data Ingestion Layer**
   - Document upload via API endpoints
   - Automatic monitoring of input directory
   - Support for PDF document formats

2. **Processing Layer**
   - Document classification using LayoutLMv2
   - Text extraction using PyMuPDF
   - Structured data extraction to CSV

3. **Reconciliation Layer**
   - PO-Invoice-GRN matching
   - Discrepancy identification
   - Reconciliation report generation

4 **Actional Insights**
   - Spend analysis and trend reporting
   - Vendor performance metrics
   - Classification and Error reports

5. **API Endpoints**
   - `/ops/generate-sample-pdfs` : Generate Sample PDFS
   - `/ops/clear-processed-data` : Generate Sample PDFS
   - `/documents/process-data-lake`: Process documents in batch
   - `/reconciliation/run`: Execute reconciliation
   - `/status/processing-stats`: View system statistics

### Future Enhancements

- **PySpark Integration** (Planned)
  - Distributed document processing
  - Scalable data pipeline
  - Enhanced batch processing capabilities

## Quick Start

1. **Start the API Server**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Generate Sample Documents**
   - Visit `http://localhost:8000/docs`
   - Use the `/ops/generate-sample-pdfs` endpoint

3. **Process Documents**
   - Copy PDFs to `simulated_data_lake/input_documents/`
   - Call `/documents/process-data-lake` endpoint

4. **View Results**
   - Check `simulated_data_lake/processed_data/` for extracted data
   - Use `/status/processing-stats` for processing statistics

## API Documentation

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Key Features

- **Robust Document Processing**
  - Intelligent document classification
  - Accurate text extraction
  - Structured data output

- **Error Handling**
  - Comprehensive logging
  - Failed document tracking
  - Process monitoring

- **Performance Monitoring**
  - Processing statistics
  - Success/failure rates
  - Processing time metrics

## Development Notes

- Uses FastAPI for async processing
- Implements background tasks for long-running operations
- Includes demo setup utilities
- Provides clear API documentation

## Future Roadmap

1. PySpark integration for distributed processing
2. Advanced ML model integration
3. Real-time processing dashboard
4. Automated test suite expansion

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT