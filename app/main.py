# app/main.py

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.document_processor import DocumentProcessor
from app.data_ingestion import enhance_data_ingestion # Using the simplified version
import uvicorn
from pathlib import Path
import os # For environment variables

# --- Configure Logging ---
# (Using a more robust setup for file and console)
log_file_path = Path("logs/document_management.log")
log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure log directory exists

# Remove default handlers from root logger if any
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Log to file
        logging.StreamHandler()             # Log to console
    ]
)
logger = logging.getLogger("document_management.api")


# --- Initialize FastAPI app ---
app = FastAPI(
    title="Complex Purchase Order and Invoice Management System",
    description="""
    API for ingesting, classifying, extracting data from, and reconciling
    procurement documents (POs, Invoices, GRNs).
    """,
    version="0.1.0"
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Initialize Document Processor ---
# Determine extractor type (e.g., from environment variable for flexibility)
# For the demo, you can switch between "pytesseract" and "advanced_model" (the placeholder for Docling)
# To use the Docling placeholder: set EXTRACTOR_CHOICE="advanced_model" in your environment
EXTRACTOR_TYPE = os.getenv("EXTRACTOR_CHOICE", "pymupdf").lower()
# MODEL_NAME = os.getenv("CLASSIFIER_MODEL", "microsoft/layoutlmv2-base-uncased") # If you want to configure classifier model

try:
    doc_processor = DocumentProcessor(extractor_type=EXTRACTOR_TYPE) #, doc_classifier_model_name=MODEL_NAME)
    logger.info(f"DocumentProcessor initialized with extractor: {EXTRACTOR_TYPE}")
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize DocumentProcessor: {e}. Application might not function correctly.")
    # Depending on severity, you might want to prevent app startup or run in a degraded mode.
    # For now, we'll let it start and endpoints might fail if doc_processor is None or not fully functional.
    doc_processor = None # Ensure it's defined, even if None

# --- Data Ingestion Setup ---
# The `enhance_data_ingestion` returns a dictionary, here we expect an APIRouter
ingestion_setup = enhance_data_ingestion()
if ingestion_setup and "api_router" in ingestion_setup and doc_processor: # Check if doc_processor initialized
    app.include_router(ingestion_setup["api_router"], prefix="/ingestion", tags=["1. Data Ingestion"])
else:
    logger.warning("Ingestion API router not mounted. Ingestion setup failed or DocumentProcessor not available.")


# --- API Endpoints ---

@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to the Document Management System API.",
        "docs_url": "/docs",
        "status": "healthy"
    }

@app.post("/ops/generate-sample-pdfs", tags=["0. Demo Setup"])
async def generate_sample_pdfs_endpoint(
    num_pos: int = 5, num_invoices: int = 4, num_grns: int = 3, clear_existing: bool = True
):
    """
    Generates synthetic PDF documents (POs, Invoices, GRNs) and saves them
    to the 'simulated_data_lake/generated_sample_pdfs' directory.
    These can then be manually copied or uploaded to 'input_documents' for processing.
    """
    if not doc_processor:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    try:
        result = doc_processor.generate_synthetic_document_pdfs(num_pos, num_invoices, num_grns, clear_output_dir=clear_existing)
        return {
            "message": "Sample PDF documents generated.",
            "details": result,
            "output_directory": str(doc_processor.generated_pdf_output_dir),
            "next_step": f"Copy desired PDFs to '{doc_processor.input_docs_dir}' and then call /documents/process-data-lake"
        }
    except Exception as e:
        logger.error(f"Error generating sample PDFs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate sample PDFs: {str(e)}")

@app.post("/ops/clear-processed-data", tags=["0. Demo Setup"])
async def clear_processed_data_endpoint():
    """
    Clears previously processed data (extracted CSVs, reconciliation reports)
    from the 'simulated_data_lake/processed_data' directory.
    Useful for running a fresh demo.
    """
    if not doc_processor:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    try:
        doc_processor.clear_simulated_data_lake_processed_outputs()
        return {"message": "Successfully cleared processed data and reset relevant stats."}
    except Exception as e:
        logger.error(f"Error clearing processed data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear processed data: {str(e)}")


@app.post("/documents/process-data-lake", tags=["2. Document Processing"])
async def process_documents_from_lake_endpoint(background_tasks: BackgroundTasks):
    """
    Triggers the processing of all PDF documents currently in the 
    'simulated_data_lake/input_documents' directory.
    This is an asynchronous operation. Check logs or the status endpoint for progress.
    """
    if not doc_processor:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    
    # Running as a background task so the API call returns immediately
    # background_tasks.add_task(doc_processor.process_documents_from_data_lake)
    # For demo, direct call might be better to see immediate results or errors, but can timeout for many files
    try:
        logger.info("Endpoint /documents/process-data-lake called. Starting processing...")
        result = doc_processor.process_documents_from_data_lake()
        logger.info(f"Processing from data lake completed with result: {result}")
        return {
            "message": "Document processing from data lake initiated and completed.",
            "summary": result
            }
    except Exception as e:
        logger.error(f"Error during /documents/process-data-lake: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process documents from data lake: {str(e)}")


@app.post("/reconciliation/run", tags=["3. Reconciliation"])
async def run_reconciliation_endpoint():
    """
    Runs the reconciliation process on the documents processed from the
    data lake (i.e., using the CSVs in 'simulated_data_lake/processed_data').
    """
    if not doc_processor:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    try:
        result = doc_processor.reconcile_processed_documents()
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        return result
    except HTTPException as http_exc: # Re-raise HTTP exceptions
        raise http_exc
    except Exception as e:
        logger.error(f"Error during reconciliation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reconciliation failed: {str(e)}")

@app.get("/status/processing-stats", tags=["4. Status & Insights"])
async def get_system_stats_endpoint():
    """
    Retrieves the current processing statistics from the DocumentProcessor.
    """
    if not doc_processor:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    stats = doc_processor.get_processing_stats()
    from app.dashboard import create_processing_dashboard
    create_processing_dashboard(stats)
    return stats


if __name__ == "__main__":
    # Make sure Poppler (for pdf2image) and Tesseract OCR are installed and in PATH
    
    logger.info("Starting Document Management System API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)