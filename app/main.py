from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.data_ingestion import enhance_data_ingestion
from app.document_processor import DocumentProcessor
from app.reconciliation import run_reconciliation
import uvicorn

# Initialize FastAPI app
app = FastAPI(title="Document Management System")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document processor
doc_processor = DocumentProcessor()

# Setup data ingestion framework
ingestion_framework = enhance_data_ingestion()
if ingestion_framework:
    ingestion_framework["document_handler"].document_processor = doc_processor
    ingestion_framework["observer"].start()
    app.mount("/upload", ingestion_framework["api"])

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/process")
async def process_documents():
    try:
        doc_processor.process_documents(['purchase_orders.csv', 'invoices.csv', 'grns.csv'])
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reconcile")
async def reconcile():
    try:
        result = run_reconciliation()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)