# app/data_ingestion.py
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import shutil

# Assuming DocumentProcessor is accessible or passed somehow
# For this example, we might not directly use it here but in main.py

logger = logging.getLogger("document_management.ingestion")

# Directory where uploaded files for the demo will be temporarily stored
# This should match the input_docs_dir used by DocumentProcessor
UPLOAD_DIR = Path("simulated_data_lake/input_documents") 
UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure it exists

router = APIRouter()

@router.post("/upload-documents/")
async def upload_documents_for_processing(files: List[UploadFile] = File(...)):
    """
    Endpoint to upload multiple documents to the simulated data lake's input directory.
    """
    uploaded_filenames = []
    for file in files:
        try:
            # Sanitize filename (basic)
            safe_filename = Path(file.filename).name 
            destination_path = UPLOAD_DIR / safe_filename
            with open(destination_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_filenames.append(file.filename)
            logger.info(f"File '{file.filename}' uploaded to '{destination_path}'")
        except Exception as e:
            logger.error(f"Could not upload file '{file.filename}': {e}")
            raise HTTPException(status_code=500, detail=f"Could not upload file: {file.filename}. Error: {e}")
        finally:
            file.file.close() # Ensure file is closed
            
    return {
        "message": f"Successfully uploaded {len(uploaded_filenames)} files.",
        "filenames": uploaded_filenames,
        "upload_directory": str(UPLOAD_DIR)
    }

def enhance_data_ingestion():
    """
    Conceptual function. For the demo, we primarily use the upload endpoint.
    A real system might have Kafka consumers or directory watchers here.
    """
    logger.info("Data ingestion framework placeholder initialized.")
    # In a more complex setup, this could return objects for watching directories,
    # handling Kafka messages, etc. For now, we'll use the API router.
    return {"api_router": router}