from fastapi import FastAPI, File, UploadFile
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import pandas as pd

def enhance_data_ingestion():
    """Setup enhanced data ingestion framework"""
    try:
        # Create directories if they don't exist
        os.makedirs("data_lake_input", exist_ok=True)
        
        # Setup document watching system using watchdog
        class DocumentHandler(PatternMatchingEventHandler):
            def __init__(self):
                super().__init__(patterns=["*.pdf", "*.csv", "*.xlsx"], 
                                ignore_directories=True)
                self.document_processor = None  # Will be set externally
                
            def on_created(self, event):
                """Process file when detected in watch directory"""
                if not self.document_processor:
                    print("Warning: DocumentProcessor not set in handler")
                    return
                    
                file_path = event.src_path
                file_type = os.path.splitext(file_path)[1].lower()
                
                print(f"New file detected: {file_path}")
                
                # Read file based on type
                try:
                    if file_type == '.pdf':
                        # For PDF files, defer to document extraction
                        from app.document_extraction import extract_data_from_pdf
                        doc_type, extracted_data = extract_data_from_pdf(file_path)
                        
                        # Process based on document type
                        if doc_type and extracted_data:
                            # Save to appropriate data store and generate PDF
                            if doc_type == "purchase_order":
                                df = pd.DataFrame([extracted_data])
                                df.to_csv("data/purchase_orders_extracted.csv", mode='a', header=False, index=False)
                                self.document_processor.generate_pdf_po(extracted_data, f"purchase_orders_{extracted_data['po_number']}.pdf")
                            elif doc_type == "invoice":
                                df = pd.DataFrame([extracted_data])
                                df.to_csv("data/invoices_extracted.csv", mode='a', header=False, index=False)
                                self.document_processor.generate_pdf_invoice(extracted_data, f"invoices_{extracted_data['invoice_number']}.pdf")
                            elif doc_type == "goods_received":
                                df = pd.DataFrame([extracted_data])
                                df.to_csv("data/grns_extracted.csv", mode='a', header=False, index=False)
                                self.document_processor.generate_pdf_grn(extracted_data, f"grns_{extracted_data['grn_number']}.pdf")
                    
                    elif file_type == '.csv':
                        df = pd.read_csv(file_path)
                        print(f"CSV read with {len(df)} rows")
                        # Determine file type from content
                        if 'po_number' in df.columns:
                            df.to_csv("data/purchase_orders_extracted.csv", mode='a', header=False, index=False)
                        elif 'invoice_number' in df.columns:
                            df.to_csv("data/invoices_extracted.csv", mode='a', header=False, index=False)
                        elif 'grn_number' in df.columns:
                            df.to_csv("data/grns_extracted.csv", mode='a', header=False, index=False)
                        
                    elif file_type == '.xlsx':
                        df = pd.read_excel(file_path)
                        print(f"Excel read with {len(df)} rows")
                        # Determine file type from content
                        if 'po_number' in df.columns:
                            df.to_csv("data/purchase_orders_extracted.csv", mode='a', header=False, index=False)
                        elif 'invoice_number' in df.columns:
                            df.to_csv("data/invoices_extracted.csv", mode='a', header=False, index=False)
                        elif 'grn_number' in df.columns:
                            df.to_csv("data/grns_extracted.csv", mode='a', header=False, index=False)
                            
                    # Update reconciliation after processing
                    from app.reconciliation import run_reconciliation
                    run_reconciliation()
                    
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
        
        # Create an instance of the handler
        document_handler = DocumentHandler()
        
        # Setup document directory observer
        observer = Observer()
        observer.schedule(document_handler, path="data_lake_input/", recursive=False)
        
        # FastAPI for document uploads
        upload_app = FastAPI(title="Document Upload API")
        
        @upload_app.post("/upload/")
        async def upload_document(file: UploadFile = File(...)):
            """API endpoint for document uploads"""
            try:
                # Save uploaded file
                file_path = f"data_lake_input/{file.filename}"
                with open(file_path, "wb") as f:
                    f.write(await file.read())
                return {"filename": file.filename, "status": "uploaded and processing"}
            except Exception as e:
                return {"error": str(e)}
        
        return {
            "api": upload_app,
            "observer": observer,
            "document_handler": document_handler
        }
    
    except Exception as e:
        print(f"Error setting up data ingestion: {str(e)}")
        return None