import re
import json
import pytesseract
from PIL import Image
import pdf2image
import spacy
import os

# Load NLP model for entity extraction
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If model isn't downloaded, use a fallback approach
    nlp = None
    print("Warning: Spacy model not found. Please download with: python -m spacy download en_core_web_sm")

def extract_data_from_pdf(pdf_path):
    """Extract data from a PDF document"""
    try:
        # Convert PDF to images
        images = pdf2image.convert_from_path(pdf_path)
        
        # Extract text using OCR
        extracted_text = ""
        for img in images:
            extracted_text += pytesseract.image_to_string(img)
            
        # Identify document type
        doc_type = classify_document(extracted_text)
        
        # Extract structured data based on document type
        if doc_type == "purchase_order":
            data = extract_po_data(extracted_text)
        elif doc_type == "invoice":
            data = extract_invoice_data(extracted_text)
        elif doc_type == "goods_received":
            data = extract_grn_data(extracted_text)
        else:
            data = {"error": "Unknown document type"}
            
        return doc_type, data
    except Exception as e:
        print(f"Error extracting data from PDF: {str(e)}")
        return None, {"error": str(e)}

def classify_document(text):
    """Classify document type based on text content"""
    text_lower = text.lower()
    
    # Simple rule-based classification
    if re.search(r'purchase\s+order|p\.?o\.?\s+number', text_lower):
        return "purchase_order"
    elif re.search(r'invoice|bill\s+to', text_lower):
        return "invoice"
    elif re.search(r'goods\s+received|receipt\s+note|delivery\s+note', text_lower):
        return "goods_received"
    else:
        return "unknown"