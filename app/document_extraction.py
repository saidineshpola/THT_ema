# app/document_extraction.py

import re
import json
import fitz  # PyMuPDF
from PIL import Image
import pdf2image
import os
import logging
from typing import Dict, Optional, Protocol, Tuple

# Import the advanced extractor from ml_models


logger = logging.getLogger("document_management.extraction")

# --- Configuration for OCR Engine ---
# Set TESSERACT_CMD_PATH to your Tesseract installation path if needed
# Example: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Ensure Poppler is in your PATH for pdf2image or specify poppler_path in convert_from_path

# --- Define an Extractor Interface (Protocol) ---
class Extractor(Protocol):
    def extract_text_from_image(self, image: Image.Image) -> str:
        """Extracts raw text from a single image."""
        ...

    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        """Extracts structured data from raw text based on document type."""
        ...

    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[Optional[str], Dict]:
        """Processes a PDF, extracts text, and then structured data."""
        ...

# --- Basic PyTesseract Implementation ---
class PyTesseractExtractor:
    def __init__(self):
        import pytesseract
        logger.info("PyTesseractExtractor initialized.")

    def extract_text_from_image(self, image: Image.Image) -> str:
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error during Tesseract OCR: {e}")
            return ""

    def _extract_invoice_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyTesseractExtractor_BasicRule"}
        # Very basic regex examples, expand significantly for real use
        if match := re.search(r"Invoice Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["invoice_id"] = match.group(1)
        if match := re.search(r"Total Amount[:\s€$£]*([0-9,]+\.?\d*)", text, re.IGNORECASE):
            data["total_amount"] = float(match.group(1).replace(',', ''))
        if match := re.search(r"Purchase Order[:\s#]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_reference"] = match.group(1)
        # Add more regex for other fields...
        return data

    def _extract_po_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyTesseractExtractor_BasicRule"}
        if match := re.search(r"Purchase Order Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_number"] = match.group(1)
        if match := re.search(r"Total Amount[:\s€$£]*([0-9,]+\.?\d*)", text, re.IGNORECASE):
            data["total_amount"] = float(match.group(1).replace(',', ''))
        # Add more regex for other fields...
        return data
        
    def _extract_grn_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyTesseractExtractor_BasicRule"}
        if match := re.search(r"GRN Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["grn_number"] = match.group(1)
        if match := re.search(r"PO Reference[:\s#]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_reference"] = match.group(1)
        if match := re.search(r"Quantity Received[:\s]*(\d+)", text, re.IGNORECASE):
             data["received_qty"] = int(match.group(1))
        # Add more regex for other fields...
        return data

    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        logger.info(f"Extracting structured data for doc_type: {doc_type} using PyTesseractExtractor.")
        if doc_type == "purchase_order":
            return self._extract_po_data_basic(text)
        elif doc_type == "invoice":
            return self._extract_invoice_data_basic(text)
        elif doc_type == "goods_received_note": # Match classifier output
            return self._extract_grn_data_basic(text)
        else:
            logger.warning(f"Unknown document type for PyTesseract basic extraction: {doc_type}")
            return {"error": "Unknown document type for basic rule-based extraction"}

    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[Optional[str], Dict]:
        """Extracts data from a PDF using PyTesseract for OCR and basic rule-based parsing."""
        full_text = ""
        try:
            # Check if poppler is available for pdf2image
            try:
                images = pdf2image.convert_from_path(pdf_path, poppler_path = r'C:\Users\user\Desktop\projects\THT_ema\.venv\Release-24.08.0-0\poppler-24.08.0\Library\bin') # Add poppler_path if not in PATH
            except pdf2image.exceptions.PDFInfoNotInstalledError:
                logger.error("Poppler not found. PDF to image conversion failed. Ensure Poppler is in your PATH.")
                return None, {"error": "Poppler not found for PDF conversion."}
            except Exception as e:
                logger.error(f"PDF to image conversion failed for {pdf_path}: {e}")
                return None, {"error": f"PDF to image conversion failed: {e}"}

            for img in images:
                full_text += self.extract_text_from_image(img) + "\n"
            
            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                return full_text, {"error": "No text extracted from PDF"}
                
            structured_data = self.extract_structured_data(full_text, doc_type)
            return full_text, structured_data

        except Exception as e:
            logger.error(f"Error extracting data from PDF {pdf_path} with PyTesseractExtractor: {str(e)}")
            return full_text, {"error": str(e)}

# ADvanced Tobe added
class AdvancedDataExtractor:
    def __init__(self, api_key: Optional[str] = None):
        # Placeholder for advanced model initialization
        self.api_key = api_key
        logger.info("AdvancedDataExtractor initialized with API key.")

    def extract_fields(self, pdf_path: str, doc_type: str) -> Dict:
        # Placeholder for advanced model extraction logic
        raise NotImplementedError("Advanced model extraction not implemented yet.")

class PyMuPDFExtractor:
    def __init__(self):
        logger.info("PyMuPDFExtractor initialized.")

    def extract_text_from_image(self, image: Image.Image) -> str:
        """
        Note: PyMuPDF doesn't directly handle PIL Images. This is provided
        for interface compatibility and falls back to Tesseract.
        """
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            logger.error(f"Error during image text extraction: {e}")
            return ""

    def _extract_invoice_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyMuPDFExtractor_BasicRule"}
        if match := re.search(r"Invoice Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["invoice_id"] = match.group(1)
        if match := re.search(r"Total Amount[:\s€$£]*([0-9,]+\.?\d*)", text, re.IGNORECASE):
            data["total_amount"] = float(match.group(1).replace(',', ''))
        if match := re.search(r"Purchase Order[:\s#]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_reference"] = match.group(1)
        return data

    def _extract_po_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyMuPDFExtractor_BasicRule"}
        if match := re.search(r"Purchase Order Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_number"] = match.group(1)
        if match := re.search(r"Total Amount[:\s€$£]*([0-9,]+\.?\d*)", text, re.IGNORECASE):
            data["total_amount"] = float(match.group(1).replace(',', ''))
        return data

    def _extract_grn_data_basic(self, text: str) -> Dict:
        data = {"extracted_with": "PyMuPDFExtractor_BasicRule"}
        if match := re.search(r"GRN Number[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["grn_number"] = match.group(1)
        if match := re.search(r"PO Reference[:\s#]*([A-Z0-9\-]+)", text, re.IGNORECASE):
            data["po_reference"] = match.group(1)
        if match := re.search(r"Quantity Received[:\s]*(\d+)", text, re.IGNORECASE):
            data["received_qty"] = int(match.group(1))
        return data

    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        logger.info(f"Extracting structured data for doc_type: {doc_type} using PyMuPDFExtractor.")
        if doc_type == "purchase_order":
            return self._extract_po_data_basic(text)
        elif doc_type == "invoice":
            return self._extract_invoice_data_basic(text)
        elif doc_type == "goods_received_note":
            return self._extract_grn_data_basic(text)
        else:
            logger.warning(f"Unknown document type for PyMuPDF extraction: {doc_type}")
            return {"error": "Unknown document type for basic rule-based extraction"}

    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[Optional[str], Dict]:
        """Extracts data from a PDF using PyMuPDF for text extraction and basic rule-based parsing."""
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                full_text += page.get_text() + "\n"
            doc.close()

            if not full_text.strip():
                logger.warning(f"No text extracted from PDF: {pdf_path}")
                # Fall back to OCR if no text was extracted
                return self._fallback_to_ocr(pdf_path, doc_type)

            structured_data = self.extract_structured_data(full_text, doc_type)
            return full_text, structured_data

        except Exception as e:
            logger.error(f"Error extracting data from PDF {pdf_path} with PyMuPDFExtractor: {str(e)}")
            return full_text, {"error": str(e)}

    def _fallback_to_ocr(self, pdf_path: str, doc_type: str) -> Tuple[Optional[str], Dict]:
        """Fallback method that uses Tesseract OCR when PyMuPDF can't extract text."""
        logger.info("Falling back to OCR-based extraction")
        try:
            pytesseract_extractor = PyTesseractExtractor()
            return pytesseract_extractor.extract_from_pdf_path(pdf_path, doc_type)
        except Exception as e:
            logger.error(f"OCR fallback failed for {pdf_path}: {e}")
            return None, {"error": "Both PyMuPDF and OCR fallback failed"}

# --- Wrapper for Advanced Extractor (Docling Placeholder) ---
class AdvancedModelExtractor:
    def __init__(self, api_key: Optional[str] = None):
        self.model = AdvancedDataExtractor(api_key=api_key) # From ml_models.py
        logger.info("AdvancedModelExtractor initialized.")

    def extract_text_from_image(self, image: Image.Image) -> str:
        # This method might not be directly used if the advanced model takes PDF paths
        # or image bytes directly and does its own OCR.
        # For consistency, or if you need to pass raw text to a part of it:
        logger.warning("extract_text_from_image called on AdvancedModelExtractor - typically it works on full documents.")
        # You could use pytesseract here as a fallback if the advanced model needs pre-extracted text
        # return pytesseract.image_to_string(image)
        return "Text extraction from single image not standard for this advanced model."

    def extract_structured_data(self, text: str, doc_type: str) -> Dict:
        # This method might also be less relevant if the model processes files directly.
        # It's here for interface consistency.
        logger.warning("extract_structured_data called on AdvancedModelExtractor with raw text - typically it works on full documents.")
        return {"error": "Advanced model typically processes files directly, not raw text for structured data."}

    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[Optional[str], Dict]:
        """
        Processes a PDF using the advanced model (e.g., Docling).
        The 'text' part of the return might be None or a summary if the model doesn't return full OCR text.
        """
        try:
            structured_data = self.model.extract_fields(pdf_path, doc_type)
            # The advanced model might not return the full raw text easily,
            # so we return None or a relevant string.
            return f"Processed by Advanced Model: {pdf_path}", structured_data
        except Exception as e:
            logger.error(f"Error extracting data from PDF {pdf_path} with AdvancedModelExtractor: {str(e)}")
            return None, {"error": str(e)}


# --- Factory function to get the desired extractor ---
def get_extractor(extractor_type: str = "pytesseract", api_key: Optional[str] = None) -> Extractor:
    """
    Factory function to get an instance of the specified extractor.
    
    Args:
        extractor_type (str): "pytesseract" or "advanced_model" (for Docling-like).
        api_key (str, optional): API key if using the advanced model.
        
    Returns:
        Extractor: An instance of the chosen extractor.
    """
    if extractor_type.lower() == "advanced_model":
        logger.info("Using AdvancedModelExtractor.")
        return AdvancedModelExtractor(api_key=api_key)
    elif extractor_type.lower() == "pymupdf":
        logger.info("Using PyMuPDFExtractor.")
        return PyMuPDFExtractor()
    elif extractor_type.lower() == "pytesseract":
        logger.info("Using PyTesseractExtractor.")
        return PyTesseractExtractor()
    else:
        logger.warning(f"Unknown extractor type '{extractor_type}'. Defaulting to PyTesseractExtractor.")
        return PyTesseractExtractor()

# Example usage (for testing this module directly)
if __name__ == "__main__":
    # This is just for testing, you'll use this from document_processor.py
    
    # Create a dummy PDF for testing (requires reportlab)
    from reportlab.pdfgen import canvas
    dummy_pdf_path = "dummy_invoice.pdf"
    c = canvas.Canvas(dummy_pdf_path)
    c.drawString(100, 750, "Sample Invoice Document")
    c.drawString(100, 700, "Invoice Number: INV-TEST-123")
    c.drawString(100, 650, "Total Amount: $150.00")
    c.save()

    # Test PyTesseract Extractor
    print("\n--- Testing PyTesseractExtractor ---")
    pytesseract_extractor = get_extractor("pytesseract")
    # Assuming you have a PDF and its type is known (e.g., from classifier)
    # For this test, we'll manually set doc_type
    text, data = pytesseract_extractor.extract_from_pdf_path(dummy_pdf_path, "invoice")
    print(f"Extracted Text (Pytesseract):\n{text[:200]}...") # Print first 200 chars
    print(f"Extracted Data (Pytesseract): {data}")

    # Test Advanced Model Extractor (Placeholder)
    print("\n--- Testing AdvancedModelExtractor (Placeholder) ---")
    # For a real test, you'd pass an API key: get_extractor("advanced_model", api_key="YOUR_API_KEY")
    advanced_extractor = get_extractor("advanced_model") 
    text_adv, data_adv = advanced_extractor.extract_from_pdf_path(dummy_pdf_path, "invoice")
    print(f"Response from Advanced Model (Placeholder Text): {text_adv}")
    print(f"Extracted Data (Advanced Model Placeholder): {data_adv}")
    
    # Clean up dummy file
    if os.path.exists(dummy_pdf_path):
        os.remove(dummy_pdf_path)