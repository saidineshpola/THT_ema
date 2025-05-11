from typing import Optional, Dict
import logging
import fitz  # PyMuPDF
from transformers import pipeline

logger = logging.getLogger("document_management.ml_models")

class DocumentClassifier:
    """
    Uses zero-shot classification to classify document text.
    """
    def __init__(self, model_name="facebook/bart-large-mnli"):
        try:
            self.classifier = pipeline("zero-shot-classification", model=model_name)
            self.classes = ["purchase_order", "invoice", "goods_received_note"]
            logger.info(f"DocumentClassifier initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing DocumentClassifier: {e}")
            self.classifier = None
            raise RuntimeError(f"Could not load transformer model: {e}")

    def extract_text(self, pdf_path: str) -> str:
        """Extract text directly from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"PyMuPDF extraction error: {str(e)}")
            return ""

    def classify_document(self, pdf_path: str) -> str:
        """Classify PDF document using zero-shot classification."""
        if not self.classifier:
            logger.error("DocumentClassifier not initialized.")
            return "unknown"
        
        try:
            # Extract text from PDF
            text = self.extract_text(pdf_path)
            if not text:
                logger.warning("No text extracted from PDF")
                return "unknown"

            # Classify using zero-shot classification
            result = self.classifier(
                text,
                candidate_labels=self.classes,
              
            )
            
            # Get the highest scoring class
            predicted_class = result['labels'][0]
            confidence = result['scores'][0]
            
            logger.info(f"Classification result: {predicted_class} (confidence: {confidence:.2f})")
            return predicted_class
                
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return "unknown"