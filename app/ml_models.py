# path ./app/ml_models.py
from typing import Optional, Dict
import logging
import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("document_management.ml_models")

class DocumentClassifier:
    """
    Uses a small transformer model to classify document text.
    """
    def __init__(self, model_name="prajjwal1/bert-tiny", num_labels=3):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels
            )
            self.model.eval()
            logger.info(f"DocumentClassifier initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing DocumentClassifier: {e}")
            self.tokenizer = None
            self.model = None
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

    def preprocess_document(self, text: str):
        """Preprocess document text for the model."""
        if not self.tokenizer:
            raise RuntimeError("DocumentClassifier tokenizer not initialized.")
        
        # Truncate text to max length supported by the model
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        return encoding

    def classify_document(self, pdf_path: str) -> str:
        """Classify PDF document by extracting text and using a small transformer."""
        if not self.model or not self.tokenizer:
            logger.error("DocumentClassifier model or tokenizer not initialized.")
            return "unknown"
        
        try:
            # Extract text from PDF
            text = self.extract_text(pdf_path)
            if not text:
                logger.warning("No text extracted from PDF")
                return "unknown"

            # Preprocess and classify
            encoding = self.preprocess_document(text)
            with torch.no_grad():
                outputs = self.model(**encoding)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class_idx = torch.argmax(predictions, dim=1).item()
            
            classes = ["purchase_order", "invoice", "goods_received_note"]
            if 0 <= predicted_class_idx < len(classes):
                return classes[predicted_class_idx]
            else:
                logger.warning(f"Predicted class index {predicted_class_idx} is out of bounds")
                return "unknown"
                
        except Exception as e:
            logger.error(f"Error classifying document: {str(e)}")
            return "unknown"