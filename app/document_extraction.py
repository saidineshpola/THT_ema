
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple, Protocol, Any, Union
from pathlib import Path
import fitz  # PyMuPDF for text extraction
from datetime import datetime
import requests

logger = logging.getLogger("document_management.extraction")

class Extractor(Protocol):
    """Protocol defining the interface for document extractors."""
    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[bool, Dict]:
        """Extract data from PDF file based on document type."""
        ...

def get_extractor(extractor_type: str = "pymupdf", api_key: Optional[str] = None) -> Extractor:
    """Factory function to create the appropriate extractor."""
    if extractor_type == "advanced_model":
        return AdvancedDataExtractor(api_key=api_key)
    elif extractor_type == "vllm":
        return VLLMExtractor(api_key=api_key)
    else:
        return PyMuPDFExtractor()

class PyMuPDFExtractor:
    """Basic extractor using pymupdf and regex patterns."""
    
    def __init__(self):
        logger.info("Initializing PyMuPDFExtractor")
        
    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[bool, Dict]:
        """Extract data from PDF using regex patterns based on document type."""
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(pdf_path)
            if not text:
                return False, {"error": "No text extracted from PDF"}
            
            # Select extraction method based on document type
            if doc_type == "purchase_order":
                return True, self._extract_purchase_order_data(text, pdf_path)
            elif doc_type == "invoice":
                return True, self._extract_invoice_data(text, pdf_path)
            elif doc_type == "goods_received_note":
                return True, self._extract_grn_data(text, pdf_path)
            else:
                return False, {"error": f"Unsupported document type: {doc_type}"}
                
        except Exception as e:
            logger.error(f"Error extracting data from {pdf_path}: {str(e)}")
            return False, {"error": str(e)}
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return ""
    
    def _extract_purchase_order_data(self, text: str, pdf_path: str) -> Dict:
        """Extract PO data using regex patterns."""
        result = {}
        
        # Extract PO number
        po_match = re.search(r'PO\s*(?:Number|#|No\.?|:)?\s*[:.]?\s*(\w+[-\d]+)', text, re.IGNORECASE)
        if po_match:
            result["po_number"] = po_match.group(1)
        else:
            # Try alternative patterns
            po_match = re.search(r'Purchase\s*Order\s*(?:Number|#|No\.?|:)?\s*[:.]?\s*(\w+[-\d]+)', text, re.IGNORECASE)
            if po_match:
                result["po_number"] = po_match.group(1)
            else:
                # Look for PO followed by digits
                po_match = re.search(r'PO\s*(\d+)', text, re.IGNORECASE)
                if po_match:
                    result["po_number"] = f"PO{po_match.group(1)}"
                else:
                    result["po_number"] = None
        
        # Extract issue date
        date_patterns = [
            r'(?:Date|Issue\s*Date|PO\s*Date)[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:Date|Issue\s*Date|PO\s*Date)[:.]?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                # Attempt to standardize date format
                try:
                    # Try to determine date format and convert to YYYY-MM-DD
                    if re.match(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["issue_date"] = f"{date_parts[2]}-{int(date_parts[1]):02d}-{int(date_parts[0]):02d}"
                    elif re.match(r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["issue_date"] = f"{date_parts[0]}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
                    else:
                        result["issue_date"] = date_str
                except Exception:
                    result["issue_date"] = date_str
                break
        
        if "issue_date" not in result:
            result["issue_date"] = None
        
        # Extract vendor name
        vendor_match = re.search(r'(?:Vendor|Supplier)[:.]?\s*([A-Za-z0-9\s&,\.]+?)(?:\n|,|\.|$)', text, re.IGNORECASE)
        if vendor_match:
            result["vendor_name"] = vendor_match.group(1).strip()
        else:
            # Alternative pattern
            vendor_match = re.search(r'TO:?\s*([A-Za-z0-9\s&,\.]+?)(?:\n|,|\.|$)', text[:500], re.IGNORECASE)  # Look in first 500 chars
            if vendor_match:
                result["vendor_name"] = vendor_match.group(1).strip()
            else:
                result["vendor_name"] = None
        
        # Extract currency and total amount
        currency_match = re.search(r'(USD|EUR|GBP|JPY|\$|€|£|¥)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text)
        if currency_match:
            currency = currency_match.group(1)
            # Normalize currency symbols to codes
            currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
            result["currency"] = currency_map.get(currency, currency)
            
            # Extract and clean amount
            amount_str = currency_match.group(2).replace(',', '')
            result["total_amount"] = float(amount_str)
        else:
            # Try pattern for total amount
            amount_match = re.search(r'Total\s*(?:Amount)?[:.]?\s*(?:[A-Z]{3})?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                result["total_amount"] = float(amount_str)
                # Look for currency nearby
                currency_match = re.search(r'(USD|EUR|GBP|JPY|\$|€|£|¥)', text[max(0, amount_match.start()-10):amount_match.end()])
                if currency_match:
                    currency = currency_match.group(1)
                    currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
                    result["currency"] = currency_map.get(currency, currency)
                else:
                    result["currency"] = "USD"  # Default
            else:
                result["total_amount"] = None
                result["currency"] = None
        
        # Extract status
        status_match = re.search(r'Status[:.]?\s*(\w+)', text, re.IGNORECASE)
        if status_match:
            result["status"] = status_match.group(1).lower()
        else:
            # Look for common PO statuses
            for status in ["approved", "pending", "rejected", "completed"]:
                if re.search(rf'\b{status}\b', text, re.IGNORECASE):
                    result["status"] = status.lower()
                    break
            else:
                result["status"] = None
        
        # Extract items if available
        # This is simplified - real implementation would be more robust
        items_section = None
        section_matches = re.search(r'(?:item|description|qty|price).+?(?=total)', text, re.IGNORECASE | re.DOTALL)
        if section_matches:
            items_section = section_matches.group(0)
            # Process items only if section was found
            if items_section:
                # Simple implementation - would be enhanced in real scenario
                items = []
                # Match item lines: extract description, quantity, price
                item_matches = re.finditer(r'([A-Za-z\s]+?)\s+(\d+)\s+(\d+(?:\.\d{2})?)', items_section)
                for match in item_matches:
                    items.append({
                        "item": match.group(1).strip(),
                        "qty": int(match.group(2)),
                        "unit_price": float(match.group(3))
                    })
                if items:
                    result["items_json"] = json.dumps(items)
        
        return result
    
    def _extract_invoice_data(self, text: str, pdf_path: str) -> Dict:
        """Extract invoice data using regex patterns."""
        result = {}
        
        # Extract invoice number
        inv_match = re.search(r'(?:Invoice|INV)[:.\s#]*(\w+[-\d]+)', text, re.IGNORECASE)
        if inv_match:
            result["invoice_number"] = inv_match.group(1)
        else:
            # Alternative patterns
            inv_match = re.search(r'(?:Invoice|INV)\s*(?:Number|No|#)[:.]?\s*(\w+[-\d]+)', text, re.IGNORECASE)
            if inv_match:
                result["invoice_number"] = inv_match.group(1)
            else:
                result["invoice_number"] = None
        
        # Extract invoice date
        date_patterns = [
            r'(?:Invoice\s*Date|Date)[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:Invoice\s*Date|Date)[:.]?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                # Attempt to standardize date format
                try:
                    if re.match(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["invoice_date"] = f"{date_parts[2]}-{int(date_parts[1]):02d}-{int(date_parts[0]):02d}"
                    elif re.match(r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["invoice_date"] = f"{date_parts[0]}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
                    else:
                        result["invoice_date"] = date_str
                except Exception:
                    result["invoice_date"] = date_str
                break
        
        if "invoice_date" not in result:
            result["invoice_date"] = None
        
        # Extract vendor name
        vendor_match = re.search(r'(?:Vendor|Supplier|From)[:.]?\s*([A-Za-z0-9\s&,\.]+?)(?:\n|,|\.|$)', text, re.IGNORECASE)
        if vendor_match:
            result["vendor_name"] = vendor_match.group(1).strip()
        else:
            result["vendor_name"] = None
        
        # Extract PO reference
        po_ref_match = re.search(r'(?:PO\s*Ref|Reference|PO|Purchase\s*Order)[:.\s#]*(\w+[-\d]+)', text, re.IGNORECASE)
        if po_ref_match:
            result["po_reference"] = po_ref_match.group(1)
        else:
            result["po_reference"] = None
        
        # Extract currency and amount paid
        currency_match = re.search(r'(?:Amount|Total|Paid|Due)[:.]?\s*(USD|EUR|GBP|JPY|\$|€|£|¥)\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
        if currency_match:
            currency = currency_match.group(1)
            # Normalize currency symbols to codes
            currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
            result["currency"] = currency_map.get(currency, currency)
            
            # Extract and clean amount
            amount_str = currency_match.group(2).replace(',', '')
            result["amount_paid"] = float(amount_str)
        else:
            # Try alternative pattern
            amount_match = re.search(r'(?:Amount|Total|Paid|Due)[:.]?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                result["amount_paid"] = float(amount_str)
                # Look for currency nearby
                currency_match = re.search(r'(USD|EUR|GBP|JPY|\$|€|£|¥)', text[max(0, amount_match.start()-10):amount_match.end()])
                if currency_match:
                    currency = currency_match.group(1)
                    currency_map = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}
                    result["currency"] = currency_map.get(currency, currency)
                else:
                    result["currency"] = "USD"  # Default
            else:
                result["amount_paid"] = None
                result["currency"] = None
        
        # Extract payment status
        payment_status_match = re.search(r'(?:Payment\s*Status|Status)[:.]?\s*(\w+)', text, re.IGNORECASE)
        if payment_status_match:
            result["payment_status"] = payment_status_match.group(1).lower()
        else:
            # Look for common payment status terms
            for status in ["paid", "unpaid", "pending", "overdue"]:
                if re.search(rf'\b{status}\b', text, re.IGNORECASE):
                    result["payment_status"] = status.lower()
                    break
            else:
                result["payment_status"] = None
        
        return result
    
    def _extract_grn_data(self, text: str, pdf_path: str) -> Dict:
        """Extract goods received note data using regex patterns."""
        result = {}
        
        # Extract GRN number
        grn_match = re.search(r'(?:GRN|Goods\s*Received\s*Note)[:.\s#]*(\w+[-\d]+)', text, re.IGNORECASE)
        if grn_match:
            result["grn_number"] = grn_match.group(1)
        else:
            # Alternative pattern
            grn_match = re.search(r'(?:GRN|Goods\s*Received\s*Note)\s*(?:Number|No|#)[:.]?\s*(\w+[-\d]+)', text, re.IGNORECASE)
            if grn_match:
                result["grn_number"] = grn_match.group(1)
            else:
                result["grn_number"] = None
        
        # Extract received date
        date_patterns = [
            r'(?:Received\s*Date|Date\s*Received|Date)[:.]?\s*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})',
            r'(?:Received\s*Date|Date\s*Received|Date)[:.]?\s*(\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2})'
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, text, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                # Attempt to standardize date format
                try:
                    if re.match(r'\d{1,2}[-/\.]\d{1,2}[-/\.]\d{4}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["received_date"] = f"{date_parts[2]}-{int(date_parts[1]):02d}-{int(date_parts[0]):02d}"
                    elif re.match(r'\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}', date_str):
                        date_parts = re.split(r'[-/\.]', date_str)
                        result["received_date"] = f"{date_parts[0]}-{int(date_parts[1]):02d}-{int(date_parts[2]):02d}"
                    else:
                        result["received_date"] = date_str
                except Exception:
                    result["received_date"] = date_str
                break
        
        if "received_date" not in result:
            result["received_date"] = None
        
        # Extract warehouse ID
        warehouse_match = re.search(r'(?:Warehouse|WH)[:.\s#]*(\w+[-\d]+)', text, re.IGNORECASE)
        if warehouse_match:
            result["warehouse_id"] = warehouse_match.group(1)
        else:
            result["warehouse_id"] = None
        
        # Extract PO reference
        po_ref_match = re.search(r'(?:PO\s*Ref|Reference|PO|Purchase\s*Order)[:.\s#]*(\w+[-\d]+)', text, re.IGNORECASE)
        if po_ref_match:
            result["po_reference"] = po_ref_match.group(1)
        else:
            result["po_reference"] = None
        
        # Extract received quantity
        qty_match = re.search(r'(?:Quantity|Qty|Received\s*Qty)[:.]?\s*(\d+)', text, re.IGNORECASE)
        if qty_match:
            result["received_qty"] = int(qty_match.group(1))
        else:
            result["received_qty"] = None
        
        # Extract receiving status
        status_match = re.search(r'(?:Receiving\s*Status|Status)[:.]?\s*(\w+)', text, re.IGNORECASE)
        if status_match:
            result["receiving_status"] = status_match.group(1).lower()
        else:
            # Look for common GRN status terms
            for status in ["complete", "partial", "damaged", "pending"]:
                if re.search(rf'\b{status}\b', text, re.IGNORECASE):
                    result["receiving_status"] = status.lower()
                    break
            else:
                result["receiving_status"] = None
        
        return result


class AdvancedDataExtractor:
    """
    Extractor based on advanced OCR/ML, either cloud service or local model.
    This is a placeholder that would be replaced by a real implementation.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        logger.info("Initializing AdvancedDataExtractor")
        # This would load a model or init a client for a cloud service
        self.model = None  # Placeholder for ML model
    
    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[bool, Dict]:
        """Extract structured data using an advanced ML model."""
        # In a real implementation, this would use either a local ML model
        # or make API calls to a cloud service with the PDF
        
        logger.info(f"Using AdvancedDataExtractor for {pdf_path} of type {doc_type}")
        
        # For demo: fall back to basic extraction via regex
        # In real impl: would use proper ML/OCR/cloud API
        basic_extractor = PyMuPDFExtractor()
        success, data = basic_extractor.extract_from_pdf_path(pdf_path, doc_type)
        
        if success:
            # Add a flag indicating this used the "advanced" extractor
            data["extraction_method"] = "advanced_model"
        
        return success, data


class VLLMExtractor:
    """
    Extractor using a vision-language model for structured data extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-vision-preview"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "demo-api-key")
        self.model_name = model_name
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        logger.info(f"Initializing VLLMExtractor with model: {model_name}")
    
    def _get_schema_for_doc_type(self, doc_type: str) -> Dict:
        """Get the appropriate JSON schema based on document type."""
        schemas = {
            "purchase_order": {
                "po_number": "string",
                "issue_date": "string (YYYY-MM-DD)",
                "vendor_name": "string",
                "items_json": "array of items with description, quantity, unit_price",
                "currency": "string (USD, EUR, GBP, etc.)",
                "total_amount": "float",
                "status": "string (approved, pending, etc.)"
            },
            "invoice": {
                "invoice_number": "string",
                "invoice_date": "string (YYYY-MM-DD)",
                "vendor_name": "string",
                "po_reference": "string",
                "currency": "string (USD, EUR, GBP, etc.)",
                "amount_paid": "float",
                "payment_status": "string (paid, pending, overdue, etc.)"
            },
            "goods_received_note": {
                "grn_number": "string",
                "received_date": "string (YYYY-MM-DD)",
                "warehouse_id": "string",
                "po_reference": "string",
                "received_qty": "integer",
                "receiving_status": "string (complete, partial, damaged, etc.)"
            }
        }
        return schemas.get(doc_type, {})
    
    def _generate_prompt(self, doc_type: str) -> str:
        """Generate prompt for the vision model based on document type."""
        schema = self._get_schema_for_doc_type(doc_type)
        schema_str = json.dumps(schema, indent=2)
        
        base_prompt = f"""
        Extract the following information from this {doc_type.replace('_', ' ')} document.
        Return the data in JSON format according to this schema:
        
        {schema_str}
        
        Only extract what you can see in the document. For any fields you cannot find, use null.
        Format dates as YYYY-MM-DD if possible. Return ONLY the JSON object, with no additional text.
        """
        return base_prompt.strip()
    
    def _convert_pdf_to_base64(self, pdf_path: str) -> Optional[str]:
        """Convert PDF to base64 for API submission."""
        try:
            with open(pdf_path, "rb") as pdf_file:
                import base64
                return base64.b64encode(pdf_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting PDF to base64: {e}")
            return None
    
    def extract_from_pdf_path(self, pdf_path: str, doc_type: str) -> Tuple[bool, Dict]:
        """Extract data from PDF using vision-language model."""
        try:
            # In demo mode, use pymupdf fallback instead of actual API calls
            if self.api_key == "demo-api-key":
                logger.warning("Using pymupdf fallback in demo mode (no valid API key)")
                basic_extractor = PyMuPDFExtractor()
                success, data = basic_extractor.extract_from_pdf_path(pdf_path, doc_type)
                if success:
                    data["extraction_method"] = "vllm_simulated"
                return success, data
            
            # Real implementation would send the PDF to the vision model API
            prompt = self._generate_prompt(doc_type)
            base64_pdf = self._convert_pdf_to_base64(pdf_path)
            
            if not base64_pdf:
                return False, {"error": "Failed to convert PDF to base64"}
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:application/pdf;base64,{base64_pdf}"}}
                        ]
                    }
                ],
                "max_tokens": 1000
            }
            
            response = requests.post(self.endpoint, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                extracted_data = json.loads(content)
                extracted_data["extraction_method"] = "vllm"
                extracted_data["original_filename"] = os.path.basename(pdf_path)
                return True, extracted_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {content}")
                return False, {"error": "Failed to parse JSON response", "raw_response": content}
                
        except requests.RequestException as e:
            logger.error(f"API request error: {str(e)}")
            return False, {"error": f"API request error: {str(e)}"}
        except Exception as e:
            logger.error(f"Error extracting data with VLLM: {str(e)}")
            return False, {"error": str(e)}


# Example usage for testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with a sample PDF
    test_pdf = "path/to/test.pdf"
    if os.path.exists(test_pdf):
        extractor = get_extractor("pymupdf")
        success, data = extractor.extract_from_pdf_path(test_pdf, "invoice")
        print(f"Extraction success: {success}")
        print(f"Extracted data: {json.dumps(data, indent=2)}")