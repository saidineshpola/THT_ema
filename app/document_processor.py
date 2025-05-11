# app/document_processor.py

import os
import json
import logging
import random
import csv
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timedelta
import shutil

import pandas as pd
import numpy as np
from PIL import Image
import pdf2image # For image conversion in classification step

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from faker import Faker
from tqdm import tqdm

# Import ML models and extraction utilities
from app.ml_models import DocumentClassifier
from app.document_extraction import AdvancedDataExtractor, get_extractor, Extractor # Import the factory and protocol

logger = logging.getLogger("document_management.processor")

# Configuration
SIMULATED_DATA_LAKE_DIR = Path("simulated_data_lake")
INPUT_DOCS_DIR = SIMULATED_DATA_LAKE_DIR / "input_documents"
PROCESSED_DATA_DIR = SIMULATED_DATA_LAKE_DIR / "processed_data" # For extracted CSVs
GENERATED_PDF_OUTPUT_DIR = SIMULATED_DATA_LAKE_DIR / "generated_sample_pdfs" # Where generated PDFs are saved
ARCHIVE_DIR = SIMULATED_DATA_LAKE_DIR / "archived_documents"


class DocumentProcessor:
    def __init__(self,
                 extractor_type: str = "pytesseract", # "pytesseract" or "advanced_model"
                 doc_classifier_model_name: str = "prajjwal1/bert-tiny"):
        self.fake = Faker()
        
        # Directories
        self.input_docs_dir = INPUT_DOCS_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.generated_pdf_output_dir = GENERATED_PDF_OUTPUT_DIR
        self.archive_dir = ARCHIVE_DIR
        
        self._setup_directories()

        self.stats = {
            "generated_pdfs": {"po": 0, "invoice": 0, "grn": 0},
            "processed_files": 0,
            "classified_docs": {"purchase_order": 0, "invoice": 0, "goods_received_note": 0, "unknown": 0},
            "extracted_docs": 0,
            "extraction_errors": 0,
            "reconciliation_errors": 0
        }
        
        try:
            self.classifier = DocumentClassifier(model_name=doc_classifier_model_name)
        except RuntimeError as e:
            logger.error(f"Failed to initialize DocumentClassifier: {e}. Classification will be skipped/defaulted.")
            raise f' {e} - Document classification will be skipped or defaulted.'
            # self.classifier = None # Allow graceful degradation if model fails to load

        # Configure your extractor. For Docling, you'd pass an API key.
        # For demo, you can switch this via environment variable or config later.
        # EXAMPLE_API_KEY = os.getenv("DOCLING_API_KEY") # Or from a config file
        self.extractor: Extractor = get_extractor(extractor_type=extractor_type) #, api_key=EXAMPLE_API_KEY)
        
        logger.info(f"DocumentProcessor initialized. Input: {self.input_docs_dir}, Output: {self.processed_data_dir}, Extractor: {extractor_type}")

    def _setup_directories(self):
        self.input_docs_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.generated_pdf_output_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Required directories ensured/created.")

    def _save_extracted_data_to_csv(self, data: Dict, doc_type: str):
        """Appends extracted data to a CSV file specific to the document type."""
        if not data or "error" in data:
            logger.warning(f"Skipping save for data with error or empty: {data}")
            return

        # Normalize doc_type to match CSV filenames (e.g., goods_received_note -> grns.csv)
        filename_map = {
            "purchase_order": "processed_purchase_orders.csv",
            "invoice": "processed_invoices.csv",
            "goods_received_note": "processed_grns.csv"
        }
        csv_filename = filename_map.get(doc_type)
        if not csv_filename:
            logger.error(f"Unknown document type for CSV saving: {doc_type}")
            return

        filepath = self.processed_data_dir / csv_filename
        
        # Basic schema - refine based on actual extracted fields
        default_fieldnames = {
            "purchase_order": ['po_number', 'issue_date', 'vendor_name', 'items_json', 'currency', 'total_amount', 'status', 'original_filename', 'extraction_method'],
            "invoice": ['invoice_number', 'invoice_date', 'vendor_name', 'po_reference', 'currency', 'amount_paid', 'payment_status', 'original_filename', 'extraction_method'],
            "goods_received_note": ['grn_number', 'received_date', 'warehouse_id', 'po_reference', 'received_qty', 'receiving_status', 'original_filename', 'extraction_method']
        }
        
        # Use fieldnames from data if more comprehensive, else default.
        # Ensure all default fields are present even if None
        fieldnames = default_fieldnames.get(doc_type, list(data.keys()))
        row_to_write = {key: data.get(key) for key in fieldnames}


        try:
            file_exists = filepath.exists()
            with open(filepath, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists or os.path.getsize(filepath) == 0:
                    writer.writeheader()
                writer.writerow(row_to_write)
            logger.info(f"Saved extracted data for {doc_type} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving extracted data to CSV {filepath}: {e}")
            self.stats["extraction_errors"] += 1
            
    def generate_synthetic_document_pdfs(self, num_pos: int = 5, num_invoices: int = 4, num_grns: int = 3, clear_output_dir: bool = True):
        """
        Generates sample PO, Invoice, and GRN data, saves them as CSVs (for reference)
        AND as PDF documents in the `generated_pdf_output_dir`.
        These PDFs can then be copied to `input_docs_dir` for the demo.
        """
        logger.info(f"Generating synthetic data: {num_pos} POs, {num_invoices} Invoices, {num_grns} GRNs.")

        if clear_output_dir:
            for f in self.generated_pdf_output_dir.glob("*.pdf"):
                try:
                    f.unlink()
                except OSError as e:
                    logger.warning(f"Could not delete old synthetic PDF {f}: {e}")
            logger.info(f"Cleared previous PDFs from {self.generated_pdf_output_dir}")

        # Generate POs
        pos_data = []
        for i in range(num_pos):
            po = {
                'po_number': f'PO{self.fake.unique.random_number(digits=5)}',
                'issue_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'vendor_name': self.fake.company(),
                'items': json.dumps([{
                    'item': self.fake.word().capitalize(),
                    'description': self.fake.sentence(nb_words=5),
                    'qty': random.randint(1, 10),
                    'unit_price': round(random.uniform(10, 200), 2)
                } for _ in range(random.randint(1,3))]),
                'currency': random.choice(['USD', 'EUR', 'GBP']),
                'total_amount': round(random.uniform(100, 5000), 2), # Recalculate if items are used
                'status': random.choice(['approved', 'pending'])
            }
            pos_data.append(po)
            pdf_path = self._generate_pdf_po(po, f"purchase_order_{po['po_number']}.pdf", self.generated_pdf_output_dir)
            if pdf_path: self.stats["generated_pdfs"]["po"] += 1

        # Generate Invoices
        invoices_data = []
        for i in range(num_invoices):
            po_ref = ""
            if pos_data and (i < len(pos_data) or random.choice([True, False])): # some match, some don't
                 po_ref = random.choice(pos_data)['po_number'] if random.random() > 0.2 else f'PO{self.fake.unique.random_number(digits=5)}' # Chance of unmatched
            else:
                po_ref = f'PO{self.fake.unique.random_number(digits=5)}' # Unmatched PO
            
            invoice = {
                'invoice_number': f'INV{self.fake.unique.random_number(digits=6)}',
                'invoice_date': (datetime.now() - timedelta(days=random.randint(1, 15))).strftime('%Y-%m-%d'),
                'vendor_name': self.fake.company() if random.random() > 0.5 else (random.choice(pos_data)['vendor_name'] if pos_data else self.fake.company()), # Sometimes same vendor
                'po_reference': po_ref,
                'currency': random.choice(['USD', 'EUR', 'GBP']),
                'amount_paid': round(random.uniform(100, 5000), 2),
                'payment_status': random.choice(['paid', 'pending', 'overdue'])
            }
            invoices_data.append(invoice)
            pdf_path = self._generate_pdf_invoice(invoice, f"invoice_{invoice['invoice_number']}.pdf", self.generated_pdf_output_dir)
            if pdf_path: self.stats["generated_pdfs"]["invoice"] += 1

        # Generate GRNs
        grns_data = []
        for i in range(num_grns):
            po_ref = ""
            if pos_data and (i < len(pos_data) or random.choice([True, False])):
                 po_ref = random.choice(pos_data)['po_number'] if random.random() > 0.3 else f'PO{self.fake.unique.random_number(digits=5)}'
            else:
                po_ref = f'PO{self.fake.unique.random_number(digits=5)}'

            grn = {
                'grn_number': f'GRN{self.fake.unique.random_number(digits=4)}',
                'received_date': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
                'warehouse_id': f'WH{random.randint(1, 3):02d}',
                'po_reference': po_ref,
                'received_qty': random.randint(5, 50),
                'receiving_status': random.choice(['complete', 'partial', 'damaged'])
            }
            grns_data.append(grn)
            pdf_path = self._generate_pdf_grn(grn, f"grn_{grn['grn_number']}.pdf", self.generated_pdf_output_dir)
            if pdf_path: self.stats["generated_pdfs"]["grn"] += 1
        
        logger.info(f"Synthetic PDF generation complete. Check {self.generated_pdf_output_dir}")
        return {
            "generated_pos_pdfs": self.stats["generated_pdfs"]["po"],
            "generated_invoice_pdfs": self.stats["generated_pdfs"]["invoice"],
            "generated_grn_pdfs": self.stats["generated_pdfs"]["grn"],
        }

    def _generate_pdf_po(self, po_data: Dict, filename: str, output_dir: Path) -> Optional[str]:
        filepath = str(output_dir / filename)
        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [Paragraph("<b>PURCHASE ORDER</b>", styles['h1']), Paragraph(f"PO Number: {po_data['po_number']}", styles['h2'])]
            elements.append(Paragraph(f"Date: {po_data['issue_date']}", styles['Normal']))
            elements.append(Paragraph(f"Vendor: {po_data['vendor_name']}", styles['Normal']))
            
            items = json.loads(po_data['items']) # Items is a JSON string
            item_data = [["Item", "Description", "Qty", "Unit Price", "Total"]]
            calculated_total = 0
            for item in items:
                qty = item.get('qty', 0)
                price = item.get('unit_price', 0)
                total = qty * price
                calculated_total += total
                item_data.append([item['item'], item['description'], str(qty), f"{po_data['currency']} {price:.2f}", f"{po_data['currency']} {total:.2f}"])
            
            # Use calculated total if items are well-defined, else use provided total_amount
            final_total = calculated_total if items else po_data.get('total_amount', 0.0)

            item_table = Table(item_data, colWidths=[doc.width/5.0]*5)
            item_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black)]))
            elements.append(item_table)
            elements.append(Paragraph(f"<b>Total Amount: {po_data['currency']} {float(final_total):.2f}</b>", styles['h3']))
            elements.append(Paragraph(f"Status: {po_data['status']}", styles['Normal']))
            doc.build(elements)
            logger.debug(f"Generated PO PDF: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating PO PDF {filename}: {e}")
            return None
            
    def _generate_pdf_invoice(self, invoice_data: Dict, filename: str, output_dir: Path) -> Optional[str]:
        filepath = str(output_dir / filename)
        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [Paragraph("<b>INVOICE</b>", styles['h1']), Paragraph(f"Invoice No: {invoice_data['invoice_number']}", styles['h2'])]
            elements.append(Paragraph(f"Date: {invoice_data['invoice_date']}", styles['Normal']))
            elements.append(Paragraph(f"Vendor: {invoice_data['vendor_name']}", styles['Normal']))
            elements.append(Paragraph(f"PO Reference: {invoice_data['po_reference']}", styles['Normal']))
            elements.append(Paragraph(f"<b>Amount: {invoice_data['currency']} {float(invoice_data['amount_paid']):.2f}</b>", styles['h3']))
            elements.append(Paragraph(f"Payment Status: {invoice_data['payment_status']}", styles['Normal']))
            doc.build(elements)
            logger.debug(f"Generated Invoice PDF: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating Invoice PDF {filename}: {e}")
            return None

    def _generate_pdf_grn(self, grn_data: Dict, filename: str, output_dir: Path) -> Optional[str]:
        filepath = str(output_dir / filename)
        try:
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = [Paragraph("<b>GOODS RECEIVED NOTE</b>", styles['h1']), Paragraph(f"GRN No: {grn_data['grn_number']}", styles['h2'])]
            elements.append(Paragraph(f"Received Date: {grn_data['received_date']}", styles['Normal']))
            elements.append(Paragraph(f"PO Reference: {grn_data['po_reference']}", styles['Normal']))
            elements.append(Paragraph(f"Warehouse ID: {grn_data['warehouse_id']}", styles['Normal']))
            elements.append(Paragraph(f"Quantity Received: {grn_data['received_qty']}", styles['Normal']))
            elements.append(Paragraph(f"Status: {grn_data['receiving_status']}", styles['Normal']))
            doc.build(elements)
            logger.debug(f"Generated GRN PDF: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error generating GRN PDF {filename}: {e}")
            return None

    def process_documents_from_data_lake(self) -> Dict:
        """
        Processes all PDF documents found in the `input_docs_dir`.
        1. Classifies the document.
        2. Extracts data using the configured extractor.
        3. Saves extracted data to CSVs in `processed_data_dir`.
        4. Archives processed PDFs.
        """
        logger.info(f"Starting document processing from data lake: {self.input_docs_dir}")
        processed_files_count = 0
        pdf_files = list(self.input_docs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.info("No PDF files found in input directory.")
            return {"message": "No PDF files found to process.", "processed_count": 0}

        for pdf_path in tqdm(pdf_files):
            logger.info(f"Processing file: {pdf_path.name}")
            doc_type = "unknown"
            extracted_data = {}

            # 1. Classify Document
            if self.classifier and self.classifier.model: # Check if classifier is available
                try:
                    # Classifier expects a PIL Image. Take the first page for classification.
                    doc_type = self.classifier.classify_document(pdf_path)
                    logger.info(f"File {pdf_path.name} classified as: {doc_type}")
                    self.stats["classified_docs"][doc_type] = self.stats["classified_docs"].get(doc_type, 0) + 1

                except Exception as e:
                    logger.error(f"Error during classification of {pdf_path.name}: {e}")
                    self.stats["classified_docs"]["unknown"] +=1 # Count as unknown if classification fails
            else:
                logger.warning(f"Document classifier not available. Skipping ML classification for {pdf_path.name}.")
                # Fallback: try to guess from filename or use a default "unknown"
                if "invoice" in pdf_path.name.lower(): doc_type = "invoice"
                elif "po" in pdf_path.name.lower() or "purchase_order" in pdf_path.name.lower(): doc_type = "purchase_order"
                elif "grn" in pdf_path.name.lower() or "goods_received" in pdf_path.name.lower(): doc_type = "goods_received_note"
                else: doc_type = "unknown"
                logger.info(f"File {pdf_path.name} heuristically typed as: {doc_type}")
                self.stats["classified_docs"][doc_type] = self.stats["classified_docs"].get(doc_type, 0) + 1


            # 2. Extract Data
            if doc_type != "unknown":
                try:
                    _, extracted_data = self.extractor.extract_from_pdf_path(str(pdf_path), doc_type)
                    if "error" not in extracted_data and extracted_data: # Check if data is not an error dict and not empty
                        logger.info(f"Successfully extracted data from {pdf_path.name} using {self.extractor.__class__.__name__}.")
                        extracted_data["original_filename"] = pdf_path.name
                        # Add extraction method to data for clarity if not already present
                        if isinstance(getattr(self.extractor, 'model', None), AdvancedDataExtractor): # crude check for advanced
                             extracted_data["extraction_method"] = "AdvancedModelExtractor"
                        else:
                             extracted_data["extraction_method"] = "PyTesseractExtractor"
                        self.stats["extracted_docs"] += 1
                    else:
                        logger.warning(f"Extraction failed or returned empty for {pdf_path.name}: {extracted_data.get('error', 'Empty data')}")
                        self.stats["extraction_errors"] += 1
                except Exception as e:
                    logger.error(f"Error during data extraction from {pdf_path.name}: {e}")
                    self.stats["extraction_errors"] += 1
                    extracted_data = {"error": str(e), "original_filename": pdf_path.name}
            else:
                logger.warning(f"Skipping data extraction for {pdf_path.name} as it was classified as 'unknown'.")

            # 3. Save Extracted Data (if successful and not 'unknown')
            if doc_type != "unknown" and "error" not in extracted_data and extracted_data:
                self._save_extracted_data_to_csv(extracted_data, doc_type)

            # 4. Archive Processed PDF
            try:
                archive_subfolder = self.archive_dir / doc_type
                archive_subfolder.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pdf_path), str(archive_subfolder / pdf_path.name))
                logger.info(f"Archived {pdf_path.name} to {archive_subfolder}")
            except Exception as e:
                logger.error(f"Error archiving file {pdf_path.name}: {e}")
            
            processed_files_count += 1
            self.stats["processed_files"] = processed_files_count
        
        logger.info(f"Finished processing from data lake. Processed {processed_files_count} files.")
        return {
            "message": f"Processed {processed_files_count} files from the data lake.",
            "summary_stats": self.get_processing_stats()
        }

    def reconcile_processed_documents(self) -> Dict:
        """Reconcile POs, Invoices and GRNs from the processed CSV files."""
        logger.info("Starting reconciliation of processed documents.")
        try:
            # Define paths to processed CSVs
            po_csv = self.processed_data_dir / 'processed_purchase_orders.csv'
            inv_csv = self.processed_data_dir / 'processed_invoices.csv'
            grn_csv = self.processed_data_dir / 'processed_grns.csv'

            # Load data, handling missing files gracefully
            pos_df = pd.read_csv(po_csv) if po_csv.exists() else pd.DataFrame()
            invoices_df = pd.read_csv(inv_csv) if inv_csv.exists() else pd.DataFrame()
            grns_df = pd.read_csv(grn_csv) if grn_csv.exists() else pd.DataFrame()

            if pos_df.empty and invoices_df.empty and grns_df.empty:
                logger.warning("No processed data found for reconciliation.")
                return {"message": "No processed data to reconcile.", "analysis": {}}

            logger.info(f"Reconciling: {len(pos_df)} POs, {len(invoices_df)} Invoices, {len(grns_df)} GRNs")

            # Ensure key columns exist, even if empty (to prevent errors on empty DFs)
            # These are example columns, adjust based on your actual CSVs from extraction
            po_cols = ['po_number', 'total_amount', 'currency', 'issue_date', 'original_filename']
            inv_cols = ['invoice_number', 'po_reference', 'amount_paid', 'currency', 'invoice_date', 'original_filename']
            grn_cols = ['grn_number', 'po_reference', 'received_qty', 'received_date', 'original_filename']

            for col in po_cols:
                if col not in pos_df.columns: pos_df[col] = None
            for col in inv_cols:
                if col not in invoices_df.columns: invoices_df[col] = None
            for col in grn_cols:
                if col not in grns_df.columns: grns_df[col] = None
            
            # Convert amount fields to numeric, coercing errors
            if 'total_amount' in pos_df.columns:
                pos_df['total_amount'] = pd.to_numeric(pos_df['total_amount'], errors='coerce')
            if 'amount_paid' in invoices_df.columns:
                invoices_df['amount_paid'] = pd.to_numeric(invoices_df['amount_paid'], errors='coerce')


            # Perform merges - ensure 'po_reference' and 'po_number' are common keys
            # Merge POs with Invoices
            if not pos_df.empty and not invoices_df.empty:
                po_invoice_match = pd.merge(
                    pos_df, invoices_df,
                    left_on='po_number', right_on='po_reference',
                    how='outer', suffixes=('_po', '_inv')
                )
            elif not pos_df.empty:
                po_invoice_match = pos_df.copy()
                for col in inv_cols: po_invoice_match[col + '_inv'] = None # Add invoice columns as None
                po_invoice_match.rename(columns={c: c + '_po' for c in pos_df.columns if c not in inv_cols}, inplace=True)


            elif not invoices_df.empty:
                po_invoice_match = invoices_df.copy()
                for col in po_cols: po_invoice_match[col + '_po'] = None # Add PO columns as None
                po_invoice_match.rename(columns={c: c + '_inv' for c in invoices_df.columns if c not in po_cols}, inplace=True)
                # Need a common reference for GRN merge, use invoice's PO ref if available
                po_invoice_match['po_number_po'] = po_invoice_match['po_reference_inv']


            else: # Both empty
                po_invoice_match = pd.DataFrame()
            
            # Merge with GRNs
            # Use the PO number from the PO side of the merge (po_number_po)
            # or if PO was missing, use the po_reference from the invoice side (po_reference_inv)
            # This requires careful handling if po_invoice_match is built from only one side.

            # Determine the correct PO reference column for GRN merge
            # If 'po_number_po' exists and is not all NaN, use it. Otherwise, try 'po_reference_inv'.
            po_ref_col_for_grn_merge = 'po_number_po' # Default
            if 'po_number_po' not in po_invoice_match.columns or po_invoice_match['po_number_po'].isna().all():
                if 'po_reference_inv' in po_invoice_match.columns: # From invoice if PO was missing
                    po_ref_col_for_grn_merge = 'po_reference_inv'
                else: # No valid PO reference available from previous merge
                    po_ref_col_for_grn_merge = None


            if not grns_df.empty and not po_invoice_match.empty and po_ref_col_for_grn_merge:
                full_match = pd.merge(
                    po_invoice_match, grns_df,
                    left_on=po_ref_col_for_grn_merge, right_on='po_reference',
                    how='outer', suffixes=('', '_grn') # GRN columns get _grn if names clash
                )
                # Rename GRN specific columns if not already suffixed (e.g. if only GRNs exist)
                for col in grn_cols:
                    if col in full_match.columns and col + '_grn' not in full_match.columns and col not in po_invoice_match.columns :
                        full_match.rename(columns={col: col + '_grn'}, inplace=True)

            elif not grns_df.empty: # Only GRNs exist or po_invoice_match was empty
                full_match = grns_df.copy()
                # Add columns from POs and Invoices as None
                for col in po_cols: full_match[col + '_po'] = None
                for col in inv_cols: full_match[col + '_inv'] = None
                full_match.rename(columns={c: c + '_grn' for c in grns_df.columns}, inplace=True)
            else: # GRNs are empty, po_invoice_match is the current full_match
                full_match = po_invoice_match
                # Add GRN columns as None if they don't exist
                for col in grn_cols:
                    if col + '_grn' not in full_match.columns: full_match[col + '_grn'] = None


            # Basic checks (ensure columns exist before trying to access)
            full_match['has_po'] = full_match['po_number_po'].notna() if 'po_number_po' in full_match else False
            full_match['has_invoice'] = full_match['invoice_number_inv'].notna() if 'invoice_number_inv' in full_match else False
            full_match['has_grn'] = full_match['grn_number_grn'].notna() if 'grn_number_grn' in full_match else False # Adjust if grn_number column name is different

            # Amount match (only if both PO and Invoice amounts are present and numeric)
            if 'total_amount_po' in full_match and 'amount_paid_inv' in full_match:
                 # Ensure they are numeric before comparison
                full_match['total_amount_po_numeric'] = pd.to_numeric(full_match['total_amount_po'], errors='coerce')
                full_match['amount_paid_inv_numeric'] = pd.to_numeric(full_match['amount_paid_inv'], errors='coerce')
                full_match['amount_match'] = np.isclose(
                    full_match['total_amount_po_numeric'].fillna(0),
                    full_match['amount_paid_inv_numeric'].fillna(0),
                    rtol=0.01 # 1% tolerance
                )
                # If one is NaN (e.g. missing PO or Invoice), amount_match should be False or Undefined
                full_match.loc[full_match['total_amount_po_numeric'].isna() | full_match['amount_paid_inv_numeric'].isna(), 'amount_match'] = pd.NA
            else:
                full_match['amount_match'] = pd.NA


            # Save reconciliation results
            recon_output_dir = self.processed_data_dir / "reconciliation_reports"
            recon_output_dir.mkdir(exist_ok=True)
            recon_file = recon_output_dir / 'reconciliation_results_detail.csv'
            full_match.to_csv(recon_file, index=False)

            # Generate analytics
            analysis = {
                'total_pos_processed': len(pos_df),
                'total_invoices_processed': len(invoices_df),
                'total_grns_processed': len(grns_df),
                'unique_po_numbers_in_reconciliation': int(full_match['po_number_po'].nunique()) if 'po_number_po' in full_match else 0,
                'pos_with_any_invoice_match': int(full_match[full_match['has_po'] & full_match['has_invoice']]['po_number_po'].nunique()) if 'po_number_po' in full_match else 0,
                'pos_with_any_grn_match': int(full_match[full_match['has_po'] & full_match['has_grn']]['po_number_po'].nunique()) if 'po_number_po' in full_match else 0,
                'pos_fully_matched_inv_grn': int(full_match[full_match['has_po'] & full_match['has_invoice'] & full_match['has_grn']]['po_number_po'].nunique()) if 'po_number_po' in full_match else 0,
                'amount_mismatches_found (amongst matched PO-Inv)': int((full_match['amount_match'] == False).sum()) if 'amount_match' in full_match else 0, # Ensure amount_match is boolean
                'invoices_without_matching_po': int(full_match[full_match['has_invoice'] & ~full_match['has_po']]['invoice_number_inv'].nunique()) if 'invoice_number_inv' in full_match else 0,
                'grns_without_matching_po': int(full_match[full_match['has_grn'] & ~full_match['has_po']]['grn_number_grn'].nunique()) if 'grn_number_grn' in full_match else 0,
            }
            
            report_file = recon_output_dir / 'reconciliation_summary_report.json'
            with open(report_file, 'w') as f:
                json.dump(analysis, f, indent=4)
                
            logger.info(f"Reconciliation completed. Detailed results: {recon_file}, Summary: {report_file}")
            return {"message": "Reconciliation successful.", "analysis": analysis, "results_file": str(recon_file)}

        except Exception as e:
            logger.error(f"Error in reconciliation: {str(e)}")
            self.stats["reconciliation_errors"] += 1
            return {"error": str(e), "analysis": {}}

    def get_processing_stats(self) -> Dict:
        return self.stats

    def clear_simulated_data_lake_processed_outputs(self):
        """Clears processed CSVs and reconciliation reports for a fresh demo run."""
        logger.info("Clearing processed data outputs (CSVs, reconciliation reports)...")
        
        # Clear processed CSVs
        for csv_file in self.processed_data_dir.glob("*.csv"):
            try:
                csv_file.unlink()
                logger.debug(f"Deleted {csv_file}")
            except OSError as e:
                logger.warning(f"Could not delete {csv_file}: {e}")
        
        # Clear reconciliation reports
        recon_report_dir = self.processed_data_dir / "reconciliation_reports"
        if recon_report_dir.exists():
            for report_file in recon_report_dir.glob("*"): # Glob for files and potentially subdirs if any
                if report_file.is_file():
                    try:
                        report_file.unlink()
                        logger.debug(f"Deleted {report_file}")
                    except OSError as e:
                        logger.warning(f"Could not delete {report_file}: {e}")
            # Optionally remove the directory itself if empty, or keep it
            # if os.listdir(recon_report_dir) == []:
            #     recon_report_dir.rmdir()

        # Reset relevant stats
        self.stats["processed_files"] = 0
        self.stats["classified_docs"] = {"purchase_order": 0, "invoice": 0, "goods_received_note": 0, "unknown": 0}
        self.stats["extracted_docs"] = 0
        self.stats["extraction_errors"] = 0
        self.stats["reconciliation_errors"] = 0
        logger.info("Cleared processed outputs and reset relevant stats.")