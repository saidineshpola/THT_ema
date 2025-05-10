import csv
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from pathlib import Path
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from faker import Faker
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='document_processing.log'
)

class DocumentProcessor:
    def __init__(self):
        self.fake = Faker()
        self.output_dir = Path("output_documents")
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def generate_sample_data(self):
        """Generate sample data for testing"""
        # Generate POs
        pos = []
        for i in range(10):
            po = {
                'po_number': f'PO{i+1:05d}',
                'issue_date': (datetime.now() - timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d'),
                'items': json.dumps([{'item': self.fake.word(), 'qty': random.randint(1, 100)} for _ in range(3)]),
                'currency': 'USD',
                'total_amount': round(random.uniform(1000, 10000), 2)
            }
            pos.append(po)

        # Generate Invoices
        invoices = []
        for i in range(8):
            invoice = {
                'invoice_number': f'INV{i+1:05d}',
                'invoice_date': (datetime.now() - timedelta(days=random.randint(1, 15))).strftime('%Y-%m-%d'),
                'po_reference': f'PO{random.randint(1,10):05d}',
                'currency': 'USD',
                'amount_paid': round(random.uniform(1000, 10000), 2)
            }
            invoices.append(invoice)

        # Generate GRNs
        grns = []
        for i in range(7):
            grn = {
                'grn_number': f'GRN{i+1:05d}',
                'received_date': (datetime.now() - timedelta(days=random.randint(1, 20))).strftime('%Y-%m-%d'),
                'po_reference': f'PO{random.randint(1,10):05d}',
                'received_qty': random.randint(1, 100)
            }
            grns.append(grn)

        # Save to CSV files
        self._save_to_csv('purchase_orders.csv', pos)
        self._save_to_csv('invoices.csv', invoices)
        self._save_to_csv('grns.csv', grns)

    def _save_to_csv(self, filename: str, data: List[Dict]):
        """Save data to CSV file"""
        if not data:
            return
        
        filepath = self.data_dir / filename
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    def validate_document_data(self, data: Dict) -> bool:
        """Validate required fields in document data"""
        required_fields = {
            'purchase_orders': ['po_number', 'issue_date', 'items', 'currency', 'total_amount'],
            'invoices': ['invoice_number', 'invoice_date', 'po_reference', 'currency', 'amount_paid'],
            'grns': ['grn_number', 'received_date', 'po_reference', 'received_qty']
        }
        
        doc_type = self._determine_document_type(data)
        if not doc_type:
            return False
            
        return all(field in data for field in required_fields[doc_type])

    def _determine_document_type(self, data: Dict) -> str:
        """Determine document type based on fields present"""
        if 'po_number' in data:
            return 'purchase_orders'
        elif 'invoice_number' in data:
            return 'invoices'
        elif 'grn_number' in data:
            return 'grns'
        return None

    def generate_pdf_po(self, po_data: Dict, filename: str):
            """Generate PDF for Purchase Order"""
            try:
                # Convert Path to string for ReportLab
                filepath = str(self.output_dir / filename)
                doc = SimpleDocTemplate(filepath, pagesize=letter)
                styles = getSampleStyleSheet()
                elements = []
                
                # Rest of the code remains the same
                elements.append(Paragraph(f"Purchase Order: {po_data['po_number']}", styles['Heading1']))
                elements.append(Paragraph(f"Date: {po_data['issue_date']}", styles['Normal']))
                
                items = json.loads(po_data['items'])
                table_data = [['Item', 'Quantity']]
                for item in items:
                    table_data.append([item['item'], str(item['qty'])])
                    
                table = Table(table_data)
                table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                elements.append(table)
                
                elements.append(Paragraph(f"Total Amount: {po_data['currency']} {po_data['total_amount']}", styles['Heading2']))
                
                doc.build(elements)
    
            except Exception as e:
                logging.error(f"Error generating PO PDF: {str(e)}")
                raise
    
    def generate_pdf_invoice(self, invoice_data: Dict, filename: str):
        """Generate PDF for Invoice"""
        try:
            filepath = str(self.output_dir / filename)
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph(f"Invoice: {invoice_data['invoice_number']}", styles['Heading1']))
            elements.append(Paragraph(f"Date: {invoice_data['invoice_date']}", styles['Normal']))
            elements.append(Paragraph(f"PO Reference: {invoice_data['po_reference']}", styles['Normal']))
            elements.append(Paragraph(f"Amount Paid: {invoice_data['currency']} {invoice_data['amount_paid']}", styles['Heading2']))
            
            doc.build(elements)

        except Exception as e:
            logging.error(f"Error generating Invoice PDF: {str(e)}")
            raise

    def generate_pdf_grn(self, grn_data: Dict, filename: str):
        """Generate PDF for Goods Received Note"""
        try:
            filepath = str(self.output_dir / filename)
            doc = SimpleDocTemplate(filepath, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            elements.append(Paragraph(f"Goods Received Note: {grn_data['grn_number']}", styles['Heading1']))
            elements.append(Paragraph(f"Date: {grn_data['received_date']}", styles['Normal']))
            elements.append(Paragraph(f"PO Reference: {grn_data['po_reference']}", styles['Normal']))
            elements.append(Paragraph(f"Quantity Received: {grn_data['received_qty']}", styles['Normal']))
            
            doc.build(elements)

        except Exception as e:
            logging.error(f"Error generating GRN PDF: {str(e)}")
            raise

    def reconcile_documents(self):
        """Reconcile POs, Invoices and GRNs"""
        try:
            pos = pd.read_csv(self.data_dir / 'purchase_orders.csv')
            invoices = pd.read_csv(self.data_dir / 'invoices.csv')
            grns = pd.read_csv(self.data_dir / 'grns.csv')

            # Merge documents on PO reference
            reconciled = pd.merge(
                pos, 
                invoices, 
                left_on='po_number', 
                right_on='po_reference', 
                how='outer'
            )
            reconciled = pd.merge(
                reconciled,
                grns,
                left_on='po_number',
                right_on='po_reference',
                how='outer'
            )

            # Generate reconciliation report
            self.generate_reconciliation_report(reconciled)

        except Exception as e:
            logging.error(f"Error in reconciliation: {str(e)}")
            raise

    def generate_reconciliation_report(self, data: pd.DataFrame):
        """Generate analytics report"""
        report = {
            'total_pos': len(data['po_number'].unique()),
            'total_invoices': len(data['invoice_number'].dropna().unique()),
            'total_grns': len(data['grn_number'].dropna().unique()),
            'unmatched_pos': len(data[data['invoice_number'].isna()]),
            'unmatched_invoices': len(data[data['po_number'].isna()]),
            'total_value': data['total_amount'].sum()
        }

        # Save report
        with open(self.output_dir / 'reconciliation_report.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=report.keys())
            writer.writeheader()
            writer.writerow(report)

    def process_documents(self, input_files: List[str]):
        """Main processing function"""
        try:
            # Generate sample data if files don't exist
            if not all((self.data_dir / file).exists() for file in input_files):
                logging.info("Generating sample data...")
                self.generate_sample_data()

            for file in input_files:
                doc_type = file.split('.')[0]
                logging.info(f"Processing {doc_type}")
                
                with open(self.data_dir / file) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if not self.validate_document_data(row):
                            logging.warning(f"Invalid data in row: {row}")
                            continue
                        # Fix: Map document types to their respective number fields
                        number_field_map = {
                            'purchase_orders': 'po_number',
                            'invoices': 'invoice_number',
                            'grns': 'grn_number'
                        }
                        filename = f"{doc_type}_{row[number_field_map[doc_type]]}.pdf"
                        if doc_type == "purchase_orders":
                            self.generate_pdf_po(row, filename)
                        elif doc_type == "invoices":
                            self.generate_pdf_invoice(row, filename)
                        elif doc_type == "grns":
                            self.generate_pdf_grn(row, filename)

            # Reconcile after processing
            self.reconcile_documents()

        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DocumentProcessor()
    input_files = ['purchase_orders.csv', 'invoices.csv', 'grns.csv']
    processor.process_documents(input_files)