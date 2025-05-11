from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import StringType, StructType, StructField
import logging
from pathlib import Path
import fitz  # PyMuPDF
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spark_document_processor")

class SparkDocumentProcessor:
    def __init__(self, input_path: str = "input_documents/"):
        self.spark = SparkSession.builder \
            .appName("DocumentProcessor") \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
        self.input_path = input_path
        self.model = None
        self.tokenizer = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the document classifier model"""
        try:
            model_name = "microsoft/layoutlmv2-base-uncased"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            # Broadcast the model to all executors
            self.broadcast_model = self.spark.sparkContext.broadcast(self.model)
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = " ".join(page.get_text() for page in doc)
            return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""

    def create_document_dataframe(self):
        """Create initial dataframe with document paths"""
        # Define schema for the document dataframe
        schema = StructType([
            StructField("filepath", StringType(), True),
            StructField("filename", StringType(), True)
        ])

        # List all PDF files in input directory
        pdf_files = list(Path(self.input_path).glob("*.pdf"))
        data = [(str(pdf), pdf.name) for pdf in pdf_files]
        
        return self.spark.createDataFrame(data, schema)

    def process_documents(self):
        """Main processing pipeline"""
        # Register UDFs
        extract_text_udf = udf(self.extract_text_from_pdf, StringType())
        
        @udf(returnType=StringType())
        def classify_document_udf(text):
            try:
                inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
                outputs = self.broadcast_model.value(**inputs)
                predicted = torch.argmax(outputs.logits, dim=1)
                labels = ["PO", "Invoice", "GRN"]
                return labels[predicted.item()]
            except Exception as e:
                logger.error(f"Classification error: {e}")
                return "Unknown"

        try:
            # Create initial dataframe
            df = self.create_document_dataframe()
            
            # Extract text
            df = df.withColumn("text", extract_text_udf(col("filepath")))
            
            # Classify documents
            df = df.withColumn("doc_type", classify_document_udf(col("text")))
            
            # Process based on document type
            pos = df.filter(col("doc_type") == "PO")
            invoices = df.filter(col("doc_type") == "Invoice")
            grns = df.filter(col("doc_type") == "GRN")
            
            # Save processed data
            pos.write.parquet("processed_data/purchase_orders/")
            invoices.write.parquet("processed_data/invoices/")
            grns.write.parquet("processed_data/grns/")
            
            return {
                "total_processed": df.count(),
                "pos_count": pos.count(),
                "invoices_count": invoices.count(),
                "grns_count": grns.count()
            }
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            raise

    def reconcile_documents(self):
        """Reconcile POs, Invoices, and GRNs"""
        try:
            pos = self.spark.read.parquet("processed_data/purchase_orders/")
            invoices = self.spark.read.parquet("processed_data/invoices/")
            grns = self.spark.read.parquet("processed_data/grns/")
            
            # Perform reconciliation logic here
            # Example: Join POs with Invoices on common fields
            reconciled = pos.join(invoices, ["order_id"], "left")
            
            return reconciled
        except Exception as e:
            logger.error(f"Reconciliation error: {e}")
            raise

if __name__ == "__main__":
    # Usage example
    processor = SparkDocumentProcessor()
    stats = processor.process_documents()
    print(f"Processing complete: {stats}")