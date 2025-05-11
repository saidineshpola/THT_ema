# app/reconciliation.py
import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger("document_management.reconciliation")
PROCESSED_DATA_DIR = "processed_data" # Should match DocumentProcessor

def load_processed_documents():
    """Loads all processed POs, Invoices, and GRNs from the JSON files."""
    docs = {"purchase_orders": [], "invoices": [], "goods_received": []}
    for doc_type_folder in docs.keys():
        folder_path = os.path.join(PROCESSED_DATA_DIR, doc_type_folder)
        if not os.path.exists(folder_path):
            logger.warning(f"Processed data folder not found for {doc_type_folder}: {folder_path}")
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Ensure essential keys are present, especially for reconciliation
                        if "extracted_data" in data and isinstance(data["extracted_data"], dict):
                            docs[doc_type_folder].append(data)
                        else:
                            logger.warning(f"Skipping {file_path}: 'extracted_data' missing or not a dict.")
                except json.JSONDecodeError:
                    logger.error(f"Error decoding JSON from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading processed document {file_path}: {e}")
    return docs

def run_reconciliation():
    """
    Performs a simple reconciliation based on PO numbers.
    Identifies matched sets, and discrepancies.
    """
    logger.info("Starting reconciliation process...")
    processed_docs = load_processed_documents()
    
    purchase_orders = processed_docs["purchase_orders"]
    invoices = processed_docs["invoices"]
    grns = processed_docs["goods_received"]

    # Group documents by PO number
    # Ensure 'po_number' exists in extracted_data and is not None
    po_map = defaultdict(lambda: {"po": None, "invoices": [], "grns": []})

    for po_doc in purchase_orders:
        po_data = po_doc.get("extracted_data", {})
        po_number = po_data.get("po_number")
        if po_number: # Ensure po_number is not None or empty
            po_map[po_number]["po"] = po_doc
        else:
            logger.debug(f"Purchase order {po_doc.get('original_filename')} lacks a PO number in extracted data.")


    for inv_doc in invoices:
        inv_data = inv_doc.get("extracted_data", {})
        po_number = inv_data.get("po_number") # Invoices should ref a PO number
        if po_number:
            po_map[po_number]["invoices"].append(inv_doc)
        else:
            logger.debug(f"Invoice {inv_doc.get('original_filename')} lacks a PO number reference in extracted data.")


    for grn_doc in grns:
        grn_data = grn_doc.get("extracted_data", {})
        po_number = grn_data.get("po_number") # GRNs should ref a PO number
        if po_number:
            po_map[po_number]["grns"].append(grn_doc)
        else:
            logger.debug(f"GRN {grn_doc.get('original_filename')} lacks a PO number reference in extracted data.")

    results = {
        "fully_matched_sets": [], # PO + Invoice + GRN (can be one PO to many Invoices/GRNs)
        "partial_matches": [],   # Missing one element
        "orphaned_pos": [],
        "orphaned_invoices": [], # Invoices that don't map to a known PO
        "orphaned_grns": [],     # GRNs that don't map to a known PO
        "discrepancies": []      # e.g., amount mismatch
    }

    processed_po_numbers_from_invoices_grns = set()

    for po_number, docs in po_map.items():
        is_fully_matched = False
        if docs["po"] and docs["invoices"] and docs["grns"]:
            results["fully_matched_sets"].append({
                "po_number": po_number,
                "po_file": docs["po"]["original_filename"],
                "invoice_files": [inv["original_filename"] for inv in docs["invoices"]],
                "grn_files": [grn["original_filename"] for grn in docs["grns"]]
            })
            is_fully_matched = True
            
            # Simple Discrepancy Check (Example: PO total vs Sum of Invoice totals for that PO)
            po_total_str = docs["po"].get("extracted_data", {}).get("total_amount")
            if po_total_str:
                try:
                    po_total = float(str(po_total_str).replace(',',''))
                    sum_invoice_totals = 0
                    for inv in docs["invoices"]:
                        inv_total_str = inv.get("extracted_data", {}).get("total_amount")
                        if inv_total_str:
                            sum_invoice_totals += float(str(inv_total_str).replace(',',''))
                    
                    if abs(po_total - sum_invoice_totals) > 0.01: # Tolerance for float comparison
                        results["discrepancies"].append({
                            "po_number": po_number,
                            "po_total": po_total,
                            "sum_invoice_totals": sum_invoice_totals,
                            "type": "PO vs Invoice Amount Mismatch"
                        })
                except ValueError:
                    logger.warning(f"Could not convert amount to float for PO {po_number} during discrepancy check.")


        if not is_fully_matched and docs["po"]: # PO exists but not full match
            results["partial_matches"].append({
                "po_number": po_number,
                "has_po": bool(docs["po"]),
                "has_invoice": bool(docs["invoices"]),
                "has_grn": bool(docs["grns"]),
                "po_file": docs["po"]["original_filename"],
                "invoice_files": [inv["original_filename"] for inv in docs["invoices"]],
                "grn_files": [grn["original_filename"] for grn in docs["grns"]]
            })
        
        if docs["po"] and not docs["invoices"] and not docs["grns"]:
             results["orphaned_pos"].append({
                "po_number": po_number,
                "po_file": docs["po"]["original_filename"]
            })
        
        processed_po_numbers_from_invoices_grns.add(po_number)


    # Check for orphaned invoices and GRNs (those whose PO number wasn't in the PO list)
    all_po_numbers_from_pos = {po_doc.get("extracted_data", {}).get("po_number") for po_doc in purchase_orders if po_doc.get("extracted_data", {}).get("po_number")}
    
    for inv_doc in invoices:
        po_number_ref = inv_doc.get("extracted_data", {}).get("po_number")
        if po_number_ref and po_number_ref not in all_po_numbers_from_pos:
            results["orphaned_invoices"].append({
                "invoice_file": inv_doc["original_filename"],
                "referenced_po_number": po_number_ref,
                "invoice_data": inv_doc.get("extracted_data")
            })
        elif not po_number_ref: # Invoice has no PO number at all
             results["orphaned_invoices"].append({
                "invoice_file": inv_doc["original_filename"],
                "referenced_po_number": "MISSING",
                "invoice_data": inv_doc.get("extracted_data")
            })


    for grn_doc in grns:
        po_number_ref = grn_doc.get("extracted_data", {}).get("po_number")
        if po_number_ref and po_number_ref not in all_po_numbers_from_pos:
            results["orphaned_grns"].append({
                "grn_file": grn_doc["original_filename"],
                "referenced_po_number": po_number_ref,
                "grn_data": grn_doc.get("extracted_data")
            })
        elif not po_number_ref: # GRN has no PO number at all
            results["orphaned_grns"].append({
                "grn_file": grn_doc["original_filename"],
                "referenced_po_number": "MISSING",
                "grn_data": grn_doc.get("extracted_data")
            })

    logger.info(f"Reconciliation complete. Found {len(results['fully_matched_sets'])} full matches, {len(results['discrepancies'])} discrepancies.")
    return results