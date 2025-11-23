"""
Table Extractor for PDFs using PyMuPDF.
Extracts tables from PDFs and converts them to Markdown format for better structure preservation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from app.core.logger import rag_logger

# Check if PyMuPDF is available
TABLE_EXTRACTION_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    TABLE_EXTRACTION_AVAILABLE = True
except ImportError:
    pass


def extract_tables_from_pdf(
    pdf_path: Path,
    silent: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using PyMuPDF and convert to Markdown format.
    
    Args:
        pdf_path: Path to the PDF file
        silent: If True, don't log progress messages
        
    Returns:
        List of table dictionaries with keys:
        - page: Page number (1-indexed)
        - table_id: Unique table identifier
        - markdown: Table in Markdown format
        - row_count: Number of rows in table
        - col_count: Number of columns in table
    """
    if not TABLE_EXTRACTION_AVAILABLE:
        if not silent:
            rag_logger.debug("PyMuPDF not available - table extraction skipped")
        return []
    
    tables_data = []
    
    try:
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        
        if not silent:
            rag_logger.info(f"📊 Extracting tables from PDF: {pdf_path.name} ({total_pages} pages)")
        
        # Log progress for large PDFs
        log_interval = 50 if total_pages > 500 else 20 if total_pages > 100 else 10
        if total_pages > 100:
            rag_logger.info(f"📊 Starting table extraction from {total_pages} pages...")
        
        for page_num in range(total_pages):
            page = doc[page_num]
            
            # Extract tables from this page
            try:
                # PyMuPDF table extraction
                tables = page.find_tables()
                
                for table_idx, table in enumerate(tables):
                    try:
                        # Convert table to pandas DataFrame
                        df = table.to_pandas()
                        
                        # Convert DataFrame to Markdown format
                        markdown_table = _dataframe_to_markdown(df)
                        
                        table_id = f"table_{page_num + 1}_{table_idx + 1}"
                        
                        tables_data.append({
                            'page': page_num + 1,
                            'table_id': table_id,
                            'markdown': markdown_table,
                            'row_count': len(df),
                            'col_count': len(df.columns),
                            'raw_data': df.to_dict('records')  # Store raw data for reference
                        })
                        
                    except Exception as e:
                        if not silent:
                            rag_logger.debug(f"Error converting table on page {page_num + 1}: {e}")
                        continue
                
                # Log progress for large files
                if (page_num + 1) % log_interval == 0:
                    rag_logger.info(f"   📄 Page {page_num + 1}/{total_pages} processed, {len(tables_data)} tables extracted so far...")
                    
            except Exception as e:
                if not silent:
                    rag_logger.debug(f"Error extracting tables from page {page_num + 1}: {e}")
                continue
        
        doc.close()
        
        # Log summary
        if tables_data:
            pages_with_tables = len(set(t['page'] for t in tables_data))
            total_rows = sum(t['row_count'] for t in tables_data)
            total_cols = sum(t['col_count'] for t in tables_data)
            
            if not silent:
                rag_logger.info(f"📊 TABLE EXTRACTION SUMMARY for {pdf_path.name}:")
                rag_logger.info(f"   - Total tables extracted: {len(tables_data)}")
                rag_logger.info(f"   - Pages with tables: {pages_with_tables}/{total_pages}")
                rag_logger.info(f"   - Total table rows: {total_rows:,}")
                rag_logger.info(f"   - Total table columns: {total_cols:,}")
        
    except Exception as e:
        if not silent:
            rag_logger.warning(f"Table extraction failed for {pdf_path.name}: {e}")
        return []
    
    return tables_data


def _dataframe_to_markdown(df) -> str:
    """
    Convert pandas DataFrame to Markdown table format.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        Markdown formatted table string
    """
    # Replace NaN with empty strings
    df = df.fillna('')
    
    # Convert all values to strings
    df = df.astype(str)
    
    # Get headers
    headers = list(df.columns)
    
    # Create markdown table
    markdown_lines = []
    
    # Header row
    markdown_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    
    # Separator row
    markdown_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    # Data rows
    for _, row in df.iterrows():
        markdown_lines.append("| " + " | ".join(str(val) for val in row.values) + " |")
    
    return "\n".join(markdown_lines)


def get_tables_for_page(
    tables_data: List[Dict[str, Any]],
    page_num: int
) -> List[Dict[str, Any]]:
    """
    Get all tables for a specific page.
    
    Args:
        tables_data: List of table dictionaries
        page_num: Page number (1-indexed)
        
    Returns:
        List of tables for the specified page
    """
    return [t for t in tables_data if t['page'] == page_num]


def format_table_reference(table_id: str, markdown: str) -> str:
    """
    Format a table with its reference for insertion into document content.
    
    Args:
        table_id: Table identifier
        markdown: Markdown formatted table
        
    Returns:
        Formatted string with table reference and markdown
    """
    return f"\n\n[Table: {table_id}]\n{markdown}\n"

