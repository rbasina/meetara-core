"""
Response formatter for detecting and enhancing structured content (tables, lists, etc.) in LLM responses.
"""

import re
from typing import Dict, List, Optional, Any


class ResponseFormatter:
    """Formatter for enhancing LLM responses with structured content detection."""
    
    @staticmethod
    def detect_tables_in_response(response: str) -> List[Dict[str, Any]]:
        """
        Detect Markdown tables in response text.
        
        Args:
            response: Response text to analyze
            
        Returns:
            List of table dictionaries with:
            - start_pos: Start position in text
            - end_pos: End position in text
            - markdown: The table markdown
            - row_count: Number of data rows
            - col_count: Number of columns
        """
        tables = []
        
        # Pattern to match Markdown tables
        # Matches: | col1 | col2 | ... followed by |---|---| ... followed by data rows
        table_pattern = r'\|[^\n]+\|\s*\n\|[-\s:]+\|\s*\n(?:\|[^\n]+\|\s*\n?)+'
        
        for match in re.finditer(table_pattern, response, re.MULTILINE):
            table_text = match.group(0)
            
            # Count rows and columns
            lines = [line.strip() for line in table_text.split('\n') if line.strip() and not re.match(r'^\|[-:\s]+\|$', line.strip())]
            if lines:
                # First line is header
                header_cols = len([c for c in lines[0].split('|') if c.strip()])
                data_rows = len(lines) - 1  # Exclude header
                
                tables.append({
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'markdown': table_text,
                    'row_count': data_rows,
                    'col_count': header_cols,
                    'table_index': len(tables) + 1
                })
        
        return tables
    
    @staticmethod
    def format_response_with_tables(response: str) -> Dict[str, Any]:
        """
        Format response and extract structured content.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary with:
            - formatted_text: Response text with detected tables
            - tables: List of detected tables
            - has_structured_content: Boolean indicating if tables were found
        """
        tables = ResponseFormatter.detect_tables_in_response(response)
        
        return {
            'formatted_text': response,  # Keep original text (tables already in Markdown)
            'tables': tables,
            'has_structured_content': len(tables) > 0,
            'table_count': len(tables)
        }
    
    @staticmethod
    def extract_table_data(markdown_table: str) -> Optional[List[Dict[str, Any]]]:
        """
        Extract structured data from Markdown table.
        
        Args:
            markdown_table: Markdown formatted table string
            
        Returns:
            List of dictionaries (one per row) with column names as keys, or None if invalid
        """
        lines = [line.strip() for line in markdown_table.split('\n') if line.strip()]
        if len(lines) < 2:  # Need at least header and separator
            return None
        
        # Parse header
        header_line = lines[0]
        headers = [col.strip() for col in header_line.split('|')[1:-1]]  # Remove empty first/last
        
        # Skip separator line (lines[1])
        # Parse data rows
        data = []
        for line in lines[2:]:
            cols = [col.strip() for col in line.split('|')[1:-1]]
            if len(cols) == len(headers):
                row_dict = {headers[i]: cols[i] for i in range(len(headers))}
                data.append(row_dict)
        
        return data if data else None
    
    @staticmethod
    def enhance_response(response: str, detected_tables: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Enhance response with better table formatting (if needed).
        
        Args:
            response: Response text
            detected_tables: Optional pre-detected tables (to avoid re-parsing)
            
        Returns:
            Enhanced response text
        """
        if detected_tables is None:
            detected_tables = ResponseFormatter.detect_tables_in_response(response)
        
        # For now, return as-is since tables are already in Markdown format
        # This can be extended to add table captions, numbering, etc.
        return response


# Global instance
_response_formatter = None

def get_response_formatter() -> ResponseFormatter:
    """Get global response formatter instance."""
    global _response_formatter
    if _response_formatter is None:
        _response_formatter = ResponseFormatter()
    return _response_formatter

