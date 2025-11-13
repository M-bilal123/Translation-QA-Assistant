"""
QLingo - File Loader Module
Handles multiple input file formats (CSV, Excel, JSON, XLIFF)
"""

import pandas as pd
import json
from xml.etree import ElementTree as ET
from pathlib import Path
from typing import Optional

class FileLoader:
    """Handle multiple input file formats"""
    
    @staticmethod
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """
        Load file based on extension and return DataFrame
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            pd.DataFrame or None if loading fails
        """
        if uploaded_file is None:
            return None
        
        file_ext = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_ext == '.csv':
                return FileLoader.parse_csv(uploaded_file)
            elif file_ext in ['.xlsx', '.xls']:
                return FileLoader.parse_excel(uploaded_file)
            elif file_ext == '.json':
                return FileLoader.parse_json(uploaded_file)
            elif file_ext == '.xliff':
                return FileLoader.parse_xliff(uploaded_file)
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def parse_csv(file) -> pd.DataFrame:
        """Parse CSV file"""
        return pd.read_csv(file)
    
    @staticmethod
    def parse_excel(file) -> pd.DataFrame:
        """Parse Excel file"""
        return pd.read_excel(file)
    
    @staticmethod
    def parse_json(file) -> pd.DataFrame:
        """Parse JSON file"""
        # Read the content
        content = file.read()
        
        # Decode if bytes
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        data = json.loads(content)
        
        # Handle both list and single object
        if isinstance(data, list):
            return pd.DataFrame(data)
        else:
            return pd.DataFrame([data])
    
    @staticmethod
    def parse_xliff(file) -> pd.DataFrame:
        """Parse XLIFF file"""
        # Read content
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        
        # Parse XML
        root = ET.fromstring(content)
        
        # Handle XLIFF namespace
        ns = {'xliff': 'urn:oasis:names:tc:xliff:document:1.2'}
        segments = []
        
        # Try with namespace
        trans_units = root.findall('.//xliff:trans-unit', ns)
        
        # If no results, try without namespace
        if not trans_units:
            trans_units = root.findall('.//{urn:oasis:names:tc:xliff:document:1.2}trans-unit')
        
        # If still no results, try generic search
        if not trans_units:
            trans_units = root.findall('.//trans-unit')
        
        for trans_unit in trans_units:
            # Try with namespace
            source = trans_unit.find('xliff:source', ns)
            target = trans_unit.find('xliff:target', ns)
            
            # Try without namespace if not found
            if source is None:
                source = trans_unit.find('.//{urn:oasis:names:tc:xliff:document:1.2}source')
            if target is None:
                target = trans_unit.find('.//{urn:oasis:names:tc:xliff:document:1.2}target')
            
            # Try generic search
            if source is None:
                source = trans_unit.find('.//source')
            if target is None:
                target = trans_unit.find('.//target')
            
            if source is not None and target is not None:
                segments.append({
                    'source': source.text or '',
                    'translation': target.text or ''
                })
        
        return pd.DataFrame(segments)
    
    @staticmethod
    def get_file_info(uploaded_file) -> dict:
        """Get information about uploaded file"""
        if uploaded_file is None:
            return {}
        
        return {
            'name': uploaded_file.name,
            'size': uploaded_file.size,
            'type': uploaded_file.type,
            'extension': Path(uploaded_file.name).suffix.lower()
        }
    
    @staticmethod
    def validate_columns(df: pd.DataFrame, required_columns: list) -> tuple:
        """
        Validate that required columns exist in DataFrame
        
        Returns:
            (is_valid: bool, message: str)
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing columns: {', '.join(missing_columns)}"
        
        return True, "All required columns found"