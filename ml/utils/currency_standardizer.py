#!/usr/bin/env python3
"""
Currency Name Standardizer

This module provides utilities to standardize currency names across all pipelines
using the PostgreSQL currency table as the single source of truth.
"""

import sys
from pathlib import Path
from typing import Dict, Set, Optional, List, Tuple
import pandas as pd
import psycopg2
from functools import lru_cache

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import MLLogger


class CurrencyStandardizer:
    """
    Standardizes currency names using PostgreSQL currency table as source of truth.
    """
    
    def __init__(self, logger: Optional[MLLogger] = None):
        """Initialize the currency standardizer."""
        self.logger = logger or MLLogger("CurrencyStandardizer")
        self._db_currencies_cache: Optional[Set[str]] = None
        self._name_to_id_cache: Optional[Dict[str, int]] = None
        
    def get_db_connection(self):
        """Get database connection - override this method in production."""
        try:
            # Try the default environment-based connection first
            from utils.database import get_db_connection
            return get_db_connection()
        except Exception:
            # Fallback to direct connection for development
            try:
                return psycopg2.connect(
                    host='localhost',
                    database='PoEconomy',
                    user='postgres',
                    password='password'
                )
            except Exception as e:
                self.logger.error(f"Database connection failed: {e}")
                return None
    
    @lru_cache(maxsize=1)
    def get_all_currency_names(self) -> Set[str]:
        """
        Get all currency names from the database (cached).
        
        Returns:
            Set of currency names from the database
        """
        if self._db_currencies_cache is not None:
            return self._db_currencies_cache
            
        try:
            conn = self.get_db_connection()
            if not conn:
                return set()
            
            query = "SELECT name FROM currency ORDER BY name"
            df = pd.read_sql(query, conn)
            conn.close()
            
            self._db_currencies_cache = set(df['name'].tolist())
            self.logger.info(f"Loaded {len(self._db_currencies_cache)} currency names from database")
            
            return self._db_currencies_cache
            
        except Exception as e:
            self.logger.error(f"Failed to load currency names: {e}")
            return set()
    
    @lru_cache(maxsize=1)
    def get_currency_name_to_id_mapping(self) -> Dict[str, int]:
        """
        Get mapping from currency names to IDs (cached).
        
        Returns:
            Dictionary mapping currency names to IDs
        """
        if self._name_to_id_cache is not None:
            return self._name_to_id_cache
            
        try:
            conn = self.get_db_connection()
            if not conn:
                return {}
                
            query = "SELECT id, name FROM currency"
            df = pd.read_sql(query, conn)
            conn.close()
            
            self._name_to_id_cache = dict(zip(df['name'], df['id']))
            self.logger.info(f"Loaded {len(self._name_to_id_cache)} currency name-to-ID mappings")
            
            return self._name_to_id_cache
            
        except Exception as e:
            self.logger.error(f"Failed to load currency mappings: {e}")
            return {}
    
    def standardize_currency_name(self, input_name: str) -> Optional[str]:
        """
        Standardize a currency name to match the database.
        
        Args:
            input_name: Input currency name (e.g., from POE Watch API)
            
        Returns:
            Standardized currency name if found, None otherwise
        """
        if not input_name:
            return None
            
        db_currencies = self.get_all_currency_names()
        if not db_currencies:
            return None
        
        # 1. Exact match (most common case)
        if input_name in db_currencies:
            return input_name
        
        # 2. Case-insensitive match
        input_lower = input_name.lower()
        for db_name in db_currencies:
            if db_name.lower() == input_lower:
                return db_name
        
        # 3. Fuzzy matching for close matches
        best_match = self._find_fuzzy_match(input_name, db_currencies)
        if best_match:
            self.logger.info(f"Fuzzy matched '{input_name}' → '{best_match}'")
            return best_match
        
        # 4. No match found
        self.logger.warning(f"No match found for currency: '{input_name}'")
        return None
    
    def _find_fuzzy_match(self, input_name: str, db_currencies: Set[str]) -> Optional[str]:
        """Find the best fuzzy match for a currency name."""
        import difflib
        
        # Use difflib to find close matches
        matches = difflib.get_close_matches(
            input_name, 
            db_currencies, 
            n=1, 
            cutoff=0.8  # 80% similarity threshold
        )
        
        return matches[0] if matches else None
    
    def get_currency_id(self, currency_name: str) -> Optional[int]:
        """
        Get the database ID for a currency name.
        
        Args:
            currency_name: Currency name
            
        Returns:
            Currency ID if found, None otherwise
        """
        mapping = self.get_currency_name_to_id_mapping()
        return mapping.get(currency_name)
    
    def get_currency_ids(self, currency_names: List[str]) -> Dict[str, int]:
        """
        Get database IDs for multiple currency names.
        
        Args:
            currency_names: List of currency names
            
        Returns:
            Dictionary mapping currency names to IDs (only found currencies)
        """
        mapping = self.get_currency_name_to_id_mapping()
        return {name: mapping[name] for name in currency_names if name in mapping}
    
    def validate_currency_name(self, currency_name: str) -> bool:
        """
        Validate that a currency name exists in the database.
        
        Args:
            currency_name: Currency name to validate
            
        Returns:
            True if currency exists, False otherwise
        """
        db_currencies = self.get_all_currency_names()
        return currency_name in db_currencies
    
    def get_standardization_report(self, input_names: List[str]) -> Dict[str, any]:
        """
        Generate a standardization report for a list of currency names.
        
        Args:
            input_names: List of input currency names
            
        Returns:
            Dictionary with standardization statistics and results
        """
        results = {
            'total_input': len(input_names),
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0,
            'standardized_names': {},
            'unmatched_names': []
        }
        
        for name in input_names:
            standardized = self.standardize_currency_name(name)
            
            if standardized:
                results['standardized_names'][name] = standardized
                
                if name == standardized:
                    results['exact_matches'] += 1
                else:
                    results['fuzzy_matches'] += 1
            else:
                results['no_matches'] += 1
                results['unmatched_names'].append(name)
        
        results['match_rate'] = (results['exact_matches'] + results['fuzzy_matches']) / results['total_input'] if results['total_input'] > 0 else 0
        
        return results
    
    def clear_cache(self):
        """Clear all cached data."""
        self._db_currencies_cache = None
        self._name_to_id_cache = None
        self.get_all_currency_names.cache_clear()
        self.get_currency_name_to_id_mapping.cache_clear()


# Global instance for easy access
_currency_standardizer = CurrencyStandardizer()

def standardize_currency_name(currency_name: str) -> Optional[str]:
    """
    Convenience function to standardize a currency name.
    
    Args:
        currency_name: Input currency name
        
    Returns:
        Standardized currency name or None
    """
    return _currency_standardizer.standardize_currency_name(currency_name)

def get_currency_id(currency_name: str) -> Optional[int]:
    """
    Convenience function to get currency ID.
    
    Args:
        currency_name: Currency name
        
    Returns:
        Currency ID or None
    """
    return _currency_standardizer.get_currency_id(currency_name)

def validate_currency_name(currency_name: str) -> bool:
    """
    Convenience function to validate currency name.
    
    Args:
        currency_name: Currency name to validate
        
    Returns:
        True if valid, False otherwise
    """
    return _currency_standardizer.validate_currency_name(currency_name)

def get_all_currency_names() -> Set[str]:
    """
    Convenience function to get all currency names.
    
    Returns:
        Set of all currency names from database
    """
    return _currency_standardizer.get_all_currency_names() 