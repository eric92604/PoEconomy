#!/usr/bin/env python3
"""
Currency Name Mapper

This module provides utilities to map currency names between different data sources
(POE Watch, ML models) that may use different naming conventions.
"""

from typing import Dict, Set, Optional, List
import re


class CurrencyNameMapper:
    """Maps currency names between different data sources."""
    
    def __init__(self):
        """Initialize the currency name mapper with known mappings."""
        # Direct mappings for known differences
        self.poe_watch_to_ml = {
            # Apostrophe differences
            "Al-Hezmin's Crest": "Al-Hezmins Crest",
            "Awakener's Orb": "Awakeners Orb",
            "Conqueror's Exalted Orb": "Conquerors Exalted Orb",
            "Crusader's Exalted Orb": "Crusaders Exalted Orb",
            "Hunter's Exalted Orb": "Hunters Exalted Orb",
            "Redeemer's Exalted Orb": "Redeemers Exalted Orb",
            "Warlord's Exalted Orb": "Warlords Exalted Orb",
            "Maven's Orb": "Mavens Orb",
            "Elderslayer's Exalted Orb": "Elderslayers Exalted Orb",
            "Shaper's Orb": "Shapers Orb",
            "Elder's Orb": "Elders Orb",
            "Valdo's Puzzle Box": "Valdos Puzzle Box",
            "Maven's Chisel of Divination": "Mavens Chisel of Divination",
            "Maven's Chisel of Avarice": "Mavens Chisel of Avarice", 
            "Maven's Chisel of Proliferation": "Mavens Chisel of Proliferation",
            "Maven's Chisel of Scarabs": "Mavens Chisel of Scarabs",
            "The Maven's Writ": "The Mavens Writ",
            "Drox's Crest": "Droxs Crest",
            "Veritania's Crest": "Veritanias Crest",
            "Baran's Crest": "Barans Crest",
            "Tul's Breachstone": "Tuls Breachstone",
            "Xoph's Breachstone": "Xophs Breachstone",
            "Esh's Breachstone": "Eshs Breachstone",
            "Uul-Netol's Breachstone": "Uul-Netols Breachstone",
            "Hinekora's Lock": "Hinekoras Lock",
            "Jeweller's Orb": "Jewellers Orb",
            "Vaal Orb": "Vaal Orb",  # Same
            "Exalted Orb": "Exalted Orb",  # Same
            "Divine Orb": "Divine Orb",  # Same
            "Chaos Orb": "Chaos Orb",  # Same
            "Ancient Orb": "Ancient Orb",  # Same
            "Annulment Orb": "Annulment Orb",  # Same
            "Blessed Orb": "Blessed Orb",  # Same
            "Chromatic Orb": "Chromatic Orb",  # Same
            "Orb of Fusing": "Orb of Fusing",  # Same
            "Orb of Alchemy": "Orb of Alchemy",  # Same
            "Orb of Alteration": "Orb of Alteration",  # Same
            "Orb of Chance": "Orb of Chance",  # Same
            "Orb of Scouring": "Orb of Scouring",  # Same
            "Orb of Regret": "Orb of Regret",  # Same
        }
        
        # Reverse mapping
        self.ml_to_poe_watch = {v: k for k, v in self.poe_watch_to_ml.items()}
    
    def normalize_name(self, name: str) -> str:
        """Normalize a currency name for comparison."""
        if not name:
            return ""
        
        # Convert to lowercase, remove apostrophes, spaces, and hyphens
        normalized = name.lower()
        normalized = re.sub(r"['\s\-]", "", normalized)
        return normalized
    
    def find_best_currency_match(self, source_name: str, target_names: Set[str]) -> Optional[str]:
        """
        Find the best matching currency name from target names.
        
        Args:
            source_name: Name to find a match for
            target_names: Set of target names to search in
            
        Returns:
            Best matching name or None if no match found
        """
        if not source_name or not target_names:
            return None
        
        # 1. Exact match
        if source_name in target_names:
            return source_name
        
        # 2. Direct mapping lookup
        if source_name in self.poe_watch_to_ml:
            mapped_name = self.poe_watch_to_ml[source_name]
            if mapped_name in target_names:
                return mapped_name
        
        if source_name in self.ml_to_poe_watch:
            mapped_name = self.ml_to_poe_watch[source_name]
            if mapped_name in target_names:
                return mapped_name
        
        # 3. Normalized matching
        source_normalized = self.normalize_name(source_name)
        
        for target_name in target_names:
            target_normalized = self.normalize_name(target_name)
            if source_normalized == target_normalized:
                return target_name
        
        # 4. Partial matching (contains)
        source_words = set(source_name.lower().split())
        best_match = None
        best_score = 0
        
        for target_name in target_names:
            target_words = set(target_name.lower().split())
            
            # Calculate overlap score
            overlap = len(source_words & target_words)
            total_words = len(source_words | target_words)
            
            if total_words > 0:
                score = overlap / total_words
                if score > best_score and score > 0.5:  # At least 50% overlap
                    best_score = score
                    best_match = target_name
        
        return best_match
    
    def add_mapping(self, poe_watch_name: str, ml_name: str):
        """Add a new mapping between POE Watch and ML names."""
        self.poe_watch_to_ml[poe_watch_name] = ml_name
        self.ml_to_poe_watch[ml_name] = poe_watch_name
    
    def get_all_mappings(self) -> Dict[str, str]:
        """Get all POE Watch to ML mappings."""
        return self.poe_watch_to_ml.copy()
    
    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about the mappings."""
        return {
            'total_mappings': len(self.poe_watch_to_ml),
            'bidirectional_mappings': len(self.ml_to_poe_watch)
        }


# Global instance for easy access
_currency_mapper = CurrencyNameMapper()

def find_best_currency_match(source_name: str, target_names: Set[str]) -> Optional[str]:
    """
    Convenience function to find the best matching currency name.
    
    Args:
        source_name: Name to find a match for
        target_names: Set of target names to search in
        
    Returns:
        Best matching name or None if no match found
    """
    return _currency_mapper.find_best_currency_match(source_name, target_names)

def add_currency_mapping(poe_watch_name: str, ml_name: str):
    """
    Convenience function to add a new currency mapping.
    
    Args:
        poe_watch_name: POE Watch currency name
        ml_name: ML model currency name
    """
    _currency_mapper.add_mapping(poe_watch_name, ml_name)

def get_currency_mappings() -> Dict[str, str]:
    """Get all currency mappings."""
    return _currency_mapper.get_all_mappings()

def get_mapping_statistics() -> Dict[str, int]:
    """Get mapping statistics."""
    return _currency_mapper.get_mapping_stats() 