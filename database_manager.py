#!/usr/bin/env python3
"""
Database management utility for bullying words and phrases
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Dict

class BullyingDatabaseManager:
    """Utility class for managing the bullying words database"""
    
    def __init__(self, db_path: str = "data/bullying_words.json"):
        self.db_path = db_path
        self.data = self._load_database()
    
    def _load_database(self) -> Dict:
        """Load the database from file"""
        if not os.path.exists(self.db_path):
            print(f"Database file not found: {self.db_path}")
            return self._create_default_database()
        
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading database: {e}")
            return self._create_default_database()
    
    def _create_default_database(self) -> Dict:
        """Create a default database structure"""
        return {
            "bullying_words": [],
            "bullying_phrases": [],
            "severity_levels": {
                "low": [],
                "medium": [],
                "high": []
            },
            "contextual_patterns": [],
            "intent_indicators": [],
            "metadata": {
                "version": "1.0",
                "last_updated": datetime.utcnow().isoformat(),
                "total_words": 0,
                "total_phrases": 0,
                "description": "Local database of cyberbullying indicators"
            }
        }
    
    def _save_database(self):
        """Save the database to file"""
        # Update metadata
        self.data['metadata']['total_words'] = len(self.data.get('bullying_words', []))
        self.data['metadata']['total_phrases'] = len(self.data.get('bullying_phrases', []))
        self.data['metadata']['last_updated'] = datetime.utcnow().isoformat()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        print(f"Database saved to {self.db_path}")
    
    def add_words(self, words: List[str], severity: str = "medium") -> int:
        """Add words to the database"""
        if severity not in ["low", "medium", "high"]:
            raise ValueError("Severity must be 'low', 'medium', or 'high'")
        
        current_words = set(self.data.get('bullying_words', []))
        added_count = 0
        
        for word in words:
            word_clean = word.strip().lower()
            if word_clean and word_clean not in current_words:
                self.data['bullying_words'].append(word_clean)
                self.data['severity_levels'][severity].append(word_clean)
                current_words.add(word_clean)
                added_count += 1
        
        return added_count
    
    def add_phrases(self, phrases: List[str], severity: str = "high") -> int:
        """Add phrases to the database"""
        if severity not in ["low", "medium", "high"]:
            raise ValueError("Severity must be 'low', 'medium', or 'high'")
        
        current_phrases = set(self.data.get('bullying_phrases', []))
        added_count = 0
        
        for phrase in phrases:
            phrase_clean = phrase.strip().lower()
            if phrase_clean and phrase_clean not in current_phrases:
                self.data['bullying_phrases'].append(phrase_clean)
                self.data['severity_levels'][severity].append(phrase_clean)
                current_phrases.add(phrase_clean)
                added_count += 1
        
        return added_count
    
    def remove_words(self, words: List[str]) -> int:
        """Remove words from the database"""
        removed_count = 0
        words_to_remove = {word.strip().lower() for word in words}
        
        # Remove from main list
        original_count = len(self.data.get('bullying_words', []))
        self.data['bullying_words'] = [
            word for word in self.data.get('bullying_words', [])
            if word.lower() not in words_to_remove
        ]
        removed_count = original_count - len(self.data['bullying_words'])
        
        # Remove from severity levels
        for severity in self.data.get('severity_levels', {}).values():
            if isinstance(severity, list):
                severity[:] = [word for word in severity if word.lower() not in words_to_remove]
        
        return removed_count
    
    def list_words(self, severity: str = None) -> List[str]:
        """List words in the database, optionally filtered by severity"""
        if severity:
            if severity not in ["low", "medium", "high"]:
                raise ValueError("Severity must be 'low', 'medium', or 'high'")
            return self.data.get('severity_levels', {}).get(severity, [])
        else:
            return self.data.get('bullying_words', [])
    
    def list_phrases(self) -> List[str]:
        """List phrases in the database"""
        return self.data.get('bullying_phrases', [])
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return {
            'total_words': len(self.data.get('bullying_words', [])),
            'total_phrases': len(self.data.get('bullying_phrases', [])),
            'severity_breakdown': {
                severity: len(words) for severity, words 
                in self.data.get('severity_levels', {}).items()
            },
            'last_updated': self.data.get('metadata', {}).get('last_updated', 'Unknown')
        }
    
    def export_to_text(self, output_file: str):
        """Export database to a simple text file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Cyberbullying Words Database\n")
            f.write(f"# Generated on: {datetime.utcnow().isoformat()}\n\n")
            
            f.write("## Bullying Words\n")
            for word in sorted(self.data.get('bullying_words', [])):
                f.write(f"{word}\n")
            
            f.write("\n## Bullying Phrases\n")
            for phrase in sorted(self.data.get('bullying_phrases', [])):
                f.write(f"{phrase}\n")
        
        print(f"Database exported to {output_file}")

def main():
    """Command-line interface for database management"""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python database_manager.py list [severity]")
        print("  python database_manager.py add-words <word1> <word2> ... [--severity=medium]")
        print("  python database_manager.py add-phrases <phrase1> <phrase2> ... [--severity=high]")
        print("  python database_manager.py remove-words <word1> <word2> ...")
        print("  python database_manager.py stats")
        print("  python database_manager.py export <output_file>")
        return
    
    db_manager = BullyingDatabaseManager()
    command = sys.argv[1]
    
    try:
        if command == "list":
            severity = sys.argv[2] if len(sys.argv) > 2 else None
            words = db_manager.list_words(severity)
            phrases = db_manager.list_phrases()
            
            if severity:
                print(f"Bullying words ({severity} severity):")
                for word in sorted(words):
                    print(f"  {word}")
            else:
                print("All bullying words:")
                for word in sorted(words):
                    print(f"  {word}")
                print(f"\nAll bullying phrases:")
                for phrase in sorted(phrases):
                    print(f"  {phrase}")
        
        elif command == "add-words":
            words = []
            severity = "medium"
            
            for arg in sys.argv[2:]:
                if arg.startswith("--severity="):
                    severity = arg.split("=")[1]
                else:
                    words.append(arg)
            
            if not words:
                print("Error: No words provided")
                return
            
            added = db_manager.add_words(words, severity)
            db_manager._save_database()
            print(f"Added {added} new words with {severity} severity")
        
        elif command == "add-phrases":
            phrases = []
            severity = "high"
            
            for arg in sys.argv[2:]:
                if arg.startswith("--severity="):
                    severity = arg.split("=")[1]
                else:
                    phrases.append(arg)
            
            if not phrases:
                print("Error: No phrases provided")
                return
            
            added = db_manager.add_phrases(phrases, severity)
            db_manager._save_database()
            print(f"Added {added} new phrases with {severity} severity")
        
        elif command == "remove-words":
            if len(sys.argv) < 3:
                print("Error: No words provided")
                return
            
            words = sys.argv[2:]
            removed = db_manager.remove_words(words)
            db_manager._save_database()
            print(f"Removed {removed} words")
        
        elif command == "stats":
            stats = db_manager.get_stats()
            print("Database Statistics:")
            print(f"  Total words: {stats['total_words']}")
            print(f"  Total phrases: {stats['total_phrases']}")
            print("  Severity breakdown:")
            for severity, count in stats['severity_breakdown'].items():
                print(f"    {severity}: {count}")
            print(f"  Last updated: {stats['last_updated']}")
        
        elif command == "export":
            if len(sys.argv) < 3:
                print("Error: No output file provided")
                return
            
            output_file = sys.argv[2]
            db_manager.export_to_text(output_file)
        
        else:
            print(f"Unknown command: {command}")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
