"""
Named Entity Recognition (NER) System
Author: BrillConsulting
Description: Extract entities (persons, organizations, locations) from text using spaCy
"""

import spacy
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict


class NERSystem:
    """Named Entity Recognition using spaCy"""

    def __init__(self, model_name='en_core_web_sm'):
        """Initialize NER system with spaCy model"""
        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model: {model_name}\n")
        except OSError:
            print(f"⚠️  Model '{model_name}' not found. Downloading...")
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)

    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            List of entities with text, label, and position
        """
        doc = self.nlp(text)

        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })

        return entities

    def analyze_document(self, text: str) -> Dict:
        """
        Analyze document and return entity statistics

        Args:
            text: Input document

        Returns:
            Dictionary with entity counts and lists
        """
        entities = self.extract_entities(text)

        # Count by type
        entity_counts = Counter([ent['label'] for ent in entities])

        # Group by type
        entities_by_type = {}
        for ent in entities:
            label = ent['label']
            if label not in entities_by_type:
                entities_by_type[label] = []
            entities_by_type[label].append(ent['text'])

        return {
            'total_entities': len(entities),
            'entity_counts': dict(entity_counts),
            'entities_by_type': entities_by_type,
            'all_entities': entities
        }

    def visualize_entities(self, text: str, save_path=None):
        """
        Visualize entities in text

        Args:
            text: Input text
            save_path: Optional path to save visualization
        """
        doc = self.nlp(text)

        # For Jupyter: use displacy.render
        # For saving: use displacy.render with file option
        from spacy import displacy

        html = displacy.render(doc, style='ent', page=True)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"📊 Visualization saved to {save_path}")

        return html

    def plot_entity_distribution(self, text: str, save_path=None):
        """Plot entity type distribution"""
        analysis = self.analyze_document(text)
        counts = analysis['entity_counts']

        if not counts:
            print("❌ No entities found")
            return

        plt.figure(figsize=(10, 6))
        plt.bar(counts.keys(), counts.values())
        plt.xlabel('Entity Type')
        plt.ylabel('Count')
        plt.title('Named Entity Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Named Entity Recognition')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='Text file to analyze')
    parser.add_argument('--model', type=str, default='en_core_web_sm',
                       help='spaCy model name')
    parser.add_argument('--output', type=str, help='Output visualization path')

    args = parser.parse_args()

    # Get text
    if args.text:
        text = args.text
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        print("❌ Provide --text or --file")
        return

    # Initialize NER
    ner = NERSystem(model_name=args.model)

    # Analyze
    print("🔍 Analyzing text...\n")
    analysis = ner.analyze_document(text)

    # Print results
    print(f"📊 Found {analysis['total_entities']} entities\n")

    print("📋 Entity Counts by Type:")
    for entity_type, count in sorted(analysis['entity_counts'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {entity_type}: {count}")

    print("\n📝 Entities by Type:")
    for entity_type, entities in analysis['entities_by_type'].items():
        unique_entities = list(set(entities))[:5]  # Top 5 unique
        print(f"  {entity_type}: {', '.join(unique_entities)}")

    # Visualize
    if args.output:
        ner.plot_entity_distribution(text, save_path=args.output)


if __name__ == "__main__":
    main()
