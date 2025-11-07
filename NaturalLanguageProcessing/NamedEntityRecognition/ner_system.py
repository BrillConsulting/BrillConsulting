"""
Advanced Named Entity Recognition (NER) System v2.0
Author: BrillConsulting
Description: Extract and classify 18+ entity types using spaCy

Supports multiple languages, custom models, batch processing, and visualization
"""

import spacy
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class NERSystem:
    """
    Advanced Named Entity Recognition System

    Recognizes 18 entity types:
    - PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT
    - PRODUCT, EVENT, LANGUAGE, LAW, WORK_OF_ART, FAC
    - NORP, QUANTITY, ORDINAL, CARDINAL
    """

    # Entity type descriptions
    ENTITY_TYPES = {
        'PERSON': 'People names',
        'ORG': 'Organizations',
        'GPE': 'Geopolitical entities (countries, cities)',
        'LOC': 'Non-GPE locations',
        'DATE': 'Dates (absolute or relative)',
        'TIME': 'Times',
        'MONEY': 'Monetary values',
        'PERCENT': 'Percentages',
        'PRODUCT': 'Products',
        'EVENT': 'Named events',
        'LANGUAGE': 'Languages',
        'LAW': 'Laws, treaties',
        'WORK_OF_ART': 'Titles of books, songs, etc.',
        'FAC': 'Facilities (buildings, airports, etc.)',
        'NORP': 'Nationalities, religious groups',
        'QUANTITY': 'Measurements',
        'ORDINAL': 'Ordinal numbers (first, second)',
        'CARDINAL': 'Cardinal numbers'
    }

    def __init__(self, model_name='en_core_web_sm', language='en'):
        """
        Initialize NER system

        Args:
            model_name: spaCy model name
            language: Language code
        """
        self.model_name = model_name
        self.language = language

        try:
            self.nlp = spacy.load(model_name)
            print(f"✅ Loaded spaCy model: {model_name}")
        except OSError:
            print(f"⚠️  Model '{model_name}' not found. Downloading...")
            import os
            os.system(f"python -m spacy download {model_name}")
            self.nlp = spacy.load(model_name)

        print(f"✓ NERSystem initialized (language={language}, model={model_name})")

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
                'end': ent.end_char,
                'description': self.ENTITY_TYPES.get(ent.label_, 'Unknown')
            })

        return entities

    def analyze_document(self, text: str) -> Dict:
        """
        Analyze document and return comprehensive entity statistics

        Args:
            text: Input document

        Returns:
            Dictionary with entity counts, lists, and statistics
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

        # Unique entities
        unique_entities = {}
        for label, ent_list in entities_by_type.items():
            unique_entities[label] = list(set(ent_list))

        return {
            'total_entities': len(entities),
            'unique_entities': sum(len(v) for v in unique_entities.values()),
            'entity_counts': dict(entity_counts),
            'entities_by_type': entities_by_type,
            'unique_entities_by_type': unique_entities,
            'all_entities': entities
        }

    def extract_entities_batch(self, texts: List[str], batch_size=100) -> List[List[Dict]]:
        """
        Extract entities from multiple texts efficiently

        Args:
            texts: List of text documents
            batch_size: Number of texts to process at once

        Returns:
            List of entity lists for each document
        """
        all_entities = []

        # Process in batches using nlp.pipe for efficiency
        for doc in self.nlp.pipe(texts, batch_size=batch_size):
            entities = []
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            all_entities.append(entities)

        return all_entities

    def filter_by_type(self, text: str, entity_types: List[str]) -> List[Dict]:
        """
        Extract only specific entity types

        Args:
            text: Input text
            entity_types: List of entity type labels to extract

        Returns:
            Filtered list of entities
        """
        all_entities = self.extract_entities(text)
        return [ent for ent in all_entities if ent['label'] in entity_types]

    def get_entity_frequency(self, texts: List[str]) -> Dict[str, Counter]:
        """
        Get entity frequency across multiple documents

        Args:
            texts: List of documents

        Returns:
            Dict mapping entity type to Counter of entity texts
        """
        entity_freq = {}

        for text in texts:
            entities = self.extract_entities(text)

            for ent in entities:
                label = ent['label']
                if label not in entity_freq:
                    entity_freq[label] = Counter()

                entity_freq[label][ent['text']] += 1

        return entity_freq

    def get_most_common_entities(self, texts: List[str], top_n=10) -> Dict[str, List[Tuple[str, int]]]:
        """
        Get most common entities by type across documents

        Args:
            texts: List of documents
            top_n: Number of top entities per type

        Returns:
            Dict mapping entity type to list of (entity, count) tuples
        """
        entity_freq = self.get_entity_frequency(texts)

        most_common = {}
        for label, counter in entity_freq.items():
            most_common[label] = counter.most_common(top_n)

        return most_common

    def visualize_entities(self, text: str, save_path=None) -> str:
        """
        Create HTML visualization of entities in text

        Args:
            text: Input text
            save_path: Optional path to save HTML

        Returns:
            HTML string
        """
        doc = self.nlp(text)

        from spacy import displacy

        # Generate HTML
        html = displacy.render(doc, style='ent', page=True)

        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"📊 Visualization saved to {save_path}")

        return html

    def plot_entity_distribution(self, text: str, save_path=None):
        """
        Plot entity type distribution

        Args:
            text: Input text
            save_path: Optional path to save plot
        """
        analysis = self.analyze_document(text)
        counts = analysis['entity_counts']

        if not counts:
            print("❌ No entities found")
            return

        # Sort by count
        sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

        plt.figure(figsize=(12, 6))
        bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color='steelblue')

        # Color top 3 differently
        if len(bars) >= 3:
            bars[0].set_color('gold')
            bars[1].set_color('silver')
            bars[2].set_color('#CD7F32')

        plt.xlabel('Entity Type', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Named Entity Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Distribution plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_entity_timeline(self, texts: List[str], dates: List[str], save_path=None):
        """
        Plot entity occurrence over time

        Args:
            texts: List of documents
            dates: List of corresponding dates
            save_path: Optional path to save plot
        """
        # Extract entities for each document
        entity_counts_over_time = []
        for text in texts:
            analysis = self.analyze_document(text)
            entity_counts_over_time.append(analysis['total_entities'])

        plt.figure(figsize=(12, 6))
        plt.plot(dates, entity_counts_over_time, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Entity Count', fontsize=12)
        plt.title('Entity Occurrence Over Time', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Timeline plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def compare_documents(self, texts: List[str], labels: List[str] = None) -> pd.DataFrame:
        """
        Compare entity distributions across documents

        Args:
            texts: List of documents
            labels: Optional labels for documents

        Returns:
            DataFrame with entity counts per document
        """
        if labels is None:
            labels = [f"Doc {i+1}" for i in range(len(texts))]

        data = []
        for text, label in zip(texts, labels):
            analysis = self.analyze_document(text)
            row = {'Document': label}
            row.update(analysis['entity_counts'])
            data.append(row)

        df = pd.DataFrame(data).fillna(0)
        return df

    def export_entities(self, text: str, format='json', output_path=None) -> str:
        """
        Export entities in various formats

        Args:
            text: Input text
            format: 'json', 'csv', or 'txt'
            output_path: Optional path to save file

        Returns:
            Formatted string
        """
        entities = self.extract_entities(text)

        if format == 'json':
            import json
            result = json.dumps(entities, indent=2)
        elif format == 'csv':
            df = pd.DataFrame(entities)
            result = df.to_csv(index=False)
        elif format == 'txt':
            lines = []
            for ent in entities:
                lines.append(f"{ent['text']} ({ent['label']})")
            result = '\n'.join(lines)
        else:
            raise ValueError(f"Unknown format: {format}")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"📁 Entities exported to {output_path}")

        return result

    def get_statistics(self) -> Dict:
        """
        Get NER system statistics

        Returns:
            Dict with model information and capabilities
        """
        return {
            'model_name': self.model_name,
            'language': self.language,
            'entity_types': list(self.ENTITY_TYPES.keys()),
            'num_entity_types': len(self.ENTITY_TYPES),
            'pipeline_components': self.nlp.pipe_names
        }


def demo_ner_system():
    """Demonstrate NER system capabilities"""
    print("=" * 80)
    print("Advanced Named Entity Recognition System v2.0 - Demo")
    print("=" * 80)

    sample_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California
    on April 1, 1976. The company's market value reached $3 trillion in 2024. Tim Cook has been
    the CEO since August 24, 2011. Apple's headquarters, Apple Park, is located at One Apple Park Way
    in Cupertino. The company develops popular products like the iPhone, iPad, and MacBook.
    In 2023, Apple announced the Vision Pro, a mixed reality headset priced at $3,499.
    """

    print(f"\n📄 Sample Text:")
    print(sample_text)

    # Initialize NER
    ner = NERSystem()

    # Extract entities
    print("\n" + "=" * 80)
    print("Extracting Entities")
    print("=" * 80)

    analysis = ner.analyze_document(sample_text)

    print(f"\n📊 Statistics:")
    print(f"  Total entities: {analysis['total_entities']}")
    print(f"  Unique entities: {analysis['unique_entities']}")

    print(f"\n📋 Entity Counts by Type:")
    for entity_type, count in sorted(analysis['entity_counts'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:15s} {count:3d}  ({ner.ENTITY_TYPES.get(entity_type, 'Unknown')})")

    print(f"\n📝 Entities by Type (sample):")
    for entity_type in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PRODUCT']:
        if entity_type in analysis['unique_entities_by_type']:
            entities = analysis['unique_entities_by_type'][entity_type][:5]
            print(f"\n  {entity_type}:")
            for ent in entities:
                print(f"    - {ent}")

    # Visualization
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    ner.plot_entity_distribution(sample_text, save_path='ner_distribution.png')
    ner.visualize_entities(sample_text, save_path='ner_entities.html')

    print("\n✓ Demo completed!")


def main():
    parser = argparse.ArgumentParser(description='Advanced Named Entity Recognition')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--file', type=str, help='Text file to analyze')
    parser.add_argument('--model', type=str, default='en_core_web_sm',
                       help='spaCy model name')
    parser.add_argument('--output', type=str, help='Output visualization path')
    parser.add_argument('--export', type=str, choices=['json', 'csv', 'txt'],
                       help='Export format')
    parser.add_argument('--export-path', type=str, help='Export file path')
    parser.add_argument('--visualize', action='store_true',
                       help='Create HTML visualization')

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
    print("\n🔍 Analyzing text...")
    analysis = ner.analyze_document(text)

    # Print results
    print(f"\n📊 Found {analysis['total_entities']} entities " +
          f"({analysis['unique_entities']} unique)\n")

    print("📋 Entity Counts by Type:")
    for entity_type, count in sorted(analysis['entity_counts'].items(),
                                     key=lambda x: x[1], reverse=True):
        print(f"  {entity_type:15s} {count:3d}")

    print("\n📝 Entities by Type (top 5 per type):")
    for entity_type, entities in sorted(analysis['unique_entities_by_type'].items()):
        unique_entities = entities[:5]
        print(f"  {entity_type}: {', '.join(unique_entities)}")

    # Export
    if args.export:
        export_path = args.export_path or f'entities.{args.export}'
        ner.export_entities(text, format=args.export, output_path=export_path)

    # Visualize
    if args.output:
        ner.plot_entity_distribution(text, save_path=args.output)

    if args.visualize:
        ner.visualize_entities(text, save_path='entities.html')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        demo_ner_system()
