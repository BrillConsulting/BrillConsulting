"""
Advanced Named Entity Linking System v2.0
Author: BrillConsulting
Description: Entity linking to knowledge bases (Wikipedia, Wikidata)

Links extracted entities to knowledge base entries for disambiguation and enrichment
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NER
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Install with: pip install spacy")

# Entity Linking (optional)
try:
    import wikipediaapi
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False
    print("Warning: wikipedia-api not available. Install with: pip install wikipedia-api")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")


class EntityLinker:
    """
    Advanced Named Entity Linking System

    Links named entities to knowledge bases:
    - Wikipedia: General knowledge
    - Wikidata: Structured data
    """

    def __init__(self, language='en', use_wikipedia=True, use_wikidata=True):
        """
        Initialize entity linker

        Args:
            language: Language code (default: 'en')
            use_wikipedia: Enable Wikipedia linking
            use_wikidata: Enable Wikidata linking
        """
        self.language = language
        self.use_wikipedia = use_wikipedia
        self.use_wikidata = use_wikidata

        # Load spaCy model for NER
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(f"{language}_core_web_sm")
            except OSError:
                print(f"Warning: spaCy model '{language}_core_web_sm' not found")
                print(f"Download with: python -m spacy download {language}_core_web_sm")

        # Initialize Wikipedia API
        self.wiki = None
        if use_wikipedia and WIKIPEDIA_AVAILABLE:
            self.wiki = wikipediaapi.Wikipedia(
                language=language,
                extract_format=wikipediaapi.ExtractFormat.PLAIN,
                user_agent='BrillConsulting-EntityLinker/2.0'
            )

        print(f"✓ EntityLinker initialized (language={language})")

    def extract_and_link(self, text: str, top_n=10) -> List[Dict]:
        """
        Extract entities from text and link to knowledge bases

        Args:
            text: Input text
            top_n: Maximum number of entities to return

        Returns:
            List of entity dictionaries with linking information
        """
        # Extract entities
        entities = self._extract_entities(text)

        # Link entities
        linked_entities = []
        for entity in entities[:top_n]:
            linked = self._link_entity(entity)
            linked_entities.append(linked)

        return linked_entities

    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy"""
        if not self.nlp:
            # Fallback: simple pattern-based extraction
            return self._simple_entity_extraction(text)

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

    def _simple_entity_extraction(self, text: str) -> List[Dict]:
        """Simple pattern-based entity extraction (fallback)"""
        # Extract capitalized phrases as potential entities
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, text)

        entities = []
        for match in matches:
            entities.append({
                'text': match.group(),
                'label': 'ENTITY',
                'start': match.start(),
                'end': match.end()
            })

        return entities

    def _link_entity(self, entity: Dict) -> Dict:
        """Link entity to knowledge bases"""
        entity_text = entity['text']

        # Add linking information
        entity['wikipedia_url'] = None
        entity['wikipedia_summary'] = None
        entity['wikidata_id'] = None
        entity['linked'] = False

        # Link to Wikipedia
        if self.use_wikipedia and self.wiki:
            wiki_result = self._link_to_wikipedia(entity_text)
            if wiki_result:
                entity['wikipedia_url'] = wiki_result['url']
                entity['wikipedia_summary'] = wiki_result['summary']
                entity['linked'] = True

        # Link to Wikidata
        if self.use_wikidata and REQUESTS_AVAILABLE:
            wikidata_id = self._link_to_wikidata(entity_text)
            if wikidata_id:
                entity['wikidata_id'] = wikidata_id
                entity['linked'] = True

        return entity

    def _link_to_wikipedia(self, entity_text: str) -> Optional[Dict]:
        """Link entity to Wikipedia"""
        try:
            # Search for Wikipedia page
            page = self.wiki.page(entity_text)

            if page.exists():
                # Get summary (first 200 chars)
                summary = page.summary[:200] + '...' if len(page.summary) > 200 else page.summary

                return {
                    'url': page.fullurl,
                    'summary': summary,
                    'title': page.title
                }
        except Exception as e:
            pass

        return None

    def _link_to_wikidata(self, entity_text: str) -> Optional[str]:
        """Link entity to Wikidata"""
        try:
            # Search Wikidata API
            url = "https://www.wikidata.org/w/api.php"
            params = {
                'action': 'wbsearchentities',
                'search': entity_text,
                'language': self.language,
                'format': 'json'
            }

            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if 'search' in data and len(data['search']) > 0:
                # Return first result ID
                return data['search'][0]['id']

        except Exception as e:
            pass

        return None

    def disambiguate(self, entity_text: str, context: str, candidates: List[str]) -> str:
        """
        Disambiguate entity mention using context

        Args:
            entity_text: Entity mention
            context: Surrounding context
            candidates: List of candidate entity URIs

        Returns:
            Best matching candidate
        """
        # Simple disambiguation: return first candidate
        # In production, use more sophisticated methods (entity embeddings, etc.)
        if candidates:
            return candidates[0]
        return entity_text

    def get_entity_info(self, entity_text: str) -> Dict:
        """
        Get detailed information about an entity

        Args:
            entity_text: Entity name

        Returns:
            Dict with entity information
        """
        info = {
            'text': entity_text,
            'wikipedia': None,
            'wikidata': None
        }

        # Wikipedia info
        if self.wiki:
            wiki_result = self._link_to_wikipedia(entity_text)
            if wiki_result:
                info['wikipedia'] = wiki_result

        # Wikidata info
        if REQUESTS_AVAILABLE:
            wikidata_id = self._link_to_wikidata(entity_text)
            if wikidata_id:
                info['wikidata'] = {
                    'id': wikidata_id,
                    'url': f'https://www.wikidata.org/wiki/{wikidata_id}'
                }

        return info

    def batch_link(self, texts: List[str]) -> List[List[Dict]]:
        """
        Link entities in multiple texts

        Args:
            texts: List of text documents

        Returns:
            List of entity lists for each document
        """
        results = []
        for text in texts:
            entities = self.extract_and_link(text)
            results.append(entities)

        return results


class SimpleEntityLinker:
    """
    Simplified entity linker without external dependencies

    Uses pattern matching and heuristics
    """

    def __init__(self):
        """Initialize simple entity linker"""
        # Common entity patterns
        self.person_indicators = ['Mr.', 'Mrs.', 'Dr.', 'Prof.']
        self.org_indicators = ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Company']
        self.location_indicators = ['City', 'State', 'Country', 'County']

        print("✓ SimpleEntityLinker initialized")

    def extract_and_link(self, text: str) -> List[Dict]:
        """
        Extract and classify entities using patterns

        Args:
            text: Input text

        Returns:
            List of entities with basic classification
        """
        entities = []

        # Extract capitalized phrases
        pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.finditer(pattern, text)

        for match in matches:
            entity_text = match.group()

            # Classify entity
            entity_type = self._classify_entity(entity_text, text)

            entities.append({
                'text': entity_text,
                'type': entity_type,
                'start': match.start(),
                'end': match.end(),
                'confidence': 0.7
            })

        return entities

    def _classify_entity(self, entity_text: str, context: str) -> str:
        """Classify entity type using heuristics"""
        # Check for person indicators
        for indicator in self.person_indicators:
            if indicator in context:
                return 'PERSON'

        # Check for organization indicators
        for indicator in self.org_indicators:
            if indicator in entity_text:
                return 'ORGANIZATION'

        # Check for location indicators
        for indicator in self.location_indicators:
            if indicator in entity_text:
                return 'LOCATION'

        # Default
        return 'ENTITY'


def demo_entity_linking():
    """Demonstrate entity linking"""
    sample_text = """
    Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in Cupertino, California.
    The company is headquartered in Cupertino and is one of the largest technology companies in the world.
    Microsoft Corporation, founded by Bill Gates and Paul Allen, is another major technology company based in Redmond, Washington.
    """

    print("=" * 80)
    print("Advanced Named Entity Linking System v2.0")
    print("Author: BrillConsulting")
    print("=" * 80)
    print(f"\n📄 Sample Text")
    print(sample_text)

    # Method 1: Full Entity Linker (if available)
    if SPACY_AVAILABLE or WIKIPEDIA_AVAILABLE:
        print("\n" + "=" * 80)
        print("Method 1: Full Entity Linking (Wikipedia + Wikidata)")
        print("=" * 80)

        linker = EntityLinker(use_wikipedia=WIKIPEDIA_AVAILABLE, use_wikidata=REQUESTS_AVAILABLE)
        entities = linker.extract_and_link(sample_text, top_n=10)

        print(f"\n🔗 Linked Entities ({len(entities)}):")
        for i, entity in enumerate(entities, 1):
            print(f"\n{i}. {entity['text']} ({entity['label']})")
            if entity.get('wikipedia_url'):
                print(f"   Wikipedia: {entity['wikipedia_url']}")
                print(f"   Summary: {entity.get('wikipedia_summary', 'N/A')[:100]}...")
            if entity.get('wikidata_id'):
                print(f"   Wikidata: {entity['wikidata_id']}")
            if not entity.get('linked'):
                print(f"   Status: Not linked")

    # Method 2: Simple Entity Linker (fallback)
    print("\n" + "=" * 80)
    print("Method 2: Simple Pattern-Based Entity Extraction")
    print("=" * 80)

    simple_linker = SimpleEntityLinker()
    simple_entities = simple_linker.extract_and_link(sample_text)

    print(f"\n🔗 Extracted Entities ({len(simple_entities)}):")
    for i, entity in enumerate(simple_entities, 1):
        print(f"{i:2d}. {entity['text']:30s} ({entity['type']})")

    # Entity Info Retrieval
    if WIKIPEDIA_AVAILABLE:
        print("\n" + "=" * 80)
        print("Entity Information Retrieval")
        print("=" * 80)

        linker = EntityLinker()
        test_entities = ["Apple Inc.", "Steve Jobs", "Microsoft"]

        for entity_name in test_entities:
            print(f"\n🔍 {entity_name}:")
            info = linker.get_entity_info(entity_name)

            if info.get('wikipedia'):
                print(f"   Wikipedia: {info['wikipedia']['url']}")
                print(f"   Summary: {info['wikipedia']['summary'][:100]}...")

            if info.get('wikidata'):
                print(f"   Wikidata: {info['wikidata']['url']}")

    print("\n" + "=" * 80)
    print("✓ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo_entity_linking()
