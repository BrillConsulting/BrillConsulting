"""
Advanced Machine Translation System v2.0
Author: BrillConsulting
Description: Neural machine translation with MarianMT and M2M100
"""

from typing import List, Dict
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, MarianMTModel, MarianTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Install: pip install transformers torch sentencepiece")


class MachineTranslator:
    """Advanced neural machine translation"""

    LANGUAGE_PAIRS = {
        'en-es': 'Helsinki-NLP/opus-mt-en-es',
        'en-fr': 'Helsinki-NLP/opus-mt-en-fr',
        'en-de': 'Helsinki-NLP/opus-mt-en-de',
        'en-zh': 'Helsinki-NLP/opus-mt-en-zh',
        'es-en': 'Helsinki-NLP/opus-mt-es-en',
        'fr-en': 'Helsinki-NLP/opus-mt-fr-en',
        'de-en': 'Helsinki-NLP/opus-mt-de-en',
    }

    def __init__(self, source_lang='en', target_lang='es'):
        """
        Initialize translator

        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")

        self.source_lang = source_lang
        self.target_lang = target_lang
        lang_pair = f"{source_lang}-{target_lang}"

        if lang_pair in self.LANGUAGE_PAIRS:
            model_name = self.LANGUAGE_PAIRS[lang_pair]
        else:
            model_name = 'facebook/m2m100_418M'

        print(f"🔧 Loading translation model: {lang_pair}")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.translator = pipeline(
            "translation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✅ Model loaded\n")

    def translate(self, text: str, max_length=512) -> Dict:
        """
        Translate text

        Args:
            text: Text to translate
            max_length: Maximum translation length

        Returns:
            Translation with metadata
        """
        result = self.translator(text, max_length=max_length)[0]

        return {
            'translation': result['translation_text'],
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'source_text': text
        }

    def translate_batch(self, texts: List[str]) -> List[Dict]:
        """Translate multiple texts"""
        return [self.translate(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description='Advanced Machine Translation v2.0')
    parser.add_argument('--text', type=str, help='Text to translate')
    parser.add_argument('--file', type=str, help='File with texts')
    parser.add_argument('--source', type=str, default='en', help='Source language')
    parser.add_argument('--target', type=str, default='es', help='Target language')

    args = parser.parse_args()

    # Get text
    texts = []
    if args.text:
        texts = [args.text]
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        print("❌ Provide --text or --file")
        return

    # Initialize
    translator = MachineTranslator(source_lang=args.source, target_lang=args.target)

    # Translate
    results = translator.translate_batch(texts)

    # Display
    print("=" * 60)
    print("🌐 TRANSLATIONS")
    print("=" * 60)
    for idx, result in enumerate(results):
        print(f"\n{idx+1}. [{result['source_lang']}] {result['source_text']}")
        print(f"   [{result['target_lang']}] {result['translation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
