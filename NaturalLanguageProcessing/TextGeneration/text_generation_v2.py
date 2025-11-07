"""
Advanced Text Generation System v2.0
Author: BrillConsulting
Description: Creative text generation with GPT-2, GPT-Neo, and controlled generation
"""

from typing import List, Dict
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, set_seed
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Install: pip install transformers torch")


class TextGenerator:
    """Advanced text generation with transformers"""

    def __init__(self, model_name='gpt2'):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Loading generation model: {model_name}")
        self.generator = pipeline("text-generation", model=model_name, device=0 if torch.cuda.is_available() else -1)
        print("✅ Model loaded\n")

    def generate(self, prompt: str, max_length=100, num_return_sequences=1, temperature=1.0, top_k=50, top_p=0.95, do_sample=True) -> Dict:
        results = self.generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences, temperature=temperature, top_k=top_k, top_p=top_p, do_sample=do_sample, pad_token_id=50256)
        return {'prompt': prompt, 'generations': [r['generated_text'] for r in results], 'num_sequences': num_return_sequences}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced Text Generation v2.0')
    parser.add_argument('--prompt', type=str, required=True, help='Generation prompt')
    parser.add_argument('--model', type=str, default='gpt2', help='Model name')
    parser.add_argument('--max-length', type=int, default=100, help='Max length')
    args = parser.parse_args()
    generator = TextGenerator(model_name=args.model)
    result = generator.generate(args.prompt, max_length=args.max_length)
    print("=" * 60)
    print(f"📝 GENERATED TEXT")
    print("=" * 60)
    for idx, text in enumerate(result['generations']):
        print(f"\n{idx+1}. {text}\n")
