"""
Advanced Question Answering System v2.0
Author: BrillConsulting
Description: Extractive & generative QA with BERT, RoBERTa, and T5
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️ Install transformers: pip install transformers torch")


class QuestionAnsweringSystem:
    """Advanced QA system with extractive and generative models"""

    def __init__(self, model_name='distilbert-base-cased-distilled-squad', qa_type='extractive'):
        """
        Initialize QA system

        Args:
            model_name: HuggingFace model name
            qa_type: 'extractive' or 'generative'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")

        self.model_name = model_name
        self.qa_type = qa_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 Loading {qa_type} QA model: {model_name}")

        if qa_type == 'extractive':
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
        elif qa_type == 'generative':
            self.qa_pipeline = pipeline(
                "text2text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )

        print("✅ Model loaded successfully\n")

    def answer_question(self, question: str, context: str, top_k=1) -> Dict:
        """
        Answer a question given context

        Args:
            question: Question to answer
            context: Context containing the answer
            top_k: Number of answers to return

        Returns:
            Answer with confidence score
        """
        if self.qa_type == 'extractive':
            if top_k == 1:
                result = self.qa_pipeline(question=question, context=context)
                return {
                    'answer': result['answer'],
                    'confidence': result['score'],
                    'start': result['start'],
                    'end': result['end'],
                    'type': 'extractive'
                }
            else:
                results = self.qa_pipeline(
                    question=question,
                    context=context,
                    top_k=top_k
                )
                return {
                    'answers': [
                        {
                            'answer': r['answer'],
                            'confidence': r['score'],
                            'start': r['start'],
                            'end': r['end']
                        }
                        for r in results
                    ],
                    'type': 'extractive'
                }

        elif self.qa_type == 'generative':
            prompt = f"question: {question} context: {context}"
            result = self.qa_pipeline(prompt, max_length=100, num_return_sequences=top_k)

            if top_k == 1:
                return {
                    'answer': result[0]['generated_text'],
                    'type': 'generative'
                }
            else:
                return {
                    'answers': [r['generated_text'] for r in result],
                    'type': 'generative'
                }

    def answer_batch(self, qa_pairs: List[Dict]) -> List[Dict]:
        """
        Answer multiple questions

        Args:
            qa_pairs: List of {'question': str, 'context': str}

        Returns:
            List of answers
        """
        results = []
        for pair in qa_pairs:
            result = self.answer_question(pair['question'], pair['context'])
            result['question'] = pair['question']
            results.append(result)
        return results

    def evaluate(self, qa_dataset: List[Dict]) -> Dict:
        """
        Evaluate on QA dataset with ground truth

        Args:
            qa_dataset: List of {'question': str, 'context': str, 'answer': str}

        Returns:
            Evaluation metrics
        """
        correct = 0
        total = len(qa_dataset)
        confidences = []

        print(f"📊 Evaluating on {total} examples...")

        for idx, item in enumerate(qa_dataset):
            result = self.answer_question(item['question'], item['context'])

            if self.qa_type == 'extractive':
                pred_answer = result['answer'].lower().strip()
                true_answer = item['answer'].lower().strip()

                # Exact match or containment
                if pred_answer == true_answer or pred_answer in true_answer or true_answer in pred_answer:
                    correct += 1

                confidences.append(result['confidence'])

            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{total}")

        accuracy = correct / total * 100

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_confidence': np.mean(confidences) if confidences else 0
        }


class MultiDocumentQA:
    """QA system that searches across multiple documents"""

    def __init__(self, model_name='distilbert-base-cased-distilled-squad'):
        """Initialize multi-document QA"""
        self.qa_system = QuestionAnsweringSystem(model_name=model_name)
        self.documents = []

    def add_documents(self, documents: List[str]):
        """Add documents to search corpus"""
        self.documents.extend(documents)
        print(f"📚 Added {len(documents)} documents (total: {len(self.documents)})")

    def answer_from_documents(self, question: str, top_k=3) -> Dict:
        """
        Answer question by searching all documents

        Args:
            question: Question to answer
            top_k: Number of candidate answers

        Returns:
            Best answer with source document
        """
        if not self.documents:
            return {'error': 'No documents available'}

        # Get answers from all documents
        candidates = []

        for doc_idx, doc in enumerate(self.documents):
            try:
                result = self.qa_system.answer_question(question, doc)
                result['doc_idx'] = doc_idx
                result['context'] = doc
                candidates.append(result)
            except:
                continue

        if not candidates:
            return {'error': 'No answer found'}

        # Sort by confidence
        candidates.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        return {
            'question': question,
            'best_answer': candidates[0],
            'alternative_answers': candidates[1:top_k]
        }


def main():
    parser = argparse.ArgumentParser(description='Advanced Question Answering v2.0')
    parser.add_argument('--question', type=str, help='Question to answer')
    parser.add_argument('--context', type=str, help='Context text')
    parser.add_argument('--context-file', type=str, help='File with context')
    parser.add_argument('--qa-file', type=str, help='JSON file with QA pairs')
    parser.add_argument('--model', type=str,
                       default='distilbert-base-cased-distilled-squad',
                       help='Model name')
    parser.add_argument('--qa-type', type=str, default='extractive',
                       choices=['extractive', 'generative'],
                       help='QA type')
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of answers')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate on dataset')

    args = parser.parse_args()

    # Initialize QA system
    qa_system = QuestionAnsweringSystem(model_name=args.model, qa_type=args.qa_type)

    # Single QA
    if args.question and (args.context or args.context_file):
        context = args.context
        if args.context_file:
            with open(args.context_file, 'r', encoding='utf-8') as f:
                context = f.read()

        print(f"❓ Question: {args.question}")
        print(f"📄 Context: {context[:200]}...\n")

        result = qa_system.answer_question(args.question, context, top_k=args.top_k)

        print("=" * 60)
        if args.top_k == 1:
            print(f"✅ Answer: {result['answer']}")
            if 'confidence' in result:
                print(f"📊 Confidence: {result['confidence']:.4f}")
            if 'start' in result:
                print(f"📍 Position: {result['start']}-{result['end']}")
        else:
            print(f"✅ Top {args.top_k} Answers:")
            for idx, ans in enumerate(result['answers']):
                if isinstance(ans, dict):
                    print(f"\n{idx+1}. {ans['answer']} (conf: {ans['confidence']:.4f})")
                else:
                    print(f"\n{idx+1}. {ans}")
        print("=" * 60)

    # Batch QA from file
    elif args.qa_file:
        with open(args.qa_file, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)

        if args.evaluate:
            # Evaluate
            metrics = qa_system.evaluate(qa_data)

            print("\n" + "=" * 60)
            print("📊 EVALUATION RESULTS")
            print("=" * 60)
            print(f"Accuracy: {metrics['accuracy']:.2f}%")
            print(f"Correct: {metrics['correct']}/{metrics['total']}")
            if metrics['avg_confidence'] > 0:
                print(f"Avg Confidence: {metrics['avg_confidence']:.4f}")
            print("=" * 60)
        else:
            # Answer
            results = qa_system.answer_batch(qa_data)

            print(f"\n📊 Answered {len(results)} questions\n")
            for idx, result in enumerate(results[:5]):
                print(f"{idx+1}. Q: {result['question']}")
                print(f"   A: {result['answer']}")
                if 'confidence' in result:
                    print(f"   Confidence: {result['confidence']:.4f}")
                print()

    else:
        print("❌ Provide --question and --context, or --qa-file")


if __name__ == "__main__":
    main()
