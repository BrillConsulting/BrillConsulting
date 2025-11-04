"""
Prompt Engineering Toolkit
===========================

Optimize prompts for better LLM performance:
- Prompt templates and patterns
- Few-shot learning examples
- Chain-of-thought prompting
- Prompt optimization and testing
- A/B testing for prompts
- Prompt versioning

Author: Brill Consulting
"""

from typing import List, Dict, Optional
import json


class PromptEngineer:
    """Prompt engineering and optimization toolkit."""

    def __init__(self):
        """Initialize prompt engineer."""
        self.prompts = {}
        self.templates = {}

    def create_template(self, name: str, template: str, variables: List[str]):
        """
        Create reusable prompt template.

        Args:
            name: Template name
            template: Template string with {placeholders}
            variables: List of variable names
        """
        self.templates[name] = {
            "template": template,
            "variables": variables
        }

    def fill_template(self, name: str, **kwargs) -> str:
        """
        Fill template with values.

        Args:
            name: Template name
            **kwargs: Variable values

        Returns:
            Filled prompt
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        template = self.templates[name]["template"]
        return template.format(**kwargs)

    def few_shot_prompt(self, task_description: str, examples: List[Dict],
                       query: str) -> str:
        """
        Create few-shot learning prompt.

        Args:
            task_description: Description of the task
            examples: List of {input, output} examples
            query: New input to process

        Returns:
            Few-shot prompt
        """
        prompt = f"{task_description}\n\n"

        # Add examples
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"
            prompt += f"Output: {ex['output']}\n\n"

        # Add query
        prompt += f"Now solve this:\n"
        prompt += f"Input: {query}\n"
        prompt += f"Output:"

        return prompt

    def chain_of_thought_prompt(self, question: str,
                                include_example: bool = True) -> str:
        """
        Create chain-of-thought prompt.

        Args:
            question: Question to answer
            include_example: Whether to include CoT example

        Returns:
            CoT prompt
        """
        prompt = ""

        if include_example:
            prompt += "Let's solve this step by step:\n\n"
            prompt += "Example:\n"
            prompt += "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?\n"
            prompt += "A: Let's think step by step:\n"
            prompt += "1. Roger starts with 5 balls\n"
            prompt += "2. He buys 2 cans with 3 balls each = 2 × 3 = 6 balls\n"
            prompt += "3. Total = 5 + 6 = 11 balls\n"
            prompt += "Answer: 11\n\n"

        prompt += f"Now solve this step by step:\n"
        prompt += f"Q: {question}\n"
        prompt += f"A: Let's think step by step:\n"

        return prompt

    def system_prompt(self, role: str, constraints: Optional[List[str]] = None,
                     style: Optional[str] = None) -> str:
        """
        Create system prompt for chat models.

        Args:
            role: Role description
            constraints: List of constraints/rules
            style: Communication style

        Returns:
            System prompt
        """
        prompt = f"You are {role}.\n\n"

        if constraints:
            prompt += "Follow these rules:\n"
            for i, constraint in enumerate(constraints, 1):
                prompt += f"{i}. {constraint}\n"
            prompt += "\n"

        if style:
            prompt += f"Communication style: {style}\n"

        return prompt.strip()

    def optimize_prompt(self, base_prompt: str,
                       improvements: List[str]) -> List[str]:
        """
        Generate prompt variations for testing.

        Args:
            base_prompt: Original prompt
            improvements: List of improvement strategies

        Returns:
            List of prompt variations
        """
        variations = [base_prompt]

        strategies = {
            "be_specific": "Be specific and detailed in your response.",
            "use_examples": "Provide concrete examples to illustrate your points.",
            "step_by_step": "Break down your reasoning step by step.",
            "cite_sources": "Cite your sources and reasoning.",
            "be_concise": "Be concise and to the point."
        }

        for improvement in improvements:
            if improvement in strategies:
                variation = f"{base_prompt}\n\n{strategies[improvement]}"
                variations.append(variation)

        return variations

    def compare_prompts(self, prompts: Dict[str, str],
                       test_query: str) -> Dict:
        """
        Compare different prompt versions.

        Args:
            prompts: Dict of {name: prompt} to compare
            test_query: Test input

        Returns:
            Comparison report
        """
        report = {
            "test_query": test_query,
            "prompts_tested": len(prompts),
            "results": {}
        }

        for name, prompt in prompts.items():
            filled_prompt = f"{prompt}\n\nQuery: {test_query}"

            # Simulate metrics (in production, call actual LLM)
            report["results"][name] = {
                "prompt": prompt[:100] + "...",
                "length": len(filled_prompt),
                "estimated_tokens": len(filled_prompt) // 4
            }

        return report

    def save_prompts(self, filepath: str):
        """Save all prompts and templates."""
        data = {
            "templates": self.templates,
            "prompts": self.prompts
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✓ Saved prompts to {filepath}")

    def load_prompts(self, filepath: str):
        """Load prompts and templates."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.templates = data.get("templates", {})
        self.prompts = data.get("prompts", {})

        print(f"✓ Loaded prompts from {filepath}")


def demo():
    """Demo prompt engineering."""
    print("Prompt Engineering Demo")
    print("="*50)

    pe = PromptEngineer()

    # 1. Create templates
    print("\n1. Creating Prompt Templates")
    print("-"*50)

    pe.create_template(
        name="classification",
        template="Classify the following {item_type} into one of these categories: {categories}.\n\n{item_type}: {text}\n\nCategory:",
        variables=["item_type", "categories", "text"]
    )

    prompt = pe.fill_template(
        "classification",
        item_type="email",
        categories="Spam, Important, Newsletter",
        text="Get 50% off now!"
    )
    print(prompt)

    # 2. Few-shot learning
    print("\n2. Few-Shot Learning Prompt")
    print("-"*50)

    examples = [
        {"input": "The movie was amazing!", "output": "Positive"},
        {"input": "Terrible experience.", "output": "Negative"},
        {"input": "It was okay.", "output": "Neutral"}
    ]

    few_shot = pe.few_shot_prompt(
        "Classify the sentiment of the following text.",
        examples,
        "I love this product!"
    )
    print(few_shot)

    # 3. Chain-of-thought
    print("\n3. Chain-of-Thought Prompt")
    print("-"*50)

    cot = pe.chain_of_thought_prompt(
        "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?",
        include_example=True
    )
    print(cot)

    # 4. System prompt
    print("\n4. System Prompt")
    print("-"*50)

    system = pe.system_prompt(
        role="a helpful data science tutor",
        constraints=[
            "Always explain concepts clearly",
            "Use examples to illustrate",
            "Be encouraging and supportive"
        ],
        style="Friendly and educational"
    )
    print(system)

    # 5. Prompt optimization
    print("\n5. Prompt Optimization")
    print("-"*50)

    base = "Explain machine learning."
    variations = pe.optimize_prompt(
        base,
        ["be_specific", "use_examples", "step_by_step"]
    )

    print(f"Generated {len(variations)} variations:")
    for i, var in enumerate(variations[:2], 1):
        print(f"\nVariation {i}:")
        print(var[:100] + "...")

    # 6. Prompt comparison
    print("\n6. Comparing Prompts")
    print("-"*50)

    prompts_to_compare = {
        "simple": "What is Python?",
        "detailed": "Explain what Python is, including its main features and use cases.",
        "structured": "Describe Python:\n1. Definition\n2. Key features\n3. Common applications"
    }

    comparison = pe.compare_prompts(prompts_to_compare, "Python programming")
    print(json.dumps(comparison, indent=2))

    # 7. Save prompts
    print("\n7. Saving Prompts")
    print("-"*50)
    pe.save_prompts("prompts.json")

    print("\n✓ Prompt Engineering Demo Complete!")


if __name__ == '__main__':
    demo()
