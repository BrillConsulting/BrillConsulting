# Prompt Engineering Toolkit

Optimize prompts for better LLM performance with templates, few-shot learning, and chain-of-thought techniques.

## Features

- **Prompt Templates**: Reusable templates with variables
- **Few-Shot Learning**: Build prompts with examples
- **Chain-of-Thought**: Step-by-step reasoning prompts
- **System Prompts**: Role and constraint definitions
- **Prompt Optimization**: Generate and test variations
- **A/B Testing**: Compare prompt effectiveness
- **Versioning**: Save and load prompt libraries

## Usage

```python
from prompt_engineering import PromptEngineer

pe = PromptEngineer()

# Create template
pe.create_template("classify", "Classify: {text}", ["text"])
prompt = pe.fill_template("classify", text="I love this!")

# Few-shot learning
examples = [{"input": "...", "output": "..."}]
prompt = pe.few_shot_prompt("Task description", examples, "New input")

# Chain-of-thought
prompt = pe.chain_of_thought_prompt("Question?")

# Compare prompts
comparison = pe.compare_prompts({"v1": "...", "v2": "..."}, "test")
```

## Demo

```bash
python prompt_engineering.py
```
