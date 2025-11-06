# Prompt Engineering Toolkit

> Production-ready prompt engineering system for optimizing LLM performance through advanced prompting techniques, versioning, and A/B testing.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

The Prompt Engineering Toolkit is a comprehensive Python library designed to help developers and researchers optimize their interactions with Large Language Models (LLMs). It provides a structured approach to prompt creation, testing, versioning, and optimization.

### Key Features

- **Advanced Template System**: Create reusable prompt templates with variable substitution and validation
- **Multiple Prompting Strategies**: Support for few-shot, chain-of-thought, tree-of-thought, and ReAct patterns
- **Prompt Versioning**: Track changes to prompts with full changelog and version comparison
- **A/B Testing Framework**: Compare prompt variants with statistical analysis and metrics
- **Template Library**: Built-in collection of professionally crafted prompt templates
- **Prompt Validation**: Automatic quality checks and optimization suggestions
- **Performance Analytics**: Track usage statistics and performance metrics
- **Import/Export**: Save and load prompt collections for team collaboration

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Features](#features)
  - [Template Management](#template-management)
  - [Few-Shot Learning](#few-shot-learning)
  - [Chain-of-Thought](#chain-of-thought)
  - [Tree-of-Thought](#tree-of-thought)
  - [ReAct Pattern](#react-pattern)
  - [Prompt Versioning](#prompt-versioning)
  - [A/B Testing](#ab-testing)
  - [Prompt Validation](#prompt-validation)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)
- [Examples](#examples)
- [Contributing](#contributing)

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/LLM.git
cd LLM/PromptEngineering

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from prompt_engineering import PromptEngineer, PromptType, PromptCategory

# Initialize the prompt engineer
pe = PromptEngineer()

# Use a built-in template
prompt = pe.fill_template(
    "classification",
    item_type="email",
    categories="Spam, Important, Newsletter",
    instructions="Consider sender, subject, and content",
    text="Get 50% off now! Limited time offer!"
)

# Create a custom template
pe.create_template(
    name="sentiment_analysis",
    template="Analyze the sentiment of: {text}\n\nSentiment:",
    variables=["text"],
    prompt_type=PromptType.CLASSIFICATION,
    category=PromptCategory.ANALYSIS
)

# Use few-shot learning
examples = [
    {"input": "I love this!", "output": "Positive"},
    {"input": "Terrible product", "output": "Negative"}
]
few_shot = pe.few_shot_prompt(
    "Classify sentiment:",
    examples,
    "This is amazing!"
)

# Create chain-of-thought prompt
cot = pe.chain_of_thought_prompt(
    "If a train travels 60 mph for 2.5 hours, how far does it go?",
    domain="math"
)

print(cot)
```

## Core Concepts

### Prompt Types

The toolkit supports several prompt engineering patterns:

1. **Standard**: Direct, straightforward prompts
2. **Few-Shot**: Learning from examples
3. **Chain-of-Thought**: Step-by-step reasoning
4. **Tree-of-Thought**: Multi-path reasoning with evaluation
5. **ReAct**: Reasoning and Acting in cycles
6. **System**: Role and constraint definitions for chat models

### Prompt Categories

Organize prompts by use case:

- Classification
- Generation
- Summarization
- Translation
- Extraction
- Reasoning
- Conversation
- Code
- Analysis

## Features

### Template Management

Create and manage reusable prompt templates with variable substitution:

```python
# Create template with validation
pe.create_template(
    name="code_review",
    template="""Review the following {language} code:

Code:
{code}

Focus on:
{focus_areas}

Provide feedback:""",
    variables=["language", "code", "focus_areas"],
    validation_rules={
        "code": lambda x: len(x) > 10,  # Code must be substantial
        "language": lambda x: x in ["Python", "JavaScript", "Java", "Go"]
    },
    tags=["code", "review"]
)

# Use template
prompt = pe.fill_template(
    "code_review",
    language="Python",
    code="def hello(): print('hi')",
    focus_areas="Style, efficiency, best practices"
)
```

### Few-Shot Learning

Create prompts that learn from examples:

```python
examples = [
    {
        "input": "Python is a programming language",
        "reasoning": "Describes a technical topic factually",
        "output": "Technical"
    },
    {
        "input": "I love sunny days!",
        "reasoning": "Expresses personal sentiment",
        "output": "Personal"
    }
]

prompt = pe.few_shot_prompt(
    task_description="Classify the type of statement",
    examples=examples,
    query="Machine learning uses algorithms",
    reasoning=True  # Include reasoning steps
)
```

### Chain-of-Thought

Enable step-by-step reasoning for complex problems:

```python
# Math domain
prompt = pe.chain_of_thought_prompt(
    "A store has 45 items. It sells 60% of them. How many remain?",
    include_example=True,
    domain="math"
)

# Logic domain
prompt = pe.chain_of_thought_prompt(
    "If all A are B, and some B are C, what can we conclude?",
    domain="logic"
)

# General domain
prompt = pe.chain_of_thought_prompt(
    "What happens when you heat water to 100°C at sea level?",
    include_example=True
)
```

### Tree-of-Thought

Explore multiple reasoning paths with evaluation:

```python
prompt = pe.tree_of_thought_prompt(
    problem="Design a scalable database schema for an e-commerce platform",
    num_thoughts=3,  # Generate 3 alternative approaches per step
    num_steps=3,     # Think through 3 steps
    evaluation_criteria=[
        "Scalability to millions of users",
        "Query performance",
        "Data integrity and consistency",
        "Cost effectiveness"
    ]
)
```

### ReAct Pattern

Combine reasoning with actions:

```python
actions = [
    {
        "name": "search_database",
        "description": "Search for information in the database",
        "parameters": "query: string"
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculation",
        "parameters": "expression: string"
    },
    {
        "name": "send_email",
        "description": "Send an email notification",
        "parameters": "to: string, subject: string, body: string"
    }
]

prompt = pe.react_prompt(
    task="Find all orders from last month and calculate total revenue",
    available_actions=actions
)
```

### Prompt Versioning

Track changes to prompts over time:

```python
# Create first version
v1 = pe.version_prompt(
    prompt_name="user_onboarding",
    prompt="Welcome! Please complete your profile: {fields}",
    changelog="Initial version",
    created_by="john@example.com"
)

# Create improved version
v2 = pe.version_prompt(
    prompt_name="user_onboarding",
    prompt="Welcome aboard! Let's personalize your experience. Please share: {fields}\n\nThis helps us serve you better.",
    changelog="Made more friendly and explained purpose",
    created_by="jane@example.com"
)

# View changelog
changelog = pe.get_prompt_changelog("user_onboarding")
print(f"Total versions: {len(changelog)}")
for entry in changelog:
    print(f"{entry['version']} - {entry['changelog']} by {entry['created_by']}")

# Compare versions
comparison = pe.compare_prompt_versions("user_onboarding", "v1.0.0", "v2.0.0")
print(f"Length change: {comparison['diff']['length_change']} characters")
```

### A/B Testing

Test and compare different prompt variants:

```python
# Create experiment
pe.create_ab_test(
    name="product_description_test",
    variants={
        "short": "Describe this product in 2 sentences: {product}",
        "detailed": "Write a compelling product description for: {product}\nInclude features, benefits, and use cases.",
        "persuasive": "Create a persuasive product description that converts:\n{product}\nHighlight unique value proposition."
    },
    success_metrics=["conversion_rate", "engagement", "clarity"],
    description="Test different approaches to product descriptions"
)

# Record test results (in production, connect to your LLM)
pe.ab_testing.record_result(
    experiment_name="product_description_test",
    variant="short",
    test_input="Wireless headphones with noise cancellation",
    output="Premium wireless headphones...",
    latency_ms=250,
    token_count=45,
    success=True,
    metrics={"conversion_rate": 0.12, "engagement": 0.85, "clarity": 0.90}
)

pe.ab_testing.record_result(
    experiment_name="product_description_test",
    variant="detailed",
    test_input="Wireless headphones with noise cancellation",
    output="Experience premium audio quality...",
    latency_ms=420,
    token_count=120,
    success=True,
    metrics={"conversion_rate": 0.18, "engagement": 0.92, "clarity": 0.88}
)

# Analyze results
analysis = pe.analyze_ab_test("product_description_test")
print(f"Winner: {analysis['winner']}")
print(f"Success rate: {analysis['winner_stats']['success_rate']:.2%}")
print(f"Avg latency: {analysis['winner_stats']['avg_latency_ms']:.0f}ms")
```

### Prompt Validation

Automatically validate prompt quality:

```python
prompt = """Analyze customer feedback and provide insights:
1. Sentiment analysis
2. Key themes
3. Actionable recommendations

Feedback: {feedback_text}"""

validation = pe.validate_prompt(prompt)

print(f"Valid: {validation['valid']}")
print(f"Quality Score: {validation['score']}/100")
print(f"Estimated tokens: {validation['statistics']['estimated_tokens']}")

if validation['warnings']:
    print("Warnings:")
    for warning in validation['warnings']:
        print(f"  - {warning}")

if validation['issues']:
    print("Issues:")
    for issue in validation['issues']:
        print(f"  - {issue}")
```

### Prompt Optimization

Generate improved variations:

```python
base_prompt = "Explain machine learning"

# Apply optimization strategies
variations = pe.optimize_prompt(
    base_prompt,
    improvements=[
        "be_specific",
        "use_examples",
        "step_by_step",
        "consider_alternatives"
    ]
)

print(f"Generated {len(variations)} variations")
for i, variant in enumerate(variations):
    print(f"\nVariation {i+1}:")
    print(variant)

# Add custom improvements
custom_improvements = {
    "add_context": "Provide relevant context and background information.",
    "target_audience": "Tailor your explanation for beginners."
}

variations = pe.optimize_prompt(
    base_prompt,
    ["be_specific", "add_context", "target_audience"],
    custom_improvements=custom_improvements
)
```

## API Reference

### PromptEngineer

Main class for prompt engineering operations.

#### Methods

- `create_template(name, template, variables, ...)` - Create a new template
- `fill_template(name, **kwargs)` - Fill template with values
- `search_templates(query, category, prompt_type, tags)` - Search templates
- `few_shot_prompt(task_description, examples, query, reasoning)` - Create few-shot prompt
- `chain_of_thought_prompt(question, include_example, domain)` - Create CoT prompt
- `tree_of_thought_prompt(problem, num_thoughts, num_steps, evaluation_criteria)` - Create ToT prompt
- `react_prompt(task, available_actions, examples)` - Create ReAct prompt
- `system_prompt(role, constraints, style, output_format, examples)` - Create system prompt
- `optimize_prompt(base_prompt, improvements, custom_improvements)` - Generate variations
- `validate_prompt(prompt)` - Validate prompt quality
- `version_prompt(prompt_name, prompt, changelog, created_by)` - Create version
- `get_prompt_changelog(prompt_name)` - Get version history
- `compare_prompt_versions(prompt_name, version1, version2)` - Compare versions
- `create_ab_test(name, variants, success_metrics, description)` - Create A/B test
- `analyze_ab_test(experiment_name)` - Analyze test results
- `export_templates(filepath, format)` - Export templates
- `import_templates(filepath)` - Import templates
- `save_state(filepath)` - Save complete state
- `load_state(filepath)` - Load complete state

### PromptTemplate

Advanced template class with validation.

#### Attributes

- `name` - Template name
- `template` - Template string
- `variables` - Required variables
- `prompt_type` - Type of prompt
- `category` - Category for organization
- `description` - Template description
- `tags` - Search tags
- `validation_rules` - Variable validation functions
- `usage_count` - Number of times used

#### Methods

- `render(**kwargs)` - Render template with values
- `validate_variables(**kwargs)` - Validate variable values
- `to_dict()` - Convert to dictionary

## Best Practices

### 1. Template Design

- Use clear, descriptive variable names
- Include instructions and context
- Add examples when appropriate
- Structure output format requirements

### 2. Few-Shot Learning

- Use 3-5 high-quality examples
- Ensure examples cover edge cases
- Balance example complexity
- Include reasoning when needed

### 3. Chain-of-Thought

- Choose appropriate domain examples
- Break down complex problems
- Guide the reasoning process
- Include verification steps

### 4. Versioning

- Write clear changelogs
- Track who made changes
- Compare versions before deployment
- Keep performance metrics

### 5. A/B Testing

- Test one variable at a time
- Collect sufficient data
- Define success metrics upfront
- Document findings

### 6. Prompt Validation

- Always validate before deployment
- Address all issues
- Review warnings
- Monitor token counts

## Examples

### Example 1: Code Review System

```python
pe = PromptEngineer()

# Create specialized template
pe.create_template(
    name="security_review",
    template="""Perform security review of {language} code:

Code:
{code}

Check for:
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication issues
- Data exposure risks
- Input validation problems

Severity levels: Critical, High, Medium, Low

Findings:""",
    variables=["language", "code"],
    category=PromptCategory.CODE,
    tags=["security", "review"]
)

# Use template
code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    return db.execute(query)
"""

prompt = pe.fill_template("security_review", language="Python", code=code)
# Send to LLM for analysis
```

### Example 2: Customer Support Automation

```python
pe = PromptEngineer()

# Create ReAct prompt for support agent
actions = [
    {"name": "search_kb", "description": "Search knowledge base", "parameters": "query"},
    {"name": "check_order", "description": "Check order status", "parameters": "order_id"},
    {"name": "create_ticket", "description": "Create support ticket", "parameters": "issue"}
]

prompt = pe.react_prompt(
    task="Customer asks: 'My order #12345 hasn't arrived. When will it ship?'",
    available_actions=actions
)

# Use prompt with LLM to guide support workflow
```

### Example 3: Content Generation Pipeline

```python
pe = PromptEngineer()

# Version 1: Basic
v1 = pe.version_prompt(
    "blog_intro",
    "Write introduction for: {topic}",
    "Initial version",
    "content_team"
)

# Version 2: Enhanced
v2 = pe.version_prompt(
    "blog_intro",
    """Write engaging blog introduction for: {topic}

Requirements:
- Hook readers in first sentence
- Establish credibility
- Preview key points
- 2-3 paragraphs
- Target audience: {audience}

Introduction:""",
    "Added requirements and structure",
    "content_team"
)

# A/B test different approaches
pe.create_ab_test(
    "intro_style_test",
    variants={
        "question": "Start with thought-provoking question: {topic}",
        "story": "Begin with brief story or anecdote: {topic}",
        "statistic": "Open with compelling statistic: {topic}"
    },
    success_metrics=["engagement", "readability", "click_through"]
)
```

## Demo

Run the comprehensive demo to see all features in action:

```bash
python prompt_engineering.py
```

The demo showcases:
- Template library usage
- All prompting strategies
- Versioning workflow
- A/B testing framework
- Validation and optimization
- Import/export functionality
- Performance analytics

## Architecture

```
PromptEngineer
├── TemplateLibrary (Built-in templates)
├── PromptVersionManager (Version control)
├── ABTestingFramework (A/B testing)
└── Templates (User templates)

Supporting Classes:
├── PromptTemplate (Template with validation)
├── TreeOfThought (ToT implementation)
├── ReActPattern (ReAct implementation)
├── PromptVersion (Version metadata)
└── TestResult (A/B test results)
```

## Performance Considerations

- **Template Caching**: Templates are loaded once and reused
- **Lazy Loading**: Built-in templates loaded on initialization
- **Efficient Storage**: JSON-based serialization for persistence
- **Memory Management**: Minimal memory footprint
- **Token Estimation**: Quick token count approximation

## Integration Examples

### With OpenAI

```python
import openai
from prompt_engineering import PromptEngineer

pe = PromptEngineer()
prompt = pe.chain_of_thought_prompt("Calculate compound interest...")

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)
```

### With LangChain

```python
from langchain import PromptTemplate as LCPromptTemplate
from prompt_engineering import PromptEngineer

pe = PromptEngineer()
our_prompt = pe.fill_template("classification", ...)

lc_template = LCPromptTemplate.from_template(our_prompt)
```

### With Custom LLM API

```python
import requests
from prompt_engineering import PromptEngineer

pe = PromptEngineer()
prompt = pe.fill_template("code_generation", ...)

response = requests.post(
    "https://api.yourlm.com/v1/generate",
    json={"prompt": prompt, "max_tokens": 500}
)
```

## Troubleshooting

### Common Issues

1. **Template not found**: Ensure template name is correct and template was created
2. **Validation errors**: Check all required variables are provided
3. **Import errors**: Verify JSON format and version compatibility
4. **Performance issues**: Use template search to find appropriate templates instead of creating many similar ones

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

pe = PromptEngineer()
# Debug output will show template operations
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [GitHub Wiki](https://github.com/BrillConsulting/LLM/wiki)
- Issues: [GitHub Issues](https://github.com/BrillConsulting/LLM/issues)
- Email: support@brillconsulting.com

## Acknowledgments

- Inspired by research in prompt engineering and LLM optimization
- Built on best practices from the ML community
- Thanks to all contributors

## Changelog

### Version 2.0.0 (Current)
- Added Tree-of-Thought prompting
- Added ReAct pattern implementation
- Enhanced version management
- Improved A/B testing framework
- Added template library
- Added prompt validation
- Added performance analytics

### Version 1.0.0
- Initial release
- Basic template system
- Few-shot learning
- Chain-of-thought prompting
- Simple versioning

---

**Built with care by Brill Consulting** | [Website](https://brillconsulting.com) | [GitHub](https://github.com/BrillConsulting)
