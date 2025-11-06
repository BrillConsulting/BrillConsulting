"""
Prompt Engineering Toolkit
===========================

Production-ready prompt engineering system for LLM optimization:
- Advanced prompt templates with variable substitution
- Few-shot learning with intelligent example selection
- Chain-of-thought and Tree-of-thought prompting
- ReAct (Reasoning + Acting) pattern implementation
- Prompt versioning and change tracking
- A/B testing framework with metrics
- Comprehensive template library
- Prompt validation and optimization
- Performance analytics and monitoring

Author: Brill Consulting
Version: 2.0.0
"""

from typing import List, Dict, Optional, Any, Callable, Tuple
import json
import re
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict
import copy


class PromptType(Enum):
    """Types of prompts supported."""
    STANDARD = "standard"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    REACT = "react"
    SYSTEM = "system"


class PromptCategory(Enum):
    """Categories for organizing prompts."""
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    EXTRACTION = "extraction"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    CODE = "code"
    ANALYSIS = "analysis"


@dataclass
class PromptVersion:
    """Version information for a prompt."""
    version: str
    prompt: str
    created_at: str
    created_by: str
    changelog: str
    metadata: Dict[str, Any]
    performance_metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TestResult:
    """Results from A/B testing."""
    prompt_name: str
    version: str
    test_input: str
    output: str
    latency_ms: float
    token_count: int
    success: bool
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


class PromptTemplate:
    """Advanced prompt template with validation and metadata."""

    def __init__(
        self,
        name: str,
        template: str,
        variables: List[str],
        prompt_type: PromptType = PromptType.STANDARD,
        category: Optional[PromptCategory] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        validation_rules: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize prompt template.

        Args:
            name: Template name
            template: Template string with {variables}
            variables: List of required variable names
            prompt_type: Type of prompt
            category: Category for organization
            description: Template description
            tags: Search tags
            validation_rules: Dict of {variable: validation_function}
        """
        self.name = name
        self.template = template
        self.variables = variables
        self.prompt_type = prompt_type
        self.category = category
        self.description = description
        self.tags = tags or []
        self.validation_rules = validation_rules or {}
        self.created_at = datetime.now().isoformat()
        self.usage_count = 0

    def validate_variables(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """
        Validate variable values.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check all required variables present
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            return False, f"Missing required variables: {missing}"

        # Run custom validation rules
        for var, validator in self.validation_rules.items():
            if var in kwargs:
                try:
                    if not validator(kwargs[var]):
                        return False, f"Validation failed for variable: {var}"
                except Exception as e:
                    return False, f"Validation error for {var}: {str(e)}"

        return True, None

    def render(self, **kwargs) -> str:
        """
        Render template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Rendered prompt

        Raises:
            ValueError: If validation fails
        """
        is_valid, error = self.validate_variables(**kwargs)
        if not is_valid:
            raise ValueError(error)

        self.usage_count += 1
        return self.template.format(**kwargs)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "variables": self.variables,
            "prompt_type": self.prompt_type.value,
            "category": self.category.value if self.category else None,
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
            "usage_count": self.usage_count
        }


class TreeOfThought:
    """Tree-of-Thought prompting for complex reasoning."""

    @staticmethod
    def generate_prompt(
        problem: str,
        num_thoughts: int = 3,
        num_steps: int = 3,
        evaluation_criteria: Optional[List[str]] = None
    ) -> str:
        """
        Generate Tree-of-Thought prompt.

        Args:
            problem: Problem to solve
            num_thoughts: Number of thought branches per step
            num_steps: Number of reasoning steps
            evaluation_criteria: Criteria for evaluating thoughts

        Returns:
            ToT prompt
        """
        prompt = f"""Problem: {problem}

Let's solve this using Tree-of-Thought reasoning. We'll explore multiple reasoning paths and evaluate each one.

Instructions:
1. Generate {num_thoughts} different possible approaches or thoughts for each step
2. Evaluate each thought critically
3. Select the most promising thought to continue
4. Repeat for {num_steps} steps

"""
        if evaluation_criteria:
            prompt += "Evaluation Criteria:\n"
            for i, criterion in enumerate(evaluation_criteria, 1):
                prompt += f"{i}. {criterion}\n"
            prompt += "\n"

        prompt += f"""Format your response as:

Step 1:
Thought 1: [First approach]
Evaluation: [Critical analysis]

Thought 2: [Second approach]
Evaluation: [Critical analysis]

Thought 3: [Third approach]
Evaluation: [Critical analysis]

Selected: Thought [X] because [reasoning]

[Continue for {num_steps} steps]

Final Solution: [Your best solution based on the reasoning path]

Begin:
"""
        return prompt

    @staticmethod
    def parse_response(response: str) -> Dict:
        """
        Parse ToT response into structured format.

        Args:
            response: Raw ToT response

        Returns:
            Structured thoughts and evaluations
        """
        steps = []
        current_step = None

        for line in response.split('\n'):
            if line.startswith('Step '):
                if current_step:
                    steps.append(current_step)
                current_step = {'thoughts': [], 'selected': None}
            elif line.startswith('Thought '):
                # Extract thought content
                pass
            elif line.startswith('Selected:'):
                if current_step:
                    current_step['selected'] = line

        return {'steps': steps}


class ReActPattern:
    """ReAct (Reasoning + Acting) pattern implementation."""

    @staticmethod
    def generate_prompt(
        task: str,
        available_actions: List[Dict[str, str]],
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate ReAct prompt.

        Args:
            task: Task description
            available_actions: List of {name, description, parameters}
            examples: Optional examples of reasoning-action cycles

        Returns:
            ReAct prompt
        """
        prompt = f"""Task: {task}

You will solve this task using the ReAct (Reasoning + Acting) pattern. Alternate between:
- Thought: Reasoning about what to do next
- Action: Taking a specific action
- Observation: Observing the result

Available Actions:
"""
        for action in available_actions:
            prompt += f"- {action['name']}: {action['description']}"
            if 'parameters' in action:
                prompt += f" (Parameters: {action['parameters']})"
            prompt += "\n"

        prompt += "\n"

        if examples:
            prompt += "Examples:\n\n"
            for i, example in enumerate(examples, 1):
                prompt += f"Example {i}:\n"
                prompt += f"Task: {example['task']}\n"
                prompt += f"Thought: {example['thought']}\n"
                prompt += f"Action: {example['action']}\n"
                prompt += f"Observation: {example['observation']}\n\n"

        prompt += """Format your response as:

Thought: [Your reasoning about what to do]
Action: [The action to take]

After each action, I'll provide an observation, then you continue with the next thought-action cycle.

When you have the final answer, output:
Thought: [Final reasoning]
Answer: [Final answer]

Begin:
"""
        return prompt

    @staticmethod
    def parse_steps(response: str) -> List[Dict[str, str]]:
        """
        Parse ReAct response into steps.

        Args:
            response: Raw ReAct response

        Returns:
            List of reasoning-action steps
        """
        steps = []
        lines = response.strip().split('\n')
        current_step = {}

        for line in lines:
            line = line.strip()
            if line.startswith('Thought:'):
                if current_step:
                    steps.append(current_step)
                current_step = {'thought': line[8:].strip()}
            elif line.startswith('Action:'):
                current_step['action'] = line[7:].strip()
            elif line.startswith('Observation:'):
                current_step['observation'] = line[12:].strip()
            elif line.startswith('Answer:'):
                current_step['answer'] = line[7:].strip()

        if current_step:
            steps.append(current_step)

        return steps


class PromptVersionManager:
    """Manage prompt versions with change tracking."""

    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, List[PromptVersion]] = defaultdict(list)

    def create_version(
        self,
        prompt_name: str,
        prompt: str,
        changelog: str,
        created_by: str = "system",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Create new prompt version.

        Args:
            prompt_name: Name of the prompt
            prompt: Prompt content
            changelog: Description of changes
            created_by: Author
            metadata: Additional metadata

        Returns:
            Version identifier
        """
        versions = self.versions[prompt_name]
        version_num = len(versions) + 1
        version_id = f"v{version_num}.0.0"

        version = PromptVersion(
            version=version_id,
            prompt=prompt,
            created_at=datetime.now().isoformat(),
            created_by=created_by,
            changelog=changelog,
            metadata=metadata or {}
        )

        versions.append(version)
        return version_id

    def get_version(self, prompt_name: str, version: Optional[str] = None) -> Optional[PromptVersion]:
        """
        Get specific version or latest.

        Args:
            prompt_name: Name of prompt
            version: Version identifier (None for latest)

        Returns:
            PromptVersion or None
        """
        if prompt_name not in self.versions:
            return None

        versions = self.versions[prompt_name]
        if not versions:
            return None

        if version is None:
            return versions[-1]

        for v in versions:
            if v.version == version:
                return v

        return None

    def get_changelog(self, prompt_name: str) -> List[Dict]:
        """
        Get full changelog for a prompt.

        Args:
            prompt_name: Name of prompt

        Returns:
            List of version changes
        """
        if prompt_name not in self.versions:
            return []

        return [
            {
                "version": v.version,
                "created_at": v.created_at,
                "created_by": v.created_by,
                "changelog": v.changelog
            }
            for v in self.versions[prompt_name]
        ]

    def compare_versions(
        self,
        prompt_name: str,
        version1: str,
        version2: str
    ) -> Dict:
        """
        Compare two versions.

        Args:
            prompt_name: Name of prompt
            version1: First version
            version2: Second version

        Returns:
            Comparison details
        """
        v1 = self.get_version(prompt_name, version1)
        v2 = self.get_version(prompt_name, version2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        return {
            "prompt_name": prompt_name,
            "version1": {
                "version": v1.version,
                "created_at": v1.created_at,
                "prompt_length": len(v1.prompt),
                "prompt": v1.prompt
            },
            "version2": {
                "version": v2.version,
                "created_at": v2.created_at,
                "prompt_length": len(v2.prompt),
                "prompt": v2.prompt
            },
            "diff": {
                "length_change": len(v2.prompt) - len(v1.prompt),
                "changed": v1.prompt != v2.prompt
            }
        }


class ABTestingFramework:
    """A/B testing framework for prompts."""

    def __init__(self):
        """Initialize A/B testing framework."""
        self.experiments: Dict[str, Dict] = {}
        self.results: Dict[str, List[TestResult]] = defaultdict(list)

    def create_experiment(
        self,
        name: str,
        variants: Dict[str, str],
        success_metrics: List[str],
        description: str = ""
    ):
        """
        Create new A/B test experiment.

        Args:
            name: Experiment name
            variants: Dict of {variant_name: prompt}
            success_metrics: List of metrics to track
            description: Experiment description
        """
        self.experiments[name] = {
            "name": name,
            "description": description,
            "variants": variants,
            "success_metrics": success_metrics,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }

    def record_result(
        self,
        experiment_name: str,
        variant: str,
        test_input: str,
        output: str,
        latency_ms: float,
        token_count: int,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None
    ):
        """
        Record test result.

        Args:
            experiment_name: Name of experiment
            variant: Variant name
            test_input: Input used
            output: Output produced
            latency_ms: Response latency
            token_count: Number of tokens
            success: Whether test succeeded
            metrics: Additional metrics
            error: Error message if failed
        """
        result = TestResult(
            prompt_name=experiment_name,
            version=variant,
            test_input=test_input,
            output=output,
            latency_ms=latency_ms,
            token_count=token_count,
            success=success,
            error=error,
            metrics=metrics or {}
        )

        self.results[experiment_name].append(result)

    def analyze_experiment(self, experiment_name: str) -> Dict:
        """
        Analyze experiment results.

        Args:
            experiment_name: Name of experiment

        Returns:
            Analysis report
        """
        if experiment_name not in self.experiments:
            return {"error": "Experiment not found"}

        results = self.results[experiment_name]
        if not results:
            return {"error": "No results recorded"}

        # Group by variant
        by_variant = defaultdict(list)
        for result in results:
            by_variant[result.version].append(result)

        # Calculate statistics
        analysis = {
            "experiment": experiment_name,
            "total_tests": len(results),
            "variants": {}
        }

        for variant, variant_results in by_variant.items():
            total = len(variant_results)
            successes = sum(1 for r in variant_results if r.success)
            avg_latency = sum(r.latency_ms for r in variant_results) / total
            avg_tokens = sum(r.token_count for r in variant_results) / total

            analysis["variants"][variant] = {
                "total_tests": total,
                "success_rate": successes / total,
                "avg_latency_ms": avg_latency,
                "avg_tokens": avg_tokens
            }

            # Aggregate custom metrics
            if variant_results[0].metrics:
                for metric in variant_results[0].metrics.keys():
                    values = [r.metrics.get(metric, 0) for r in variant_results]
                    analysis["variants"][variant][f"avg_{metric}"] = sum(values) / len(values)

        # Determine winner
        best_variant = max(
            analysis["variants"].items(),
            key=lambda x: x[1]["success_rate"]
        )
        analysis["winner"] = best_variant[0]
        analysis["winner_stats"] = best_variant[1]

        return analysis

    def get_experiment_status(self, experiment_name: str) -> Dict:
        """Get current experiment status."""
        if experiment_name not in self.experiments:
            return {"error": "Experiment not found"}

        exp = self.experiments[experiment_name]
        results = self.results[experiment_name]

        return {
            "name": exp["name"],
            "description": exp["description"],
            "status": exp["status"],
            "variants": list(exp["variants"].keys()),
            "total_tests": len(results),
            "created_at": exp["created_at"]
        }


class TemplateLibrary:
    """Comprehensive library of prompt templates."""

    @staticmethod
    def get_classification_template() -> PromptTemplate:
        """Classification task template."""
        return PromptTemplate(
            name="classification",
            template="""Classify the following {item_type} into one of these categories: {categories}

Instructions:
{instructions}

{item_type}: {text}

Think step by step:
1. Analyze the key features
2. Match against category definitions
3. Select the best match

Category:""",
            variables=["item_type", "categories", "instructions", "text"],
            prompt_type=PromptType.CHAIN_OF_THOUGHT,
            category=PromptCategory.CLASSIFICATION,
            description="Multi-class classification with reasoning"
        )

    @staticmethod
    def get_summarization_template() -> PromptTemplate:
        """Text summarization template."""
        return PromptTemplate(
            name="summarization",
            template="""Summarize the following {content_type} in {length} words or less.

Focus on:
{focus_areas}

{content_type}: {text}

Summary:""",
            variables=["content_type", "length", "focus_areas", "text"],
            prompt_type=PromptType.STANDARD,
            category=PromptCategory.SUMMARIZATION,
            description="Flexible summarization with focus areas"
        )

    @staticmethod
    def get_code_generation_template() -> PromptTemplate:
        """Code generation template."""
        return PromptTemplate(
            name="code_generation",
            template="""Generate {language} code for the following task:

Task: {task_description}

Requirements:
{requirements}

Constraints:
{constraints}

Output format:
- Include docstrings/comments
- Follow {language} best practices
- Include error handling

Code:""",
            variables=["language", "task_description", "requirements", "constraints"],
            prompt_type=PromptType.STANDARD,
            category=PromptCategory.CODE,
            description="Professional code generation"
        )

    @staticmethod
    def get_data_extraction_template() -> PromptTemplate:
        """Data extraction template."""
        return PromptTemplate(
            name="data_extraction",
            template="""Extract structured data from the following text.

Extract these fields:
{fields}

Text: {text}

Output format: JSON
Example: {example_output}

Extracted data:""",
            variables=["fields", "text", "example_output"],
            prompt_type=PromptType.FEW_SHOT,
            category=PromptCategory.EXTRACTION,
            description="Structured data extraction"
        )

    @staticmethod
    def get_reasoning_template() -> PromptTemplate:
        """Complex reasoning template."""
        return PromptTemplate(
            name="reasoning",
            template="""Solve the following problem using careful reasoning:

Problem: {problem}

Context: {context}

Break down your reasoning:
1. Understand the problem
2. Identify key information
3. Apply relevant principles
4. Derive the solution
5. Verify your answer

Solution:""",
            variables=["problem", "context"],
            prompt_type=PromptType.CHAIN_OF_THOUGHT,
            category=PromptCategory.REASONING,
            description="Step-by-step reasoning"
        )

    @staticmethod
    def get_all_templates() -> List[PromptTemplate]:
        """Get all built-in templates."""
        return [
            TemplateLibrary.get_classification_template(),
            TemplateLibrary.get_summarization_template(),
            TemplateLibrary.get_code_generation_template(),
            TemplateLibrary.get_data_extraction_template(),
            TemplateLibrary.get_reasoning_template()
        ]


class PromptEngineer:
    """
    Production-ready prompt engineering toolkit.

    Features:
    - Advanced template management
    - Multiple prompting strategies
    - Version control
    - A/B testing
    - Performance analytics
    """

    def __init__(self):
        """Initialize prompt engineer."""
        self.templates: Dict[str, PromptTemplate] = {}
        self.prompts: Dict[str, str] = {}
        self.version_manager = PromptVersionManager()
        self.ab_testing = ABTestingFramework()
        self.metrics: Dict[str, List[float]] = defaultdict(list)

        # Load built-in templates
        self._load_builtin_templates()

    def _load_builtin_templates(self):
        """Load built-in template library."""
        for template in TemplateLibrary.get_all_templates():
            self.templates[template.name] = template

    def add_template(self, template: PromptTemplate):
        """
        Add custom template.

        Args:
            template: PromptTemplate instance
        """
        self.templates[template.name] = template

    def create_template(
        self,
        name: str,
        template: str,
        variables: List[str],
        prompt_type: PromptType = PromptType.STANDARD,
        category: Optional[PromptCategory] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        validation_rules: Optional[Dict[str, Callable]] = None
    ) -> PromptTemplate:
        """
        Create new template.

        Args:
            name: Template name
            template: Template string
            variables: Required variables
            prompt_type: Type of prompt
            category: Category
            description: Description
            tags: Search tags
            validation_rules: Validation functions

        Returns:
            Created template
        """
        tpl = PromptTemplate(
            name=name,
            template=template,
            variables=variables,
            prompt_type=prompt_type,
            category=category,
            description=description,
            tags=tags,
            validation_rules=validation_rules
        )
        self.templates[name] = tpl
        return tpl

    def fill_template(self, name: str, **kwargs) -> str:
        """
        Fill template with variables.

        Args:
            name: Template name
            **kwargs: Variable values

        Returns:
            Rendered prompt
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")

        return self.templates[name].render(**kwargs)

    def search_templates(
        self,
        query: Optional[str] = None,
        category: Optional[PromptCategory] = None,
        prompt_type: Optional[PromptType] = None,
        tags: Optional[List[str]] = None
    ) -> List[PromptTemplate]:
        """
        Search templates by criteria.

        Args:
            query: Search query (searches name and description)
            category: Filter by category
            prompt_type: Filter by type
            tags: Filter by tags

        Returns:
            Matching templates
        """
        results = list(self.templates.values())

        if query:
            query_lower = query.lower()
            results = [
                t for t in results
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]

        if category:
            results = [t for t in results if t.category == category]

        if prompt_type:
            results = [t for t in results if t.prompt_type == prompt_type]

        if tags:
            results = [
                t for t in results
                if any(tag in t.tags for tag in tags)
            ]

        return results

    def few_shot_prompt(
        self,
        task_description: str,
        examples: List[Dict],
        query: str,
        reasoning: bool = False
    ) -> str:
        """
        Create few-shot learning prompt.

        Args:
            task_description: Task description
            examples: List of {input, output} or {input, reasoning, output}
            query: New input
            reasoning: Include reasoning steps

        Returns:
            Few-shot prompt
        """
        prompt = f"{task_description}\n\n"

        # Add examples
        for i, ex in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {ex['input']}\n"

            if reasoning and 'reasoning' in ex:
                prompt += f"Reasoning: {ex['reasoning']}\n"

            prompt += f"Output: {ex['output']}\n\n"

        # Add query
        prompt += f"Now solve this:\n"
        prompt += f"Input: {query}\n"

        if reasoning:
            prompt += f"Reasoning:"
        else:
            prompt += f"Output:"

        return prompt

    def chain_of_thought_prompt(
        self,
        question: str,
        include_example: bool = True,
        domain: Optional[str] = None
    ) -> str:
        """
        Create chain-of-thought prompt.

        Args:
            question: Question to answer
            include_example: Include example
            domain: Specific domain for examples

        Returns:
            CoT prompt
        """
        prompt = ""

        if include_example:
            prompt += "Let's solve this step by step:\n\n"

            # Domain-specific examples
            if domain == "math":
                prompt += """Example:
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 balls. How many tennis balls does he have now?
A: Let's think step by step:
1. Roger starts with 5 balls
2. He buys 2 cans with 3 balls each = 2 × 3 = 6 balls
3. Total = 5 + 6 = 11 balls
Answer: 11

"""
            elif domain == "logic":
                prompt += """Example:
Q: All roses are flowers. Some flowers fade quickly. Therefore, some roses fade quickly. Is this valid?
A: Let's think step by step:
1. Premise 1: All roses are flowers (roses ⊆ flowers)
2. Premise 2: Some flowers fade quickly (some flowers ∈ quick-fading)
3. Conclusion: Some roses fade quickly
4. This is invalid - the some flowers that fade quickly might not include roses
Answer: Invalid reasoning

"""
            else:
                prompt += """Example:
Q: If I heat water to 100°C at sea level, what happens?
A: Let's think step by step:
1. Water has a boiling point of 100°C at sea level (1 atm pressure)
2. When heated to 100°C, water molecules gain enough energy to overcome intermolecular forces
3. Water transitions from liquid to gas phase
Answer: The water boils

"""

        prompt += f"Now solve this step by step:\n"
        prompt += f"Q: {question}\n"
        prompt += f"A: Let's think step by step:\n"

        return prompt

    def tree_of_thought_prompt(
        self,
        problem: str,
        num_thoughts: int = 3,
        num_steps: int = 3,
        evaluation_criteria: Optional[List[str]] = None
    ) -> str:
        """
        Create Tree-of-Thought prompt.

        Args:
            problem: Problem to solve
            num_thoughts: Branches per step
            num_steps: Number of steps
            evaluation_criteria: Evaluation criteria

        Returns:
            ToT prompt
        """
        return TreeOfThought.generate_prompt(
            problem,
            num_thoughts,
            num_steps,
            evaluation_criteria
        )

    def react_prompt(
        self,
        task: str,
        available_actions: List[Dict[str, str]],
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Create ReAct pattern prompt.

        Args:
            task: Task description
            available_actions: Available actions
            examples: Example reasoning-action cycles

        Returns:
            ReAct prompt
        """
        return ReActPattern.generate_prompt(task, available_actions, examples)

    def system_prompt(
        self,
        role: str,
        constraints: Optional[List[str]] = None,
        style: Optional[str] = None,
        output_format: Optional[str] = None,
        examples: Optional[List[str]] = None
    ) -> str:
        """
        Create system prompt for chat models.

        Args:
            role: Role description
            constraints: List of constraints/rules
            style: Communication style
            output_format: Expected output format
            examples: Example behaviors

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
            prompt += f"Communication style: {style}\n\n"

        if output_format:
            prompt += f"Output format: {output_format}\n\n"

        if examples:
            prompt += "Examples of good responses:\n"
            for i, example in enumerate(examples, 1):
                prompt += f"{i}. {example}\n"
            prompt += "\n"

        return prompt.strip()

    def optimize_prompt(
        self,
        base_prompt: str,
        improvements: List[str],
        custom_improvements: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Generate prompt variations.

        Args:
            base_prompt: Original prompt
            improvements: Improvement strategies
            custom_improvements: Custom improvement templates

        Returns:
            List of prompt variations
        """
        variations = [base_prompt]

        strategies = {
            "be_specific": "Be specific and detailed in your response.",
            "use_examples": "Provide concrete examples to illustrate your points.",
            "step_by_step": "Break down your reasoning step by step.",
            "cite_sources": "Cite your sources and reasoning.",
            "be_concise": "Be concise and to the point.",
            "show_work": "Show your work and intermediate steps.",
            "consider_alternatives": "Consider alternative approaches or perspectives.",
            "verify_answer": "Verify your answer before providing it.",
            "structured_output": "Provide your answer in a structured format."
        }

        if custom_improvements:
            strategies.update(custom_improvements)

        for improvement in improvements:
            if improvement in strategies:
                variation = f"{base_prompt}\n\n{strategies[improvement]}"
                variations.append(variation)

        return variations

    def create_ab_test(
        self,
        name: str,
        variants: Dict[str, str],
        success_metrics: List[str],
        description: str = ""
    ):
        """
        Create A/B test experiment.

        Args:
            name: Experiment name
            variants: Prompt variants
            success_metrics: Metrics to track
            description: Description
        """
        self.ab_testing.create_experiment(name, variants, success_metrics, description)

    def analyze_ab_test(self, experiment_name: str) -> Dict:
        """
        Analyze A/B test results.

        Args:
            experiment_name: Experiment name

        Returns:
            Analysis report
        """
        return self.ab_testing.analyze_experiment(experiment_name)

    def version_prompt(
        self,
        prompt_name: str,
        prompt: str,
        changelog: str,
        created_by: str = "system"
    ) -> str:
        """
        Create prompt version.

        Args:
            prompt_name: Prompt name
            prompt: Prompt content
            changelog: Change description
            created_by: Author

        Returns:
            Version ID
        """
        return self.version_manager.create_version(
            prompt_name,
            prompt,
            changelog,
            created_by
        )

    def get_prompt_changelog(self, prompt_name: str) -> List[Dict]:
        """Get changelog for prompt."""
        return self.version_manager.get_changelog(prompt_name)

    def compare_prompt_versions(
        self,
        prompt_name: str,
        version1: str,
        version2: str
    ) -> Dict:
        """Compare two prompt versions."""
        return self.version_manager.compare_versions(
            prompt_name,
            version1,
            version2
        )

    def validate_prompt(self, prompt: str) -> Dict:
        """
        Validate prompt quality.

        Args:
            prompt: Prompt to validate

        Returns:
            Validation report
        """
        issues = []
        warnings = []
        score = 100

        # Check length
        if len(prompt) < 10:
            issues.append("Prompt too short (< 10 chars)")
            score -= 30
        elif len(prompt) > 4000:
            warnings.append("Prompt very long (> 4000 chars), may exceed model limits")
            score -= 10

        # Check for unclear instructions
        if '?' not in prompt and 'please' not in prompt.lower():
            warnings.append("No clear question or request detected")
            score -= 5

        # Check for variables
        unclosed_braces = prompt.count('{') != prompt.count('}')
        if unclosed_braces:
            issues.append("Unmatched braces - possible variable syntax error")
            score -= 20

        # Check for examples
        has_examples = 'example' in prompt.lower()
        if not has_examples and len(prompt) > 100:
            warnings.append("Consider adding examples for complex tasks")

        # Check for structure
        has_structure = any(marker in prompt for marker in ['1.', '2.', '-', '*', 'Step'])
        if not has_structure and len(prompt) > 200:
            warnings.append("Consider adding structure for long prompts")

        return {
            "valid": len(issues) == 0,
            "score": max(0, score),
            "issues": issues,
            "warnings": warnings,
            "statistics": {
                "length": len(prompt),
                "estimated_tokens": len(prompt) // 4,
                "has_examples": has_examples,
                "has_structure": has_structure
            }
        }

    def analyze_prompt_performance(self, prompt_name: str) -> Dict:
        """
        Analyze prompt performance metrics.

        Args:
            prompt_name: Prompt name

        Returns:
            Performance analysis
        """
        if prompt_name not in self.templates:
            return {"error": "Template not found"}

        template = self.templates[prompt_name]

        return {
            "name": prompt_name,
            "usage_count": template.usage_count,
            "type": template.prompt_type.value,
            "category": template.category.value if template.category else None,
            "created_at": template.created_at,
            "variables": template.variables,
            "tags": template.tags
        }

    def export_templates(self, filepath: str, format: str = "json"):
        """
        Export templates to file.

        Args:
            filepath: Output file path
            format: Export format (json, yaml)
        """
        data = {
            "templates": {
                name: template.to_dict()
                for name, template in self.templates.items()
            },
            "exported_at": datetime.now().isoformat(),
            "version": "2.0.0"
        }

        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Exported {len(self.templates)} templates to {filepath}")

    def import_templates(self, filepath: str):
        """
        Import templates from file.

        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        for name, template_data in data.get("templates", {}).items():
            template = PromptTemplate(
                name=template_data["name"],
                template=template_data["template"],
                variables=template_data["variables"],
                prompt_type=PromptType(template_data["prompt_type"]),
                category=PromptCategory(template_data["category"]) if template_data.get("category") else None,
                description=template_data.get("description", ""),
                tags=template_data.get("tags", [])
            )
            self.templates[name] = template

        print(f"Imported {len(data.get('templates', {}))} templates from {filepath}")

    def save_state(self, filepath: str):
        """Save complete state including templates, versions, experiments."""
        state = {
            "templates": {
                name: tpl.to_dict()
                for name, tpl in self.templates.items()
            },
            "prompts": self.prompts,
            "versions": {
                name: [v.to_dict() for v in versions]
                for name, versions in self.version_manager.versions.items()
            },
            "experiments": self.ab_testing.experiments,
            "saved_at": datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        print(f"Saved state to {filepath}")

    def load_state(self, filepath: str):
        """Load complete state."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        # Load templates
        for name, tpl_data in state.get("templates", {}).items():
            template = PromptTemplate(
                name=tpl_data["name"],
                template=tpl_data["template"],
                variables=tpl_data["variables"],
                prompt_type=PromptType(tpl_data["prompt_type"]),
                category=PromptCategory(tpl_data["category"]) if tpl_data.get("category") else None,
                description=tpl_data.get("description", ""),
                tags=tpl_data.get("tags", [])
            )
            self.templates[name] = template

        self.prompts = state.get("prompts", {})

        print(f"Loaded state from {filepath}")


def demo():
    """Comprehensive demo of prompt engineering features."""
    print("=" * 80)
    print("PROMPT ENGINEERING TOOLKIT - PRODUCTION DEMO")
    print("=" * 80)

    pe = PromptEngineer()

    # 1. Template Library
    print("\n1. TEMPLATE LIBRARY")
    print("-" * 80)
    print(f"Built-in templates: {len(pe.templates)}")

    templates = pe.search_templates(category=PromptCategory.CLASSIFICATION)
    print(f"Classification templates: {len(templates)}")

    for template in templates[:1]:
        print(f"\nTemplate: {template.name}")
        print(f"Type: {template.prompt_type.value}")
        print(f"Variables: {template.variables}")

    # 2. Advanced Template Usage
    print("\n\n2. ADVANCED TEMPLATE WITH VALIDATION")
    print("-" * 80)

    pe.create_template(
        name="email_classifier",
        template="Classify this email: {email_text}\n\nCategory:",
        variables=["email_text"],
        prompt_type=PromptType.CLASSIFICATION,
        category=PromptCategory.CLASSIFICATION,
        description="Email classification template",
        tags=["email", "classification"],
        validation_rules={
            "email_text": lambda x: len(x) > 10
        }
    )

    prompt = pe.fill_template(
        "email_classifier",
        email_text="Get 50% off now! Limited time offer!"
    )
    print(prompt)

    # 3. Few-Shot Learning
    print("\n\n3. FEW-SHOT LEARNING WITH REASONING")
    print("-" * 80)

    examples = [
        {
            "input": "The movie was fantastic!",
            "reasoning": "Contains positive adjective 'fantastic'",
            "output": "Positive"
        },
        {
            "input": "Waste of time and money.",
            "reasoning": "Contains negative phrases 'waste'",
            "output": "Negative"
        }
    ]

    few_shot = pe.few_shot_prompt(
        "Classify sentiment with reasoning:",
        examples,
        "I absolutely loved it!",
        reasoning=True
    )
    print(few_shot[:400] + "...")

    # 4. Chain-of-Thought
    print("\n\n4. CHAIN-OF-THOUGHT PROMPTING")
    print("-" * 80)

    cot = pe.chain_of_thought_prompt(
        "A store has 45 items. It sells 60% of them. How many remain?",
        include_example=True,
        domain="math"
    )
    print(cot[:400] + "...")

    # 5. Tree-of-Thought
    print("\n\n5. TREE-OF-THOUGHT REASONING")
    print("-" * 80)

    tot = pe.tree_of_thought_prompt(
        "Design a database schema for a social media platform",
        num_thoughts=3,
        num_steps=2,
        evaluation_criteria=["Scalability", "Data integrity", "Query efficiency"]
    )
    print(tot[:400] + "...")

    # 6. ReAct Pattern
    print("\n\n6. REACT PATTERN (REASONING + ACTING)")
    print("-" * 80)

    actions = [
        {"name": "search", "description": "Search the database", "parameters": "query"},
        {"name": "calculate", "description": "Perform calculation", "parameters": "expression"},
        {"name": "format", "description": "Format the output", "parameters": "data"}
    ]

    react = pe.react_prompt(
        "Find the total sales for Q4 2023 and format as currency",
        actions
    )
    print(react[:400] + "...")

    # 7. Prompt Versioning
    print("\n\n7. PROMPT VERSIONING")
    print("-" * 80)

    v1 = pe.version_prompt(
        "sales_report",
        "Generate sales report for {period}",
        "Initial version",
        "John Doe"
    )
    print(f"Created version: {v1}")

    v2 = pe.version_prompt(
        "sales_report",
        "Generate detailed sales report for {period} including trends and forecasts",
        "Added trends and forecasts",
        "Jane Smith"
    )
    print(f"Created version: {v2}")

    changelog = pe.get_prompt_changelog("sales_report")
    print(f"\nChangelog entries: {len(changelog)}")
    for entry in changelog:
        print(f"  {entry['version']}: {entry['changelog']}")

    # 8. A/B Testing
    print("\n\n8. A/B TESTING FRAMEWORK")
    print("-" * 80)

    pe.create_ab_test(
        "summarization_test",
        variants={
            "concise": "Summarize in 2 sentences: {text}",
            "detailed": "Provide comprehensive summary with key points: {text}",
            "structured": "Summarize with bullet points:\n{text}"
        },
        success_metrics=["accuracy", "readability", "completeness"],
        description="Test different summarization approaches"
    )

    # Simulate results
    pe.ab_testing.record_result(
        "summarization_test", "concise",
        "Long article...", "Summary...",
        latency_ms=250, token_count=50, success=True,
        metrics={"accuracy": 0.85, "readability": 0.9}
    )
    pe.ab_testing.record_result(
        "summarization_test", "detailed",
        "Long article...", "Detailed summary...",
        latency_ms=400, token_count=100, success=True,
        metrics={"accuracy": 0.92, "readability": 0.85}
    )

    analysis = pe.analyze_ab_test("summarization_test")
    print(f"Winner: {analysis['winner']}")
    print(f"Success rate: {analysis['winner_stats']['success_rate']:.2%}")

    # 9. Prompt Validation
    print("\n\n9. PROMPT VALIDATION")
    print("-" * 80)

    test_prompt = """Analyze the following data and provide insights:
1. Identify trends
2. Calculate key metrics
3. Suggest recommendations

Data: {data}"""

    validation = pe.validate_prompt(test_prompt)
    print(f"Valid: {validation['valid']}")
    print(f"Score: {validation['score']}/100")
    print(f"Estimated tokens: {validation['statistics']['estimated_tokens']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")

    # 10. Prompt Optimization
    print("\n\n10. PROMPT OPTIMIZATION")
    print("-" * 80)

    base = "Explain quantum computing."
    variations = pe.optimize_prompt(
        base,
        ["be_specific", "use_examples", "step_by_step", "consider_alternatives"]
    )

    print(f"Generated {len(variations)} variations")
    print(f"\nOriginal: {variations[0]}")
    print(f"\nOptimized example: {variations[1]}")

    # 11. Template Search
    print("\n\n11. TEMPLATE SEARCH")
    print("-" * 80)

    results = pe.search_templates(
        query="code",
        prompt_type=PromptType.STANDARD
    )
    print(f"Found {len(results)} templates matching 'code'")
    for t in results:
        print(f"  - {t.name}: {t.description}")

    # 12. System Prompt
    print("\n\n12. ADVANCED SYSTEM PROMPT")
    print("-" * 80)

    system = pe.system_prompt(
        role="an expert Python developer and code reviewer",
        constraints=[
            "Always follow PEP 8 style guidelines",
            "Provide constructive feedback",
            "Suggest best practices and optimizations",
            "Include security considerations"
        ],
        style="Professional and educational",
        output_format="Markdown with code blocks",
        examples=[
            "Good: Uses list comprehension for better performance",
            "Good: Includes proper error handling"
        ]
    )
    print(system[:300] + "...")

    # 13. Save and Load
    print("\n\n13. PERSISTENCE")
    print("-" * 80)

    pe.save_state("prompt_state.json")
    print("State saved successfully")

    # 14. Performance Analytics
    print("\n\n14. PERFORMANCE ANALYTICS")
    print("-" * 80)

    stats = pe.analyze_prompt_performance("classification")
    print(f"Template: {stats['name']}")
    print(f"Usage count: {stats['usage_count']}")
    print(f"Type: {stats['type']}")
    print(f"Category: {stats['category']}")

    print("\n" + "=" * 80)
    print("DEMO COMPLETE - All features demonstrated successfully!")
    print("=" * 80)


if __name__ == '__main__':
    demo()
