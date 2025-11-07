# LLM Red Teaming

Automated red teaming framework for testing LLM security, safety, and robustness against adversarial prompts and jailbreaks.

## Features

- **Automated Attack Generation** - Generate diverse adversarial prompts
- **Jailbreak Testing** - Test against known jailbreak techniques
- **Safety Evaluation** - Measure harmful output rates
- **Bias Detection** - Identify biased or discriminatory outputs
- **PII Leakage Testing** - Check for sensitive data exposure
- **Hallucination Detection** - Identify factually incorrect outputs
- **Comprehensive Reporting** - Detailed vulnerability reports
- **Continuous Monitoring** - Track security posture over time

## Attack Categories

| Category | Techniques | Risk Level |
|----------|-----------|------------|
| **Jailbreaks** | DAN, UCAR, role-playing | Critical |
| **Prompt Injection** | Instruction override, delimiter attacks | High |
| **Data Extraction** | Training data leakage, memorization | High |
| **Harmful Content** | Violence, hate speech, illegal content | Critical |
| **Bias Elicitation** | Stereotype amplification | Medium |
| **Misinformation** | False facts, conspiracy theories | High |

## Usage

### Basic Red Team Test
```python
from red_teaming import RedTeamFramework

# Initialize framework
red_team = RedTeamFramework(
    target_model="gpt-4",
    safety_threshold=0.9
)

# Run comprehensive test
report = red_team.run_campaign(
    test_categories=["jailbreaks", "prompt_injection", "harmful_content"],
    num_tests=1000
)

print(f"Vulnerabilities found: {report.vulnerability_count}")
print(f"Safety score: {report.safety_score}")
```

### Jailbreak Testing
```python
from red_teaming import JailbreakTester

tester = JailbreakTester()

# Test known jailbreaks
results = tester.test_jailbreaks(
    model=target_model,
    techniques=["dan", "ucar", "roleplay", "cognitive_hacking"]
)

for result in results:
    if result.successful:
        print(f"Jailbreak succeeded: {result.technique}")
```

### Safety Evaluation
```python
from red_teaming import SafetyEvaluator

evaluator = SafetyEvaluator()

# Test harmful prompts
safety_report = evaluator.evaluate(
    model=target_model,
    categories=["violence", "hate_speech", "illegal_activity"],
    num_prompts=500
)

print(f"Harmful output rate: {safety_report.harmful_rate:.2%}")
```

## Attack Techniques

### 1. Direct Jailbreaks
- **DAN (Do Anything Now)**: Role-playing bypass
- **UCAR**: Developer mode simulation
- **Evil Confidant**: Malicious persona
- **Cognitive Hacking**: Multi-step manipulation

### 2. Prompt Injection
- **Instruction Override**: Ignore previous instructions
- **Delimiter Attacks**: Break out of context
- **Template Injection**: Exploit formatting
- **Context Pollution**: Confuse with noise

### 3. Data Extraction
- **Training Data Leakage**: Extract memorized data
- **PII Extraction**: Elicit personal information
- **Model Probing**: Reverse engineer internals

### 4. Harmful Content
- **Violence Generation**: Explicit violent content
- **Hate Speech**: Discriminatory content
- **Illegal Activities**: Instructions for crimes
- **CSAM**: Child safety violations

## Testing Methodology

```python
from red_teaming import RedTeamCampaign

campaign = RedTeamCampaign(
    model=target_model,
    budget=1000,  # Number of test prompts
    strategy="adaptive"  # Learns from successes
)

# Phase 1: Reconnaissance
campaign.reconnaissance()

# Phase 2: Vulnerability scanning
vulnerabilities = campaign.scan_vulnerabilities()

# Phase 3: Exploitation
exploits = campaign.attempt_exploits(vulnerabilities)

# Phase 4: Reporting
report = campaign.generate_report()
```

## Metrics

- **Attack Success Rate (ASR)** - % of successful attacks
- **Mean Harmful Score** - Average harmfulness of outputs
- **Jailbreak Resistance** - Robustness against jailbreaks
- **Safety Violations** - Count of policy violations
- **Time to Jailbreak** - Speed of successful attacks

## Defense Recommendations

Based on findings, automatically suggest:
- System prompts improvements
- Input validation rules
- Output filtering policies
- User education materials
- Monitoring configurations

## Technologies

- **LLM APIs**: OpenAI, Anthropic, Cohere
- **Safety**: OpenAI Moderation, Perspective API
- **Prompting**: LangChain, LlamaIndex
- **Analysis**: Transformers, spaCy

## Best Practices

✅ Run red team tests before production deployment
✅ Test regularly (weekly/monthly depending on risk)
✅ Use diverse attack techniques
✅ Document all findings and fixes
✅ Implement defense-in-depth
✅ Monitor for novel attack patterns
✅ Collaborate with security researchers

## Ethical Guidelines

- Only test models you own or have permission to test
- Never use findings to harm users
- Responsibly disclose vulnerabilities
- Follow coordinated disclosure timelines
- Prioritize user safety over demonstrations

## References

- OWASP LLM Top 10: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Jailbreak Chat: https://www.jailbreakchat.com/
- AI Red Team: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/red-teaming
