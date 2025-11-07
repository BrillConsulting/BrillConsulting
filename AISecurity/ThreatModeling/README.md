# AI Threat Modeling

Comprehensive threat modeling framework for AI systems based on MITRE ATLAS, OWASP ML Top 10, and custom risk assessments.

## Features

- **MITRE ATLAS Integration** - Map threats to ATLAS framework
- **Risk Assessment** - Quantify likelihood and impact
- **Threat Intelligence** - Track emerging AI threats
- **Attack Surface Analysis** - Identify vulnerabilities
- **Mitigation Strategies** - Recommend controls
- **Compliance Mapping** - Map to security standards
- **Automated Scanning** - Detect common vulnerabilities
- **Threat Reports** - Generate comprehensive reports

## Threat Categories

| Category | Example Threats | Severity |
|----------|----------------|----------|
| **Model Attacks** | Adversarial examples, model inversion | High |
| **Data Poisoning** | Training data manipulation, backdoors | Critical |
| **Privacy Leakage** | Membership inference, model extraction | High |
| **Supply Chain** | Compromised pre-trained models | Critical |
| **Infrastructure** | ML pipeline compromise, API attacks | High |
| **Compliance** | GDPR violations, bias issues | Medium |

## MITRE ATLAS Tactics

1. **Reconnaissance** - Discover ML system details
2. **Resource Development** - Prepare attack resources
3. **Initial Access** - Gain entry to ML systems
4. **ML Attack Staging** - Craft adversarial inputs
5. **Execution** - Run malicious code
6. **Persistence** - Maintain access
7. **Defense Evasion** - Avoid detection
8. **Discovery** - Learn about ML environment
9. **Collection** - Gather training data
10. **Impact** - Manipulate model behavior

## Usage

### Threat Assessment
```python
from threat_modeling import ThreatModeler, ThreatCategory

modeler = ThreatModeler(framework="mitre_atlas")

# Assess system
assessment = modeler.assess_system(
    system_type="image_classifier",
    deployment="cloud",
    data_sensitivity="high"
)

print(f"Risk score: {assessment.risk_score}")
print(f"Critical threats: {len(assessment.critical_threats)}")
```

### MITRE ATLAS Mapping
```python
from threat_modeling import ATLASMapper

mapper = ATLASMapper()

# Map threat to ATLAS
threat = "adversarial_perturbation"
atlas_info = mapper.map_threat(threat)

print(f"Tactic: {atlas_info.tactic}")
print(f"Technique: {atlas_info.technique}")
print(f"Mitigations: {atlas_info.mitigations}")
```

### Risk Scoring
```python
from threat_modeling import RiskAssessor

assessor = RiskAssessor()

risk = assessor.calculate_risk(
    likelihood="high",
    impact="critical",
    existing_controls=["input_validation", "monitoring"]
)

print(f"Risk level: {risk.level}")
print(f"Priority: {risk.priority}")
```

## Threat Database

Built-in threat intelligence for:
- Adversarial attacks (FGSM, PGD, C&W, etc.)
- Data poisoning attacks
- Model extraction / stealing
- Membership inference
- Privacy attacks
- Supply chain compromises
- Infrastructure vulnerabilities

## Risk Assessment Matrix

| Impact / Likelihood | Low | Medium | High | Critical |
|---------------------|-----|--------|------|----------|
| **Critical** | Medium | High | Critical | Critical |
| **High** | Low | Medium | High | Critical |
| **Medium** | Low | Medium | Medium | High |
| **Low** | Low | Low | Medium | Medium |

## Mitigation Strategies

Each identified threat includes:
- **Preventive controls** - Stop attacks before they happen
- **Detective controls** - Identify ongoing attacks
- **Corrective controls** - Respond to incidents
- **Cost-benefit analysis** - ROI of each control

## Technologies

- **Framework**: MITRE ATLAS
- **Standards**: OWASP ML Top 10, NIST AI RMF
- **Analysis**: Custom risk scoring algorithms
- **Reporting**: Jinja2 templates

## Best Practices

✅ Perform threat modeling during design phase
✅ Update threat model with each major change
✅ Map threats to MITRE ATLAS for standardization
✅ Quantify risks with likelihood × impact
✅ Prioritize mitigations by risk score
✅ Include supply chain in threat model
✅ Consider both technical and organizational threats

## References

- MITRE ATLAS: https://atlas.mitre.org/
- OWASP ML Top 10: https://owasp.org/www-project-machine-learning-security-top-10/
- NIST AI RMF: https://www.nist.gov/itl/ai-risk-management-framework
