"""
AI Threat Modeling
==================

Threat modeling framework for AI systems based on MITRE ATLAS

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime


class ThreatCategory(Enum):
    """Threat categories."""
    MODEL_ATTACK = "model_attack"
    DATA_POISONING = "data_poisoning"
    PRIVACY_LEAKAGE = "privacy_leakage"
    SUPPLY_CHAIN = "supply_chain"
    INFRASTRUCTURE = "infrastructure"
    COMPLIANCE = "compliance"


class Severity(Enum):
    """Severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Likelihood(Enum):
    """Likelihood levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Threat:
    """Threat definition."""
    threat_id: str
    name: str
    category: ThreatCategory
    description: str
    severity: Severity
    likelihood: Likelihood
    atlas_tactic: Optional[str]
    atlas_technique: Optional[str]
    mitigations: List[str]


@dataclass
class RiskScore:
    """Risk assessment score."""
    likelihood: str
    impact: str
    risk_level: str
    risk_score: float
    priority: int
    existing_controls: List[str]


@dataclass
class ThreatAssessment:
    """System threat assessment."""
    system_type: str
    deployment: str
    identified_threats: List[Threat]
    risk_score: float
    critical_threats: List[Threat]
    recommended_controls: List[str]
    compliance_gaps: List[str]
    timestamp: str


class ATLASMapper:
    """Map threats to MITRE ATLAS framework."""

    def __init__(self):
        """Initialize ATLAS mapper."""
        self.tactics = self._initialize_tactics()
        self.techniques = self._initialize_techniques()

        print(f"🗺️  MITRE ATLAS Mapper initialized")
        print(f"   Tactics: {len(self.tactics)}")
        print(f"   Techniques: {len(self.techniques)}")

    def _initialize_tactics(self) -> Dict[str, str]:
        """Initialize ATLAS tactics."""
        return {
            "reconnaissance": "Discover ML system details",
            "resource_development": "Prepare attack resources",
            "initial_access": "Gain entry to ML systems",
            "ml_attack_staging": "Craft adversarial inputs",
            "execution": "Run malicious code",
            "persistence": "Maintain access",
            "defense_evasion": "Avoid detection",
            "discovery": "Learn about ML environment",
            "collection": "Gather training data",
            "impact": "Manipulate model behavior"
        }

    def _initialize_techniques(self) -> Dict[str, Dict[str, Any]]:
        """Initialize ATLAS techniques."""
        return {
            "adversarial_perturbation": {
                "tactic": "ml_attack_staging",
                "description": "Craft adversarial examples",
                "mitigations": ["adversarial_training", "input_validation"]
            },
            "model_extraction": {
                "tactic": "collection",
                "description": "Steal model via API queries",
                "mitigations": ["rate_limiting", "query_monitoring"]
            },
            "data_poisoning": {
                "tactic": "initial_access",
                "description": "Inject malicious training data",
                "mitigations": ["data_validation", "anomaly_detection"]
            },
            "membership_inference": {
                "tactic": "collection",
                "description": "Determine if data was used in training",
                "mitigations": ["differential_privacy", "output_perturbation"]
            },
            "backdoor_injection": {
                "tactic": "persistence",
                "description": "Embed trigger in model",
                "mitigations": ["model_inspection", "activation_analysis"]
            }
        }

    def map_threat(self, threat_name: str) -> Dict[str, Any]:
        """Map threat to ATLAS framework."""
        print(f"\n🔍 Mapping threat: {threat_name}")

        if threat_name in self.techniques:
            technique = self.techniques[threat_name]
            tactic = technique["tactic"]

            print(f"   Tactic: {tactic}")
            print(f"   Mitigations: {', '.join(technique['mitigations'])}")

            return {
                "threat": threat_name,
                "tactic": tactic,
                "technique": technique,
                "atlas_ref": f"ATLAS.{tactic}.{threat_name}"
            }

        return {"threat": threat_name, "tactic": "unknown"}


class RiskAssessor:
    """Assess and quantify AI security risks."""

    def __init__(self):
        """Initialize risk assessor."""
        self.risk_matrix = self._initialize_risk_matrix()
        print(f"⚖️  Risk Assessor initialized")

    def _initialize_risk_matrix(self) -> Dict[str, Dict[str, str]]:
        """Initialize risk scoring matrix."""
        return {
            "critical": {
                "low": "medium",
                "medium": "high",
                "high": "critical",
                "critical": "critical"
            },
            "high": {
                "low": "low",
                "medium": "medium",
                "high": "high",
                "critical": "critical"
            },
            "medium": {
                "low": "low",
                "medium": "medium",
                "high": "medium",
                "critical": "high"
            },
            "low": {
                "low": "low",
                "medium": "low",
                "high": "medium",
                "critical": "medium"
            }
        }

    def calculate_risk(
        self,
        likelihood: str,
        impact: str,
        existing_controls: Optional[List[str]] = None
    ) -> RiskScore:
        """Calculate risk score."""
        print(f"\n📊 Calculating risk")
        print(f"   Likelihood: {likelihood}")
        print(f"   Impact: {impact}")

        existing_controls = existing_controls or []

        # Get risk level from matrix
        risk_level = self.risk_matrix.get(impact, {}).get(likelihood, "medium")

        # Calculate numerical score (0-100)
        likelihood_score = {"low": 25, "medium": 50, "high": 75, "critical": 100}
        impact_score = {"low": 25, "medium": 50, "high": 75, "critical": 100}

        raw_score = (likelihood_score[likelihood] + impact_score[impact]) / 2

        # Adjust for existing controls
        control_reduction = len(existing_controls) * 5
        adjusted_score = max(0, raw_score - control_reduction)

        # Determine priority
        priority_map = {"critical": 1, "high": 2, "medium": 3, "low": 4}
        priority = priority_map[risk_level]

        risk_score = RiskScore(
            likelihood=likelihood,
            impact=impact,
            risk_level=risk_level,
            risk_score=adjusted_score,
            priority=priority,
            existing_controls=existing_controls
        )

        print(f"   Risk level: {risk_level.upper()}")
        print(f"   Risk score: {adjusted_score:.1f}/100")
        print(f"   Priority: P{priority}")

        return risk_score


class ThreatModeler:
    """Comprehensive threat modeling for AI systems."""

    def __init__(self, framework: str = "mitre_atlas"):
        """Initialize threat modeler."""
        self.framework = framework
        self.atlas_mapper = ATLASMapper()
        self.risk_assessor = RiskAssessor()
        self.threat_database = self._initialize_threat_database()

        print(f"🛡️  Threat Modeler initialized")
        print(f"   Framework: {framework}")
        print(f"   Threat database: {len(self.threat_database)} threats")

    def _initialize_threat_database(self) -> Dict[str, Threat]:
        """Initialize threat database."""
        threats = {}

        # Model attacks
        threats["adversarial_attack"] = Threat(
            threat_id="T001",
            name="Adversarial Attack",
            category=ThreatCategory.MODEL_ATTACK,
            description="Craft inputs to fool model",
            severity=Severity.HIGH,
            likelihood=Likelihood.HIGH,
            atlas_tactic="ml_attack_staging",
            atlas_technique="adversarial_perturbation",
            mitigations=["adversarial_training", "input_validation", "ensemble_methods"]
        )

        threats["model_extraction"] = Threat(
            threat_id="T002",
            name="Model Extraction",
            category=ThreatCategory.PRIVACY_LEAKAGE,
            description="Steal model via API queries",
            severity=Severity.HIGH,
            likelihood=Likelihood.MEDIUM,
            atlas_tactic="collection",
            atlas_technique="model_extraction",
            mitigations=["rate_limiting", "query_monitoring", "output_perturbation"]
        )

        # Data poisoning
        threats["training_poisoning"] = Threat(
            threat_id="T003",
            name="Training Data Poisoning",
            category=ThreatCategory.DATA_POISONING,
            description="Inject malicious training data",
            severity=Severity.CRITICAL,
            likelihood=Likelihood.MEDIUM,
            atlas_tactic="initial_access",
            atlas_technique="data_poisoning",
            mitigations=["data_validation", "anomaly_detection", "data_provenance"]
        )

        threats["backdoor"] = Threat(
            threat_id="T004",
            name="Model Backdoor",
            category=ThreatCategory.DATA_POISONING,
            description="Embed trigger in model",
            severity=Severity.CRITICAL,
            likelihood=Likelihood.LOW,
            atlas_tactic="persistence",
            atlas_technique="backdoor_injection",
            mitigations=["model_inspection", "activation_analysis", "trigger_detection"]
        )

        # Privacy attacks
        threats["membership_inference"] = Threat(
            threat_id="T005",
            name="Membership Inference",
            category=ThreatCategory.PRIVACY_LEAKAGE,
            description="Determine if data was in training set",
            severity=Severity.HIGH,
            likelihood=Likelihood.HIGH,
            atlas_tactic="collection",
            atlas_technique="membership_inference",
            mitigations=["differential_privacy", "output_clipping", "regularization"]
        )

        # Supply chain
        threats["compromised_pretrained"] = Threat(
            threat_id="T006",
            name="Compromised Pre-trained Model",
            category=ThreatCategory.SUPPLY_CHAIN,
            description="Use backdoored pre-trained model",
            severity=Severity.CRITICAL,
            likelihood=Likelihood.MEDIUM,
            atlas_tactic="resource_development",
            atlas_technique="supply_chain_compromise",
            mitigations=["model_verification", "trusted_sources", "model_scanning"]
        )

        # Infrastructure
        threats["api_abuse"] = Threat(
            threat_id="T007",
            name="ML API Abuse",
            category=ThreatCategory.INFRASTRUCTURE,
            description="Exploit ML API vulnerabilities",
            severity=Severity.MEDIUM,
            likelihood=Likelihood.HIGH,
            atlas_tactic="execution",
            atlas_technique="api_abuse",
            mitigations=["authentication", "rate_limiting", "input_validation"]
        )

        return threats

    def assess_system(
        self,
        system_type: str,
        deployment: str,
        data_sensitivity: str
    ) -> ThreatAssessment:
        """Assess threats for system."""
        print(f"\n🔍 Assessing system")
        print(f"   Type: {system_type}")
        print(f"   Deployment: {deployment}")
        print(f"   Data sensitivity: {data_sensitivity}")

        # Identify relevant threats
        relevant_threats = self._identify_threats(
            system_type, deployment, data_sensitivity
        )

        # Calculate overall risk
        risk_scores = []
        critical_threats = []

        for threat in relevant_threats:
            risk = self.risk_assessor.calculate_risk(
                likelihood=threat.likelihood.value,
                impact=threat.severity.value
            )
            risk_scores.append(risk.risk_score)

            if threat.severity == Severity.CRITICAL:
                critical_threats.append(threat)

        overall_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0

        # Recommend controls
        recommended_controls = self._recommend_controls(relevant_threats)

        # Identify compliance gaps
        compliance_gaps = self._check_compliance(system_type, data_sensitivity)

        assessment = ThreatAssessment(
            system_type=system_type,
            deployment=deployment,
            identified_threats=relevant_threats,
            risk_score=overall_risk,
            critical_threats=critical_threats,
            recommended_controls=recommended_controls,
            compliance_gaps=compliance_gaps,
            timestamp=datetime.now().isoformat()
        )

        # Summary
        print(f"\n{'='*60}")
        print(f"Assessment Summary")
        print(f"{'='*60}")
        print(f"Identified threats: {len(relevant_threats)}")
        print(f"Critical threats: {len(critical_threats)}")
        print(f"Overall risk score: {overall_risk:.1f}/100")
        print(f"Recommended controls: {len(recommended_controls)}")

        return assessment

    def _identify_threats(
        self,
        system_type: str,
        deployment: str,
        sensitivity: str
    ) -> List[Threat]:
        """Identify relevant threats."""
        relevant = []

        # All systems face certain threats
        relevant.extend([
            self.threat_database["adversarial_attack"],
            self.threat_database["model_extraction"],
            self.threat_database["api_abuse"]
        ])

        # High sensitivity data
        if sensitivity == "high":
            relevant.append(self.threat_database["membership_inference"])

        # Cloud deployment
        if deployment == "cloud":
            relevant.append(self.threat_database["compromised_pretrained"])

        # Training systems
        if "train" in system_type.lower():
            relevant.extend([
                self.threat_database["training_poisoning"],
                self.threat_database["backdoor"]
            ])

        return relevant

    def _recommend_controls(self, threats: List[Threat]) -> List[str]:
        """Recommend security controls."""
        controls = set()

        for threat in threats:
            controls.update(threat.mitigations)

        return sorted(list(controls))

    def _check_compliance(self, system_type: str, sensitivity: str) -> List[str]:
        """Check compliance gaps."""
        gaps = []

        if sensitivity == "high":
            gaps.append("Requires GDPR Article 22 explanation")
            gaps.append("Need differential privacy for high-risk data")

        if "healthcare" in system_type.lower():
            gaps.append("HIPAA compliance required")

        return gaps

    def generate_report(self, assessment: ThreatAssessment) -> str:
        """Generate threat modeling report."""
        print(f"\n📄 Generating threat model report")

        report = f"""
AI Threat Modeling Report
=========================

System: {assessment.system_type}
Deployment: {assessment.deployment}
Assessment Date: {assessment.timestamp}

Overall Risk Score: {assessment.risk_score:.1f}/100

Identified Threats ({len(assessment.identified_threats)}):
"""

        for threat in assessment.identified_threats:
            report += f"\n{threat.threat_id}: {threat.name}"
            report += f"\n   Category: {threat.category.value}"
            report += f"\n   Severity: {threat.severity.value.upper()}"
            report += f"\n   Mitigations: {', '.join(threat.mitigations)}"

        report += f"\n\nRecommended Controls ({len(assessment.recommended_controls)}):"
        for control in assessment.recommended_controls:
            report += f"\n- {control}"

        print(f"   ✓ Report generated")

        return report


def demo():
    """Demonstrate threat modeling."""
    print("=" * 60)
    print("AI Threat Modeling Demo")
    print("=" * 60)

    # ATLAS Mapping
    print(f"\n{'='*60}")
    print("MITRE ATLAS Mapping")
    print(f"{'='*60}")

    mapper = ATLASMapper()

    threats_to_map = ["adversarial_perturbation", "model_extraction", "data_poisoning"]

    for threat in threats_to_map:
        atlas_info = mapper.map_threat(threat)

    # Risk Assessment
    print(f"\n{'='*60}")
    print("Risk Assessment")
    print(f"{'='*60}")

    assessor = RiskAssessor()

    # Assess different risks
    risk1 = assessor.calculate_risk(
        likelihood="high",
        impact="critical",
        existing_controls=[]
    )

    risk2 = assessor.calculate_risk(
        likelihood="medium",
        impact="high",
        existing_controls=["input_validation", "monitoring"]
    )

    # System Threat Modeling
    print(f"\n{'='*60}")
    print("System Threat Assessment")
    print(f"{'='*60}")

    modeler = ThreatModeler(framework="mitre_atlas")

    # Assess image classification system
    assessment = modeler.assess_system(
        system_type="image_classifier",
        deployment="cloud",
        data_sensitivity="high"
    )

    # Generate report
    print(f"\n{'='*60}")
    print("Threat Model Report")
    print(f"{'='*60}")

    report = modeler.generate_report(assessment)
    print(report)


if __name__ == "__main__":
    demo()
