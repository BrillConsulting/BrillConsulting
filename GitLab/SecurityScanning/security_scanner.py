"""
GitLab Security Scanning
Author: BrillConsulting
Description: Comprehensive security scanning with SAST, DAST, dependency scanning, container scanning, secret detection, and license compliance
"""

import json
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum


class ScanType(Enum):
    """Security scan types."""
    SAST = "sast"  # Static Application Security Testing
    DAST = "dast"  # Dynamic Application Security Testing
    DEPENDENCY = "dependency_scanning"
    CONTAINER = "container_scanning"
    SECRET_DETECTION = "secret_detection"
    LICENSE_SCANNING = "license_scanning"
    API_FUZZING = "api_fuzzing"
    COVERAGE_FUZZING = "coverage_fuzzing"


class Severity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


class ScanStatus(Enum):
    """Scan execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELED = "canceled"


class VulnerabilityState(Enum):
    """Vulnerability triage states."""
    DETECTED = "detected"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class SASTScannerManager:
    """SAST (Static Application Security Testing) management."""

    def __init__(self):
        self.sast_scans = {}
        self.supported_scanners = ['semgrep', 'bandit', 'eslint', 'brakeman', 'spotbugs', 'gosec']

    def run_sast_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run SAST scan on source code.

        Config:
            project_id: str - Project identifier
            branch: str - Branch to scan (default: main)
            scanner: str - Scanner to use (semgrep, bandit, eslint, etc.)
            file_patterns: List[str] - Files to scan (default: all)
            exclude_patterns: List[str] - Files to exclude
            severity_threshold: str - Minimum severity to report
        """
        scan_id = f"sast-{len(self.sast_scans) + 1}"
        scanner = config.get('scanner', 'semgrep')

        scan = {
            'scan_id': scan_id,
            'type': ScanType.SAST.value,
            'project_id': config['project_id'],
            'branch': config.get('branch', 'main'),
            'scanner': scanner,
            'status': ScanStatus.SUCCESS.value,
            'file_patterns': config.get('file_patterns', ['**/*']),
            'exclude_patterns': config.get('exclude_patterns', ['**/test/**', '**/node_modules/**']),
            'severity_threshold': config.get('severity_threshold', Severity.LOW.value),
            'files_scanned': config.get('files_scanned', 150),
            'lines_scanned': config.get('lines_scanned', 12500),
            'scan_duration_seconds': config.get('duration', 45),
            'vulnerabilities': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 45))).isoformat()
        }

        # Generate sample vulnerabilities
        vuln_count = config.get('vulnerability_count', 5)
        for i in range(vuln_count):
            scan['vulnerabilities'].append({
                'id': f"{scan_id}-vuln-{i+1}",
                'severity': config.get('severity', Severity.MEDIUM.value),
                'confidence': config.get('confidence', 'high'),
                'category': config.get('category', 'injection'),
                'name': f"SQL Injection in {config.get('file', 'users.py')}",
                'description': 'User input is used in SQL query without sanitization',
                'file': config.get('file', 'src/users.py'),
                'line': config.get('line', 45),
                'cwe': config.get('cwe', 'CWE-89'),
                'owasp': config.get('owasp', 'A3:2017-Injection'),
                'solution': 'Use parameterized queries or ORM',
                'state': VulnerabilityState.DETECTED.value
            })

        self.sast_scans[scan_id] = scan
        return scan

    def get_sast_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get SAST scan details."""
        return self.sast_scans.get(scan_id)

    def list_sast_scans(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List SAST scans with optional filters."""
        scans = list(self.sast_scans.values())

        if filters:
            if 'project_id' in filters:
                scans = [s for s in scans if s['project_id'] == filters['project_id']]
            if 'branch' in filters:
                scans = [s for s in scans if s['branch'] == filters['branch']]
            if 'status' in filters:
                scans = [s for s in scans if s['status'] == filters['status']]

        return scans


class DASTScannerManager:
    """DAST (Dynamic Application Security Testing) management."""

    def __init__(self):
        self.dast_scans = {}

    def run_dast_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run DAST scan on running application.

        Config:
            target_url: str - Application URL to scan
            authentication: Dict - Auth config (type, username, password)
            spider_timeout: int - Max spider time in minutes
            scan_profile: str - passive, active, full
            excluded_urls: List[str] - URLs to exclude
            api_specification: str - OpenAPI/Swagger URL for API testing
        """
        scan_id = f"dast-{len(self.dast_scans) + 1}"

        scan = {
            'scan_id': scan_id,
            'type': ScanType.DAST.value,
            'target_url': config['target_url'],
            'status': ScanStatus.SUCCESS.value,
            'scan_profile': config.get('scan_profile', 'active'),
            'authentication': {
                'enabled': 'authentication' in config,
                'type': config.get('authentication', {}).get('type', 'form'),
                'username_field': config.get('authentication', {}).get('username_field', 'email')
            },
            'spider_timeout': config.get('spider_timeout', 30),
            'excluded_urls': config.get('excluded_urls', []),
            'api_specification': config.get('api_specification'),
            'requests_sent': config.get('requests_sent', 2500),
            'pages_crawled': config.get('pages_crawled', 75),
            'scan_duration_seconds': config.get('duration', 1800),
            'vulnerabilities': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 1800))).isoformat()
        }

        # Generate sample vulnerabilities
        vuln_count = config.get('vulnerability_count', 3)
        for i in range(vuln_count):
            scan['vulnerabilities'].append({
                'id': f"{scan_id}-vuln-{i+1}",
                'severity': config.get('severity', Severity.HIGH.value),
                'category': config.get('category', 'xss'),
                'name': 'Cross-Site Scripting (XSS)',
                'description': 'User input reflected in response without encoding',
                'url': f"{config['target_url']}/search?q=<script>alert(1)</script>",
                'method': 'GET',
                'parameter': 'q',
                'evidence': '<script>alert(1)</script>',
                'cwe': 'CWE-79',
                'owasp': 'A7:2017-Cross-Site Scripting (XSS)',
                'solution': 'Encode user input before rendering in HTML',
                'state': VulnerabilityState.DETECTED.value
            })

        self.dast_scans[scan_id] = scan
        return scan

    def get_dast_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get DAST scan details."""
        return self.dast_scans.get(scan_id)


class DependencyScannerManager:
    """Dependency scanning for vulnerable packages."""

    def __init__(self):
        self.dependency_scans = {}

    def run_dependency_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan project dependencies for known vulnerabilities.

        Config:
            project_id: str - Project identifier
            package_manager: str - npm, pip, maven, bundler, etc.
            manifest_file: str - package.json, requirements.txt, pom.xml
            lock_file: str - package-lock.json, Pipfile.lock, etc.
        """
        scan_id = f"dep-{len(self.dependency_scans) + 1}"

        scan = {
            'scan_id': scan_id,
            'type': ScanType.DEPENDENCY.value,
            'project_id': config['project_id'],
            'package_manager': config.get('package_manager', 'npm'),
            'manifest_file': config.get('manifest_file', 'package.json'),
            'lock_file': config.get('lock_file', 'package-lock.json'),
            'status': ScanStatus.SUCCESS.value,
            'dependencies_scanned': config.get('dependencies_scanned', 245),
            'scan_duration_seconds': config.get('duration', 30),
            'vulnerabilities': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 30))).isoformat()
        }

        # Generate sample vulnerable dependencies
        vuln_count = config.get('vulnerability_count', 4)
        for i in range(vuln_count):
            scan['vulnerabilities'].append({
                'id': f"{scan_id}-vuln-{i+1}",
                'severity': config.get('severity', Severity.HIGH.value),
                'package_name': config.get('package_name', 'lodash'),
                'installed_version': config.get('installed_version', '4.17.15'),
                'fixed_version': config.get('fixed_version', '4.17.21'),
                'cve': config.get('cve', 'CVE-2021-23337'),
                'cvss_score': config.get('cvss_score', 7.5),
                'description': 'Command injection vulnerability in lodash',
                'solution': 'Upgrade to version 4.17.21 or later',
                'published_date': '2021-02-15',
                'state': VulnerabilityState.DETECTED.value
            })

        self.dependency_scans[scan_id] = scan
        return scan

    def get_dependency_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get dependency scan details."""
        return self.dependency_scans.get(scan_id)


class ContainerScannerManager:
    """Container image scanning for vulnerabilities."""

    def __init__(self):
        self.container_scans = {}

    def run_container_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan container image for vulnerabilities.

        Config:
            image: str - Image name and tag
            registry: str - Container registry URL
            scanner: str - trivy, clair, grype
            severity_threshold: str - Minimum severity to report
        """
        scan_id = f"container-{len(self.container_scans) + 1}"
        scanner = config.get('scanner', 'trivy')

        scan = {
            'scan_id': scan_id,
            'type': ScanType.CONTAINER.value,
            'image': config['image'],
            'registry': config.get('registry', 'registry.gitlab.com'),
            'scanner': scanner,
            'status': ScanStatus.SUCCESS.value,
            'image_digest': f"sha256:{hashlib.sha256(config['image'].encode()).hexdigest()}",
            'base_image': config.get('base_image', 'ubuntu:20.04'),
            'layers_scanned': config.get('layers_scanned', 12),
            'packages_scanned': config.get('packages_scanned', 356),
            'scan_duration_seconds': config.get('duration', 60),
            'vulnerabilities': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 60))).isoformat()
        }

        # Generate sample vulnerabilities
        vuln_count = config.get('vulnerability_count', 8)
        for i in range(vuln_count):
            scan['vulnerabilities'].append({
                'id': f"{scan_id}-vuln-{i+1}",
                'severity': config.get('severity', Severity.CRITICAL.value),
                'package_name': config.get('package_name', 'openssl'),
                'installed_version': config.get('installed_version', '1.1.1f'),
                'fixed_version': config.get('fixed_version', '1.1.1k'),
                'cve': config.get('cve', 'CVE-2021-3450'),
                'cvss_score': config.get('cvss_score', 9.8),
                'layer': config.get('layer', 'sha256:abc123...'),
                'description': 'CA certificate check bypass vulnerability',
                'solution': 'Update base image to ubuntu:22.04',
                'state': VulnerabilityState.DETECTED.value
            })

        self.container_scans[scan_id] = scan
        return scan

    def get_container_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get container scan details."""
        return self.container_scans.get(scan_id)


class SecretDetectionManager:
    """Secret detection in source code."""

    def __init__(self):
        self.secret_scans = {}

    def run_secret_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect hardcoded secrets in source code.

        Config:
            project_id: str - Project identifier
            branch: str - Branch to scan
            paths: List[str] - Paths to scan
            custom_patterns: List[Dict] - Custom regex patterns
        """
        scan_id = f"secret-{len(self.secret_scans) + 1}"

        scan = {
            'scan_id': scan_id,
            'type': ScanType.SECRET_DETECTION.value,
            'project_id': config['project_id'],
            'branch': config.get('branch', 'main'),
            'status': ScanStatus.SUCCESS.value,
            'files_scanned': config.get('files_scanned', 180),
            'scan_duration_seconds': config.get('duration', 25),
            'secrets_found': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 25))).isoformat()
        }

        # Generate sample secrets found
        secret_count = config.get('secret_count', 2)
        for i in range(secret_count):
            scan['secrets_found'].append({
                'id': f"{scan_id}-secret-{i+1}",
                'type': config.get('type', 'aws_access_key'),
                'description': 'AWS Access Key detected',
                'file': config.get('file', 'src/config.py'),
                'line': config.get('line', 12),
                'commit': config.get('commit', 'abc123...'),
                'match': 'AKIA...[REDACTED]',
                'severity': Severity.CRITICAL.value,
                'state': VulnerabilityState.DETECTED.value
            })

        self.secret_scans[scan_id] = scan
        return scan

    def get_secret_scan(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get secret detection scan details."""
        return self.secret_scans.get(scan_id)


class LicenseScanningManager:
    """License compliance scanning."""

    def __init__(self):
        self.license_scans = {}
        self.license_policies = {}

    def create_license_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create license compliance policy.

        Config:
            name: str - Policy name
            approved_licenses: List[str] - Allowed licenses
            denied_licenses: List[str] - Prohibited licenses
        """
        policy_id = f"policy-{len(self.license_policies) + 1}"

        policy = {
            'policy_id': policy_id,
            'name': config['name'],
            'approved_licenses': config.get('approved_licenses', ['MIT', 'Apache-2.0', 'BSD-3-Clause']),
            'denied_licenses': config.get('denied_licenses', ['GPL-3.0', 'AGPL-3.0']),
            'created_at': datetime.now().isoformat()
        }

        self.license_policies[policy_id] = policy
        return policy

    def run_license_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scan project dependencies for license compliance.

        Config:
            project_id: str - Project identifier
            policy_id: str - License policy to enforce
        """
        scan_id = f"license-{len(self.license_scans) + 1}"

        scan = {
            'scan_id': scan_id,
            'type': ScanType.LICENSE_SCANNING.value,
            'project_id': config['project_id'],
            'policy_id': config.get('policy_id'),
            'status': ScanStatus.SUCCESS.value,
            'dependencies_scanned': config.get('dependencies_scanned', 245),
            'scan_duration_seconds': config.get('duration', 20),
            'licenses_found': [],
            'violations': [],
            'started_at': datetime.now().isoformat(),
            'completed_at': (datetime.now() + timedelta(seconds=config.get('duration', 20))).isoformat()
        }

        # Generate sample license findings
        license_count = config.get('license_count', 5)
        for i in range(license_count):
            license_info = {
                'package_name': f"package-{i+1}",
                'license': config.get('license', 'MIT'),
                'version': config.get('version', '1.0.0'),
                'approved': True
            }
            scan['licenses_found'].append(license_info)

        # Add violations if any
        violation_count = config.get('violation_count', 1)
        for i in range(violation_count):
            scan['violations'].append({
                'package_name': config.get('violation_package', 'gpl-package'),
                'license': config.get('violation_license', 'GPL-3.0'),
                'version': '2.1.0',
                'severity': Severity.HIGH.value,
                'reason': 'Copyleft license conflicts with proprietary codebase'
            })

        self.license_scans[scan_id] = scan
        return scan


class VulnerabilityManager:
    """Vulnerability tracking and management."""

    def __init__(self):
        self.vulnerabilities = {}

    def create_vulnerability(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create/track vulnerability from scan results.

        Config:
            scan_id: str - Source scan ID
            severity: str - Severity level
            category: str - Vulnerability category
            title: str - Vulnerability title
            description: str - Description
            cve: str - CVE identifier (optional)
        """
        vuln_id = f"vuln-{len(self.vulnerabilities) + 1}"

        vulnerability = {
            'vulnerability_id': vuln_id,
            'scan_id': config['scan_id'],
            'severity': config['severity'],
            'category': config.get('category', 'security'),
            'title': config['title'],
            'description': config['description'],
            'cve': config.get('cve'),
            'cvss_score': config.get('cvss_score'),
            'state': VulnerabilityState.DETECTED.value,
            'assignee': config.get('assignee'),
            'due_date': config.get('due_date'),
            'detected_at': datetime.now().isoformat(),
            'resolved_at': None
        }

        self.vulnerabilities[vuln_id] = vulnerability
        return vulnerability

    def update_vulnerability_state(self, vuln_id: str, state: str, comment: Optional[str] = None) -> Dict[str, Any]:
        """Update vulnerability state (confirmed, dismissed, resolved, false_positive)."""
        if vuln_id not in self.vulnerabilities:
            raise ValueError(f"Vulnerability {vuln_id} not found")

        self.vulnerabilities[vuln_id]['state'] = state
        self.vulnerabilities[vuln_id]['state_updated_at'] = datetime.now().isoformat()

        if comment:
            self.vulnerabilities[vuln_id]['state_comment'] = comment

        if state == VulnerabilityState.RESOLVED.value:
            self.vulnerabilities[vuln_id]['resolved_at'] = datetime.now().isoformat()

        return self.vulnerabilities[vuln_id]

    def get_vulnerability_statistics(self, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get vulnerability statistics."""
        vulns = list(self.vulnerabilities.values())

        if filters:
            if 'severity' in filters:
                vulns = [v for v in vulns if v['severity'] == filters['severity']]
            if 'state' in filters:
                vulns = [v for v in vulns if v['state'] == filters['state']]

        stats = {
            'total': len(vulns),
            'by_severity': {},
            'by_state': {},
            'avg_time_to_resolve': None
        }

        for vuln in vulns:
            # Count by severity
            severity = vuln['severity']
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1

            # Count by state
            state = vuln['state']
            stats['by_state'][state] = stats['by_state'].get(state, 0) + 1

        return stats


class SecurityPolicyManager:
    """Security policies and compliance rules."""

    def __init__(self):
        self.policies = {}

    def create_security_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create security policy.

        Config:
            name: str - Policy name
            scan_types: List[str] - Required scan types
            severity_thresholds: Dict - Max allowed vulnerabilities by severity
            block_on_violation: bool - Block pipeline on policy violation
        """
        policy_id = f"policy-{len(self.policies) + 1}"

        policy = {
            'policy_id': policy_id,
            'name': config['name'],
            'scan_types': config.get('scan_types', [ScanType.SAST.value, ScanType.DEPENDENCY.value]),
            'severity_thresholds': config.get('severity_thresholds', {
                Severity.CRITICAL.value: 0,
                Severity.HIGH.value: 5,
                Severity.MEDIUM.value: 20
            }),
            'block_on_violation': config.get('block_on_violation', True),
            'created_at': datetime.now().isoformat()
        }

        self.policies[policy_id] = policy
        return policy

    def evaluate_policy(self, policy_id: str, scan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate scan results against security policy."""
        if policy_id not in self.policies:
            raise ValueError(f"Policy {policy_id} not found")

        policy = self.policies[policy_id]

        # Count vulnerabilities by severity
        vuln_counts = {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 0,
            Severity.MEDIUM.value: 0,
            Severity.LOW.value: 0
        }

        for scan in scan_results:
            for vuln in scan.get('vulnerabilities', []):
                severity = vuln.get('severity', Severity.UNKNOWN.value)
                if severity in vuln_counts:
                    vuln_counts[severity] += 1

        # Check against thresholds
        violations = []
        for severity, threshold in policy['severity_thresholds'].items():
            if vuln_counts.get(severity, 0) > threshold:
                violations.append({
                    'severity': severity,
                    'threshold': threshold,
                    'actual': vuln_counts[severity],
                    'excess': vuln_counts[severity] - threshold
                })

        evaluation = {
            'policy_id': policy_id,
            'policy_name': policy['name'],
            'passed': len(violations) == 0,
            'vulnerability_counts': vuln_counts,
            'violations': violations,
            'block_pipeline': policy['block_on_violation'] and len(violations) > 0,
            'evaluated_at': datetime.now().isoformat()
        }

        return evaluation


class SecurityScanningManager:
    """Main security scanning orchestration."""

    def __init__(self, gitlab_url: str = 'https://gitlab.com'):
        self.gitlab_url = gitlab_url
        self.sast = SASTScannerManager()
        self.dast = DASTScannerManager()
        self.dependencies = DependencyScannerManager()
        self.containers = ContainerScannerManager()
        self.secrets = SecretDetectionManager()
        self.licenses = LicenseScanningManager()
        self.vulnerabilities = VulnerabilityManager()
        self.policies = SecurityPolicyManager()

    def run_full_security_scan(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run comprehensive security scan across all scan types.

        Config:
            project_id: str - Project identifier
            branch: str - Branch to scan
            target_url: str - Application URL for DAST
            image: str - Container image for scanning
            scan_types: List[str] - Scan types to run
        """
        scan_types = config.get('scan_types', [
            ScanType.SAST.value,
            ScanType.DEPENDENCY.value,
            ScanType.SECRET_DETECTION.value
        ])

        results = {
            'project_id': config['project_id'],
            'branch': config.get('branch', 'main'),
            'scan_types': scan_types,
            'scans': {},
            'started_at': datetime.now().isoformat()
        }

        # Run SAST if requested
        if ScanType.SAST.value in scan_types:
            sast_result = self.sast.run_sast_scan(config)
            results['scans']['sast'] = sast_result

        # Run DAST if requested and URL provided
        if ScanType.DAST.value in scan_types and 'target_url' in config:
            dast_result = self.dast.run_dast_scan(config)
            results['scans']['dast'] = dast_result

        # Run dependency scan if requested
        if ScanType.DEPENDENCY.value in scan_types:
            dep_result = self.dependencies.run_dependency_scan(config)
            results['scans']['dependency'] = dep_result

        # Run container scan if requested and image provided
        if ScanType.CONTAINER.value in scan_types and 'image' in config:
            container_result = self.containers.run_container_scan(config)
            results['scans']['container'] = container_result

        # Run secret detection if requested
        if ScanType.SECRET_DETECTION.value in scan_types:
            secret_result = self.secrets.run_secret_detection(config)
            results['scans']['secret_detection'] = secret_result

        # Run license scan if requested
        if ScanType.LICENSE_SCANNING.value in scan_types:
            license_result = self.licenses.run_license_scan(config)
            results['scans']['license'] = license_result

        results['completed_at'] = datetime.now().isoformat()
        return results

    def get_security_dashboard(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive security dashboard for project."""
        dashboard = {
            'project_id': project_id,
            'sast_scans': len([s for s in self.sast.sast_scans.values() if s['project_id'] == project_id]),
            'dependency_scans': len([s for s in self.dependencies.dependency_scans.values() if s['project_id'] == project_id]),
            'secret_scans': len([s for s in self.secrets.secret_scans.values() if s['project_id'] == project_id]),
            'total_vulnerabilities': len(self.vulnerabilities.vulnerabilities),
            'vulnerability_statistics': self.vulnerabilities.get_vulnerability_statistics(),
            'generated_at': datetime.now().isoformat()
        }

        return dashboard


def demo():
    """Demonstrate security scanning capabilities."""
    print("=" * 70)
    print("GitLab Security Scanning - Comprehensive Demo")
    print("=" * 70)

    mgr = SecurityScanningManager()

    # 1. Run SAST scan
    print("\n1. Running SAST scan...")
    sast_result = mgr.sast.run_sast_scan({
        'project_id': 'myorg/webapp',
        'branch': 'main',
        'scanner': 'semgrep',
        'vulnerability_count': 3,
        'severity': Severity.HIGH.value
    })
    print(f"   ✓ SAST scan completed: {sast_result['scan_id']}")
    print(f"   Files scanned: {sast_result['files_scanned']}")
    print(f"   Vulnerabilities found: {len(sast_result['vulnerabilities'])}")

    # 2. Run dependency scan
    print("\n2. Running dependency scan...")
    dep_result = mgr.dependencies.run_dependency_scan({
        'project_id': 'myorg/webapp',
        'package_manager': 'npm',
        'vulnerability_count': 4,
        'severity': Severity.CRITICAL.value
    })
    print(f"   ✓ Dependency scan completed: {dep_result['scan_id']}")
    print(f"   Dependencies scanned: {dep_result['dependencies_scanned']}")
    print(f"   Vulnerable packages: {len(dep_result['vulnerabilities'])}")

    # 3. Run container scan
    print("\n3. Running container scan...")
    container_result = mgr.containers.run_container_scan({
        'image': 'myapp:1.2.3',
        'scanner': 'trivy',
        'vulnerability_count': 6,
        'severity': Severity.CRITICAL.value
    })
    print(f"   ✓ Container scan completed: {container_result['scan_id']}")
    print(f"   Image: {container_result['image']}")
    print(f"   Vulnerabilities: {len(container_result['vulnerabilities'])}")

    # 4. Run secret detection
    print("\n4. Running secret detection...")
    secret_result = mgr.secrets.run_secret_detection({
        'project_id': 'myorg/webapp',
        'branch': 'main',
        'secret_count': 2
    })
    print(f"   ✓ Secret detection completed: {secret_result['scan_id']}")
    print(f"   Files scanned: {secret_result['files_scanned']}")
    print(f"   Secrets found: {len(secret_result['secrets_found'])}")

    # 5. Create security policy
    print("\n5. Creating security policy...")
    policy = mgr.policies.create_security_policy({
        'name': 'Production Security Policy',
        'severity_thresholds': {
            Severity.CRITICAL.value: 0,
            Severity.HIGH.value: 3,
            Severity.MEDIUM.value: 15
        },
        'block_on_violation': True
    })
    print(f"   ✓ Policy created: {policy['name']}")
    print(f"   Thresholds: {policy['severity_thresholds']}")

    # 6. Evaluate policy
    print("\n6. Evaluating security policy...")
    evaluation = mgr.policies.evaluate_policy(
        policy['policy_id'],
        [sast_result, dep_result, container_result]
    )
    print(f"   ✓ Policy evaluation completed")
    print(f"   Passed: {evaluation['passed']}")
    print(f"   Vulnerabilities: {evaluation['vulnerability_counts']}")
    if evaluation['violations']:
        print(f"   Violations: {len(evaluation['violations'])}")

    # 7. Get security dashboard
    print("\n7. Security dashboard...")
    dashboard = mgr.get_security_dashboard('myorg/webapp')
    print(f"   ✓ Dashboard generated")
    print(f"   SAST scans: {dashboard['sast_scans']}")
    print(f"   Dependency scans: {dashboard['dependency_scans']}")
    print(f"   Total vulnerabilities: {dashboard['total_vulnerabilities']}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo()
