"""
GitLab Security Scanning
Author: BrillConsulting
Description: SAST, DAST, dependency scanning, and container scanning integration
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class SecurityScanner:
    """GitLab Security Scanning management"""

    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.scans = []

    def run_sast_scan(self, scan_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run SAST (Static Application Security Testing) scan"""
        scan = {
            'scan_id': len(self.scans) + 1,
            'type': 'sast',
            'project_id': scan_config.get('project_id', 1),
            'branch': scan_config.get('branch', 'main'),
            'vulnerabilities': {
                'critical': scan_config.get('critical', 0),
                'high': scan_config.get('high', 2),
                'medium': scan_config.get('medium', 5),
                'low': scan_config.get('low', 10)
            },
            'scanned_at': datetime.now().isoformat()
        }

        gitlab_ci_sast = """include:
  - template: Security/SAST.gitlab-ci.yml

sast:
  stage: test
  artifacts:
    reports:
      sast: gl-sast-report.json
"""

        self.scans.append(scan)
        print(f"✓ SAST scan completed: {scan['scan_id']}")
        print(f"  Vulnerabilities: C={scan['vulnerabilities']['critical']}, H={scan['vulnerabilities']['high']}, M={scan['vulnerabilities']['medium']}, L={scan['vulnerabilities']['low']}")
        return scan

    def run_dast_scan(self, scan_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run DAST (Dynamic Application Security Testing) scan"""
        scan = {
            'scan_id': len(self.scans) + 1,
            'type': 'dast',
            'target_url': scan_config.get('target_url', 'https://example.com'),
            'vulnerabilities_found': scan_config.get('vulnerabilities', 3),
            'scanned_at': datetime.now().isoformat()
        }

        gitlab_ci_dast = f"""include:
  - template: Security/DAST.gitlab-ci.yml

dast:
  variables:
    DAST_WEBSITE: {scan['target_url']}
"""

        self.scans.append(scan)
        print(f"✓ DAST scan completed: {scan['target_url']}")
        print(f"  Vulnerabilities found: {scan['vulnerabilities_found']}")
        return scan

    def run_dependency_scan(self, scan_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run dependency scanning"""
        scan = {
            'scan_id': len(self.scans) + 1,
            'type': 'dependency_scanning',
            'project_id': scan_config.get('project_id', 1),
            'vulnerable_dependencies': scan_config.get('vulnerable_deps', 4),
            'scanned_at': datetime.now().isoformat()
        }

        print(f"✓ Dependency scan completed")
        print(f"  Vulnerable dependencies: {scan['vulnerable_dependencies']}")
        return scan

    def run_container_scan(self, scan_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run container scanning"""
        scan = {
            'scan_id': len(self.scans) + 1,
            'type': 'container_scanning',
            'image': scan_config.get('image', 'myapp:latest'),
            'vulnerabilities_found': scan_config.get('vulnerabilities', 8),
            'scanned_at': datetime.now().isoformat()
        }

        print(f"✓ Container scan completed: {scan['image']}")
        print(f"  Vulnerabilities found: {scan['vulnerabilities_found']}")
        return scan


def demo():
    """Demonstrate security scanning"""
    print("=" * 60)
    print("GitLab Security Scanning Demo")
    print("=" * 60)

    scanner = SecurityScanner('https://gitlab.example.com', 'token')

    print("\n1. Running SAST scan...")
    scanner.run_sast_scan({'project_id': 1, 'critical': 0, 'high': 2, 'medium': 5, 'low': 10})

    print("\n2. Running DAST scan...")
    scanner.run_dast_scan({'target_url': 'https://example.com', 'vulnerabilities': 3})

    print("\n3. Running dependency scan...")
    scanner.run_dependency_scan({'project_id': 1, 'vulnerable_deps': 4})

    print("\n4. Running container scan...")
    scanner.run_container_scan({'image': 'myapp:latest', 'vulnerabilities': 8})

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
