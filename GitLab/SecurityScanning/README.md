# Security Scanning - Comprehensive Application Security

GitLab security scanning system providing SAST, DAST, dependency scanning, container scanning, secret detection, license compliance, vulnerability management, and security policies.

## Features

### SAST (Static Application Security Testing)
- **Code Analysis**: Scan source code for vulnerabilities
- **Multi-Language**: Python (Bandit), JavaScript (ESLint), Ruby (Brakeman), Java (SpotBugs), Go (Gosec)
- **Semgrep**: Advanced static analysis with custom rules
- **Vulnerability Detection**: SQL injection, XSS, CSRF, etc.
- **CWE/OWASP Mapping**: Industry-standard classifications

### DAST (Dynamic Application Security Testing)
- **Runtime Scanning**: Test running applications
- **Authentication**: Form-based, HTTP basic, token auth
- **API Testing**: OpenAPI/Swagger specification support
- **Spider Crawling**: Automatic page discovery
- **Scan Profiles**: Passive, active, full coverage

### Dependency Scanning
- **CVE Detection**: Known vulnerabilities in dependencies
- **Package Managers**: npm, pip, Maven, Bundler, Composer, Go modules
- **CVSS Scores**: Severity ratings for vulnerabilities
- **Fix Recommendations**: Upgrade paths to patched versions

### Container Scanning
- **Image Analysis**: Trivy, Clair, Grype scanners
- **Layer Scanning**: Vulnerability tracking per layer
- **Base Image Checks**: Identify vulnerable base images
- **Package Detection**: OS packages and libraries

### Secret Detection
- **Pattern Matching**: AWS keys, API tokens, passwords
- **Custom Patterns**: Configurable regex patterns
- **Commit History**: Scan historical commits
- **Severity Classification**: CRITICAL severity for exposed secrets

### License Compliance
- **License Detection**: Identify package licenses
- **Policy Enforcement**: Approved/denied license lists
- **Violation Reporting**: Flag non-compliant licenses
- **Copyleft Detection**: GPL, AGPL warnings

### Vulnerability Management
- **Tracking**: Centralized vulnerability database
- **Triage States**: Detected, confirmed, dismissed, resolved, false positive
- **Assignment**: Assign vulnerabilities to team members
- **Due Dates**: Track remediation deadlines
- **Statistics**: Vulnerability metrics and trends

### Security Policies
- **Severity Thresholds**: Max allowed vulnerabilities per severity
- **Pipeline Blocking**: Fail pipelines on policy violations
- **Scan Requirements**: Enforce specific scan types
- **Compliance Rules**: Customizable security rules

## Usage Example

```python
from security_scanner import SecurityScanningManager, ScanType, Severity

# Initialize
mgr = SecurityScanningManager()

# 1. Run SAST scan
sast_scan = mgr.sast.run_sast_scan({
    'project_id': 'myorg/webapp',
    'branch': 'main',
    'scanner': 'semgrep',
    'file_patterns': ['**/*.py', '**/*.js'],
    'exclude_patterns': ['**/test/**', '**/node_modules/**'],
    'severity_threshold': Severity.MEDIUM.value
})

# 2. Run dependency scan
dep_scan = mgr.dependencies.run_dependency_scan({
    'project_id': 'myorg/webapp',
    'package_manager': 'npm',
    'manifest_file': 'package.json',
    'lock_file': 'package-lock.json'
})

# 3. Run container scan
container_scan = mgr.containers.run_container_scan({
    'image': 'myapp:1.2.3',
    'registry': 'registry.gitlab.com',
    'scanner': 'trivy',
    'severity_threshold': Severity.HIGH.value
})

# 4. Run secret detection
secret_scan = mgr.secrets.run_secret_detection({
    'project_id': 'myorg/webapp',
    'branch': 'main',
    'paths': ['src/**']
})

# 5. Run DAST scan
dast_scan = mgr.dast.run_dast_scan({
    'target_url': 'https://staging.example.com',
    'authentication': {
        'type': 'form',
        'username': 'testuser',
        'password': 'testpass',
        'username_field': 'email',
        'password_field': 'password'
    },
    'scan_profile': 'active',
    'api_specification': 'https://staging.example.com/openapi.json'
})

# 6. Run license scan with policy
license_policy = mgr.licenses.create_license_policy({
    'name': 'Corporate License Policy',
    'approved_licenses': ['MIT', 'Apache-2.0', 'BSD-3-Clause', 'ISC'],
    'denied_licenses': ['GPL-3.0', 'AGPL-3.0', 'SSPL']
})

license_scan = mgr.licenses.run_license_scan({
    'project_id': 'myorg/webapp',
    'policy_id': license_policy['policy_id']
})

# 7. Create security policy
security_policy = mgr.policies.create_security_policy({
    'name': 'Production Security Policy',
    'scan_types': [ScanType.SAST.value, ScanType.DEPENDENCY.value, ScanType.SECRET_DETECTION.value],
    'severity_thresholds': {
        Severity.CRITICAL.value: 0,
        Severity.HIGH.value: 5,
        Severity.MEDIUM.value: 20,
        Severity.LOW.value: 50
    },
    'block_on_violation': True
})

# 8. Evaluate policy against scans
evaluation = mgr.policies.evaluate_policy(
    security_policy['policy_id'],
    [sast_scan, dep_scan, container_scan, secret_scan]
)

if not evaluation['passed']:
    print(f"Security policy violations: {evaluation['violations']}")
    if evaluation['block_pipeline']:
        print("Pipeline blocked due to security policy violations")

# 9. Track vulnerabilities
for vuln in sast_scan['vulnerabilities']:
    tracked = mgr.vulnerabilities.create_vulnerability({
        'scan_id': sast_scan['scan_id'],
        'severity': vuln['severity'],
        'title': vuln['name'],
        'description': vuln['description'],
        'cve': vuln.get('cwe'),
        'assignee': 'security-team'
    })

# 10. Update vulnerability states
mgr.vulnerabilities.update_vulnerability_state(
    'vuln-1',
    'confirmed',
    comment='Confirmed SQL injection, high priority fix'
)

mgr.vulnerabilities.update_vulnerability_state(
    'vuln-2',
    'false_positive',
    comment='False positive - using parameterized queries'
)

# 11. Get vulnerability statistics
vuln_stats = mgr.vulnerabilities.get_vulnerability_statistics()
print(f"Total vulnerabilities: {vuln_stats['total']}")
print(f"By severity: {vuln_stats['by_severity']}")
print(f"By state: {vuln_stats['by_state']}")

# 12. Get security dashboard
dashboard = mgr.get_security_dashboard('myorg/webapp')
print(f"Security Dashboard for {dashboard['project_id']}")
print(f"SAST scans: {dashboard['sast_scans']}")
print(f"Dependency scans: {dashboard['dependency_scans']}")
print(f"Vulnerabilities: {dashboard['vulnerability_statistics']}")

# 13. Run full security scan
full_scan = mgr.run_full_security_scan({
    'project_id': 'myorg/webapp',
    'branch': 'main',
    'target_url': 'https://staging.example.com',
    'image': 'myapp:1.2.3',
    'scan_types': [
        ScanType.SAST.value,
        ScanType.DAST.value,
        ScanType.DEPENDENCY.value,
        ScanType.CONTAINER.value,
        ScanType.SECRET_DETECTION.value,
        ScanType.LICENSE_SCANNING.value
    ]
})
```

## Scan Types

| Scan Type | Purpose | When to Use |
|-----------|---------|-------------|
| **SAST** | Static code analysis | Every commit, pre-merge |
| **DAST** | Dynamic app testing | Staging/production deployments |
| **Dependency** | Vulnerable packages | Every build, daily |
| **Container** | Image vulnerabilities | Container builds, deployments |
| **Secret Detection** | Exposed credentials | Every commit, historical scans |
| **License Scanning** | License compliance | Release builds, audits |

## Vulnerability Severity Levels

| Severity | CVSS Score | Response Time | Examples |
|----------|------------|---------------|----------|
| **CRITICAL** | 9.0-10.0 | Immediate | RCE, authentication bypass |
| **HIGH** | 7.0-8.9 | 1-7 days | SQL injection, XSS |
| **MEDIUM** | 4.0-6.9 | 30 days | CSRF, information disclosure |
| **LOW** | 0.1-3.9 | 90 days | Minor information leaks |
| **INFO** | 0.0 | No action | Best practice violations |

## GitLab CI/CD Integration

### SAST Scanning
```yaml
include:
  - template: Security/SAST.gitlab-ci.yml

sast:
  stage: test
  variables:
    SAST_EXCLUDED_PATHS: "spec,test,tests,tmp"
  artifacts:
    reports:
      sast: gl-sast-report.json
```

### Dependency Scanning
```yaml
include:
  - template: Security/Dependency-Scanning.gitlab-ci.yml

dependency_scanning:
  stage: test
  artifacts:
    reports:
      dependency_scanning: gl-dependency-scanning-report.json
```

### Container Scanning
```yaml
include:
  - template: Security/Container-Scanning.gitlab-ci.yml

container_scanning:
  stage: test
  variables:
    CS_IMAGE: $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  artifacts:
    reports:
      container_scanning: gl-container-scanning-report.json
```

### Secret Detection
```yaml
include:
  - template: Security/Secret-Detection.gitlab-ci.yml

secret_detection:
  stage: test
  artifacts:
    reports:
      secret_detection: gl-secret-detection-report.json
```

### DAST Scanning
```yaml
include:
  - template: Security/DAST.gitlab-ci.yml

dast:
  stage: test
  variables:
    DAST_WEBSITE: https://staging.example.com
    DAST_AUTH_URL: https://staging.example.com/login
    DAST_USERNAME_FIELD: email
    DAST_PASSWORD_FIELD: password
  artifacts:
    reports:
      dast: gl-dast-report.json
```

### License Scanning
```yaml
include:
  - template: Security/License-Scanning.gitlab-ci.yml

license_scanning:
  stage: test
  artifacts:
    reports:
      license_scanning: gl-license-scanning-report.json
```

### Complete Security Pipeline
```yaml
stages:
  - build
  - test
  - security
  - deploy

include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Dependency-Scanning.gitlab-ci.yml
  - template: Security/Container-Scanning.gitlab-ci.yml
  - template: Security/Secret-Detection.gitlab-ci.yml
  - template: Security/License-Scanning.gitlab-ci.yml

# Run all security scans in parallel
security_scan:
  stage: security
  script:
    - echo "Security scans running in parallel..."
  needs: ['build']

# Block deployment on critical vulnerabilities
deploy:
  stage: deploy
  script:
    - echo "Deploying application..."
  rules:
    - if: '$CI_COMMIT_BRANCH == "main"'
  # This will fail if critical vulnerabilities are found
  needs:
    - job: sast
      artifacts: true
    - job: dependency_scanning
      artifacts: true
```

## SAST Scanners by Language

| Language | Scanner | Coverage |
|----------|---------|----------|
| **Python** | Bandit | SQL injection, hardcoded passwords, etc. |
| **JavaScript** | ESLint, Semgrep | XSS, prototype pollution, etc. |
| **Ruby** | Brakeman | Rails vulnerabilities, SQL injection |
| **Java** | SpotBugs, FindSecBugs | SQL injection, XXE, etc. |
| **Go** | Gosec | SQL injection, file inclusion |
| **PHP** | phpcs-security-audit | SQL injection, XSS |
| **C/C++** | Flawfinder | Buffer overflows, format strings |
| **Multi-language** | Semgrep | Custom rules, all languages |

## Dependency Scanners

| Package Manager | Scanner | Vulnerability Database |
|-----------------|---------|------------------------|
| **npm** | npm audit, Snyk | National Vulnerability Database |
| **pip** | Safety, pip-audit | PyPI Advisory Database |
| **Maven** | OWASP Dependency Check | NVD, OSS Index |
| **Bundler** | bundler-audit | RubySec Advisory Database |
| **Composer** | Local PHP Security Checker | FriendsOfPHP Security Advisories |
| **Go** | govulncheck | Go Vulnerability Database |
| **NuGet** | dotnet list package | NuGet Advisory Database |

## Container Scanners

| Scanner | Strengths | Use Case |
|---------|-----------|----------|
| **Trivy** | Fast, comprehensive, easy to use | General purpose, CI/CD |
| **Clair** | CoreOS ecosystem, layer analysis | Container registries |
| **Grype** | Anchore, policy enforcement | Enterprise compliance |
| **Snyk** | Developer-friendly, fix PRs | Development workflow |

## Security Policy Examples

### Strict Production Policy
```python
mgr.policies.create_security_policy({
    'name': 'Production - Zero Tolerance',
    'scan_types': [
        ScanType.SAST.value,
        ScanType.DEPENDENCY.value,
        ScanType.CONTAINER.value,
        ScanType.SECRET_DETECTION.value
    ],
    'severity_thresholds': {
        Severity.CRITICAL.value: 0,
        Severity.HIGH.value: 0,
        Severity.MEDIUM.value: 5
    },
    'block_on_violation': True
})
```

### Development Policy
```python
mgr.policies.create_security_policy({
    'name': 'Development - Balanced',
    'scan_types': [ScanType.SAST.value, ScanType.DEPENDENCY.value],
    'severity_thresholds': {
        Severity.CRITICAL.value: 0,
        Severity.HIGH.value: 10,
        Severity.MEDIUM.value: 30
    },
    'block_on_violation': False
})
```

### Feature Branch Policy
```python
mgr.policies.create_security_policy({
    'name': 'Feature Branches - Permissive',
    'scan_types': [ScanType.SAST.value, ScanType.SECRET_DETECTION.value],
    'severity_thresholds': {
        Severity.CRITICAL.value: 2,
        Severity.HIGH.value: 20,
        Severity.MEDIUM.value: 100
    },
    'block_on_violation': False
})
```

## Vulnerability Triage Workflow

```python
# 1. Scan detects vulnerability
sast_scan = mgr.sast.run_sast_scan({
    'project_id': 'myorg/webapp',
    'branch': 'main'
})

# 2. Create tracked vulnerability
for vuln in sast_scan['vulnerabilities']:
    tracked = mgr.vulnerabilities.create_vulnerability({
        'scan_id': sast_scan['scan_id'],
        'severity': vuln['severity'],
        'title': vuln['name'],
        'description': vuln['description']
    })

# 3. Security team reviews
mgr.vulnerabilities.update_vulnerability_state(
    'vuln-1',
    'confirmed',
    comment='Confirmed vulnerability, assigning to dev team'
)

# 4. Developer fixes
mgr.vulnerabilities.update_vulnerability_state(
    'vuln-1',
    'resolved',
    comment='Fixed in commit abc123, using parameterized queries'
)

# 5. False positive handling
mgr.vulnerabilities.update_vulnerability_state(
    'vuln-2',
    'false_positive',
    comment='Input is sanitized by framework, false positive'
)

# 6. Risk acceptance
mgr.vulnerabilities.update_vulnerability_state(
    'vuln-3',
    'dismissed',
    comment='Low risk, accepted by security team, will fix in next release'
)
```

## Best Practices

### SAST Scanning
1. **Early Integration**: Scan on every commit, not just before release
2. **Custom Rules**: Add project-specific security rules to Semgrep
3. **Exclude Tests**: Don't scan test/mock code to reduce noise
4. **Incremental Scans**: Scan only changed files for faster feedback
5. **Fix Immediately**: Address CRITICAL and HIGH findings before merge

### DAST Scanning
1. **Staging Environment**: Scan staging, not production
2. **Authentication**: Configure proper auth to scan protected pages
3. **API Specs**: Provide OpenAPI specs for comprehensive API testing
4. **Off-Peak**: Schedule heavy scans during low traffic periods
5. **Rate Limiting**: Don't overwhelm applications with requests

### Dependency Scanning
1. **Daily Scans**: Run daily to catch new CVEs
2. **Auto-Updates**: Enable automated dependency updates (Dependabot, Renovate)
3. **Lock Files**: Commit lock files for reproducible builds
4. **Transitive Deps**: Check indirect dependencies, not just direct
5. **Version Pinning**: Pin versions in production, ranges in development

### Container Scanning
1. **Scan on Build**: Every container build triggers scan
2. **Minimal Base**: Use minimal base images (alpine, distroless)
3. **Multi-Stage**: Use multi-stage builds to reduce attack surface
4. **Image Signing**: Sign images to verify integrity
5. **Regular Rebuilds**: Rebuild images weekly to get security patches

### Secret Detection
1. **Pre-Commit Hooks**: Detect secrets before commit
2. **Historical Scans**: Scan entire git history, not just new commits
3. **Immediate Rotation**: Rotate exposed secrets immediately
4. **Secret Managers**: Use GitLab variables, HashiCorp Vault, AWS Secrets Manager
5. **Custom Patterns**: Add organization-specific secret patterns

### License Compliance
1. **Policy First**: Define license policy before first scan
2. **Legal Review**: Have legal team approve license policy
3. **Continuous Monitoring**: Scan on every dependency change
4. **Copyleft Awareness**: Understand GPL/AGPL implications
5. **SBOM Generation**: Generate Software Bill of Materials for audits

## Common Vulnerabilities

### SAST Detects
- **SQL Injection** (CWE-89): Unsanitized input in SQL queries
- **Cross-Site Scripting** (CWE-79): Unescaped output in HTML
- **Command Injection** (CWE-78): User input in system commands
- **Path Traversal** (CWE-22): Unsanitized file paths
- **Hardcoded Credentials** (CWE-798): Passwords in source code

### DAST Detects
- **XSS** (CWE-79): Reflected/stored XSS in web apps
- **CSRF** (CWE-352): Missing CSRF tokens
- **Insecure Headers**: Missing security headers (CSP, HSTS)
- **SSL/TLS Issues**: Weak ciphers, expired certificates
- **Authentication Bypass**: Broken access control

### Dependency Scanning Detects
- **Known CVEs**: Published vulnerabilities in packages
- **Outdated Packages**: Old versions with known issues
- **Abandoned Packages**: Unmaintained dependencies
- **License Violations**: Non-compliant licenses

### Container Scanning Detects
- **OS Vulnerabilities**: Vulnerable system packages
- **Malware**: Known malicious files
- **Misconfigurations**: Insecure container settings
- **Secrets in Layers**: Exposed credentials in image layers

## Troubleshooting

**Issue**: SAST scan finds too many false positives
- Review exclude patterns to skip test/mock code
- Adjust severity threshold to focus on critical issues
- Use Semgrep custom rules to reduce noise
- Mark false positives to train scanner

**Issue**: DAST scan can't authenticate
- Verify authentication credentials are correct
- Check DAST_AUTH_URL points to login page
- Ensure username/password field names match
- Use API tokens instead of form auth if possible

**Issue**: Dependency scan missing vulnerabilities
- Ensure lock files are committed
- Update scanner database
- Check manifest files are in standard format
- Verify all package managers are configured

**Issue**: Container scan takes too long
- Use lighter scanners (Trivy > Clair for speed)
- Scan specific severity levels only
- Cache scanner databases
- Use minimal base images to reduce packages

**Issue**: Secrets detected in old commits
- Rotate exposed secrets immediately
- Use git-filter-repo to remove from history (caution!)
- Add to .gitignore to prevent future commits
- Enable pre-commit hooks to prevent

**Issue**: Pipeline blocked by security policy
- Review violations in scan reports
- Fix critical/high vulnerabilities
- Adjust policy thresholds if too strict
- Use vulnerability dismissal for false positives

## Requirements

```
hashlib (standard library)
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

```python
from security_scanner import SecurityScanningManager

mgr = SecurityScanningManager(gitlab_url='https://gitlab.com')
```

## Author

BrillConsulting - Enterprise Cloud Solutions
