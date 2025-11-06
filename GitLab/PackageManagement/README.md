# Package Management - GitLab Package Registry

Comprehensive package registry system supporting 11+ package types (NPM, Maven, PyPI, NuGet, Composer, Conan, Go, Helm, Debian, RPM, Generic) with version management, dependency tracking, security scanning, and analytics.

## Features

### Package Registry
- **Multi-Type Support**: 11+ package types including NPM, Maven, PyPI, NuGet, Composer
- **Publish Packages**: Upload packages with metadata and visibility controls
- **List & Filter**: Filter by type, status, visibility, name
- **Delete Packages**: Remove packages from registry
- **Deprecate Packages**: Mark packages as deprecated with reason
- **Download Tracking**: Track package downloads

### Version Management
- **Semantic Versioning**: Support for semver (1.2.3)
- **Version History**: Complete version timeline
- **Latest Version**: Quick access to most recent version
- **Version Comparison**: Compare semantic versions
- **Release Notes**: Document changes per version
- **Breaking Changes**: Flag breaking changes

### Dependency Management
- **Dependency Tracking**: Track package dependencies
- **Dependency Types**: runtime, development, peer, optional
- **Dependency Trees**: Build complete dependency graphs
- **Conflict Detection**: Detect version conflicts
- **Version Constraints**: Support version ranges (^18.0.0, ~1.2.0)

### Security Scanning
- **Vulnerability Scanning**: Trivy, Snyk, Clair, Grype integration
- **CVE Tracking**: Track Common Vulnerabilities and Exposures
- **Severity Levels**: Critical, High, Medium, Low
- **Quarantine**: Isolate packages with security issues
- **Scan History**: Track all security scans
- **Automated Scanning**: Scan on publish

### Statistics & Analytics
- **Download Counts**: Track total and per-package downloads
- **Popular Packages**: Most downloaded packages
- **Registry Statistics**: Total packages, average downloads
- **Usage Trends**: Track package adoption

### Publish Commands
- **NPM**: npm publish with authentication
- **Maven**: pom.xml configuration and mvn deploy
- **PyPI**: twine upload with .pypirc
- **NuGet**: nuget push with source config
- **Helm**: Chart packaging and upload
- **Docker**: Container image push

## Usage Example

```python
from package_manager import PackageManager, PackageType, PackageVisibility

# Initialize manager
mgr = PackageManager(project_id='myorg/myproject')

# 1. Publish NPM package with dependencies
npm_pkg = mgr.publish_with_tracking({
    'package_type': PackageType.NPM.value,
    'package_name': '@myorg/react-components',
    'version': '1.2.3',
    'file_path': 'dist/package.tgz',
    'description': 'Reusable React components',
    'visibility': PackageVisibility.PUBLIC.value,
    'dependencies': [
        {'name': 'react', 'version': '^18.0.0', 'type': 'peer'},
        {'name': 'react-dom', 'version': '^18.0.0', 'type': 'peer'}
    ]
})

# 2. Publish PyPI package
pypi_pkg = mgr.registry.publish_package({
    'package_type': PackageType.PYPI.value,
    'package_name': 'myorg-utils',
    'version': '2.1.0',
    'file_path': 'dist/myorg_utils-2.1.0.tar.gz',
    'description': 'Utility functions',
    'visibility': PackageVisibility.INTERNAL.value
})

# 3. Version management
latest = mgr.versions.get_latest_version('@myorg/react-components')
print(f"Latest: {latest['version']}")

comparison = mgr.versions.compare_versions('1.2.3', '1.2.0')
# Output: "1.2.3 > 1.2.0"

# 4. Dependency analysis
deps = mgr.dependencies.get_dependencies(npm_pkg['package']['package_id'])
tree = mgr.dependencies.build_dependency_tree(npm_pkg['package']['package_id'])
conflicts = mgr.dependencies.detect_conflicts(npm_pkg['package']['package_id'])

# 5. Security scanning
scan = mgr.security.scan_package(npm_pkg['package']['package_id'], 'trivy')
# Returns: status, vulnerabilities_found, severity_counts

# Add vulnerability if found
mgr.security.add_vulnerability(npm_pkg['package']['package_id'], {
    'cve_id': 'CVE-2023-12345',
    'severity': 'high',
    'description': 'XSS vulnerability',
    'affected_versions': '1.0.0-1.2.2',
    'fixed_version': '1.2.3'
})

# 6. Statistics
mgr.statistics.record_download(npm_pkg['package']['package_id'], '@myorg/react-components')
popular = mgr.statistics.get_popular_packages(10)
stats = mgr.statistics.get_registry_statistics()

# 7. List and filter packages
all_packages = mgr.registry.list_packages()
npm_packages = mgr.registry.list_packages({'package_type': PackageType.NPM.value})
public_packages = mgr.registry.list_packages({'visibility': PackageVisibility.PUBLIC.value})

# 8. Get publish commands
npm_cmd = mgr.commands.get_npm_publish_command('@myorg/react-components', '1.2.3')
pypi_cmd = mgr.commands.get_pypi_publish_command('myorg-utils', '2.1.0')
```

## Supported Package Types

### NPM (Node Package Manager)
- **Registry**: `/packages/npm/`
- **Auth**: CI_JOB_TOKEN or personal access token
- **Scopes**: @organization/package-name
- **Files**: package.json, .npmrc

### Maven (Java)
- **Registry**: `/packages/maven/`
- **Config**: pom.xml distributionManagement
- **Format**: groupId:artifactId:version
- **Files**: pom.xml, settings.xml

### PyPI (Python Package Index)
- **Registry**: `/packages/pypi/`
- **Config**: .pypirc
- **Upload**: twine
- **Files**: setup.py, pyproject.toml, wheel

### NuGet (.NET)
- **Registry**: `/packages/nuget/index.json`
- **Config**: nuget.config
- **Format**: PackageName.Version.nupkg
- **Files**: .nuspec, .nupkg

### Composer (PHP)
- **Registry**: `/packages/composer/`
- **Config**: composer.json
- **Format**: vendor/package
- **Files**: composer.json, composer.lock

### Conan (C/C++)
- **Registry**: `/packages/conan/`
- **Config**: conanfile.py or conanfile.txt
- **Format**: name/version@user/channel
- **Files**: conanfile.py

### Go Modules
- **Registry**: `/packages/go/`
- **Format**: module/path/v2
- **Files**: go.mod, go.sum
- **Import**: import "gitlab.com/org/project/package"

### Helm Charts
- **Registry**: `/packages/helm/`
- **Format**: chart-name-version.tgz
- **Files**: Chart.yaml, values.yaml
- **Commands**: helm package, helm push

### Debian Packages
- **Registry**: `/packages/debian/`
- **Format**: .deb files
- **Distribution**: focal, jammy, bullseye
- **Component**: main, contrib, non-free

### RPM Packages
- **Registry**: `/packages/rpm/`
- **Format**: .rpm files
- **Distribution**: el8, el9, fedora38
- **Files**: .spec files

### Generic Packages
- **Registry**: `/packages/generic/`
- **Format**: Any file type
- **Use**: Binaries, archives, assets
- **Upload**: curl with multipart/form-data

## Package Publishing

### NPM Package

```bash
# Configure registry
npm config set @myorg:registry https://gitlab.com/api/v4/projects/123/packages/npm/
npm config set '//gitlab.com/api/v4/projects/123/packages/npm/:_authToken' "${CI_JOB_TOKEN}"

# Publish
npm publish
```

### Maven Package

```xml
<!-- pom.xml -->
<distributionManagement>
  <repository>
    <id>gitlab-maven</id>
    <url>https://gitlab.com/api/v4/projects/123/packages/maven</url>
  </repository>
</distributionManagement>
```

```bash
# Publish
mvn deploy
```

### PyPI Package

```ini
# ~/.pypirc
[distutils]
index-servers = gitlab

[gitlab]
repository = https://gitlab.com/api/v4/projects/123/packages/pypi
username = __token__
password = ${CI_JOB_TOKEN}
```

```bash
# Build and publish
python setup.py sdist bdist_wheel
twine upload --repository gitlab dist/*
```

### NuGet Package

```bash
# Add source
nuget sources Add -Name "GitLab" \
  -Source "https://gitlab.com/api/v4/projects/123/packages/nuget/index.json" \
  -UserName gitlab-ci-token \
  -Password ${CI_JOB_TOKEN}

# Publish
nuget push MyPackage.1.0.0.nupkg -Source "GitLab"
```

### Helm Chart

```bash
# Package chart
helm package mychart/

# Publish
curl --request POST \
  --form 'chart=@mychart-1.0.0.tgz' \
  --user gitlab-ci-token:${CI_JOB_TOKEN} \
  https://gitlab.com/api/v4/projects/123/packages/helm/api/stable/charts
```

### Generic Package

```bash
# Upload file
curl --header "JOB-TOKEN: ${CI_JOB_TOKEN}" \
  --upload-file path/to/file.zip \
  "https://gitlab.com/api/v4/projects/123/packages/generic/mypackage/1.0.0/file.zip"
```

## Version Management

### Semantic Versioning

Follows semver specification: MAJOR.MINOR.PATCH

- **MAJOR**: Breaking changes (2.0.0)
- **MINOR**: New features, backward compatible (1.1.0)
- **PATCH**: Bug fixes (1.0.1)

```python
# Add version
mgr.versions.add_version('my-package', {
    'version': '2.0.0',
    'package_id': 'pkg-1',
    'release_notes': 'Complete rewrite with new API',
    'breaking_changes': True
})

# Get latest
latest = mgr.versions.get_latest_version('my-package')

# Version history
history = mgr.versions.get_version_history('my-package')

# Compare versions
result = mgr.versions.compare_versions('2.1.0', '2.0.5')
# Output: "2.1.0 > 2.0.5"
```

### Version Constraints

| Constraint | Meaning | Example |
|------------|---------|---------|
| `1.2.3` | Exact version | `1.2.3` |
| `^1.2.3` | Compatible (>=1.2.3 <2.0.0) | `1.9.9` |
| `~1.2.3` | Patch level (>=1.2.3 <1.3.0) | `1.2.9` |
| `>=1.2.3` | Greater or equal | `2.0.0` |
| `1.2.x` | Any patch version | `1.2.5` |
| `*` | Any version | `99.99.99` |

## Dependency Management

### Dependency Types

```python
# Runtime dependency (required)
mgr.dependencies.add_dependency('pkg-1', {
    'name': 'express',
    'version': '^4.18.0',
    'type': 'runtime'
})

# Development dependency
mgr.dependencies.add_dependency('pkg-1', {
    'name': 'jest',
    'version': '^29.0.0',
    'type': 'development'
})

# Peer dependency (must be installed by consumer)
mgr.dependencies.add_dependency('pkg-1', {
    'name': 'react',
    'version': '^18.0.0',
    'type': 'peer'
})

# Optional dependency
mgr.dependencies.add_dependency('pkg-1', {
    'name': 'fsevents',
    'version': '^2.3.0',
    'type': 'optional'
})
```

### Dependency Tree

```python
# Build dependency tree
tree = mgr.dependencies.build_dependency_tree('pkg-1', max_depth=5)

# Returns:
{
    'package_id': 'pkg-1',
    'dependencies': [
        {'name': 'react', 'version': '^18.0.0', 'type': 'peer'},
        {'name': 'express', 'version': '^4.18.0', 'type': 'runtime'}
    ],
    'depth': 0,
    'total_dependencies': 2
}
```

### Conflict Detection

```python
# Detect version conflicts
conflicts = mgr.dependencies.detect_conflicts('pkg-1')

# Returns:
[
    {
        'package': 'lodash',
        'versions': ['4.17.21', '3.10.1'],
        'conflict_type': 'version_mismatch'
    }
]
```

## Security Scanning

### Vulnerability Scanners

| Scanner | Type | Features |
|---------|------|----------|
| **Trivy** | Container/OS | Fast, comprehensive |
| **Snyk** | Dependencies | Developer-first |
| **Clair** | Container | Static analysis |
| **Grype** | OS/Dependencies | Fast scanning |

### Scan Package

```python
# Scan with Trivy
scan = mgr.security.scan_package('pkg-1', scanner='trivy')

# Returns:
{
    'scan_id': 'scan-1',
    'package_id': 'pkg-1',
    'scanner': 'trivy',
    'scanned_at': '2025-11-06T10:30:00',
    'vulnerabilities_found': 0,
    'severity_counts': {
        'critical': 0,
        'high': 0,
        'medium': 0,
        'low': 0
    },
    'status': 'clean'
}
```

### Add Vulnerability

```python
mgr.security.add_vulnerability('pkg-1', {
    'cve_id': 'CVE-2023-12345',
    'severity': 'critical',
    'description': 'Remote code execution in lodash',
    'affected_versions': '4.0.0-4.17.20',
    'fixed_version': '4.17.21'
})
```

### Quarantine Package

```python
# Quarantine package with critical vulnerabilities
mgr.security.quarantine_package('pkg-1', 'Critical CVE detected: CVE-2023-12345')
```

### Filter Vulnerabilities

```python
# Get all vulnerabilities
all_vulns = mgr.security.get_vulnerabilities('pkg-1')

# Get high and critical only
critical_vulns = mgr.security.get_vulnerabilities('pkg-1', min_severity='high')
```

## Package Statistics

### Track Downloads

```python
# Record download
mgr.statistics.record_download('pkg-1', '@myorg/react-components')

# Returns:
{
    'package_name': '@myorg/react-components',
    'total_downloads': 1234
}
```

### Popular Packages

```python
# Get top 10 most downloaded
popular = mgr.statistics.get_popular_packages(limit=10)

# Returns:
[
    {'package_name': '@myorg/react-components', 'downloads': 5420},
    {'package_name': 'myorg-utils', 'downloads': 3210},
    {'package_name': '@myorg/api-client', 'downloads': 2100}
]
```

### Registry Statistics

```python
stats = mgr.statistics.get_registry_statistics()

# Returns:
{
    'total_downloads': 15230,
    'total_packages': 47,
    'average_downloads': 324.0
}
```

## Package Visibility

### Visibility Levels

| Level | Description | Access |
|-------|-------------|--------|
| **PUBLIC** | Available to everyone | All users |
| **INTERNAL** | Within GitLab instance | Authenticated users |
| **PRIVATE** | Project members only | Project members |

```python
# Public package
mgr.registry.publish_package({
    'package_name': 'my-public-lib',
    'version': '1.0.0',
    'visibility': PackageVisibility.PUBLIC.value
})

# Private package
mgr.registry.publish_package({
    'package_name': 'internal-tools',
    'version': '1.0.0',
    'visibility': PackageVisibility.PRIVATE.value
})
```

## Best Practices

### Package Publishing
1. **Semantic Versioning**: Use semver consistently
2. **Release Notes**: Document changes for each version
3. **Breaking Changes**: Flag breaking changes clearly
4. **Metadata**: Include comprehensive package metadata
5. **README**: Include installation and usage docs

### Version Management
1. **Version Bumps**: MAJOR for breaking, MINOR for features, PATCH for fixes
2. **Pre-release Tags**: Use -alpha, -beta, -rc for pre-releases
3. **Version Locking**: Lock dependencies for reproducible builds
4. **Changelog**: Maintain CHANGELOG.md
5. **Git Tags**: Tag releases in Git

### Dependency Management
1. **Minimal Dependencies**: Keep dependencies minimal
2. **Version Constraints**: Use ^ for libraries, exact for applications
3. **Regular Updates**: Update dependencies regularly
4. **Audit**: Run security audits frequently
5. **Lock Files**: Commit lock files (package-lock.json, yarn.lock)

### Security
1. **Scan on Publish**: Automatically scan new packages
2. **Monitor CVEs**: Track new vulnerabilities
3. **Quick Updates**: Update vulnerable dependencies immediately
4. **Quarantine**: Isolate packages with critical issues
5. **Private First**: Default to private visibility

### Package Naming
1. **NPM Scopes**: Use @org/package for NPM
2. **Maven GroupId**: Use reverse domain (com.myorg)
3. **Lowercase**: Use lowercase names
4. **Hyphens**: Use hyphens not underscores (except Python)
5. **Descriptive**: Use clear, descriptive names

## CI/CD Integration

### GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - publish

build:
  stage: build
  script:
    - npm install
    - npm run build

test:
  stage: test
  script:
    - npm test

publish:
  stage: publish
  only:
    - tags
  script:
    - |
      npm config set @myorg:registry ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/npm/
      npm config set -- "//${CI_SERVER_HOST}:${CI_SERVER_PORT}/api/v4/projects/${CI_PROJECT_ID}/packages/npm/:_authToken" "${CI_JOB_TOKEN}"
    - npm publish
```

### Maven CI Example

```yaml
publish-maven:
  stage: publish
  only:
    - tags
  script:
    - mvn deploy -s ci_settings.xml
```

### PyPI CI Example

```yaml
publish-pypi:
  stage: publish
  only:
    - tags
  script:
    - pip install twine
    - python setup.py sdist bdist_wheel
    - TWINE_PASSWORD=${CI_JOB_TOKEN} TWINE_USERNAME=gitlab-ci-token python -m twine upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi dist/*
```

## Common Use Cases

### Private NPM Registry

```python
# Setup private NPM registry
mgr.registry.publish_package({
    'package_type': PackageType.NPM.value,
    'package_name': '@mycompany/internal-lib',
    'version': '1.0.0',
    'visibility': PackageVisibility.PRIVATE.value
})

# Install in other projects
# npm config set @mycompany:registry https://gitlab.com/api/v4/projects/123/packages/npm/
# npm install @mycompany/internal-lib
```

### Maven Artifact Repository

```python
# Publish Maven artifact
mgr.registry.publish_package({
    'package_type': PackageType.MAVEN.value,
    'package_name': 'com.mycompany:my-library',
    'version': '2.1.0',
    'visibility': PackageVisibility.INTERNAL.value
})
```

### Python Package Distribution

```python
# Publish Python package
mgr.registry.publish_package({
    'package_type': PackageType.PYPI.value,
    'package_name': 'mycompany-utils',
    'version': '3.2.1',
    'description': 'Utility functions for MyCompany',
    'visibility': PackageVisibility.PUBLIC.value
})
```

### Container Image Registry

```python
# Publish Docker image
mgr.registry.publish_package({
    'package_type': PackageType.GENERIC.value,
    'package_name': 'my-app',
    'version': '1.5.0',
    'metadata': {
        'image': 'registry.gitlab.com/myorg/myapp:1.5.0',
        'platform': 'linux/amd64'
    }
})
```

## Troubleshooting

**Issue**: Package publish fails with 401 Unauthorized
- Verify CI_JOB_TOKEN or personal access token is valid
- Check project permissions (Developer role required)
- Ensure token has api or write_package_registry scope

**Issue**: Package not found when installing
- Verify package visibility matches user permissions
- Check registry URL configuration
- Ensure package was published successfully

**Issue**: Dependency conflict errors
- Run `detect_conflicts()` to identify issues
- Update conflicting dependencies to compatible versions
- Use dependency resolution strategies (npm overrides, Maven exclusions)

**Issue**: Vulnerability scan failing
- Check scanner is properly configured
- Verify network access to vulnerability databases
- Update scanner to latest version

**Issue**: Large package upload timeout
- Increase upload timeout in GitLab settings
- Use chunked uploads for large files
- Consider splitting large packages

## Requirements

```
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

### Environment Variables

```bash
export GITLAB_URL="https://gitlab.com"
export CI_JOB_TOKEN="your-token"
export CI_PROJECT_ID="123"
```

### Python Configuration

```python
from package_manager import PackageManager

mgr = PackageManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com'
)
```

## Author

BrillConsulting - Enterprise Cloud Solutions
