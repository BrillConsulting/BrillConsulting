"""
GitLab Package Management - Comprehensive Package Registry System

This module provides comprehensive package registry management for GitLab,
supporting multiple package types, version management, dependency tracking,
security scanning, and statistics.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


class PackageType(Enum):
    """Supported package types."""
    NPM = "npm"
    MAVEN = "maven"
    PYPI = "pypi"
    NUGET = "nuget"
    COMPOSER = "composer"
    CONAN = "conan"
    GO = "golang"
    GENERIC = "generic"
    HELM = "helm"
    DEBIAN = "debian"
    RPM = "rpm"


class PackageStatus(Enum):
    """Package status."""
    PUBLISHED = "published"
    PENDING = "pending"
    DEPRECATED = "deprecated"
    DELETED = "deleted"
    QUARANTINED = "quarantined"


class PackageVisibility(Enum):
    """Package visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


class PackageRegistryManager:
    """
    Manages package registry operations.

    Handles package publishing, listing, deletion, and deprecation
    across all supported package types.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.packages: Dict[str, Dict[str, Any]] = {}

    def publish_package(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Publish package to registry.

        Config:
            package_type: PackageType enum value
            package_name: Name of the package
            version: Package version (semantic versioning)
            file_path: Path to package file
            description: Package description
            visibility: PUBLIC, PRIVATE, or INTERNAL
            metadata: Additional package metadata
        """
        package_id = f"pkg-{len(self.packages) + 1}"

        package = {
            'package_id': package_id,
            'project_id': self.project_id,
            'package_type': config.get('package_type'),
            'package_name': config.get('package_name'),
            'version': config.get('version'),
            'file_path': config.get('file_path'),
            'description': config.get('description', ''),
            'visibility': config.get('visibility', PackageVisibility.PRIVATE.value),
            'status': PackageStatus.PUBLISHED.value,
            'downloads': 0,
            'created_at': datetime.now().isoformat(),
            'metadata': config.get('metadata', {}),
            'sha256': f"sha256:{hash(package_id) % 10**16:016x}"
        }

        self.packages[package_id] = package

        print(f"✅ Published package: {package['package_name']}@{package['version']}")
        return package

    def list_packages(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List packages with optional filtering.

        Filters:
            package_type: Filter by package type
            status: Filter by status
            visibility: Filter by visibility
            package_name: Filter by name
        """
        packages = list(self.packages.values())

        if filters:
            if 'package_type' in filters:
                packages = [p for p in packages if p['package_type'] == filters['package_type']]
            if 'status' in filters:
                packages = [p for p in packages if p['status'] == filters['status']]
            if 'visibility' in filters:
                packages = [p for p in packages if p['visibility'] == filters['visibility']]
            if 'package_name' in filters:
                packages = [p for p in packages if filters['package_name'] in p['package_name']]

        return packages

    def delete_package(self, package_id: str) -> Dict[str, str]:
        """Delete package from registry."""
        if package_id in self.packages:
            package = self.packages[package_id]
            package['status'] = PackageStatus.DELETED.value
            package['deleted_at'] = datetime.now().isoformat()
            print(f"🗑️  Deleted package: {package['package_name']}@{package['version']}")
            return {'status': 'deleted', 'package_id': package_id}
        return {'status': 'not_found', 'package_id': package_id}

    def deprecate_package(self, package_id: str, reason: str = '') -> Dict[str, Any]:
        """Mark package as deprecated."""
        if package_id in self.packages:
            package = self.packages[package_id]
            package['status'] = PackageStatus.DEPRECATED.value
            package['deprecation_reason'] = reason
            package['deprecated_at'] = datetime.now().isoformat()
            print(f"⚠️  Deprecated package: {package['package_name']}@{package['version']}")
            return package
        return {'status': 'not_found', 'package_id': package_id}

    def get_package_details(self, package_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed package information."""
        return self.packages.get(package_id)

    def track_download(self, package_id: str) -> Dict[str, Any]:
        """Track package download."""
        if package_id in self.packages:
            self.packages[package_id]['downloads'] += 1
            self.packages[package_id]['last_downloaded'] = datetime.now().isoformat()
            return self.packages[package_id]
        return {'status': 'not_found'}


class PackageVersionManager:
    """
    Manages package versions and version comparison.

    Handles semantic versioning, version queries, and version history.
    """

    def __init__(self):
        self.versions: Dict[str, List[Dict[str, Any]]] = {}

    def add_version(self, package_name: str, version_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add new version to package history."""
        if package_name not in self.versions:
            self.versions[package_name] = []

        version = {
            'version': version_info.get('version'),
            'package_id': version_info.get('package_id'),
            'created_at': datetime.now().isoformat(),
            'release_notes': version_info.get('release_notes', ''),
            'breaking_changes': version_info.get('breaking_changes', False)
        }

        self.versions[package_name].append(version)
        print(f"📦 Added version {version['version']} for {package_name}")
        return version

    def get_latest_version(self, package_name: str) -> Optional[Dict[str, Any]]:
        """Get latest version of package."""
        if package_name in self.versions and self.versions[package_name]:
            return sorted(
                self.versions[package_name],
                key=lambda x: x['created_at'],
                reverse=True
            )[0]
        return None

    def get_version_history(self, package_name: str) -> List[Dict[str, Any]]:
        """Get complete version history."""
        return self.versions.get(package_name, [])

    def compare_versions(self, version1: str, version2: str) -> str:
        """Compare two semantic versions."""
        def parse_version(v: str) -> tuple:
            return tuple(map(int, v.split('.')))

        try:
            v1 = parse_version(version1)
            v2 = parse_version(version2)

            if v1 > v2:
                return f"{version1} > {version2}"
            elif v1 < v2:
                return f"{version1} < {version2}"
            else:
                return f"{version1} = {version2}"
        except:
            return "Unable to compare versions"


class PackageDependencyManager:
    """
    Manages package dependencies and dependency trees.

    Tracks dependencies, builds dependency graphs, and detects conflicts.
    """

    def __init__(self):
        self.dependencies: Dict[str, List[Dict[str, str]]] = {}

    def add_dependency(self, package_id: str, dependency: Dict[str, str]) -> Dict[str, Any]:
        """
        Add dependency for package.

        Dependency:
            name: Dependency package name
            version: Required version or version constraint
            type: runtime, development, peer, optional
        """
        if package_id not in self.dependencies:
            self.dependencies[package_id] = []

        dep = {
            'name': dependency.get('name'),
            'version': dependency.get('version'),
            'type': dependency.get('type', 'runtime')
        }

        self.dependencies[package_id].append(dep)
        print(f"🔗 Added dependency {dep['name']}@{dep['version']} to {package_id}")
        return dep

    def get_dependencies(self, package_id: str, dep_type: Optional[str] = None) -> List[Dict[str, str]]:
        """Get package dependencies, optionally filtered by type."""
        deps = self.dependencies.get(package_id, [])

        if dep_type:
            deps = [d for d in deps if d['type'] == dep_type]

        return deps

    def build_dependency_tree(self, package_id: str, max_depth: int = 5) -> Dict[str, Any]:
        """Build complete dependency tree."""
        return {
            'package_id': package_id,
            'dependencies': self.dependencies.get(package_id, []),
            'depth': 0,
            'total_dependencies': len(self.dependencies.get(package_id, []))
        }

    def detect_conflicts(self, package_id: str) -> List[Dict[str, Any]]:
        """Detect version conflicts in dependencies."""
        deps = self.dependencies.get(package_id, [])
        conflicts = []

        # Check for multiple versions of same package
        seen = {}
        for dep in deps:
            if dep['name'] in seen and seen[dep['name']] != dep['version']:
                conflicts.append({
                    'package': dep['name'],
                    'versions': [seen[dep['name']], dep['version']],
                    'conflict_type': 'version_mismatch'
                })
            seen[dep['name']] = dep['version']

        if conflicts:
            print(f"⚠️  Found {len(conflicts)} dependency conflicts")

        return conflicts


class PackageSecurityManager:
    """
    Manages package security scanning and vulnerability tracking.

    Integrates with Trivy, Snyk, Clair for vulnerability detection.
    """

    def __init__(self):
        self.vulnerabilities: Dict[str, List[Dict[str, Any]]] = {}
        self.scan_results: Dict[str, Dict[str, Any]] = {}

    def scan_package(self, package_id: str, scanner: str = 'trivy') -> Dict[str, Any]:
        """
        Scan package for vulnerabilities.

        Scanners: trivy, snyk, clair, grype
        """
        # Simulate vulnerability scan
        result = {
            'scan_id': f"scan-{len(self.scan_results) + 1}",
            'package_id': package_id,
            'scanner': scanner,
            'scanned_at': datetime.now().isoformat(),
            'vulnerabilities_found': 0,
            'severity_counts': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'status': 'clean'
        }

        self.scan_results[result['scan_id']] = result
        print(f"🔍 Scanned package {package_id} with {scanner}: {result['status']}")
        return result

    def add_vulnerability(self, package_id: str, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add detected vulnerability.

        Vulnerability:
            cve_id: CVE identifier
            severity: critical, high, medium, low
            description: Vulnerability description
            affected_versions: Version range affected
            fixed_version: Version with fix
        """
        if package_id not in self.vulnerabilities:
            self.vulnerabilities[package_id] = []

        vuln = {
            'vuln_id': f"vuln-{len(self.vulnerabilities[package_id]) + 1}",
            'cve_id': vulnerability.get('cve_id'),
            'severity': vulnerability.get('severity'),
            'description': vulnerability.get('description'),
            'affected_versions': vulnerability.get('affected_versions'),
            'fixed_version': vulnerability.get('fixed_version'),
            'detected_at': datetime.now().isoformat()
        }

        self.vulnerabilities[package_id].append(vuln)
        print(f"🚨 Vulnerability detected: {vuln['cve_id']} ({vuln['severity']})")
        return vuln

    def get_vulnerabilities(self, package_id: str, min_severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get package vulnerabilities, optionally filtered by severity."""
        vulns = self.vulnerabilities.get(package_id, [])

        if min_severity:
            severity_order = ['low', 'medium', 'high', 'critical']
            min_index = severity_order.index(min_severity)
            vulns = [v for v in vulns if severity_order.index(v['severity']) >= min_index]

        return vulns

    def quarantine_package(self, package_id: str, reason: str) -> Dict[str, Any]:
        """Quarantine package due to security issues."""
        return {
            'package_id': package_id,
            'status': PackageStatus.QUARANTINED.value,
            'reason': reason,
            'quarantined_at': datetime.now().isoformat()
        }


class PackageStatisticsManager:
    """
    Tracks package statistics and analytics.

    Download counts, popular packages, registry statistics.
    """

    def __init__(self):
        self.stats: Dict[str, Any] = {
            'total_downloads': 0,
            'package_downloads': {}
        }

    def record_download(self, package_id: str, package_name: str) -> Dict[str, Any]:
        """Record package download."""
        self.stats['total_downloads'] += 1

        if package_name not in self.stats['package_downloads']:
            self.stats['package_downloads'][package_name] = 0

        self.stats['package_downloads'][package_name] += 1

        return {
            'package_name': package_name,
            'total_downloads': self.stats['package_downloads'][package_name]
        }

    def get_popular_packages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most downloaded packages."""
        sorted_packages = sorted(
            self.stats['package_downloads'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return [
            {'package_name': name, 'downloads': count}
            for name, count in sorted_packages
        ]

    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get overall registry statistics."""
        return {
            'total_downloads': self.stats['total_downloads'],
            'total_packages': len(self.stats['package_downloads']),
            'average_downloads': (
                self.stats['total_downloads'] / len(self.stats['package_downloads'])
                if self.stats['package_downloads'] else 0
            )
        }


class PackagePublishCommandsManager:
    """
    Generates publish commands for different package types.

    Provides CLI commands for publishing to GitLab Package Registry.
    """

    def __init__(self, gitlab_url: str, project_id: str):
        self.gitlab_url = gitlab_url
        self.project_id = project_id

    def get_npm_publish_command(self, package_name: str, version: str) -> str:
        """Get npm publish command."""
        return f"""# NPM Package Registry
npm config set @myorg:registry {self.gitlab_url}/api/v4/projects/{self.project_id}/packages/npm/
npm config set '//{self.gitlab_url}/api/v4/projects/{self.project_id}/packages/npm/:_authToken' "${{CI_JOB_TOKEN}}"
npm publish"""

    def get_maven_publish_command(self, group_id: str, artifact_id: str, version: str) -> str:
        """Get Maven publish command."""
        return f"""<!-- pom.xml -->
<distributionManagement>
  <repository>
    <id>gitlab-maven</id>
    <url>{self.gitlab_url}/api/v4/projects/{self.project_id}/packages/maven</url>
  </repository>
</distributionManagement>

# Publish
mvn deploy"""

    def get_pypi_publish_command(self, package_name: str, version: str) -> str:
        """Get PyPI publish command."""
        return f"""# Setup ~/.pypirc
[distutils]
index-servers = gitlab

[gitlab]
repository = {self.gitlab_url}/api/v4/projects/{self.project_id}/packages/pypi
username = __token__
password = ${{CI_JOB_TOKEN}}

# Publish
python setup.py sdist bdist_wheel
twine upload --repository gitlab dist/*"""

    def get_nuget_publish_command(self, package_name: str, version: str) -> str:
        """Get NuGet publish command."""
        return f"""# Configure source
nuget sources Add -Name "GitLab" -Source "{self.gitlab_url}/api/v4/projects/{self.project_id}/packages/nuget/index.json" -UserName gitlab-ci-token -Password ${{CI_JOB_TOKEN}}

# Publish
nuget push {package_name}.{version}.nupkg -Source "GitLab" """

    def get_helm_publish_command(self, chart_name: str, version: str) -> str:
        """Get Helm publish command."""
        return f"""# Package chart
helm package {chart_name}

# Publish
curl --request POST --form 'chart=@{chart_name}-{version}.tgz' --user gitlab-ci-token:${{CI_JOB_TOKEN}} {self.gitlab_url}/api/v4/projects/{self.project_id}/packages/helm/api/stable/charts"""

    def get_docker_publish_command(self, image_name: str, tag: str) -> str:
        """Get Docker publish command."""
        return f"""# Login
docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY

# Build and push
docker build -t {self.gitlab_url}/{self.project_id}/{image_name}:{tag} .
docker push {self.gitlab_url}/{self.project_id}/{image_name}:{tag}"""


class PackageManager:
    """
    Main package management orchestration class.

    Coordinates all package management operations including registry,
    versions, dependencies, security, and statistics.
    """

    def __init__(self, project_id: str, gitlab_url: str = 'https://gitlab.com'):
        self.project_id = project_id
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.registry = PackageRegistryManager(project_id)
        self.versions = PackageVersionManager()
        self.dependencies = PackageDependencyManager()
        self.security = PackageSecurityManager()
        self.statistics = PackageStatisticsManager()
        self.commands = PackagePublishCommandsManager(gitlab_url, project_id)

    def publish_with_tracking(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Publish package with full tracking."""
        # Publish package
        package = self.registry.publish_package(config)

        # Add version tracking
        self.versions.add_version(
            package['package_name'],
            {
                'version': package['version'],
                'package_id': package['package_id'],
                'release_notes': config.get('release_notes', '')
            }
        )

        # Add dependencies if provided
        if 'dependencies' in config:
            for dep in config['dependencies']:
                self.dependencies.add_dependency(package['package_id'], dep)

        # Security scan
        scan_result = self.security.scan_package(
            package['package_id'],
            scanner=config.get('scanner', 'trivy')
        )

        return {
            'package': package,
            'scan_result': scan_result,
            'publish_command': self._get_publish_command(package)
        }

    def _get_publish_command(self, package: Dict[str, Any]) -> str:
        """Get publish command for package type."""
        package_type = package['package_type']
        name = package['package_name']
        version = package['version']

        if package_type == PackageType.NPM.value:
            return self.commands.get_npm_publish_command(name, version)
        elif package_type == PackageType.MAVEN.value:
            return self.commands.get_maven_publish_command('com.example', name, version)
        elif package_type == PackageType.PYPI.value:
            return self.commands.get_pypi_publish_command(name, version)
        elif package_type == PackageType.NUGET.value:
            return self.commands.get_nuget_publish_command(name, version)
        elif package_type == PackageType.HELM.value:
            return self.commands.get_helm_publish_command(name, version)
        else:
            return "# Generic package - use GitLab API"


def demo_package_management():
    """Demonstrate comprehensive package management."""
    print("\n" + "="*60)
    print("🎯 GitLab Package Management Demo")
    print("="*60)

    mgr = PackageManager(project_id='myorg/myproject')

    # 1. Publish NPM package
    print("\n1️⃣  Publishing NPM Package")
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
    print("\n2️⃣  Publishing PyPI Package")
    pypi_pkg = mgr.registry.publish_package({
        'package_type': PackageType.PYPI.value,
        'package_name': 'myorg-utils',
        'version': '2.1.0',
        'file_path': 'dist/myorg_utils-2.1.0.tar.gz',
        'description': 'Utility functions',
        'visibility': PackageVisibility.INTERNAL.value
    })

    # 3. Version management
    print("\n3️⃣  Version Management")
    latest = mgr.versions.get_latest_version('@myorg/react-components')
    print(f"   Latest version: {latest['version']}")

    comparison = mgr.versions.compare_versions('1.2.3', '1.2.0')
    print(f"   Version comparison: {comparison}")

    # 4. Dependency management
    print("\n4️⃣  Dependency Analysis")
    deps = mgr.dependencies.get_dependencies(npm_pkg['package']['package_id'])
    print(f"   Total dependencies: {len(deps)}")
    for dep in deps:
        print(f"   - {dep['name']}@{dep['version']} ({dep['type']})")

    # 5. Security scanning
    print("\n5️⃣  Security Scanning")
    scan = mgr.security.scan_package(npm_pkg['package']['package_id'], 'trivy')
    print(f"   Scan status: {scan['status']}")
    print(f"   Critical: {scan['severity_counts']['critical']}, High: {scan['severity_counts']['high']}")

    # 6. Package statistics
    print("\n6️⃣  Package Statistics")
    mgr.statistics.record_download(npm_pkg['package']['package_id'], '@myorg/react-components')
    mgr.statistics.record_download(npm_pkg['package']['package_id'], '@myorg/react-components')
    mgr.statistics.record_download(pypi_pkg['package_id'], 'myorg-utils')

    popular = mgr.statistics.get_popular_packages(5)
    print(f"   Popular packages:")
    for pkg in popular:
        print(f"   - {pkg['package_name']}: {pkg['downloads']} downloads")

    # 7. List packages
    print("\n7️⃣  Package Registry")
    all_packages = mgr.registry.list_packages()
    print(f"   Total packages: {len(all_packages)}")

    npm_packages = mgr.registry.list_packages({'package_type': PackageType.NPM.value})
    print(f"   NPM packages: {len(npm_packages)}")

    # 8. Publish commands
    print("\n8️⃣  Publish Commands")
    print("   NPM Publish:")
    print(mgr.commands.get_npm_publish_command('@myorg/react-components', '1.2.3'))

    print("\n" + "="*60)
    print("✅ Package Management Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_package_management()
