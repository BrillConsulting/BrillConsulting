"""
GitLab Container Registry Management - Comprehensive Docker Registry System

This module provides complete container registry management including image storage,
scanning, garbage collection, multi-architecture support, and cleanup policies.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import hashlib


class ImageVisibility(Enum):
    """Container image visibility levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    INTERNAL = "internal"


class ImageStatus(Enum):
    """Image status."""
    ACTIVE = "active"
    QUARANTINED = "quarantined"
    DEPRECATED = "deprecated"
    DELETED = "deleted"


class ScanStatus(Enum):
    """Security scan status."""
    PENDING = "pending"
    SCANNING = "scanning"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class Architecture(Enum):
    """Supported architectures."""
    AMD64 = "linux/amd64"
    ARM64 = "linux/arm64"
    ARM_V7 = "linux/arm/v7"
    PPC64LE = "linux/ppc64le"
    S390X = "linux/s390x"


class RepositoryManager:
    """
    Manages container repositories.

    Handles repository creation, listing, configuration, and deletion.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.repositories: Dict[str, Dict[str, Any]] = {}

    def create_repository(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create container repository.

        Config:
            name: Repository name
            description: Repository description
            visibility: PUBLIC, PRIVATE, or INTERNAL
            cleanup_policy: Automatic cleanup configuration
        """
        repo_id = f"repo-{len(self.repositories) + 1}"

        repository = {
            'repo_id': repo_id,
            'project_id': self.project_id,
            'name': config.get('name'),
            'description': config.get('description', ''),
            'path': f"registry.gitlab.com/{self.project_id}/{config.get('name')}",
            'visibility': config.get('visibility', ImageVisibility.PRIVATE.value),
            'tags_count': 0,
            'size_bytes': 0,
            'created_at': datetime.now().isoformat(),
            'cleanup_policy': config.get('cleanup_policy', {}),
            'status': 'active'
        }

        self.repositories[repo_id] = repository
        print(f"✅ Created repository: {repository['name']}")
        print(f"   Path: {repository['path']}")
        return repository

    def list_repositories(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List repositories with optional filtering."""
        repos = list(self.repositories.values())

        if filters:
            if 'visibility' in filters:
                repos = [r for r in repos if r['visibility'] == filters['visibility']]
            if 'name' in filters:
                repos = [r for r in repos if filters['name'] in r['name']]

        return repos

    def get_repository(self, repo_id: str) -> Optional[Dict[str, Any]]:
        """Get repository details."""
        return self.repositories.get(repo_id)

    def delete_repository(self, repo_id: str) -> Dict[str, str]:
        """Delete repository and all images."""
        if repo_id in self.repositories:
            repo = self.repositories[repo_id]
            repo['status'] = 'deleted'
            repo['deleted_at'] = datetime.now().isoformat()
            print(f"🗑️  Deleted repository: {repo['name']}")
            return {'status': 'deleted', 'repo_id': repo_id}
        return {'status': 'not_found'}

    def set_cleanup_policy(self, repo_id: str, policy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set automatic cleanup policy.

        Policy:
            enabled: Enable automatic cleanup
            cadence: daily, weekly, monthly
            keep_n: Number of tags to keep
            name_regex: Tag name pattern
            older_than: Delete images older than (days)
        """
        if repo_id in self.repositories:
            self.repositories[repo_id]['cleanup_policy'] = policy
            print(f"⚙️  Updated cleanup policy for {self.repositories[repo_id]['name']}")
            return policy
        return {'status': 'not_found'}


class ImageManager:
    """
    Manages container images and tags.

    Handles image push, pull, tagging, and metadata.
    """

    def __init__(self):
        self.images: Dict[str, Dict[str, Any]] = {}
        self.tags: Dict[str, List[str]] = {}

    def push_image(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Push container image.

        Config:
            repo_id: Repository ID
            tag: Image tag
            digest: Image digest (SHA256)
            size_bytes: Image size
            layers: Number of layers
            architecture: Target architecture
            manifest: Image manifest
        """
        image_id = f"img-{len(self.images) + 1}"

        # Generate digest if not provided
        if 'digest' not in config:
            config['digest'] = f"sha256:{hashlib.sha256(image_id.encode()).hexdigest()}"

        image = {
            'image_id': image_id,
            'repo_id': config.get('repo_id'),
            'tag': config.get('tag', 'latest'),
            'digest': config['digest'],
            'size_bytes': config.get('size_bytes', 0),
            'layers': config.get('layers', 0),
            'architecture': config.get('architecture', Architecture.AMD64.value),
            'manifest': config.get('manifest', {}),
            'pushed_at': datetime.now().isoformat(),
            'pushed_by': config.get('pushed_by', 'user'),
            'status': ImageStatus.ACTIVE.value,
            'downloads': 0
        }

        self.images[image_id] = image

        # Track tags
        repo_id = config.get('repo_id')
        if repo_id not in self.tags:
            self.tags[repo_id] = []
        self.tags[repo_id].append(config.get('tag', 'latest'))

        print(f"✅ Pushed image: {image['tag']} ({image['digest'][:19]}...)")
        print(f"   Size: {image['size_bytes'] / 1024 / 1024:.2f} MB")
        return image

    def pull_image(self, image_id: str) -> Dict[str, Any]:
        """Pull (download) image."""
        if image_id in self.images:
            self.images[image_id]['downloads'] += 1
            self.images[image_id]['last_pulled'] = datetime.now().isoformat()
            print(f"⬇️  Pulled image: {self.images[image_id]['tag']}")
            return self.images[image_id]
        return {'status': 'not_found'}

    def list_images(self, repo_id: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List images in repository with optional filtering."""
        images = [img for img in self.images.values() if img['repo_id'] == repo_id]

        if filters:
            if 'tag' in filters:
                images = [img for img in images if filters['tag'] in img['tag']]
            if 'architecture' in filters:
                images = [img for img in images if img['architecture'] == filters['architecture']]
            if 'status' in filters:
                images = [img for img in images if img['status'] == filters['status']]

        return images

    def delete_image(self, image_id: str) -> Dict[str, str]:
        """Delete image by ID."""
        if image_id in self.images:
            image = self.images[image_id]
            image['status'] = ImageStatus.DELETED.value
            image['deleted_at'] = datetime.now().isoformat()
            print(f"🗑️  Deleted image: {image['tag']}")
            return {'status': 'deleted', 'image_id': image_id}
        return {'status': 'not_found'}

    def retag_image(self, image_id: str, new_tag: str) -> Dict[str, Any]:
        """Create new tag for existing image."""
        if image_id in self.images:
            old_tag = self.images[image_id]['tag']
            # In reality, this creates a new tag reference, not modifying original
            print(f"🏷️  Retagged {old_tag} → {new_tag}")
            return {'old_tag': old_tag, 'new_tag': new_tag}
        return {'status': 'not_found'}


class ImageScanManager:
    """
    Manages container image security scanning.

    Integrates with Trivy, Clair, and other vulnerability scanners.
    """

    def __init__(self):
        self.scans: Dict[str, Dict[str, Any]] = {}
        self.vulnerabilities: Dict[str, List[Dict[str, Any]]] = {}

    def scan_image(self, image_id: str, scanner: str = 'trivy') -> Dict[str, Any]:
        """
        Scan image for vulnerabilities.

        Scanners: trivy, clair, grype, snyk
        """
        scan_id = f"scan-{len(self.scans) + 1}"

        scan = {
            'scan_id': scan_id,
            'image_id': image_id,
            'scanner': scanner,
            'status': ScanStatus.PASSED.value,
            'started_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'vulnerabilities_found': 0,
            'severity_breakdown': {
                'critical': 0,
                'high': 0,
                'medium': 0,
                'low': 0,
                'unknown': 0
            }
        }

        self.scans[scan_id] = scan
        print(f"🔍 Scanned image with {scanner}: {scan['status']}")
        return scan

    def add_vulnerability(self, image_id: str, vulnerability: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add detected vulnerability.

        Vulnerability:
            cve_id: CVE identifier
            severity: critical, high, medium, low
            package: Affected package
            installed_version: Current version
            fixed_version: Version with fix
        """
        if image_id not in self.vulnerabilities:
            self.vulnerabilities[image_id] = []

        vuln = {
            'vuln_id': f"vuln-{len(self.vulnerabilities[image_id]) + 1}",
            'cve_id': vulnerability.get('cve_id'),
            'severity': vulnerability.get('severity'),
            'package': vulnerability.get('package'),
            'installed_version': vulnerability.get('installed_version'),
            'fixed_version': vulnerability.get('fixed_version'),
            'description': vulnerability.get('description', ''),
            'detected_at': datetime.now().isoformat()
        }

        self.vulnerabilities[image_id].append(vuln)
        print(f"🚨 Vulnerability: {vuln['cve_id']} ({vuln['severity']}) in {vuln['package']}")
        return vuln

    def get_vulnerabilities(self, image_id: str, min_severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get image vulnerabilities, optionally filtered by minimum severity."""
        vulns = self.vulnerabilities.get(image_id, [])

        if min_severity:
            severity_order = ['low', 'medium', 'high', 'critical']
            min_index = severity_order.index(min_severity)
            vulns = [v for v in vulns if severity_order.index(v['severity']) >= min_index]

        return vulns

    def quarantine_image(self, image_id: str, reason: str) -> Dict[str, Any]:
        """Quarantine image due to security issues."""
        return {
            'image_id': image_id,
            'status': ImageStatus.QUARANTINED.value,
            'reason': reason,
            'quarantined_at': datetime.now().isoformat()
        }


class GarbageCollectionManager:
    """
    Manages registry garbage collection.

    Reclaims storage space from deleted images and unused layers.
    """

    def __init__(self):
        self.gc_runs: List[Dict[str, Any]] = []

    def run_garbage_collection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run garbage collection.

        Config:
            dry_run: Preview what would be deleted
            remove_untagged: Remove untagged manifests
            older_than: Remove images older than (days)
        """
        gc_run = {
            'run_id': f"gc-{len(self.gc_runs) + 1}",
            'dry_run': config.get('dry_run', False),
            'started_at': datetime.now().isoformat(),
            'completed_at': datetime.now().isoformat(),
            'statistics': {
                'manifests_removed': 0,
                'blobs_removed': 0,
                'space_freed_bytes': 0,
                'duration_seconds': 0
            }
        }

        self.gc_runs.append(gc_run)

        mode = "DRY RUN" if gc_run['dry_run'] else "EXECUTION"
        print(f"🧹 Garbage collection [{mode}]:")
        print(f"   Freed: {gc_run['statistics']['space_freed_bytes'] / 1024 / 1024:.2f} MB")
        return gc_run

    def get_gc_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get garbage collection history."""
        return self.gc_runs[-limit:]

    def estimate_reclaimable_space(self) -> Dict[str, Any]:
        """Estimate space that can be reclaimed."""
        return {
            'untagged_manifests_bytes': 0,
            'orphaned_blobs_bytes': 0,
            'total_reclaimable_bytes': 0,
            'estimated_at': datetime.now().isoformat()
        }


class MultiArchManager:
    """
    Manages multi-architecture images.

    Handles manifest lists for cross-platform images.
    """

    def __init__(self):
        self.manifest_lists: Dict[str, Dict[str, Any]] = {}

    def create_manifest_list(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create manifest list for multi-arch image.

        Config:
            repo_id: Repository ID
            tag: Tag for manifest list
            manifests: List of architecture-specific manifests
        """
        list_id = f"mlist-{len(self.manifest_lists) + 1}"

        manifest_list = {
            'list_id': list_id,
            'repo_id': config.get('repo_id'),
            'tag': config.get('tag'),
            'manifests': config.get('manifests', []),
            'created_at': datetime.now().isoformat()
        }

        self.manifest_lists[list_id] = manifest_list

        print(f"📦 Created manifest list: {manifest_list['tag']}")
        print(f"   Architectures: {[m['architecture'] for m in manifest_list['manifests']]}")
        return manifest_list

    def get_manifest_for_arch(self, list_id: str, architecture: str) -> Optional[Dict[str, Any]]:
        """Get manifest for specific architecture."""
        if list_id in self.manifest_lists:
            manifests = self.manifest_lists[list_id]['manifests']
            for manifest in manifests:
                if manifest['architecture'] == architecture:
                    return manifest
        return None


class CleanupPolicyManager:
    """
    Manages automatic cleanup policies.

    Applies retention policies to remove old images automatically.
    """

    def __init__(self):
        self.policies: Dict[str, Dict[str, Any]] = {}
        self.policy_runs: List[Dict[str, Any]] = []

    def create_policy(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create cleanup policy.

        Config:
            repo_id: Repository ID
            enabled: Enable policy
            cadence: daily, weekly, monthly
            keep_n: Keep N most recent tags
            name_regex: Tag name pattern to match
            name_regex_keep: Tags matching this are kept
            older_than: Delete tags older than (days)
        """
        policy_id = f"policy-{len(self.policies) + 1}"

        policy = {
            'policy_id': policy_id,
            'repo_id': config.get('repo_id'),
            'enabled': config.get('enabled', True),
            'cadence': config.get('cadence', 'daily'),
            'keep_n': config.get('keep_n', 10),
            'name_regex': config.get('name_regex', '.*'),
            'name_regex_keep': config.get('name_regex_keep', ''),
            'older_than': config.get('older_than', 90),
            'created_at': datetime.now().isoformat()
        }

        self.policies[policy_id] = policy
        print(f"⚙️  Created cleanup policy: keep {policy['keep_n']} tags, delete older than {policy['older_than']} days")
        return policy

    def apply_policy(self, policy_id: str) -> Dict[str, Any]:
        """Apply cleanup policy."""
        if policy_id not in self.policies:
            return {'status': 'not_found'}

        policy = self.policies[policy_id]

        run = {
            'run_id': f"run-{len(self.policy_runs) + 1}",
            'policy_id': policy_id,
            'executed_at': datetime.now().isoformat(),
            'tags_removed': 0,
            'space_freed_bytes': 0
        }

        self.policy_runs.append(run)
        print(f"🧹 Applied policy: removed {run['tags_removed']} tags")
        return run

    def get_policy_runs(self, policy_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get policy execution history."""
        runs = [r for r in self.policy_runs if r['policy_id'] == policy_id]
        return runs[-limit:]


class RegistryStatisticsManager:
    """
    Tracks registry statistics and analytics.

    Storage usage, pull/push counts, popular images.
    """

    def __init__(self):
        self.stats = {
            'total_repositories': 0,
            'total_images': 0,
            'total_size_bytes': 0,
            'total_pulls': 0,
            'total_pushes': 0
        }

    def record_push(self, image_id: str, size_bytes: int) -> None:
        """Record image push."""
        self.stats['total_pushes'] += 1
        self.stats['total_size_bytes'] += size_bytes

    def record_pull(self, image_id: str) -> None:
        """Record image pull."""
        self.stats['total_pulls'] += 1

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            'total_size_bytes': self.stats['total_size_bytes'],
            'total_size_gb': self.stats['total_size_bytes'] / 1024 / 1024 / 1024,
            'total_repositories': self.stats['total_repositories'],
            'total_images': self.stats['total_images']
        }

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_pulls': self.stats['total_pulls'],
            'total_pushes': self.stats['total_pushes'],
            'pull_push_ratio': (
                self.stats['total_pulls'] / self.stats['total_pushes']
                if self.stats['total_pushes'] > 0 else 0
            )
        }


class ContainerRegistryManager:
    """
    Main container registry orchestration class.

    Coordinates all registry operations including repositories, images,
    scanning, garbage collection, and cleanup policies.
    """

    def __init__(self, project_id: str, gitlab_url: str = 'https://gitlab.com'):
        self.project_id = project_id
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.repositories = RepositoryManager(project_id)
        self.images = ImageManager()
        self.scanning = ImageScanManager()
        self.garbage_collection = GarbageCollectionManager()
        self.multi_arch = MultiArchManager()
        self.cleanup_policies = CleanupPolicyManager()
        self.statistics = RegistryStatisticsManager()

    def get_docker_commands(self, repo_name: str) -> Dict[str, str]:
        """Get Docker commands for working with registry."""
        registry_path = f"{self.gitlab_url}/{self.project_id}/{repo_name}"

        return {
            'login': f"docker login {self.gitlab_url}",
            'build': f"docker build -t {registry_path}:latest .",
            'tag': f"docker tag myimage:latest {registry_path}:v1.0.0",
            'push': f"docker push {registry_path}:latest",
            'pull': f"docker pull {registry_path}:latest"
        }


def demo_container_registry():
    """Demonstrate comprehensive container registry management."""
    print("\n" + "="*60)
    print("🐳 GitLab Container Registry Demo")
    print("="*60)

    mgr = ContainerRegistryManager(project_id='myorg/myproject')

    # 1. Create repository
    print("\n1️⃣  Creating Repository")
    repo = mgr.repositories.create_repository({
        'name': 'webapp',
        'description': 'Web application container',
        'visibility': ImageVisibility.PRIVATE.value
    })

    # 2. Push images
    print("\n2️⃣  Pushing Images")
    amd64_image = mgr.images.push_image({
        'repo_id': repo['repo_id'],
        'tag': 'v1.0.0',
        'size_bytes': 450 * 1024 * 1024,
        'layers': 12,
        'architecture': Architecture.AMD64.value
    })

    arm64_image = mgr.images.push_image({
        'repo_id': repo['repo_id'],
        'tag': 'v1.0.0-arm64',
        'size_bytes': 430 * 1024 * 1024,
        'layers': 12,
        'architecture': Architecture.ARM64.value
    })

    # 3. Create multi-arch manifest
    print("\n3️⃣  Creating Multi-Architecture Manifest")
    manifest_list = mgr.multi_arch.create_manifest_list({
        'repo_id': repo['repo_id'],
        'tag': 'v1.0.0',
        'manifests': [
            {'architecture': Architecture.AMD64.value, 'digest': amd64_image['digest']},
            {'architecture': Architecture.ARM64.value, 'digest': arm64_image['digest']}
        ]
    })

    # 4. Security scanning
    print("\n4️⃣  Security Scanning")
    scan = mgr.scanning.scan_image(amd64_image['image_id'], 'trivy')

    # 5. Cleanup policy
    print("\n5️⃣  Setting Cleanup Policy")
    policy = mgr.cleanup_policies.create_policy({
        'repo_id': repo['repo_id'],
        'keep_n': 5,
        'older_than': 90
    })

    # 6. Garbage collection
    print("\n6️⃣  Running Garbage Collection (Dry Run)")
    gc_run = mgr.garbage_collection.run_garbage_collection({
        'dry_run': True,
        'remove_untagged': True
    })

    # 7. Docker commands
    print("\n7️⃣  Docker Commands")
    commands = mgr.get_docker_commands('webapp')
    print(f"   Login:  {commands['login']}")
    print(f"   Build:  {commands['build']}")
    print(f"   Push:   {commands['push']}")

    print("\n" + "="*60)
    print("✅ Container Registry Demo Complete!")
    print("="*60)


if __name__ == '__main__':
    demo_container_registry()
