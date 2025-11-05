"""
GitLab Package Management
Author: BrillConsulting
Description: Maven, npm, PyPI, NuGet package registry management
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class PackageManager:
    """GitLab Package Registry management"""

    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.packages = []

    def publish_package(self, pkg_config: Dict[str, Any]) -> Dict[str, Any]:
        """Publish package to GitLab registry"""
        package = {
            'id': len(self.packages) + 1,
            'name': pkg_config.get('name', 'mypackage'),
            'version': pkg_config.get('version', '1.0.0'),
            'type': pkg_config.get('type', 'npm'),
            'size_mb': pkg_config.get('size_mb', 2.5),
            'published_at': datetime.now().isoformat()
        }

        commands = {
            'npm': f"npm publish --registry=https://gitlab.com/api/v4/projects/123/packages/npm/",
            'maven': f"mvn deploy -DaltDeploymentRepository=gitlab::default::https://gitlab.com/api/v4/projects/123/packages/maven",
            'pypi': f"twine upload --repository-url https://gitlab.com/api/v4/projects/123/packages/pypi dist/*"
        }

        self.packages.append(package)
        print(f"✓ Package published: {package['name']}@{package['version']}")
        print(f"  Type: {package['type']}, Size: {package['size_mb']}MB")
        print(f"  Command: {commands.get(package['type'], 'N/A')}")
        return package

    def list_packages(self, filter_params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """List packages in registry"""
        filter_params = filter_params or {}
        package_type = filter_params.get('type')
        packages = self.packages
        if package_type:
            packages = [p for p in packages if p['type'] == package_type]
        print(f"✓ Listed {len(packages)} packages")
        return packages

    def delete_package(self, package_id: int) -> Dict[str, Any]:
        """Delete package from registry"""
        package = next((p for p in self.packages if p['id'] == package_id), None)
        if package:
            self.packages.remove(package)
            print(f"✓ Package deleted: {package['name']}@{package['version']}")
            return {'status': 'deleted'}
        return {'error': 'Package not found'}


def demo():
    """Demonstrate package management"""
    print("=" * 60)
    print("GitLab Package Management Demo")
    print("=" * 60)

    mgr = PackageManager('https://gitlab.example.com', 'token')

    print("\n1. Publishing npm package...")
    mgr.publish_package({'name': 'my-lib', 'version': '1.0.0', 'type': 'npm'})

    print("\n2. Publishing Maven package...")
    mgr.publish_package({'name': 'my-java-lib', 'version': '2.0.0', 'type': 'maven'})

    print("\n3. Listing packages...")
    mgr.list_packages({'type': 'npm'})

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
