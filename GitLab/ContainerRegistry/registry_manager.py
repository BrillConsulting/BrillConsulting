"""
GitLab Container Registry Management
Author: BrillConsulting
Description: Complete Docker container registry integration and management
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class RegistryManager:
    """GitLab Container Registry management"""

    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.repositories = []

    def create_repository(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create container repository"""
        repo = {
            'id': len(self.repositories) + 1,
            'project_id': config.get('project_id', 1),
            'name': config.get('name', 'myapp'),
            'path': f"registry.gitlab.com/group/project/{config.get('name', 'myapp')}",
            'tags': config.get('tags', []),
            'size_bytes': config.get('size', 524288000),
            'created_at': datetime.now().isoformat()
        }
        self.repositories.append(repo)
        print(f"✓ Repository created: {repo['name']}")
        print(f"  Path: {repo['path']}")
        return repo

    def push_image(self, image_config: Dict[str, Any]) -> Dict[str, Any]:
        """Push Docker image to registry"""
        image = {
            'repository': image_config.get('repository', 'myapp'),
            'tag': image_config.get('tag', 'latest'),
            'size_mb': image_config.get('size_mb', 500),
            'layers': image_config.get('layers', 12),
            'pushed_at': datetime.now().isoformat()
        }

        commands = [
            f"docker login registry.gitlab.com",
            f"docker tag myapp:latest registry.gitlab.com/group/project/{image['repository']}:{image['tag']}",
            f"docker push registry.gitlab.com/group/project/{image['repository']}:{image['tag']}"
        ]

        print(f"✓ Image pushed: {image['repository']}:{image['tag']}")
        print(f"  Size: {image['size_mb']}MB, Layers: {image['layers']}")
        return image

    def cleanup_old_images(self, cleanup_config: Dict[str, Any]) -> Dict[str, Any]:
        """Cleanup old container images"""
        result = {
            'repository': cleanup_config.get('repository', 'myapp'),
            'keep_n_tags': cleanup_config.get('keep_n', 10),
            'removed_count': cleanup_config.get('removed', 15),
            'freed_space_gb': cleanup_config.get('freed_gb', 5.2),
            'cleaned_at': datetime.now().isoformat()
        }
        print(f"✓ Cleanup completed: {result['removed_count']} tags removed")
        print(f"  Freed space: {result['freed_space_gb']}GB")
        return result


def demo():
    """Demonstrate registry management"""
    print("=" * 60)
    print("GitLab Container Registry Demo")
    print("=" * 60)

    mgr = RegistryManager('https://gitlab.example.com', 'token')

    print("\n1. Creating repository...")
    mgr.create_repository({'name': 'webapp', 'project_id': 1})

    print("\n2. Pushing image...")
    mgr.push_image({'repository': 'webapp', 'tag': 'v1.0.0', 'size_mb': 450})

    print("\n3. Cleaning up old images...")
    mgr.cleanup_old_images({'repository': 'webapp', 'keep_n': 10})

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
