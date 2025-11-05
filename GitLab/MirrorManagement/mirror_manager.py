"""
GitLab Mirror Management
Author: BrillConsulting
Description: Repository mirroring and synchronization
"""

import json
from typing import Dict, List, Any
from datetime import datetime


class MirrorManager:
    """GitLab Repository Mirror management"""

    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.mirrors = []

    def create_pull_mirror(self, mirror_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create pull mirror (import from external repo)"""
        mirror = {
            'id': len(self.mirrors) + 1,
            'type': 'pull',
            'project_id': mirror_config.get('project_id', 1),
            'url': mirror_config.get('url', 'https://github.com/user/repo.git'),
            'enabled': mirror_config.get('enabled', True),
            'update_interval': mirror_config.get('update_interval', 300),
            'only_protected_branches': mirror_config.get('only_protected', False),
            'created_at': datetime.now().isoformat()
        }
        self.mirrors.append(mirror)
        print(f"✓ Pull mirror created")
        print(f"  URL: {mirror['url']}")
        print(f"  Update interval: {mirror['update_interval']}s")
        return mirror

    def create_push_mirror(self, mirror_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create push mirror (export to external repo)"""
        mirror = {
            'id': len(self.mirrors) + 1,
            'type': 'push',
            'project_id': mirror_config.get('project_id', 1),
            'url': mirror_config.get('url', 'https://bitbucket.org/user/repo.git'),
            'enabled': mirror_config.get('enabled', True),
            'only_protected_branches': mirror_config.get('only_protected', True),
            'keep_divergent_refs': mirror_config.get('keep_divergent', False),
            'created_at': datetime.now().isoformat()
        }
        self.mirrors.append(mirror)
        print(f"✓ Push mirror created")
        print(f"  URL: {mirror['url']}")
        print(f"  Only protected: {mirror['only_protected_branches']}")
        return mirror

    def trigger_mirror_update(self, mirror_id: int) -> Dict[str, Any]:
        """Trigger mirror update manually"""
        mirror = next((m for m in self.mirrors if m['id'] == mirror_id), None)
        if mirror:
            result = {
                'mirror_id': mirror_id,
                'status': 'updating',
                'triggered_at': datetime.now().isoformat()
            }
            print(f"✓ Mirror update triggered: #{mirror_id}")
            return result
        return {'error': 'Mirror not found'}


def demo():
    """Demonstrate mirror management"""
    print("=" * 60)
    print("GitLab Mirror Management Demo")
    print("=" * 60)

    mgr = MirrorManager('https://gitlab.example.com', 'token')

    print("\n1. Creating pull mirror...")
    mgr.create_pull_mirror({
        'project_id': 1,
        'url': 'https://github.com/user/repo.git',
        'update_interval': 300
    })

    print("\n2. Creating push mirror...")
    mgr.create_push_mirror({
        'project_id': 1,
        'url': 'https://bitbucket.org/user/repo.git',
        'only_protected': True
    })

    print("\n3. Triggering mirror update...")
    mgr.trigger_mirror_update(1)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    demo()
