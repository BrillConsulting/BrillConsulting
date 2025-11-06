# Container Registry - Docker Image Management

Comprehensive GitLab container registry system providing Docker image storage, multi-architecture support, security scanning, garbage collection, and automated cleanup policies.

## Features

### Repository Management
- **Create Repositories**: Organize images by repository
- **Visibility Control**: PUBLIC, PRIVATE, INTERNAL
- **List & Filter**: Search repositories by name, visibility
- **Delete Repositories**: Remove repositories and all images
- **Cleanup Policies**: Automatic retention policies

### Image Management
- **Push Images**: Upload container images with metadata
- **Pull Images**: Download images with tracking
- **Tag Management**: Create, list, and delete tags
- **Retag Images**: Create additional tags for images
- **List Images**: Filter by tag, architecture, status
- **Delete Images**: Remove specific images
- **Image Digests**: SHA256 content addressing

### Multi-Architecture Support
- **Manifest Lists**: Cross-platform image support
- **AMD64**: x86_64 Linux
- **ARM64**: 64-bit ARM
- **ARM v7**: 32-bit ARM
- **PPC64LE**: IBM POWER
- **S390X**: IBM Z mainframe

### Security Scanning
- **Vulnerability Scanning**: Trivy, Clair, Grype, Snyk
- **CVE Tracking**: Common Vulnerabilities and Exposures
- **Severity Levels**: Critical, High, Medium, Low
- **Quarantine**: Isolate vulnerable images
- **Scan History**: Track all security scans

### Garbage Collection
- **Space Reclamation**: Remove deleted images and layers
- **Dry Run Mode**: Preview what would be deleted
- **Untagged Removal**: Clean up untagged manifests
- **Orphaned Blobs**: Remove unreferenced layers
- **GC History**: Track garbage collection runs

### Cleanup Policies
- **Automated Retention**: Remove old images automatically
- **Tag Limits**: Keep N most recent tags
- **Age-Based**: Delete images older than X days
- **Regex Patterns**: Match tags by regular expression
- **Exclusion Rules**: Protect specific tags
- **Cadence**: Daily, weekly, monthly schedules

### Statistics & Analytics
- **Storage Usage**: Total storage by repository
- **Pull/Push Counts**: Track image operations
- **Popular Images**: Most downloaded images
- **Usage Trends**: Monitor registry growth

## Usage Example

```python
from registry_manager import ContainerRegistryManager, ImageVisibility, Architecture

# Initialize manager
mgr = ContainerRegistryManager(project_id='myorg/myproject')

# 1. Create repository
repo = mgr.repositories.create_repository({
    'name': 'webapp',
    'description': 'Web application container',
    'visibility': ImageVisibility.PRIVATE.value
})

# 2. Push AMD64 image
amd64_image = mgr.images.push_image({
    'repo_id': repo['repo_id'],
    'tag': 'v1.0.0',
    'size_bytes': 450 * 1024 * 1024,  # 450 MB
    'layers': 12,
    'architecture': Architecture.AMD64.value
})

# 3. Push ARM64 image
arm64_image = mgr.images.push_image({
    'repo_id': repo['repo_id'],
    'tag': 'v1.0.0-arm64',
    'size_bytes': 430 * 1024 * 1024,
    'layers': 12,
    'architecture': Architecture.ARM64.value
})

# 4. Create multi-arch manifest list
manifest_list = mgr.multi_arch.create_manifest_list({
    'repo_id': repo['repo_id'],
    'tag': 'v1.0.0',
    'manifests': [
        {'architecture': Architecture.AMD64.value, 'digest': amd64_image['digest']},
        {'architecture': Architecture.ARM64.value, 'digest': arm64_image['digest']}
    ]
})

# 5. Security scan
scan = mgr.scanning.scan_image(amd64_image['image_id'], scanner='trivy')

# Add vulnerability if found
mgr.scanning.add_vulnerability(amd64_image['image_id'], {
    'cve_id': 'CVE-2023-12345',
    'severity': 'high',
    'package': 'openssl',
    'installed_version': '1.1.1k',
    'fixed_version': '1.1.1t'
})

# Quarantine if critical
vulns = mgr.scanning.get_vulnerabilities(amd64_image['image_id'], min_severity='critical')
if vulns:
    mgr.scanning.quarantine_image(amd64_image['image_id'], 'Critical vulnerabilities detected')

# 6. Set cleanup policy
policy = mgr.cleanup_policies.create_policy({
    'repo_id': repo['repo_id'],
    'enabled': True,
    'cadence': 'weekly',
    'keep_n': 10,
    'older_than': 90,
    'name_regex': '.*',
    'name_regex_keep': 'latest|stable'
})

# Apply policy
mgr.cleanup_policies.apply_policy(policy['policy_id'])

# 7. Garbage collection
gc_run = mgr.garbage_collection.run_garbage_collection({
    'dry_run': True,
    'remove_untagged': True
})

estimate = mgr.garbage_collection.estimate_reclaimable_space()
print(f"Reclaimable: {estimate['total_reclaimable_bytes'] / 1024 / 1024} MB")

# 8. List images
all_images = mgr.images.list_images(repo['repo_id'])
amd64_only = mgr.images.list_images(repo['repo_id'], {'architecture': Architecture.AMD64.value})

# 9. Pull image
pulled = mgr.images.pull_image(amd64_image['image_id'])

# 10. Docker commands
commands = mgr.get_docker_commands('webapp')
print(commands['login'])
print(commands['push'])
```

## Supported Architectures

| Architecture | Description | Use Case |
|--------------|-------------|----------|
| **linux/amd64** | x86_64 (64-bit Intel/AMD) | Most servers, desktops |
| **linux/arm64** | ARM 64-bit (aarch64) | Apple Silicon, AWS Graviton |
| **linux/arm/v7** | ARM 32-bit | Raspberry Pi, IoT devices |
| **linux/ppc64le** | IBM POWER (Little Endian) | Enterprise servers |
| **linux/s390x** | IBM Z mainframe | Enterprise mainframe |

## Docker Registry Operations

### Login to Registry

```bash
# Using personal access token
docker login registry.gitlab.com -u username -p YOUR_TOKEN

# Using deploy token
docker login registry.gitlab.com -u deploy_token_username -p DEPLOY_TOKEN

# Using CI/CD job token
docker login -u gitlab-ci-token -p $CI_JOB_TOKEN $CI_REGISTRY
```

### Build and Push

```bash
# Build image
docker build -t myapp:v1.0.0 .

# Tag for GitLab registry
docker tag myapp:v1.0.0 registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Push to registry
docker push registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Push multiple tags
docker push registry.gitlab.com/myorg/myproject/myapp:v1.0.0
docker push registry.gitlab.com/myorg/myproject/myapp:latest
```

### Pull Image

```bash
# Pull specific tag
docker pull registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Pull latest
docker pull registry.gitlab.com/myorg/myproject/myapp:latest

# Pull and run
docker run -d registry.gitlab.com/myorg/myproject/myapp:v1.0.0
```

### Multi-Architecture Build

```bash
# Create and use buildx builder
docker buildx create --name multiarch --use

# Build and push multi-arch image
docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 \
  -t registry.gitlab.com/myorg/myproject/myapp:v1.0.0 \
  --push .

# Inspect manifest list
docker buildx imagetools inspect registry.gitlab.com/myorg/myproject/myapp:v1.0.0
```

## Security Scanning

### Vulnerability Scanners

| Scanner | Type | Best For |
|---------|------|----------|
| **Trivy** | Container/OS | Fast, comprehensive, offline DB |
| **Clair** | Container | Static analysis, layer scanning |
| **Grype** | Container/OS | Fast, accurate, easy to use |
| **Snyk** | Container/Dependencies | Developer-friendly, CI/CD |

### Scan with Trivy

```bash
# Scan image
trivy image registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Scan with severity filter
trivy image --severity HIGH,CRITICAL registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Output as JSON
trivy image --format json --output report.json registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Fail on vulnerabilities
trivy image --exit-code 1 --severity CRITICAL registry.gitlab.com/myorg/myproject/myapp:v1.0.0
```

### Scan with Clair

```bash
# Start Clair
docker run -d --name clair -p 6060:6060 -p 6061:6061 quay.io/coreos/clair:latest

# Scan image
clairctl analyze registry.gitlab.com/myorg/myproject/myapp:v1.0.0

# Generate report
clairctl report registry.gitlab.com/myorg/myproject/myapp:v1.0.0
```

## Cleanup Policies

### Policy Configuration

```python
# Keep 10 most recent tags
policy = mgr.cleanup_policies.create_policy({
    'repo_id': 'repo-1',
    'enabled': True,
    'keep_n': 10,
    'older_than': 0  # No age restriction
})

# Delete tags older than 90 days
policy = mgr.cleanup_policies.create_policy({
    'repo_id': 'repo-1',
    'enabled': True,
    'keep_n': 999,  # Keep many tags
    'older_than': 90  # But delete if older than 90 days
})

# Regex-based cleanup
policy = mgr.cleanup_policies.create_policy({
    'repo_id': 'repo-1',
    'enabled': True,
    'name_regex': '^v[0-9]+\\.[0-9]+\\.[0-9]+$',  # Match semver tags
    'name_regex_keep': 'latest|stable|production',  # Never delete these
    'keep_n': 5,
    'older_than': 30
})
```

### Policy Cadence

| Cadence | Description | Use Case |
|---------|-------------|----------|
| **daily** | Run every day | High-activity repos |
| **weekly** | Run every 7 days | Medium-activity repos |
| **monthly** | Run every 30 days | Low-activity, stable repos |

## Garbage Collection

### Run Garbage Collection

```python
# Dry run (preview)
gc_run = mgr.garbage_collection.run_garbage_collection({
    'dry_run': True,
    'remove_untagged': True,
    'older_than': 30
})

# Actual execution
gc_run = mgr.garbage_collection.run_garbage_collection({
    'dry_run': False,
    'remove_untagged': True
})

# Get history
history = mgr.garbage_collection.get_gc_history(limit=10)

# Estimate reclaimable space
estimate = mgr.garbage_collection.estimate_reclaimable_space()
print(f"Can reclaim: {estimate['total_reclaimable_bytes']} bytes")
```

### What Gets Collected

- **Untagged Manifests**: Images without any tags
- **Orphaned Blobs**: Layers not referenced by any manifest
- **Deleted Images**: Images marked for deletion
- **Old Layers**: Unused layers from deleted images

## Image Visibility

### Visibility Levels

| Level | Access | Use Case |
|-------|--------|----------|
| **PUBLIC** | Anyone can pull | Open source projects |
| **PRIVATE** | Project members only | Private applications |
| **INTERNAL** | All authenticated users | Company-wide images |

```python
# Public repository
repo = mgr.repositories.create_repository({
    'name': 'opensource-tool',
    'visibility': ImageVisibility.PUBLIC.value
})

# Private repository
repo = mgr.repositories.create_repository({
    'name': 'company-app',
    'visibility': ImageVisibility.PRIVATE.value
})

# Internal repository
repo = mgr.repositories.create_repository({
    'name': 'shared-base-image',
    'visibility': ImageVisibility.INTERNAL.value
})
```

## Best Practices

### Image Tagging
1. **Semantic Versioning**: Use v1.0.0, v1.1.0, v2.0.0
2. **Git SHA**: Include git commit hash
3. **Environment Tags**: dev, staging, production
4. **Latest Tag**: Always maintain :latest
5. **Immutable Tags**: Never overwrite version tags

### Security
1. **Scan on Push**: Automatically scan all new images
2. **Quarantine**: Isolate images with critical vulnerabilities
3. **Regular Updates**: Update base images frequently
4. **Minimal Images**: Use alpine or distroless
5. **Non-Root**: Run containers as non-root user

### Storage Management
1. **Cleanup Policies**: Enable automatic cleanup
2. **Tag Limits**: Keep only recent versions
3. **Garbage Collection**: Run GC weekly
4. **Monitor Usage**: Track storage trends
5. **Compress Layers**: Use multi-stage builds

### Multi-Architecture
1. **Build for Common Platforms**: amd64, arm64
2. **Manifest Lists**: Use for cross-platform support
3. **Test All Architectures**: Verify all platform builds
4. **Consistent Tags**: Same tag for all architectures
5. **Platform-Specific**: Use separate tags if needed

### Registry Performance
1. **Layer Caching**: Optimize layer ordering
2. **Parallel Pushes**: Push multiple tags concurrently
3. **Compression**: Use gzip for layers
4. **CDN**: Use content delivery network
5. **Regional Mirrors**: Deploy close to users

## CI/CD Integration

### GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - scan
  - push

variables:
  IMAGE_TAG: $CI_REGISTRY_IMAGE:$CI_COMMIT_REF_SLUG

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $IMAGE_TAG .
    - docker save $IMAGE_TAG -o image.tar
  artifacts:
    paths:
      - image.tar

test:
  stage: test
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker load -i image.tar
    - docker run --rm $IMAGE_TAG test

scan:
  stage: scan
  image: aquasec/trivy:latest
  script:
    - trivy image --exit-code 0 --severity HIGH,CRITICAL --no-progress $IMAGE_TAG
  allow_failure: true

push:
  stage: push
  image: docker:latest
  services:
    - docker:dind
  only:
    - main
    - tags
  script:
    - docker load -i image.tar
    - docker login -u $CI_REGISTRY_USER -p $CI_REGISTRY_PASSWORD $CI_REGISTRY
    - docker push $IMAGE_TAG
    - |
      if [[ "$CI_COMMIT_TAG" ]]; then
        docker tag $IMAGE_TAG $CI_REGISTRY_IMAGE:latest
        docker push $CI_REGISTRY_IMAGE:latest
      fi
```

### Multi-Arch Build CI

```yaml
build-multiarch:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  before_script:
    - docker buildx create --use
  script:
    - docker buildx build
      --platform linux/amd64,linux/arm64,linux/arm/v7
      -t $CI_REGISTRY_IMAGE:$CI_COMMIT_TAG
      -t $CI_REGISTRY_IMAGE:latest
      --push .
```

## Common Use Cases

### Base Image Repository

```python
# Create base image repository
base_repo = mgr.repositories.create_repository({
    'name': 'base-images/python',
    'description': 'Python base images for all projects',
    'visibility': ImageVisibility.INTERNAL.value
})

# Push different Python versions
for version in ['3.9', '3.10', '3.11', '3.12']:
    mgr.images.push_image({
        'repo_id': base_repo['repo_id'],
        'tag': version
    })
```

### Microservices Registry

```python
# Create repository for each service
services = ['user-service', 'order-service', 'payment-service']

for service in services:
    repo = mgr.repositories.create_repository({
        'name': f'microservices/{service}',
        'visibility': ImageVisibility.PRIVATE.value
    })

    # Set aggressive cleanup for dev images
    mgr.cleanup_policies.create_policy({
        'repo_id': repo['repo_id'],
        'keep_n': 3,
        'older_than': 7
    })
```

### Production Image Management

```python
# Create production repository
prod_repo = mgr.repositories.create_repository({
    'name': 'production/webapp',
    'visibility': ImageVisibility.PRIVATE.value
})

# Conservative cleanup policy
policy = mgr.cleanup_policies.create_policy({
    'repo_id': prod_repo['repo_id'],
    'keep_n': 20,  # Keep many versions
    'older_than': 180,  # Keep for 6 months
    'name_regex_keep': 'production|stable'  # Never delete these
})
```

## Troubleshooting

**Issue**: Image push fails with "unauthorized"
- Verify you're logged in: `docker login registry.gitlab.com`
- Check token has write_registry scope
- Ensure you have Developer role or higher

**Issue**: Multi-arch image not working on ARM
- Verify manifest list includes ARM architecture
- Check ARM-specific image was built and pushed
- Inspect with: `docker buildx imagetools inspect IMAGE`

**Issue**: Storage usage too high
- Run garbage collection
- Enable cleanup policies
- Remove unused repositories
- Check for duplicate layers

**Issue**: Scan reports false positives
- Update scanner to latest version
- Use allowlist for known safe packages
- Check if vulnerability is actually exploitable
- Update base images

**Issue**: Slow image pulls
- Use regional registry mirrors
- Optimize layer sizes
- Enable layer caching
- Check network bandwidth

## Requirements

```
hashlib (standard library)
datetime (standard library)
typing (standard library)
enum (standard library)
```

No external dependencies required.

## Configuration

### Environment Variables

```bash
export CI_REGISTRY="registry.gitlab.com"
export CI_REGISTRY_IMAGE="registry.gitlab.com/myorg/myproject"
export CI_REGISTRY_USER="gitlab-ci-token"
export CI_REGISTRY_PASSWORD="${CI_JOB_TOKEN}"
```

### Python Configuration

```python
from registry_manager import ContainerRegistryManager

mgr = ContainerRegistryManager(
    project_id='myorg/myproject',
    gitlab_url='https://gitlab.com'
)
```

## Author

BrillConsulting - Enterprise Cloud Solutions
