"""
ProjectTemplates - Project Template Management
Author: BrillConsulting
Description: Comprehensive GitLab project template system for quick project creation with predefined structures and configurations
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import json


class TemplateCategory(Enum):
    """Template categories."""
    WEB_APPLICATION = "web_application"
    MICROSERVICE = "microservice"
    MOBILE_APP = "mobile_app"
    DATA_SCIENCE = "data_science"
    DEVOPS = "devops"
    DOCUMENTATION = "documentation"
    LIBRARY = "library"
    API = "api"
    MONOREPO = "monorepo"
    STATIC_SITE = "static_site"
    CUSTOM = "custom"


class TemplateLanguage(Enum):
    """Programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"
    CSHARP = "csharp"
    CPP = "cpp"
    SWIFT = "swift"
    KOTLIN = "kotlin"


class TemplateFramework(Enum):
    """Frameworks."""
    DJANGO = "django"
    FLASK = "flask"
    FASTAPI = "fastapi"
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    NEXTJS = "nextjs"
    EXPRESS = "express"
    SPRING_BOOT = "spring_boot"
    LARAVEL = "laravel"
    RAILS = "rails"


class TemplateManager:
    """Manage project templates."""

    def __init__(self):
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.template_counter = 1

    def create_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create project template.

        Config:
        - name: Template name
        - description: Template description
        - category: TemplateCategory
        - language: TemplateLanguage
        - framework: TemplateFramework (optional)
        - files: List of template files
        - variables: Dict of template variables
        - ci_config: CI/CD configuration
        - is_public: Public template (default: True)
        """
        template_id = f"template-{self.template_counter}"
        self.template_counter += 1

        template = {
            "template_id": template_id,
            "name": config.get('name'),
            "description": config.get('description'),
            "category": config.get('category'),
            "language": config.get('language'),
            "framework": config.get('framework'),
            "files": config.get('files', []),
            "variables": config.get('variables', {}),
            "ci_config": config.get('ci_config'),
            "is_public": config.get('is_public', True),
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0"
        }

        self.templates[template_id] = template
        return template

    def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def list_templates(self, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        List templates with optional filters.

        Filters:
        - category: TemplateCategory
        - language: TemplateLanguage
        - framework: TemplateFramework
        - is_public: Boolean
        """
        templates = list(self.templates.values())

        if filters:
            if 'category' in filters:
                category = filters['category']
                category_value = category.value if isinstance(category, TemplateCategory) else category
                templates = [t for t in templates if t['category'] == category_value]

            if 'language' in filters:
                language = filters['language']
                language_value = language.value if isinstance(language, TemplateLanguage) else language
                templates = [t for t in templates if t['language'] == language_value]

            if 'framework' in filters:
                framework = filters['framework']
                framework_value = framework.value if isinstance(framework, TemplateFramework) else framework
                templates = [t for t in templates if t['framework'] == framework_value]

            if 'is_public' in filters:
                templates = [t for t in templates if t['is_public'] == filters['is_public']]

        return templates

    def delete_template(self, template_id: str) -> Dict[str, Any]:
        """Delete template."""
        if template_id in self.templates:
            del self.templates[template_id]
            return {"status": "deleted", "template_id": template_id}
        return {"status": "not_found", "template_id": template_id}


class BuiltInTemplatesManager:
    """Manage built-in project templates."""

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager

    def create_python_django_template(self) -> Dict[str, Any]:
        """Create Django web application template."""
        return self.template_manager.create_template({
            "name": "Django Web Application",
            "description": "Full-stack Django web application with PostgreSQL and CI/CD",
            "category": TemplateCategory.WEB_APPLICATION.value,
            "language": TemplateLanguage.PYTHON.value,
            "framework": TemplateFramework.DJANGO.value,
            "files": [
                {"path": "manage.py", "type": "file"},
                {"path": "requirements.txt", "type": "file"},
                {"path": "app/__init__.py", "type": "file"},
                {"path": "app/settings.py", "type": "file"},
                {"path": "app/urls.py", "type": "file"},
                {"path": "app/wsgi.py", "type": "file"},
                {"path": "Dockerfile", "type": "file"},
                {"path": "docker-compose.yml", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "project_name": "myproject",
                "database": "postgresql",
                "python_version": "3.11"
            },
            "ci_config": {
                "stages": ["test", "build", "deploy"],
                "docker": True
            }
        })

    def create_react_app_template(self) -> Dict[str, Any]:
        """Create React application template."""
        return self.template_manager.create_template({
            "name": "React Application",
            "description": "Modern React app with TypeScript and Vite",
            "category": TemplateCategory.WEB_APPLICATION.value,
            "language": TemplateLanguage.TYPESCRIPT.value,
            "framework": TemplateFramework.REACT.value,
            "files": [
                {"path": "package.json", "type": "file"},
                {"path": "tsconfig.json", "type": "file"},
                {"path": "vite.config.ts", "type": "file"},
                {"path": "src/App.tsx", "type": "file"},
                {"path": "src/main.tsx", "type": "file"},
                {"path": "src/index.css", "type": "file"},
                {"path": "public/index.html", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "app_name": "my-react-app",
                "node_version": "18"
            },
            "ci_config": {
                "stages": ["test", "build", "deploy"],
                "node": True
            }
        })

    def create_python_fastapi_template(self) -> Dict[str, Any]:
        """Create FastAPI microservice template."""
        return self.template_manager.create_template({
            "name": "FastAPI Microservice",
            "description": "FastAPI microservice with Docker and OpenAPI",
            "category": TemplateCategory.MICROSERVICE.value,
            "language": TemplateLanguage.PYTHON.value,
            "framework": TemplateFramework.FASTAPI.value,
            "files": [
                {"path": "main.py", "type": "file"},
                {"path": "requirements.txt", "type": "file"},
                {"path": "app/__init__.py", "type": "file"},
                {"path": "app/api/v1/__init__.py", "type": "file"},
                {"path": "app/models.py", "type": "file"},
                {"path": "app/schemas.py", "type": "file"},
                {"path": "Dockerfile", "type": "file"},
                {"path": "docker-compose.yml", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "service_name": "api-service",
                "python_version": "3.11"
            },
            "ci_config": {
                "stages": ["test", "build", "deploy"],
                "docker": True
            }
        })

    def create_go_microservice_template(self) -> Dict[str, Any]:
        """Create Go microservice template."""
        return self.template_manager.create_template({
            "name": "Go Microservice",
            "description": "Go microservice with gRPC and Docker",
            "category": TemplateCategory.MICROSERVICE.value,
            "language": TemplateLanguage.GO.value,
            "files": [
                {"path": "main.go", "type": "file"},
                {"path": "go.mod", "type": "file"},
                {"path": "go.sum", "type": "file"},
                {"path": "internal/handlers/handler.go", "type": "file"},
                {"path": "internal/services/service.go", "type": "file"},
                {"path": "pkg/proto/service.proto", "type": "file"},
                {"path": "Dockerfile", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "module_name": "github.com/myorg/service",
                "go_version": "1.21"
            },
            "ci_config": {
                "stages": ["test", "build", "deploy"],
                "docker": True
            }
        })

    def create_nodejs_express_template(self) -> Dict[str, Any]:
        """Create Node.js Express API template."""
        return self.template_manager.create_template({
            "name": "Node.js Express API",
            "description": "RESTful API with Express and TypeScript",
            "category": TemplateCategory.API.value,
            "language": TemplateLanguage.TYPESCRIPT.value,
            "framework": TemplateFramework.EXPRESS.value,
            "files": [
                {"path": "package.json", "type": "file"},
                {"path": "tsconfig.json", "type": "file"},
                {"path": "src/index.ts", "type": "file"},
                {"path": "src/routes/api.ts", "type": "file"},
                {"path": "src/controllers/controller.ts", "type": "file"},
                {"path": "src/models/model.ts", "type": "file"},
                {"path": "src/middleware/auth.ts", "type": "file"},
                {"path": "Dockerfile", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "api_name": "my-api",
                "node_version": "18"
            },
            "ci_config": {
                "stages": ["test", "build", "deploy"],
                "node": True,
                "docker": True
            }
        })

    def create_python_data_science_template(self) -> Dict[str, Any]:
        """Create Python data science template."""
        return self.template_manager.create_template({
            "name": "Python Data Science Project",
            "description": "Data science project with Jupyter and MLflow",
            "category": TemplateCategory.DATA_SCIENCE.value,
            "language": TemplateLanguage.PYTHON.value,
            "files": [
                {"path": "requirements.txt", "type": "file"},
                {"path": "notebooks/exploration.ipynb", "type": "file"},
                {"path": "src/data_processing.py", "type": "file"},
                {"path": "src/model_training.py", "type": "file"},
                {"path": "src/evaluation.py", "type": "file"},
                {"path": "data/.gitkeep", "type": "file"},
                {"path": "models/.gitkeep", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "project_name": "ds-project",
                "python_version": "3.11"
            },
            "ci_config": {
                "stages": ["test", "train", "deploy"],
                "jupyter": True
            }
        })

    def create_terraform_template(self) -> Dict[str, Any]:
        """Create Terraform infrastructure template."""
        return self.template_manager.create_template({
            "name": "Terraform Infrastructure",
            "description": "Infrastructure as Code with Terraform",
            "category": TemplateCategory.DEVOPS.value,
            "language": TemplateLanguage.GO.value,  # Terraform uses HCL but Go for providers
            "files": [
                {"path": "main.tf", "type": "file"},
                {"path": "variables.tf", "type": "file"},
                {"path": "outputs.tf", "type": "file"},
                {"path": "terraform.tfvars", "type": "file"},
                {"path": "modules/networking/main.tf", "type": "file"},
                {"path": "modules/compute/main.tf", "type": "file"},
                {"path": ".gitlab-ci.yml", "type": "file"},
                {"path": "README.md", "type": "file"}
            ],
            "variables": {
                "cloud_provider": "aws",
                "terraform_version": "1.5"
            },
            "ci_config": {
                "stages": ["validate", "plan", "apply"],
                "terraform": True
            }
        })


class TemplateFileManager:
    """Manage template files and content."""

    def __init__(self):
        self.file_templates: Dict[str, str] = {}

    def add_file_template(self, filename: str, content: str) -> Dict[str, Any]:
        """Add file template content."""
        self.file_templates[filename] = content
        return {
            "filename": filename,
            "size": len(content),
            "added_at": datetime.now().isoformat()
        }

    def get_file_template(self, filename: str) -> Optional[str]:
        """Get file template content."""
        return self.file_templates.get(filename)

    def render_file(self, filename: str, variables: Dict[str, Any]) -> str:
        """Render file template with variables."""
        template_content = self.file_templates.get(filename, "")

        # Simple variable substitution
        for var_name, var_value in variables.items():
            placeholder = f"{{{{{var_name}}}}}"  # {{variable}}
            template_content = template_content.replace(placeholder, str(var_value))

        return template_content


class TemplateApplicationManager:
    """Apply templates to create new projects."""

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        self.applications: Dict[str, Dict[str, Any]] = {}
        self.application_counter = 1

    def apply_template(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply template to create new project.

        Config:
        - template_id: Template ID to apply
        - project_name: New project name
        - project_path: Project path
        - variables: Variable values for template
        """
        template_id = config.get('template_id')
        template = self.template_manager.get_template(template_id)

        if not template:
            return {"status": "error", "message": "Template not found"}

        application_id = f"app-{self.application_counter}"
        self.application_counter += 1

        # Merge template variables with provided variables
        merged_variables = {**template['variables'], **config.get('variables', {})}

        application = {
            "application_id": application_id,
            "template_id": template_id,
            "template_name": template['name'],
            "project_name": config.get('project_name'),
            "project_path": config.get('project_path'),
            "variables": merged_variables,
            "files_created": len(template['files']),
            "status": "created",
            "created_at": datetime.now().isoformat()
        }

        self.applications[application_id] = application
        return application

    def get_application(self, application_id: str) -> Optional[Dict[str, Any]]:
        """Get template application details."""
        return self.applications.get(application_id)

    def list_applications(self) -> List[Dict[str, Any]]:
        """List all template applications."""
        return list(self.applications.values())


class TemplateImportExportManager:
    """Import and export templates."""

    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager

    def export_template(self, template_id: str) -> str:
        """Export template to JSON."""
        template = self.template_manager.get_template(template_id)
        if not template:
            return json.dumps({"error": "Template not found"})

        return json.dumps(template, indent=2)

    def import_template(self, template_json: str) -> Dict[str, Any]:
        """Import template from JSON."""
        try:
            template_data = json.loads(template_json)

            # Remove ID to generate new one
            if 'template_id' in template_data:
                del template_data['template_id']

            return self.template_manager.create_template(template_data)
        except json.JSONDecodeError as e:
            return {"status": "error", "message": f"Invalid JSON: {str(e)}"}

    def export_all_templates(self) -> str:
        """Export all templates to JSON."""
        templates = list(self.template_manager.templates.values())
        return json.dumps(templates, indent=2)


class TemplateVersionManager:
    """Manage template versions."""

    def __init__(self):
        self.versions: Dict[str, List[Dict[str, Any]]] = {}

    def create_version(self, template_id: str, version: str, changes: str) -> Dict[str, Any]:
        """Create new template version."""
        version_record = {
            "version": version,
            "changes": changes,
            "created_at": datetime.now().isoformat()
        }

        if template_id not in self.versions:
            self.versions[template_id] = []

        self.versions[template_id].append(version_record)
        return version_record

    def get_versions(self, template_id: str) -> List[Dict[str, Any]]:
        """Get all versions for template."""
        return self.versions.get(template_id, [])

    def get_latest_version(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get latest version for template."""
        versions = self.versions.get(template_id, [])
        return versions[-1] if versions else None


class ProjectTemplatesManager:
    """Main project templates manager."""

    def __init__(self, gitlab_url: str = 'https://gitlab.com'):
        self.gitlab_url = gitlab_url

        # Initialize all managers
        self.templates = TemplateManager()
        self.built_in = BuiltInTemplatesManager(self.templates)
        self.files = TemplateFileManager()
        self.applications = TemplateApplicationManager(self.templates)
        self.import_export = TemplateImportExportManager(self.templates)
        self.versions = TemplateVersionManager()

    def initialize_built_in_templates(self) -> Dict[str, Any]:
        """Initialize all built-in templates."""
        templates_created = []

        # Create all built-in templates
        templates_created.append(self.built_in.create_python_django_template())
        templates_created.append(self.built_in.create_react_app_template())
        templates_created.append(self.built_in.create_python_fastapi_template())
        templates_created.append(self.built_in.create_go_microservice_template())
        templates_created.append(self.built_in.create_nodejs_express_template())
        templates_created.append(self.built_in.create_python_data_science_template())
        templates_created.append(self.built_in.create_terraform_template())

        return {
            "status": "success",
            "templates_created": len(templates_created),
            "templates": templates_created
        }

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get template usage statistics."""
        all_templates = list(self.templates.templates.values())

        # Count by category
        category_counts = {}
        for template in all_templates:
            category = template['category']
            category_counts[category] = category_counts.get(category, 0) + 1

        # Count by language
        language_counts = {}
        for template in all_templates:
            language = template['language']
            language_counts[language] = language_counts.get(language, 0) + 1

        return {
            "total_templates": len(all_templates),
            "by_category": category_counts,
            "by_language": language_counts,
            "public_templates": len([t for t in all_templates if t['is_public']]),
            "private_templates": len([t for t in all_templates if not t['is_public']]),
            "total_applications": len(self.applications.applications)
        }

    def info(self) -> Dict[str, Any]:
        """Get project templates system information."""
        return {
            "gitlab_url": self.gitlab_url,
            "statistics": self.get_template_statistics(),
            "supported_categories": [c.value for c in TemplateCategory],
            "supported_languages": [l.value for l in TemplateLanguage],
            "supported_frameworks": [f.value for f in TemplateFramework]
        }


def demo():
    """Demonstrate project templates capabilities."""
    print("=" * 80)
    print("GitLab Project Templates - Comprehensive Demo")
    print("=" * 80)

    # Initialize manager
    mgr = ProjectTemplatesManager()

    print("\n📋 1. Initialize Built-In Templates")
    print("-" * 80)

    # Initialize built-in templates
    result = mgr.initialize_built_in_templates()
    print(f"✓ Created {result['templates_created']} built-in templates:")
    for template in result['templates']:
        print(f"  - {template['name']} ({template['language']}, {template['category']})")

    print("\n🔍 2. Browse Templates")
    print("-" * 80)

    # List all templates
    all_templates = mgr.templates.list_templates()
    print(f"✓ Total templates available: {len(all_templates)}")

    # Filter by category
    web_templates = mgr.templates.list_templates({
        'category': TemplateCategory.WEB_APPLICATION
    })
    print(f"✓ Web application templates: {len(web_templates)}")

    microservice_templates = mgr.templates.list_templates({
        'category': TemplateCategory.MICROSERVICE
    })
    print(f"✓ Microservice templates: {len(microservice_templates)}")

    # Filter by language
    python_templates = mgr.templates.list_templates({
        'language': TemplateLanguage.PYTHON
    })
    print(f"✓ Python templates: {len(python_templates)}")

    print("\n🏗️ 3. Create Custom Template")
    print("-" * 80)

    # Create custom template
    custom_template = mgr.templates.create_template({
        'name': 'Vue.js SPA',
        'description': 'Single Page Application with Vue 3 and Vite',
        'category': TemplateCategory.WEB_APPLICATION.value,
        'language': TemplateLanguage.JAVASCRIPT.value,
        'framework': TemplateFramework.VUE.value,
        'files': [
            {'path': 'package.json', 'type': 'file'},
            {'path': 'vite.config.js', 'type': 'file'},
            {'path': 'src/App.vue', 'type': 'file'},
            {'path': 'src/main.js', 'type': 'file'}
        ],
        'variables': {
            'app_name': 'my-vue-app',
            'vue_version': '3.3'
        },
        'is_public': True
    })
    print(f"✓ Created custom template: {custom_template['name']}")
    print(f"  Template ID: {custom_template['template_id']}")

    print("\n📝 4. Manage Template Files")
    print("-" * 80)

    # Add file templates
    mgr.files.add_file_template('README.md', """# {{project_name}}

{{description}}

## Installation

```bash
npm install
```

## Usage

```bash
npm run dev
```
""")
    print("✓ Added README.md template")

    mgr.files.add_file_template('package.json', """{
  "name": "{{app_name}}",
  "version": "1.0.0",
  "description": "{{description}}"
}
""")
    print("✓ Added package.json template")

    # Render file with variables
    rendered_readme = mgr.files.render_file('README.md', {
        'project_name': 'My Awesome App',
        'description': 'A cool application'
    })
    print(f"✓ Rendered README.md: {len(rendered_readme)} characters")

    print("\n🚀 5. Apply Templates to Create Projects")
    print("-" * 80)

    # Apply Django template
    django_app = mgr.applications.apply_template({
        'template_id': result['templates'][0]['template_id'],  # Django template
        'project_name': 'ecommerce-platform',
        'project_path': 'myorg/ecommerce',
        'variables': {
            'project_name': 'ecommerce',
            'database': 'postgresql',
            'python_version': '3.11'
        }
    })
    print(f"✓ Created Django project: {django_app['project_name']}")
    print(f"  Files created: {django_app['files_created']}")

    # Apply React template
    react_app = mgr.applications.apply_template({
        'template_id': result['templates'][1]['template_id'],  # React template
        'project_name': 'admin-dashboard',
        'project_path': 'myorg/dashboard',
        'variables': {
            'app_name': 'admin-dashboard',
            'node_version': '18'
        }
    })
    print(f"✓ Created React project: {react_app['project_name']}")

    # Apply FastAPI template
    fastapi_service = mgr.applications.apply_template({
        'template_id': result['templates'][2]['template_id'],  # FastAPI template
        'project_name': 'user-service',
        'project_path': 'myorg/services/users',
        'variables': {
            'service_name': 'user-service',
            'python_version': '3.11'
        }
    })
    print(f"✓ Created FastAPI microservice: {fastapi_service['project_name']}")

    print("\n💾 6. Import/Export Templates")
    print("-" * 80)

    # Export template
    template_json = mgr.import_export.export_template(custom_template['template_id'])
    print(f"✓ Exported template: {len(template_json)} bytes")

    # Export all templates
    all_templates_json = mgr.import_export.export_all_templates()
    print(f"✓ Exported all templates: {len(all_templates_json)} bytes")

    # Import template (simulate)
    imported = mgr.import_export.import_template(template_json)
    print(f"✓ Imported template: {imported['name']}")

    print("\n🔄 7. Template Versioning")
    print("-" * 80)

    # Create template versions
    v1 = mgr.versions.create_version(
        custom_template['template_id'],
        '1.0.0',
        'Initial release'
    )
    print(f"✓ Created version {v1['version']}: {v1['changes']}")

    v2 = mgr.versions.create_version(
        custom_template['template_id'],
        '1.1.0',
        'Added TypeScript support'
    )
    print(f"✓ Created version {v2['version']}: {v2['changes']}")

    v3 = mgr.versions.create_version(
        custom_template['template_id'],
        '2.0.0',
        'Major rewrite with Composition API'
    )
    print(f"✓ Created version {v3['version']}: {v3['changes']}")

    # Get all versions
    all_versions = mgr.versions.get_versions(custom_template['template_id'])
    print(f"✓ Total versions: {len(all_versions)}")

    # Get latest version
    latest = mgr.versions.get_latest_version(custom_template['template_id'])
    print(f"✓ Latest version: {latest['version']}")

    print("\n📊 8. Template Statistics")
    print("-" * 80)

    stats = mgr.get_template_statistics()
    print(f"Total templates: {stats['total_templates']}")
    print(f"  - Public: {stats['public_templates']}")
    print(f"  - Private: {stats['private_templates']}")
    print(f"Total applications: {stats['total_applications']}")

    print("\nTemplates by category:")
    for category, count in stats['by_category'].items():
        print(f"  - {category}: {count}")

    print("\nTemplates by language:")
    for language, count in stats['by_language'].items():
        print(f"  - {language}: {count}")

    print("\n📈 9. System Information")
    print("-" * 80)

    info = mgr.info()
    print(f"GitLab URL: {info['gitlab_url']}")
    print(f"Supported categories: {len(info['supported_categories'])}")
    print(f"Supported languages: {len(info['supported_languages'])}")
    print(f"Supported frameworks: {len(info['supported_frameworks'])}")

    print("\n✅ Demo completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    demo()
