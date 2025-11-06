# Project Templates - Quick Project Creation

Comprehensive GitLab project template system for rapidly creating new projects with predefined structures, configurations, and CI/CD pipelines. Includes 7 built-in templates and support for custom templates.

## Features

### Template Management
- **Create Templates**: Define project structures with files and variables
- **Built-In Templates**: 7 production-ready templates
- **Custom Templates**: Create organization-specific templates
- **Template Filtering**: Filter by category, language, framework
- **Public/Private**: Control template visibility
- **Template Deletion**: Remove unused templates

### Built-In Templates
- **Django Web Application**: Full-stack Django with PostgreSQL
- **React Application**: Modern React with TypeScript and Vite
- **FastAPI Microservice**: Python microservice with Docker
- **Go Microservice**: Go service with gRPC
- **Node.js Express API**: RESTful API with TypeScript
- **Python Data Science**: Jupyter notebooks with MLflow
- **Terraform Infrastructure**: Infrastructure as Code

### Template Files
- **File Structure**: Define directory hierarchy
- **File Templates**: Create reusable file content
- **Variable Substitution**: {{variable}} placeholders
- **Rendering**: Generate files with custom values

### Template Application
- **Apply to Projects**: Create new projects from templates
- **Variable Merging**: Override template defaults
- **Tracking**: Track all template applications
- **Status Monitoring**: Monitor creation status

### Import/Export
- **JSON Export**: Export templates to JSON
- **JSON Import**: Import external templates
- **Bulk Export**: Export all templates at once
- **Template Sharing**: Share templates across teams

### Version Management
- **Semantic Versioning**: Track template versions
- **Change Log**: Document version changes
- **Version History**: View all template versions
- **Latest Version**: Get most recent version

## Usage Example

```python
from project_templates import ProjectTemplatesManager, TemplateCategory, TemplateLanguage

# Initialize manager
mgr = ProjectTemplatesManager()

# 1. Initialize built-in templates
result = mgr.initialize_built_in_templates()
# Creates 7 built-in templates

# 2. Browse templates
all_templates = mgr.templates.list_templates()

# Filter by category
web_templates = mgr.templates.list_templates({
    'category': TemplateCategory.WEB_APPLICATION
})

# Filter by language
python_templates = mgr.templates.list_templates({
    'language': TemplateLanguage.PYTHON
})

# 3. Create custom template
custom = mgr.templates.create_template({
    'name': 'Vue.js SPA',
    'description': 'Single Page Application with Vue 3',
    'category': TemplateCategory.WEB_APPLICATION.value,
    'language': TemplateLanguage.JAVASCRIPT.value,
    'framework': TemplateFramework.VUE.value,
    'files': [
        {'path': 'package.json', 'type': 'file'},
        {'path': 'src/App.vue', 'type': 'file'}
    ],
    'variables': {
        'app_name': 'my-vue-app'
    }
})

# 4. Manage template files
mgr.files.add_file_template('README.md', """# {{project_name}}

{{description}}

## Installation
npm install
""")

# Render with variables
rendered = mgr.files.render_file('README.md', {
    'project_name': 'My App',
    'description': 'Cool application'
})

# 5. Apply template to create project
project = mgr.applications.apply_template({
    'template_id': 'template-1',
    'project_name': 'ecommerce-platform',
    'project_path': 'myorg/ecommerce',
    'variables': {
        'database': 'postgresql',
        'python_version': '3.11'
    }
})

# 6. Export/Import templates
json_export = mgr.import_export.export_template('template-1')
imported = mgr.import_export.import_template(json_export)

# 7. Version management
mgr.versions.create_version('template-1', '1.1.0', 'Added feature X')
latest = mgr.versions.get_latest_version('template-1')
```

## Built-In Templates

### Django Web Application
- **Category**: Web Application
- **Language**: Python 3.11
- **Framework**: Django
- **Database**: PostgreSQL
- **Files**: 10 (manage.py, settings.py, urls.py, Dockerfile, etc.)
- **CI/CD**: Test, build, deploy stages

### React Application
- **Category**: Web Application
- **Language**: TypeScript
- **Framework**: React + Vite
- **Build Tool**: Vite
- **Files**: 9 (App.tsx, vite.config.ts, tsconfig.json, etc.)
- **CI/CD**: Test, build, deploy stages

### FastAPI Microservice
- **Category**: Microservice
- **Language**: Python 3.11
- **Framework**: FastAPI
- **API**: OpenAPI/Swagger
- **Files**: 10 (main.py, models.py, schemas.py, Dockerfile, etc.)
- **CI/CD**: Test, build, deploy stages

### Go Microservice
- **Category**: Microservice
- **Language**: Go 1.21
- **Protocol**: gRPC
- **Files**: 9 (main.go, go.mod, proto files, Dockerfile, etc.)
- **CI/CD**: Test, build, deploy stages

### Node.js Express API
- **Category**: API
- **Language**: TypeScript
- **Framework**: Express
- **Files**: 10 (index.ts, routes, controllers, models, Dockerfile, etc.)
- **CI/CD**: Test, build, deploy stages

### Python Data Science
- **Category**: Data Science
- **Language**: Python 3.11
- **Tools**: Jupyter, MLflow
- **Files**: 9 (notebooks, data processing, model training, etc.)
- **CI/CD**: Test, train, deploy stages

### Terraform Infrastructure
- **Category**: DevOps
- **Language**: HCL
- **Tool**: Terraform 1.5
- **Files**: 8 (main.tf, variables.tf, modules, etc.)
- **CI/CD**: Validate, plan, apply stages

## Template Categories

### Available Categories
- **web_application**: Web applications (React, Django, Vue)
- **microservice**: Microservices (FastAPI, Go)
- **mobile_app**: Mobile applications
- **data_science**: ML/Data projects
- **devops**: Infrastructure and automation
- **documentation**: Documentation sites
- **library**: Reusable libraries
- **api**: REST/GraphQL APIs
- **monorepo**: Monorepo projects
- **static_site**: Static websites
- **custom**: Custom categories

## Supported Languages

- Python
- JavaScript
- TypeScript
- Java
- Go
- Rust
- Ruby
- PHP
- C#
- C++
- Swift
- Kotlin

## Supported Frameworks

- Django
- Flask
- FastAPI
- React
- Vue
- Angular
- Next.js
- Express
- Spring Boot
- Laravel
- Rails

## Variable Substitution

Templates support {{variable}} placeholders:

```python
# Template file content
"""
# {{project_name}}

Version: {{version}}
Author: {{author}}
"""

# Render with variables
rendered = mgr.files.render_file('README.md', {
    'project_name': 'My Project',
    'version': '1.0.0',
    'author': 'Team Name'
})

# Result:
# My Project
#
# Version: 1.0.0
# Author: Team Name
```

## Template Structure

```python
template = {
    'name': 'Template Name',
    'description': 'Template description',
    'category': 'web_application',
    'language': 'python',
    'framework': 'django',
    'files': [
        {'path': 'manage.py', 'type': 'file'},
        {'path': 'app/__init__.py', 'type': 'file'},
        {'path': 'requirements.txt', 'type': 'file'}
    ],
    'variables': {
        'project_name': 'myproject',
        'python_version': '3.11'
    },
    'ci_config': {
        'stages': ['test', 'build', 'deploy'],
        'docker': True
    }
}
```

## Best Practices

### Template Creation
1. **Descriptive Names**: Use clear, descriptive template names
2. **Complete Documentation**: Include comprehensive descriptions
3. **Sensible Defaults**: Provide working default values
4. **File Structure**: Organize files logically
5. **CI/CD Integration**: Include .gitlab-ci.yml

### Variable Usage
1. **Consistent Naming**: Use snake_case for variables
2. **Required vs Optional**: Mark required variables clearly
3. **Default Values**: Provide defaults when possible
4. **Documentation**: Document all available variables
5. **Validation**: Validate variable values

### Template Maintenance
1. **Version Control**: Use semantic versioning
2. **Change Logs**: Document all changes
3. **Testing**: Test templates before sharing
4. **Updates**: Keep dependencies current
5. **Deprecation**: Mark old templates as deprecated

### Template Sharing
1. **Public Templates**: Share organization-wide
2. **Private Templates**: Keep team-specific templates private
3. **Export Standards**: Use consistent export format
4. **Documentation**: Include usage examples
5. **Support**: Provide support channels

## Common Use Cases

### Quick Project Creation
```python
# Create new microservice
service = mgr.applications.apply_template({
    'template_id': 'fastapi-template',
    'project_name': 'user-service',
    'project_path': 'myorg/services/users'
})
```

### Team Standards
```python
# Create company standard React template
company_react = mgr.templates.create_template({
    'name': 'Company React Standard',
    'description': 'React app following company standards',
    'category': TemplateCategory.WEB_APPLICATION.value,
    'language': TemplateLanguage.TYPESCRIPT.value,
    'files': [
        # Company-specific structure
    ]
})
```

### Multi-Project Deployment
```python
# Create multiple microservices
services = ['users', 'orders', 'payments']
for service in services:
    mgr.applications.apply_template({
        'template_id': 'go-microservice-template',
        'project_name': f'{service}-service',
        'project_path': f'myorg/services/{service}'
    })
```

### Template Versioning
```python
# Track template evolution
mgr.versions.create_version('react-template', '1.0.0', 'Initial release')
mgr.versions.create_version('react-template', '1.1.0', 'Added TypeScript')
mgr.versions.create_version('react-template', '2.0.0', 'Migrated to Vite')

# Get latest
latest = mgr.versions.get_latest_version('react-template')
```

## Template Statistics

```python
stats = mgr.get_template_statistics()

# Returns:
{
    'total_templates': 10,
    'by_category': {
        'web_application': 4,
        'microservice': 3,
        'data_science': 1
    },
    'by_language': {
        'python': 3,
        'typescript': 3,
        'go': 2
    },
    'public_templates': 8,
    'private_templates': 2,
    'total_applications': 25
}
```

## Requirements

```
json (standard library)
datetime (standard library)
```

No external dependencies required.

## Configuration

### Environment Variables
```bash
export GITLAB_URL="https://gitlab.com"
```

### Python Configuration
```python
from project_templates import ProjectTemplatesManager

mgr = ProjectTemplatesManager(
    gitlab_url='https://gitlab.com'
)
```

## Integration with GitLab

### Create Project from Template
1. Select template from catalog
2. Provide project details
3. Override variable defaults
4. Apply template
5. GitLab creates project with all files

### CI/CD Integration
Templates include `.gitlab-ci.yml` with:
- Test stages
- Build stages
- Deploy stages
- Environment-specific configurations

## Troubleshooting

**Issue**: Template not appearing in list
- Check template is public
- Verify filters are not excluding it
- Check template category and language

**Issue**: Variable substitution not working
- Verify variable names match exactly
- Check {{variable}} syntax
- Ensure variables provided during application

**Issue**: Template application failing
- Verify template ID is correct
- Check all required variables provided
- Review file paths for conflicts

**Issue**: Import failing
- Verify JSON is valid
- Check template structure matches schema
- Review error messages for specific issues

## Author

BrillConsulting - Enterprise Cloud Solutions
