# Azure Logic Apps Integration

Advanced implementation of Azure Logic Apps with workflow automation, integration connectors, and enterprise orchestration capabilities.

**Author:** BrillConsulting
**Contact:** clientbrill@gmail.com
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Overview

This project provides a comprehensive Python implementation for Azure Logic Apps, featuring workflow automation, integration with 400+ connectors, conditional logic, loops, error handling, and stateful orchestration. Built for enterprise applications requiring low-code/no-code workflow automation, system integration, and business process orchestration.

## Features

### Core Capabilities
- **Workflow Designer**: Visual workflow creation and management
- **Built-in Connectors**: Integration with Azure services and third-party apps
- **HTTP Triggers**: Webhook and request-response patterns
- **Scheduled Triggers**: Time-based workflow execution
- **Event-based Triggers**: React to Azure events and messages
- **Conditional Logic**: If/else statements and switch cases
- **Loops**: For-each and until loops for iteration
- **Error Handling**: Try-catch scopes and retry policies

### Advanced Features
- **Stateful Workflows**: Long-running processes with state persistence
- **Parallel Processing**: Execute multiple actions concurrently
- **Data Transformation**: JSON parsing, XML conversion, and data mapping
- **Batch Processing**: Collect and process messages in batches
- **Integration Account**: B2B messaging with EDI and AS2
- **Custom Connectors**: Create connectors for proprietary APIs
- **Managed Identity**: Secure authentication without credentials
- **API Management Integration**: Enterprise API gateway capabilities

## Architecture

```
LogicApps/
├── logic_apps.py              # Main implementation
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Key Components

1. **LogicAppsManager**: Main service interface
   - Workflow management
   - Trigger configuration
   - Run history monitoring

2. **Workflow Designer**:
   - Visual workflow creation
   - Action configuration
   - Connector selection

3. **Trigger Types**:
   - HTTP triggers (request/webhook)
   - Recurrence (schedule)
   - Event Grid
   - Service Bus
   - Blob storage

4. **Actions**:
   - HTTP actions
   - Data operations
   - Control actions (condition, loop, scope)
   - Connector actions

5. **Integration**:
   - API connections
   - Managed connectors
   - Custom connectors
   - On-premises data gateway

## Installation

```bash
# Clone the repository
git clone https://github.com/BrillConsulting/BrillConsulting.git
cd BrillConsulting/Azure/LogicApps

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Set up your Azure Logic Apps credentials:

```python
from logic_apps import LogicAppsManager

manager = LogicAppsManager(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    location="eastus"
)
```

### Environment Variables (Recommended)

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_LOGIC_APPS_LOCATION="eastus"
export AZURE_LOGIC_APP_NAME="your-logic-app"
```

## Usage Examples

### 1. Create HTTP-Triggered Workflow

```python
from logic_apps import LogicAppsManager

manager = LogicAppsManager(
    subscription_id="your-subscription-id",
    resource_group="your-resource-group",
    location="eastus"
)

# Define workflow with HTTP trigger
workflow_definition = {
    "triggers": {
        "manual": {
            "type": "Request",
            "kind": "Http",
            "inputs": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }
            }
        }
    },
    "actions": {
        "Send_Email": {
            "type": "ApiConnection",
            "inputs": {
                "host": {
                    "connection": {
                        "name": "@parameters('$connections')['office365']['connectionId']"
                    }
                },
                "method": "post",
                "path": "/v2/Mail",
                "body": {
                    "To": "@triggerBody()?['email']",
                    "Subject": "Welcome!",
                    "Body": "Hello @{triggerBody()?['name']}, welcome to our service!"
                }
            }
        },
        "Response": {
            "type": "Response",
            "kind": "Http",
            "inputs": {
                "statusCode": 200,
                "body": {
                    "message": "Email sent successfully"
                }
            },
            "runAfter": {
                "Send_Email": ["Succeeded"]
            }
        }
    }
}

# Create Logic App
logic_app = manager.create_logic_app(
    logic_app_name="WelcomeEmailWorkflow",
    workflow_definition=workflow_definition
)

print(f"Logic App created: {logic_app['name']}")
print(f"Callback URL: {logic_app['callback_url']}")
```

### 2. Scheduled Workflow with Recurrence

```python
# Define scheduled workflow
workflow_definition = {
    "triggers": {
        "Recurrence": {
            "type": "Recurrence",
            "recurrence": {
                "frequency": "Day",
                "interval": 1,
                "schedule": {
                    "hours": ["9"],
                    "minutes": [0]
                },
                "timeZone": "Eastern Standard Time"
            }
        }
    },
    "actions": {
        "Get_Database_Records": {
            "type": "ApiConnection",
            "inputs": {
                "host": {
                    "connection": {
                        "name": "@parameters('$connections')['sql']['connectionId']"
                    }
                },
                "method": "get",
                "path": "/v2/datasets/@{encodeURIComponent('your-database')}/tables/@{encodeURIComponent('Orders')}/items"
            }
        },
        "For_Each_Order": {
            "type": "Foreach",
            "foreach": "@body('Get_Database_Records')?['value']",
            "actions": {
                "Process_Order": {
                    "type": "Http",
                    "inputs": {
                        "method": "POST",
                        "uri": "https://api.example.com/process",
                        "body": "@item()"
                    }
                }
            },
            "runAfter": {
                "Get_Database_Records": ["Succeeded"]
            }
        }
    }
}

# Create scheduled Logic App
logic_app = manager.create_logic_app(
    logic_app_name="DailyOrderProcessing",
    workflow_definition=workflow_definition
)

print(f"Scheduled workflow created: {logic_app['name']}")
```

### 3. Event-Driven Workflow with Service Bus

```python
# Service Bus triggered workflow
workflow_definition = {
    "triggers": {
        "When_a_message_is_received": {
            "type": "ServiceBus",
            "inputs": {
                "host": {
                    "connection": {
                        "name": "@parameters('$connections')['servicebus']['connectionId']"
                    }
                },
                "path": "orders/messages/head",
                "method": "GET"
            },
            "recurrence": {
                "frequency": "Second",
                "interval": 30
            }
        }
    },
    "actions": {
        "Parse_JSON": {
            "type": "ParseJson",
            "inputs": {
                "content": "@triggerBody()?['ContentData']",
                "schema": {
                    "type": "object",
                    "properties": {
                        "orderId": {"type": "string"},
                        "amount": {"type": "number"},
                        "customerId": {"type": "string"}
                    }
                }
            }
        },
        "Condition_Check_Amount": {
            "type": "If",
            "expression": {
                "and": [
                    {
                        "greater": [
                            "@body('Parse_JSON')?['amount']",
                            1000
                        ]
                    }
                ]
            },
            "actions": {
                "Send_Approval_Request": {
                    "type": "ApiConnection",
                    "inputs": {
                        "host": {
                            "connection": {
                                "name": "@parameters('$connections')['teams']['connectionId']"
                            }
                        },
                        "method": "post",
                        "path": "/v2/approvals",
                        "body": {
                            "title": "Approve Order @{body('Parse_JSON')?['orderId']}",
                            "assignedTo": "manager@company.com"
                        }
                    }
                }
            },
            "runAfter": {
                "Parse_JSON": ["Succeeded"]
            }
        }
    }
}

logic_app = manager.create_logic_app(
    logic_app_name="OrderApprovalWorkflow",
    workflow_definition=workflow_definition
)

print(f"Event-driven workflow created: {logic_app['name']}")
```

### 4. Error Handling with Try-Catch

```python
# Workflow with comprehensive error handling
workflow_definition = {
    "triggers": {
        "manual": {
            "type": "Request",
            "kind": "Http"
        }
    },
    "actions": {
        "Try_Scope": {
            "type": "Scope",
            "actions": {
                "Call_External_API": {
                    "type": "Http",
                    "inputs": {
                        "method": "POST",
                        "uri": "https://api.example.com/process",
                        "body": "@triggerBody()",
                        "retryPolicy": {
                            "type": "exponential",
                            "count": 3,
                            "interval": "PT10S"
                        }
                    }
                },
                "Update_Database": {
                    "type": "ApiConnection",
                    "inputs": {
                        "host": {
                            "connection": {
                                "name": "@parameters('$connections')['sql']['connectionId']"
                            }
                        },
                        "method": "patch",
                        "path": "/v2/datasets/@{encodeURIComponent('database')}/tables/@{encodeURIComponent('Transactions')}/items/@{encodeURIComponent(triggerBody()?['id'])}",
                        "body": {
                            "status": "completed",
                            "processedAt": "@utcNow()"
                        }
                    },
                    "runAfter": {
                        "Call_External_API": ["Succeeded"]
                    }
                }
            }
        },
        "Catch_Scope": {
            "type": "Scope",
            "actions": {
                "Log_Error": {
                    "type": "ApiConnection",
                    "inputs": {
                        "host": {
                            "connection": {
                                "name": "@parameters('$connections')['azureloganalytics']['connectionId']"
                            }
                        },
                        "method": "post",
                        "path": "/api/logs",
                        "body": {
                            "error": "@result('Try_Scope')?['outputs']",
                            "timestamp": "@utcNow()"
                        }
                    }
                },
                "Send_Alert": {
                    "type": "ApiConnection",
                    "inputs": {
                        "host": {
                            "connection": {
                                "name": "@parameters('$connections')['teams']['connectionId']"
                            }
                        },
                        "method": "post",
                        "path": "/v2/teams/channels/messages",
                        "body": {
                            "message": "Workflow failed: @{result('Try_Scope')?['error']}"
                        }
                    },
                    "runAfter": {
                        "Log_Error": ["Succeeded"]
                    }
                }
            },
            "runAfter": {
                "Try_Scope": ["Failed", "Skipped", "TimedOut"]
            }
        }
    }
}

logic_app = manager.create_logic_app(
    logic_app_name="ResilientWorkflow",
    workflow_definition=workflow_definition
)

print(f"Workflow with error handling created: {logic_app['name']}")
```

### 5. Parallel Processing

```python
# Workflow with parallel branches
workflow_definition = {
    "triggers": {
        "manual": {
            "type": "Request",
            "kind": "Http"
        }
    },
    "actions": {
        "Parse_Request": {
            "type": "ParseJson",
            "inputs": {
                "content": "@triggerBody()",
                "schema": {
                    "type": "object",
                    "properties": {
                        "userId": {"type": "string"},
                        "data": {"type": "object"}
                    }
                }
            }
        },
        "Parallel_Branch": {
            "type": "Parallel",
            "branches": {
                "Branch_1_Update_CRM": {
                    "actions": {
                        "Update_CRM": {
                            "type": "Http",
                            "inputs": {
                                "method": "POST",
                                "uri": "https://crm.example.com/api/users",
                                "body": "@body('Parse_Request')"
                            }
                        }
                    }
                },
                "Branch_2_Send_Email": {
                    "actions": {
                        "Send_Notification": {
                            "type": "ApiConnection",
                            "inputs": {
                                "host": {
                                    "connection": {
                                        "name": "@parameters('$connections')['sendgrid']['connectionId']"
                                    }
                                },
                                "method": "post",
                                "path": "/v3/mail/send",
                                "body": {
                                    "to": "@body('Parse_Request')?['email']",
                                    "subject": "Update Notification"
                                }
                            }
                        }
                    }
                },
                "Branch_3_Log_Analytics": {
                    "actions": {
                        "Track_Event": {
                            "type": "ApiConnection",
                            "inputs": {
                                "host": {
                                    "connection": {
                                        "name": "@parameters('$connections')['applicationinsights']['connectionId']"
                                    }
                                },
                                "method": "post",
                                "path": "/v2/track",
                                "body": {
                                    "name": "UserUpdated",
                                    "properties": "@body('Parse_Request')"
                                }
                            }
                        }
                    }
                }
            },
            "runAfter": {
                "Parse_Request": ["Succeeded"]
            }
        }
    }
}

logic_app = manager.create_logic_app(
    logic_app_name="ParallelProcessingWorkflow",
    workflow_definition=workflow_definition
)

print(f"Parallel workflow created: {logic_app['name']}")
```

### 6. Batch Processing Workflow

```python
# Collect and batch process messages
workflow_definition = {
    "triggers": {
        "Batch_Trigger": {
            "type": "Batch",
            "inputs": {
                "mode": "Inline",
                "configurations": {
                    "OrderBatch": {
                        "releaseCriteria": {
                            "messageCount": 100,
                            "recurrence": {
                                "frequency": "Minute",
                                "interval": 15
                            }
                        }
                    }
                }
            }
        }
    },
    "actions": {
        "Process_Batch": {
            "type": "Foreach",
            "foreach": "@triggerBody()?['items']",
            "actions": {
                "Process_Order": {
                    "type": "Http",
                    "inputs": {
                        "method": "POST",
                        "uri": "https://api.example.com/orders/bulk",
                        "body": "@item()?['content']"
                    }
                }
            }
        },
        "Send_Batch_Report": {
            "type": "ApiConnection",
            "inputs": {
                "host": {
                    "connection": {
                        "name": "@parameters('$connections')['office365']['connectionId']"
                    }
                },
                "method": "post",
                "path": "/v2/Mail",
                "body": {
                    "To": "admin@company.com",
                    "Subject": "Batch Processing Complete",
                    "Body": "Processed @{length(triggerBody()?['items'])} orders"
                }
            },
            "runAfter": {
                "Process_Batch": ["Succeeded"]
            }
        }
    }
}

logic_app = manager.create_logic_app(
    logic_app_name="BatchOrderProcessing",
    workflow_definition=workflow_definition
)

print(f"Batch processing workflow created: {logic_app['name']}")
```

### 7. Workflow Management and Monitoring

```python
# Trigger a workflow run
run_result = manager.trigger_workflow_run(
    logic_app_name="WelcomeEmailWorkflow",
    trigger_name="manual",
    trigger_body={
        "name": "John Doe",
        "email": "john@example.com"
    }
)

print(f"Run ID: {run_result['run_id']}")
print(f"Status: {run_result['status']}")

# Get run history
runs = manager.get_workflow_runs(
    logic_app_name="WelcomeEmailWorkflow",
    top=10
)

for run in runs:
    print(f"Run: {run['name']}, Status: {run['status']}, Start Time: {run['startTime']}")

# Get specific run details
run_details = manager.get_run_details(
    logic_app_name="WelcomeEmailWorkflow",
    run_name=run_result['run_id']
)

print(f"Duration: {run_details['duration']}")
print(f"Actions: {len(run_details['actions'])}")

# Get action outputs
for action_name, action_details in run_details['actions'].items():
    print(f"Action: {action_name}")
    print(f"Status: {action_details['status']}")
    print(f"Output: {action_details.get('outputs')}")

# Cancel a running workflow
manager.cancel_workflow_run(
    logic_app_name="LongRunningWorkflow",
    run_name="run-id-123"
)

# Disable/Enable workflow
manager.disable_logic_app("WelcomeEmailWorkflow")
manager.enable_logic_app("WelcomeEmailWorkflow")

# Delete workflow
manager.delete_logic_app("OldWorkflow")
```

## Running Demos

```bash
# Run the implementation
python logic_apps.py
```

Demo output includes:
- Workflow creation and configuration
- Trigger setup
- Action orchestration
- Run history monitoring

## Common Connectors

### Microsoft Services
- **Office 365**: Email, Calendar, Contacts
- **SharePoint**: Document management
- **Teams**: Messaging and collaboration
- **Dynamics 365**: CRM operations
- **Power BI**: Data visualization

### Data & Storage
- **SQL Server**: Database operations
- **Cosmos DB**: NoSQL operations
- **Blob Storage**: File operations
- **Azure Table Storage**: Key-value storage

### Messaging & Events
- **Service Bus**: Message queuing
- **Event Grid**: Event routing
- **Event Hubs**: Streaming data

### External Services
- **Salesforce**: CRM integration
- **SAP**: ERP integration
- **Twitter**: Social media
- **Twilio**: SMS messaging
- **SendGrid**: Email delivery

## API Reference

### LogicAppsManager

#### Workflow Management Methods

**`create_logic_app(logic_app_name, workflow_definition, location, tags)`**
- Creates a new Logic App workflow
- **Parameters**: logic_app_name (str), workflow_definition (Dict), location (str), tags (Dict)
- **Returns**: `Dict[str, Any]`

**`update_logic_app(logic_app_name, workflow_definition)`**
- Updates an existing Logic App
- **Returns**: `Dict[str, Any]`

**`delete_logic_app(logic_app_name)`**
- Deletes a Logic App
- **Returns**: `None`

**`get_logic_app(logic_app_name)`**
- Gets Logic App details
- **Returns**: `Dict[str, Any]`

**`list_logic_apps()`**
- Lists all Logic Apps in resource group
- **Returns**: `List[Dict[str, Any]]`

**`enable_logic_app(logic_app_name)`**
- Enables a disabled Logic App
- **Returns**: `Dict[str, Any]`

**`disable_logic_app(logic_app_name)`**
- Disables a Logic App
- **Returns**: `Dict[str, Any]`

#### Workflow Run Methods

**`trigger_workflow_run(logic_app_name, trigger_name, trigger_body)`**
- Manually triggers a workflow run
- **Returns**: `Dict[str, Any]`

**`get_workflow_runs(logic_app_name, top, filter)`**
- Gets workflow run history
- **Returns**: `List[Dict[str, Any]]`

**`get_run_details(logic_app_name, run_name)`**
- Gets detailed information about a specific run
- **Returns**: `Dict[str, Any]`

**`cancel_workflow_run(logic_app_name, run_name)`**
- Cancels a running workflow
- **Returns**: `None`

**`resubmit_workflow_run(logic_app_name, run_name)`**
- Resubmits a failed workflow run
- **Returns**: `Dict[str, Any]`

#### Trigger Methods

**`get_trigger_callback_url(logic_app_name, trigger_name)`**
- Gets the callback URL for HTTP triggers
- **Returns**: `str`

**`run_trigger(logic_app_name, trigger_name)`**
- Manually runs a trigger
- **Returns**: `Dict[str, Any]`

#### Connection Methods

**`create_api_connection(connection_name, api_id, parameters)`**
- Creates an API connection
- **Returns**: `Dict[str, Any]`

**`list_api_connections()`**
- Lists all API connections
- **Returns**: `List[Dict[str, Any]]`

**`delete_api_connection(connection_name)`**
- Deletes an API connection
- **Returns**: `None`

## Best Practices

### 1. Use Managed Identity for Authentication
```python
# Configure workflow with managed identity
workflow_definition = {
    "triggers": {...},
    "actions": {
        "Call_API": {
            "type": "Http",
            "inputs": {
                "method": "GET",
                "uri": "https://api.example.com/data",
                "authentication": {
                    "type": "ManagedServiceIdentity"
                }
            }
        }
    }
}
```

### 2. Implement Idempotency
```python
# Use unique identifiers for idempotent operations
workflow_definition = {
    "triggers": {...},
    "actions": {
        "Check_Duplicate": {
            "type": "ApiConnection",
            "inputs": {
                "host": {...},
                "path": "/check/@{triggerBody()?['requestId']}"
            }
        },
        "Condition_Process_If_New": {
            "type": "If",
            "expression": {
                "equals": ["@body('Check_Duplicate')?['exists']", false]
            },
            "actions": {...}
        }
    }
}
```

### 3. Configure Retry Policies
```python
# Add retry policies for resilience
"Call_External_API": {
    "type": "Http",
    "inputs": {
        "method": "POST",
        "uri": "https://api.example.com/endpoint",
        "retryPolicy": {
            "type": "exponential",
            "count": 4,
            "interval": "PT7S",
            "minimumInterval": "PT5S",
            "maximumInterval": "PT1H"
        }
    }
}
```

### 4. Use Scopes for Error Handling
```python
# Organize actions in scopes
"Try": {
    "type": "Scope",
    "actions": {...}
},
"Catch": {
    "type": "Scope",
    "actions": {...},
    "runAfter": {
        "Try": ["Failed", "Skipped", "TimedOut"]
    }
}
```

### 5. Optimize Connector Usage
```python
# Minimize API calls with batching
"Get_Multiple_Records": {
    "type": "ApiConnection",
    "inputs": {
        "host": {...},
        "method": "get",
        "path": "/items",
        "queries": {
            "$top": 100,
            "$select": "id,name,status"
        }
    }
}
```

### 6. Monitor and Alert
```python
# Add monitoring actions
"Send_Alert_On_Failure": {
    "type": "ApiConnection",
    "inputs": {
        "host": {...},
        "method": "post",
        "body": {
            "severity": "high",
            "message": "Workflow @{workflow().name} failed"
        }
    },
    "runAfter": {
        "Main_Process": ["Failed"]
    }
}
```

### 7. Use Parameters and Variables
```python
# Parameterize workflows for reusability
workflow_definition = {
    "parameters": {
        "environment": {
            "type": "string",
            "defaultValue": "production"
        },
        "apiEndpoint": {
            "type": "string"
        }
    },
    "triggers": {...},
    "actions": {
        "Call_API": {
            "inputs": {
                "uri": "@parameters('apiEndpoint')"
            }
        }
    }
}
```

## Use Cases

### 1. Automated Approval Workflows
Implement multi-stage approval processes with Teams, email, or custom systems.

### 2. Data Integration
Synchronize data between CRM, ERP, and other business systems automatically.

### 3. Business Process Automation
Automate repetitive business processes like invoice processing, onboarding, and reporting.

### 4. Event-Driven Architectures
React to events from Azure services, third-party systems, or custom applications.

### 5. ETL Pipelines
Extract, transform, and load data between systems without writing code.

### 6. Monitoring and Alerting
Monitor systems and send alerts through multiple channels when issues occur.

## Troubleshooting

### Common Issues

**Issue**: Workflow not triggering
**Solution**: Verify trigger configuration, check connection status, review trigger conditions

**Issue**: Action fails with authentication error
**Solution**: Verify API connection credentials, check managed identity permissions, renew expired connections

**Issue**: Workflow timeout
**Solution**: Increase timeout settings, optimize long-running actions, use asynchronous patterns

**Issue**: Throttling errors
**Solution**: Implement retry policies, use batching, review connector limits

**Issue**: Large payload errors
**Solution**: Use chunking for large data, leverage blob storage, implement pagination

**Issue**: Cannot access on-premises resources
**Solution**: Configure on-premises data gateway, verify network connectivity, check firewall rules

## Deployment

### Azure CLI Deployment
```bash
# Create Logic App
az logic workflow create \
    --resource-group my-resource-group \
    --location eastus \
    --name my-logic-app \
    --definition @workflow-definition.json

# Update Logic App
az logic workflow update \
    --resource-group my-resource-group \
    --name my-logic-app \
    --definition @updated-workflow.json

# Get callback URL
az logic workflow trigger callback \
    --resource-group my-resource-group \
    --name my-logic-app \
    --trigger-name manual

# Enable/Disable
az logic workflow update \
    --resource-group my-resource-group \
    --name my-logic-app \
    --state Disabled
```

### Infrastructure as Code
```python
# Terraform example
resource "azurerm_logic_app_workflow" "example" {
  name                = "my-logic-app"
  location            = azurerm_resource_group.example.location
  resource_group_name = azurerm_resource_group.example.name

  workflow_parameters = {
    "$connections" = jsonencode({
      defaultValue = {}
      type         = "Object"
    })
  }
}

resource "azurerm_logic_app_trigger_http_request" "example" {
  name         = "manual"
  logic_app_id = azurerm_logic_app_workflow.example.id

  schema = <<SCHEMA
{
  "type": "object",
  "properties": {
    "name": {"type": "string"}
  }
}
SCHEMA
}
```

### Container Deployment (Standard Tier)
```dockerfile
FROM mcr.microsoft.com/azure-functions/dotnet:3.0
WORKDIR /home/site/wwwroot
COPY . .
ENV AzureWebJobsStorage="your-storage-connection"
CMD ["dotnet", "LogicApp.dll"]
```

## Monitoring

### Key Metrics
- Workflow runs (success/failure)
- Run duration
- Action success rate
- Trigger frequency
- Throttled requests
- Billable action executions

### Azure Monitor Integration
```bash
# Enable diagnostic logs
az monitor diagnostic-settings create \
    --resource /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Logic/workflows/{workflow} \
    --name LogicAppDiagnostics \
    --logs '[{"category":"WorkflowRuntime","enabled":true}]' \
    --metrics '[{"category":"AllMetrics","enabled":true}]' \
    --workspace /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.OperationalInsights/workspaces/{workspace}

# Create alert
az monitor metrics alert create \
    --name HighFailureRate \
    --resource-group my-resource-group \
    --scopes /subscriptions/{sub}/resourceGroups/{rg}/providers/Microsoft.Logic/workflows/{workflow} \
    --condition "avg RunsFailed > 5" \
    --window-size 5m \
    --evaluation-frequency 1m
```

### Application Insights Integration
```python
# Add Application Insights tracking
"Track_Custom_Event": {
    "type": "ApiConnection",
    "inputs": {
        "host": {
            "connection": {
                "name": "@parameters('$connections')['applicationinsights']['connectionId']"
            }
        },
        "method": "post",
        "path": "/v2/track",
        "body": {
            "name": "OrderProcessed",
            "properties": {
                "orderId": "@body('Parse_Order')?['id']",
                "amount": "@body('Parse_Order')?['amount']"
            }
        }
    }
}
```

## Dependencies

```
Python >= 3.8
azure-core >= 1.26.0
azure-mgmt-logic >= 10.0.0
azure-identity >= 1.12.0
typing
datetime
json
```

See `requirements.txt` for complete list.

## Version History

- **v1.0.0**: Initial release with basic workflow management
- **v1.1.0**: Added connector support and error handling
- **v1.2.0**: Enhanced monitoring and run history features
- **v2.0.0**: Added Standard tier support and advanced patterns

## Contributing

Contributions are welcome! Please submit pull requests or open issues on GitHub.

## License

This project is part of the Brill Consulting portfolio.

## Support

For questions or support:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## Related Projects

- [Azure Functions](../Functions/)
- [Azure Automation](../Automation/)
- [Power Automate](../PowerAutomate/)

---

**Built with Azure Logic Apps** | **Brill Consulting © 2024**
