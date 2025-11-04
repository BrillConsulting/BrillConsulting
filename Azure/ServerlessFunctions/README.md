# Azure Functions

Serverless computing with Azure Functions.

## Features

- HTTP triggered functions
- Timer triggered functions (cron)
- Blob storage triggered functions
- Queue storage triggered functions
- Durable Function orchestration
- Function App management

## Usage

```python
from azure_functions import HttpTrigger, TimerTrigger, AzureFunctionApp

# Create function app
app = AzureFunctionApp("func-demo", "rg-demo")

# Register HTTP function
http_func = HttpTrigger("ProcessOrder", methods=["POST"])
app.register_function("httpTrigger", http_func)

# Invoke function
result = app.invoke("ProcessOrder", {"method": "POST", "body": {"order_id": "123"}})
```

## Demo

```bash
python azure_functions.py
```
