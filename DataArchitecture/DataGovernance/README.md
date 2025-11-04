# Data Governance Framework

Comprehensive data governance with metadata, lineage, and compliance tracking.

## Features

- Metadata registry and management
- Data lineage tracking
- Policy creation and enforcement
- Compliance checking
- Access control
- Audit logging

## Usage

```python
from data_governance import DataGovernance

gov = DataGovernance()

# Register dataset
gov.register_dataset("dataset_id", metadata={...})

# Track lineage
gov.track_lineage("output_id", sources=["input_id"], transformation="...")

# Create policy
gov.create_policy("policy_id", policy={...})

# Check compliance
compliance = gov.check_compliance("dataset_id", "policy_id")
```

## Demo

```bash
python data_governance.py
```
