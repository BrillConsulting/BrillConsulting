# Model Monitoring

Monitor ML models in production for drift, performance, and data quality.

## Features

- Data drift detection (KS test)
- Prediction distribution monitoring
- Performance degradation alerts
- Metrics history tracking
- Automated reporting

## Usage

```python
from model_monitor import ModelMonitor

# Initialize with baseline
monitor = ModelMonitor(baseline_data)

# Detect drift
drift = monitor.detect_data_drift(new_data)

# Monitor predictions
metrics = monitor.monitor_predictions(predictions)

# Check performance
degradation = monitor.check_performance_degradation(0.85, 0.90)

# Generate report
report = monitor.generate_report()
```

## Demo

```bash
python model_monitor.py
```
