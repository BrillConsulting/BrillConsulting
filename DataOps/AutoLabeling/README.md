# Automated Data Labeling

Intelligent data labeling pipelines using Active Learning, semi-supervised learning, and integration with CVAT/Label Studio for efficient annotation.

## Features

- **Active Learning** - Uncertainty sampling, query-by-committee, diversity sampling
- **Semi-Supervised** - Self-training, co-training, pseudo-labeling
- **Label Studio Integration** - Automated annotation workflows
- **CVAT Integration** - Computer vision annotation automation
- **Human-in-the-Loop** - Smart sample selection for manual review
- **Model-Assisted Labeling** - Pre-annotations with ML models
- **Quality Control** - Inter-annotator agreement, consensus labeling
- **Cost Optimization** - Reduce labeling costs by 60-80%

## Architecture

```
[Unlabeled Data] → [Active Learning] → [Sample Selection] → [Labeling Tool]
                          ↓                                         ↓
                    [ML Model]                              [Human Annotators]
                          ↑                                         ↓
                          └──────── [Labeled Data] ←───────────────┘
```

## Usage

### Active Learning Strategy

```python
from auto_labeling import ActiveLearner, UncertaintySampling

learner = ActiveLearner(
    model=base_model,
    strategy="uncertainty",  # or "query_by_committee", "diversity"
    batch_size=100
)

# Select most informative samples
unlabeled_pool = load_unlabeled_data()

informative_samples = learner.query(
    unlabeled_pool=unlabeled_pool,
    n_instances=100
)

# Send to labeling
learner.send_to_labeling(
    samples=informative_samples,
    tool="label_studio"
)
```

### Label Studio Integration

```python
from auto_labeling import LabelStudioClient

client = LabelStudioClient(
    url="http://localhost:8080",
    api_key="your-api-key"
)

# Create project
project = client.create_project(
    name="Image Classification",
    label_config="""
    <View>
      <Image name="image" value="$image"/>
      <Choices name="label" toName="image">
        <Choice value="cat"/>
        <Choice value="dog"/>
      </Choices>
    </View>
    """
)

# Upload tasks with pre-annotations
client.upload_tasks(
    project_id=project.id,
    tasks=tasks_with_predictions
)

# Export labeled data
labeled_data = client.export_annotations(
    project_id=project.id,
    format="JSON"
)
```

### CVAT Integration

```python
from auto_labeling import CVATClient

cvat = CVATClient(
    host="localhost:8080",
    credentials=("user", "password")
)

# Create annotation task
task = cvat.create_task(
    name="Object Detection",
    labels=["car", "person", "bicycle"],
    subset="train"
)

# Upload images with pre-annotations
cvat.upload_images(
    task_id=task.id,
    images=image_paths,
    annotations=model_predictions  # Pre-fill with model
)

# Export annotations
annotations = cvat.export_annotations(
    task_id=task.id,
    format="YOLO"
)
```

## Active Learning Strategies

### Uncertainty Sampling

Select samples where model is least confident:

```python
strategy = UncertaintySampling(method="least_confident")

# or "margin", "entropy"
selected = strategy.query(
    model=model,
    unlabeled_pool=pool,
    n_instances=100
)
```

### Query by Committee

Multiple models vote on disagreements:

```python
from auto_labeling import QueryByCommittee

committee = QueryByCommittee(
    models=[model1, model2, model3],
    disagreement_measure="vote_entropy"
)

selected = committee.query(
    unlabeled_pool=pool,
    n_instances=100
)
```

### Diversity Sampling

Select diverse samples to cover data distribution:

```python
from auto_labeling import DiversitySampling

diversity = DiversitySampling(
    method="kmeans",  # or "coreset"
    n_clusters=100
)

selected = diversity.query(
    unlabeled_pool=pool,
    n_instances=100
)
```

## Semi-Supervised Learning

### Self-Training

Use confident predictions as pseudo-labels:

```python
from auto_labeling import SelfTrainer

trainer = SelfTrainer(
    base_model=model,
    confidence_threshold=0.95
)

# Iterative self-training
for iteration in range(10):
    # Predict on unlabeled
    pseudo_labeled = trainer.label_unlabeled(
        unlabeled_data=pool,
        confidence_threshold=0.95
    )

    # Retrain with pseudo-labels
    trainer.fit(
        labeled_data=labeled_data,
        pseudo_labeled_data=pseudo_labeled
    )
```

### Co-Training

Train multiple models on different views:

```python
from auto_labeling import CoTrainer

co_trainer = CoTrainer(
    model1=text_model,  # Text features
    model2=image_model,  # Image features
    confidence_threshold=0.9
)

co_trainer.fit(
    labeled_data=labeled_data,
    unlabeled_data=unlabeled_data,
    iterations=10
)
```

## Model-Assisted Labeling

### Pre-Annotations

```python
from auto_labeling import PreAnnotator

annotator = PreAnnotator(
    model=yolov8_model,
    confidence_threshold=0.7
)

# Generate pre-annotations
pre_annotations = annotator.predict(
    images=unlabeled_images,
    format="coco"  # or "yolo", "pascal_voc"
)

# Send to labeling tool for review
client.upload_with_predictions(
    images=unlabeled_images,
    predictions=pre_annotations
)
```

## Quality Control

### Inter-Annotator Agreement

```python
from auto_labeling import QualityControl

qc = QualityControl()

# Calculate agreement
agreement = qc.inter_annotator_agreement(
    annotations=[annotator1_labels, annotator2_labels],
    metric="cohen_kappa"  # or "fleiss_kappa", "iou"
)

print(f"Agreement: {agreement:.2%}")

# Find disagreements
disagreements = qc.find_disagreements(
    annotations=[annotator1_labels, annotator2_labels],
    threshold=0.5
)
```

### Consensus Labeling

```python
# Majority vote
consensus = qc.consensus_labeling(
    annotations=[ann1, ann2, ann3],
    method="majority_vote"  # or "weighted_vote"
)
```

## Complete Pipeline

### End-to-End Workflow

```python
from auto_labeling import AutoLabelingPipeline

pipeline = AutoLabelingPipeline(
    model=base_model,
    active_learning="uncertainty",
    labeling_tool="label_studio",
    api_key="your-key"
)

# Configure
pipeline.configure(
    batch_size=100,
    confidence_threshold=0.95,
    max_iterations=20,
    budget=10000  # Maximum labels
)

# Run pipeline
results = pipeline.run(
    labeled_data=initial_labeled,
    unlabeled_data=large_unlabeled_pool,
    target_accuracy=0.95
)

print(f"Final accuracy: {results.accuracy:.2%}")
print(f"Labels used: {results.labels_used}/{pipeline.budget}")
print(f"Cost savings: {results.cost_savings:.0%}")
```

## Use Cases

### Image Classification

```python
# Select uncertain images
learner = ActiveLearner(model=resnet50, strategy="uncertainty")
samples = learner.query(unlabeled_images, n_instances=100)

# Label with Label Studio
client.upload_tasks(project_id, samples)
```

### Object Detection

```python
# Pre-annotate with YOLOv8
annotator = PreAnnotator(model=yolov8)
pre_boxes = annotator.predict(images)

# Review in CVAT
cvat.upload_with_annotations(task_id, images, pre_boxes)
```

### Text Classification

```python
# Query by committee for text
committee = QueryByCommittee(models=[bert1, bert2, bert3])
uncertain_texts = committee.query(unlabeled_texts, n_instances=200)

# Label with Label Studio
client.upload_text_tasks(project_id, uncertain_texts)
```

### Medical Image Segmentation

```python
# Diversity sampling for rare cases
diversity = DiversitySampling(method="coreset")
diverse_scans = diversity.query(unlabeled_scans, n_instances=50)

# Expert annotation
cvat.create_segmentation_task(diverse_scans, labels=["tumor", "healthy"])
```

## Technologies

- **Active Learning**: modAL, libact, ALiPy
- **Labeling Tools**: Label Studio, CVAT, Labelbox
- **ML Frameworks**: PyTorch, TensorFlow, scikit-learn
- **APIs**: Label Studio SDK, CVAT SDK
- **Semi-Supervised**: scikit-learn, self-training
- **Computer Vision**: Ultralytics, Detectron2

## Performance

### Labeling Efficiency

| Method | Labels Needed | Accuracy | Cost Savings |
|--------|---------------|----------|--------------|
| Random sampling | 10,000 | 90% | 0% |
| Uncertainty sampling | 4,000 | 90% | 60% |
| Query by committee | 3,500 | 90% | 65% |
| Diversity sampling | 4,500 | 90% | 55% |
| Pre-annotation | 5,000 | 90% | 70%* |

*70% time savings with pre-annotations

### Active Learning Curves

```
Accuracy vs. Labels:
Random:      1000 labels → 75%, 5000 → 85%, 10000 → 90%
Uncertainty: 500 labels → 75%, 2000 → 85%, 4000 → 90%
```

## Best Practices

✅ Start with diversity sampling for initial batch
✅ Use uncertainty sampling for iterative refinement
✅ Pre-annotate with models (>70% confidence)
✅ Implement quality control (inter-annotator agreement)
✅ Monitor model confidence over iterations
✅ Set stopping criteria (accuracy or budget)
✅ Use human-in-the-loop for edge cases
✅ Version labeled datasets

## Cost Analysis

### Traditional Labeling
- 10,000 images × $0.10/image = **$1,000**
- Time: 40 hours

### Active Learning + Pre-Annotation
- 3,000 images × $0.10/image = **$300**
- Pre-annotation cost: $50 (compute)
- **Total: $350 (65% savings)**
- Time: 15 hours

## Integration Examples

### Label Studio Webhook

```python
# Auto-submit completed annotations
@app.route('/webhook', methods=['POST'])
def label_studio_webhook():
    annotation = request.json

    # Add to training data
    learner.add_labeled_sample(annotation)

    # Retrain incrementally
    learner.update_model()

    # Query next batch
    next_batch = learner.query(unlabeled_pool, n_instances=50)
    client.upload_tasks(project_id, next_batch)

    return {"status": "ok"}
```

### CVAT Auto-Annotation Plugin

```python
# Custom auto-annotation function
@cvat_plugin
def auto_annotate(task_id, frames):
    model = load_model("yolov8n.pt")

    for frame in frames:
        predictions = model.predict(frame)
        cvat.create_annotations(task_id, frame.id, predictions)
```

## References

- Active Learning Literature Survey: [Settles, 2009]
- Label Studio: https://labelstud.io/
- CVAT: https://github.com/opencv/cvat
- modAL: https://modal-python.readthedocs.io/
- Self-training with Noisy Student: https://arxiv.org/abs/1911.04252
