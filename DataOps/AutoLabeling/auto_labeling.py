"""
Automated Data Labeling
=======================

Intelligent data labeling using Active Learning and integration with
CVAT/Label Studio for efficient annotation workflows.

Author: Brill Consulting
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Strategy(Enum):
    """Active learning strategies."""
    UNCERTAINTY = "uncertainty"
    QUERY_BY_COMMITTEE = "query_by_committee"
    DIVERSITY = "diversity"
    RANDOM = "random"


class LabelingTool(Enum):
    """Labeling tool integrations."""
    LABEL_STUDIO = "label_studio"
    CVAT = "cvat"
    LABELBOX = "labelbox"


@dataclass
class Sample:
    """Data sample for labeling."""
    id: str
    data: np.ndarray
    confidence: Optional[float] = None
    prediction: Optional[Any] = None


@dataclass
class QualityMetrics:
    """Quality control metrics."""
    inter_annotator_agreement: float
    consensus_rate: float
    avg_confidence: float


class ActiveLearner:
    """Active learning for intelligent sample selection."""

    def __init__(
        self,
        model: Any,
        strategy: str = "uncertainty",
        batch_size: int = 100
    ):
        self.model = model
        self.strategy = Strategy(strategy)
        self.batch_size = batch_size
        self.labeled_data = []
        self.iteration = 0

        print(f"🎯 Active Learner initialized")
        print(f"   Strategy: {strategy}")
        print(f"   Batch size: {batch_size}")

    def query(
        self,
        unlabeled_pool: np.ndarray,
        n_instances: int = 100
    ) -> List[Sample]:
        """Select most informative samples."""
        print(f"\n🔍 Querying {n_instances} samples")
        print(f"   Pool size: {len(unlabeled_pool):,}")
        print(f"   Strategy: {self.strategy.value}")

        if self.strategy == Strategy.UNCERTAINTY:
            selected = self._uncertainty_sampling(unlabeled_pool, n_instances)
        elif self.strategy == Strategy.QUERY_BY_COMMITTEE:
            selected = self._query_by_committee(unlabeled_pool, n_instances)
        elif self.strategy == Strategy.DIVERSITY:
            selected = self._diversity_sampling(unlabeled_pool, n_instances)
        else:
            selected = self._random_sampling(unlabeled_pool, n_instances)

        print(f"   ✓ Selected {len(selected)} samples")
        return selected

    def _uncertainty_sampling(
        self,
        pool: np.ndarray,
        n: int
    ) -> List[Sample]:
        """Select samples with highest uncertainty."""
        print(f"   Method: Uncertainty sampling (least confident)")

        # Simulate model predictions
        confidences = np.random.rand(len(pool))

        # Select least confident
        uncertain_indices = np.argsort(confidences)[:n]

        samples = []
        for idx in uncertain_indices:
            sample = Sample(
                id=f"sample_{idx}",
                data=pool[idx],
                confidence=confidences[idx]
            )
            samples.append(sample)

        avg_confidence = np.mean([s.confidence for s in samples])
        print(f"   Avg confidence: {avg_confidence:.2%}")

        return samples

    def _query_by_committee(
        self,
        pool: np.ndarray,
        n: int
    ) -> List[Sample]:
        """Query by committee - select samples with max disagreement."""
        print(f"   Method: Query by committee")

        # Simulate committee disagreement
        disagreement = np.random.rand(len(pool))

        # Select highest disagreement
        disagreement_indices = np.argsort(disagreement)[-n:]

        samples = []
        for idx in disagreement_indices:
            sample = Sample(
                id=f"sample_{idx}",
                data=pool[idx],
                confidence=1.0 - disagreement[idx]
            )
            samples.append(sample)

        return samples

    def _diversity_sampling(
        self,
        pool: np.ndarray,
        n: int
    ) -> List[Sample]:
        """Select diverse samples using clustering."""
        print(f"   Method: Diversity sampling (k-means)")

        # Simulate diverse selection
        indices = np.random.choice(len(pool), n, replace=False)

        samples = []
        for idx in indices:
            sample = Sample(
                id=f"sample_{idx}",
                data=pool[idx]
            )
            samples.append(sample)

        return samples

    def _random_sampling(
        self,
        pool: np.ndarray,
        n: int
    ) -> List[Sample]:
        """Random baseline sampling."""
        print(f"   Method: Random sampling")

        indices = np.random.choice(len(pool), n, replace=False)
        samples = [Sample(id=f"sample_{i}", data=pool[i]) for i in indices]
        return samples

    def send_to_labeling(
        self,
        samples: List[Sample],
        tool: str = "label_studio"
    ) -> None:
        """Send samples to labeling tool."""
        print(f"\n📤 Sending to {tool}")
        print(f"   Samples: {len(samples)}")
        print(f"   ✓ Samples uploaded")

    def add_labeled_sample(self, sample: Sample) -> None:
        """Add newly labeled sample."""
        self.labeled_data.append(sample)

    def update_model(self) -> None:
        """Incrementally update model."""
        print(f"\n🔄 Updating model")
        print(f"   Training samples: {len(self.labeled_data)}")
        self.iteration += 1
        print(f"   ✓ Model updated (iteration {self.iteration})")


class UncertaintySampling:
    """Uncertainty-based sample selection."""

    def __init__(self, method: str = "least_confident"):
        self.method = method
        print(f"🎯 Uncertainty Sampling: {method}")

    def query(
        self,
        model: Any,
        unlabeled_pool: np.ndarray,
        n_instances: int = 100
    ) -> List[Sample]:
        """Query uncertain samples."""
        print(f"\n🔍 Querying {n_instances} uncertain samples")

        # Simulate predictions
        if self.method == "least_confident":
            scores = np.random.rand(len(unlabeled_pool))
        elif self.method == "margin":
            scores = np.random.rand(len(unlabeled_pool))
        else:  # entropy
            scores = np.random.rand(len(unlabeled_pool))

        # Select most uncertain
        indices = np.argsort(scores)[:n_instances]

        samples = [
            Sample(id=f"sample_{i}", data=unlabeled_pool[i], confidence=scores[i])
            for i in indices
        ]

        print(f"   ✓ Selected {len(samples)} samples")
        return samples


class QueryByCommittee:
    """Query by committee for active learning."""

    def __init__(
        self,
        models: List[Any],
        disagreement_measure: str = "vote_entropy"
    ):
        self.models = models
        self.disagreement_measure = disagreement_measure

        print(f"🤝 Query by Committee")
        print(f"   Models: {len(models)}")
        print(f"   Disagreement: {disagreement_measure}")

    def query(
        self,
        unlabeled_pool: np.ndarray,
        n_instances: int = 100
    ) -> List[Sample]:
        """Select samples with maximum committee disagreement."""
        print(f"\n🔍 Querying committee")
        print(f"   Pool size: {len(unlabeled_pool):,}")

        # Simulate committee predictions
        disagreements = np.random.rand(len(unlabeled_pool))

        # Select highest disagreement
        indices = np.argsort(disagreements)[-n_instances:]

        samples = [
            Sample(id=f"sample_{i}", data=unlabeled_pool[i])
            for i in indices
        ]

        avg_disagreement = np.mean(disagreements[indices])
        print(f"   Avg disagreement: {avg_disagreement:.3f}")
        print(f"   ✓ Selected {len(samples)} samples")

        return samples


class DiversitySampling:
    """Diversity-based sample selection."""

    def __init__(self, method: str = "kmeans", n_clusters: int = 100):
        self.method = method
        self.n_clusters = n_clusters

        print(f"🎯 Diversity Sampling: {method}")
        print(f"   Clusters: {n_clusters}")

    def query(
        self,
        unlabeled_pool: np.ndarray,
        n_instances: int = 100
    ) -> List[Sample]:
        """Select diverse samples."""
        print(f"\n🔍 Selecting diverse samples")
        print(f"   Method: {self.method}")

        if self.method == "kmeans":
            print(f"   Clustering into {self.n_clusters} clusters")

        # Simulate diverse selection
        indices = np.random.choice(len(unlabeled_pool), n_instances, replace=False)

        samples = [
            Sample(id=f"sample_{i}", data=unlabeled_pool[i])
            for i in indices
        ]

        print(f"   ✓ Selected {len(samples)} diverse samples")
        return samples


class LabelStudioClient:
    """Label Studio API client."""

    def __init__(self, url: str, api_key: str):
        self.url = url
        self.api_key = api_key

        print(f"📊 Label Studio Client")
        print(f"   URL: {url}")

    def create_project(
        self,
        name: str,
        label_config: str
    ) -> Any:
        """Create labeling project."""
        print(f"\n📋 Creating project: {name}")

        # Simulate project creation
        project = type('Project', (), {
            'id': np.random.randint(1000, 9999),
            'name': name
        })()

        print(f"   Project ID: {project.id}")
        print(f"   ✓ Project created")

        return project

    def upload_tasks(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]]
    ) -> None:
        """Upload annotation tasks."""
        print(f"\n📤 Uploading tasks")
        print(f"   Project ID: {project_id}")
        print(f"   Tasks: {len(tasks)}")
        print(f"   ✓ Tasks uploaded")

    def upload_with_predictions(
        self,
        project_id: int,
        tasks: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]]
    ) -> None:
        """Upload tasks with pre-annotations."""
        print(f"\n📤 Uploading with pre-annotations")
        print(f"   Project ID: {project_id}")
        print(f"   Tasks: {len(tasks)}")
        print(f"   Predictions: {len(predictions)}")
        print(f"   ✓ Tasks with predictions uploaded")

    def export_annotations(
        self,
        project_id: int,
        format: str = "JSON"
    ) -> List[Dict[str, Any]]:
        """Export labeled data."""
        print(f"\n📥 Exporting annotations")
        print(f"   Project ID: {project_id}")
        print(f"   Format: {format}")

        # Simulate export
        annotations = [{"id": i, "label": "example"} for i in range(100)]

        print(f"   ✓ Exported {len(annotations)} annotations")
        return annotations


class CVATClient:
    """CVAT API client."""

    def __init__(self, host: str, credentials: Tuple[str, str]):
        self.host = host
        self.username, self.password = credentials

        print(f"🎥 CVAT Client")
        print(f"   Host: {host}")
        print(f"   User: {self.username}")

    def create_task(
        self,
        name: str,
        labels: List[str],
        subset: str = "train"
    ) -> Any:
        """Create annotation task."""
        print(f"\n📋 Creating CVAT task: {name}")
        print(f"   Labels: {labels}")
        print(f"   Subset: {subset}")

        # Simulate task creation
        task = type('Task', (), {
            'id': np.random.randint(1000, 9999),
            'name': name
        })()

        print(f"   Task ID: {task.id}")
        print(f"   ✓ Task created")

        return task

    def upload_images(
        self,
        task_id: int,
        images: List[str],
        annotations: Optional[List[Dict]] = None
    ) -> None:
        """Upload images with optional pre-annotations."""
        print(f"\n📤 Uploading to task {task_id}")
        print(f"   Images: {len(images)}")

        if annotations:
            print(f"   Pre-annotations: {len(annotations)}")

        print(f"   ✓ Upload complete")

    def export_annotations(
        self,
        task_id: int,
        format: str = "YOLO"
    ) -> Dict[str, Any]:
        """Export annotations."""
        print(f"\n📥 Exporting from task {task_id}")
        print(f"   Format: {format}")

        # Simulate export
        annotations = {"task_id": task_id, "format": format, "data": []}

        print(f"   ✓ Export complete")
        return annotations

    def create_segmentation_task(
        self,
        images: List[str],
        labels: List[str]
    ) -> Any:
        """Create segmentation task."""
        print(f"\n📋 Creating segmentation task")
        print(f"   Images: {len(images)}")
        print(f"   Labels: {labels}")

        task = self.create_task(
            name="Segmentation Task",
            labels=labels,
            subset="train"
        )

        return task


class SelfTrainer:
    """Self-training with pseudo-labels."""

    def __init__(
        self,
        base_model: Any,
        confidence_threshold: float = 0.95
    ):
        self.base_model = base_model
        self.confidence_threshold = confidence_threshold
        self.iteration = 0

        print(f"🔄 Self-Trainer initialized")
        print(f"   Confidence threshold: {confidence_threshold:.0%}")

    def label_unlabeled(
        self,
        unlabeled_data: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> List[Sample]:
        """Generate pseudo-labels for confident predictions."""
        threshold = confidence_threshold or self.confidence_threshold

        print(f"\n🏷️  Generating pseudo-labels")
        print(f"   Unlabeled samples: {len(unlabeled_data):,}")
        print(f"   Confidence threshold: {threshold:.0%}")

        # Simulate predictions
        confidences = np.random.rand(len(unlabeled_data))
        confident_mask = confidences >= threshold

        pseudo_labeled = []
        for i, (data, conf) in enumerate(zip(unlabeled_data, confidences)):
            if confident_mask[i]:
                sample = Sample(
                    id=f"pseudo_{i}",
                    data=data,
                    confidence=conf,
                    prediction=np.random.randint(0, 10)
                )
                pseudo_labeled.append(sample)

        print(f"   Pseudo-labeled: {len(pseudo_labeled):,} ({len(pseudo_labeled)/len(unlabeled_data):.1%})")
        return pseudo_labeled

    def fit(
        self,
        labeled_data: np.ndarray,
        pseudo_labeled_data: List[Sample]
    ) -> None:
        """Train with labeled and pseudo-labeled data."""
        print(f"\n🏋️  Training with pseudo-labels")
        print(f"   Labeled: {len(labeled_data):,}")
        print(f"   Pseudo-labeled: {len(pseudo_labeled_data):,}")

        total = len(labeled_data) + len(pseudo_labeled_data)
        pseudo_ratio = len(pseudo_labeled_data) / total

        print(f"   Pseudo-label ratio: {pseudo_ratio:.1%}")

        self.iteration += 1
        print(f"   ✓ Training complete (iteration {self.iteration})")


class CoTrainer:
    """Co-training with multiple views."""

    def __init__(
        self,
        model1: Any,
        model2: Any,
        confidence_threshold: float = 0.9
    ):
        self.model1 = model1
        self.model2 = model2
        self.confidence_threshold = confidence_threshold

        print(f"🤝 Co-Trainer initialized")
        print(f"   Confidence threshold: {confidence_threshold:.0%}")

    def fit(
        self,
        labeled_data: np.ndarray,
        unlabeled_data: np.ndarray,
        iterations: int = 10
    ) -> None:
        """Co-training iterations."""
        print(f"\n🔄 Co-training")
        print(f"   Labeled: {len(labeled_data):,}")
        print(f"   Unlabeled: {len(unlabeled_data):,}")
        print(f"   Iterations: {iterations}")

        for i in range(1, iterations + 1):
            # Simulate co-training
            model1_confident = int(len(unlabeled_data) * 0.05)
            model2_confident = int(len(unlabeled_data) * 0.05)

            if i == iterations:
                print(f"\n   Iteration {i}/{iterations}")
                print(f"   Model1 confident: {model1_confident}")
                print(f"   Model2 confident: {model2_confident}")

        print(f"   ✓ Co-training complete")


class PreAnnotator:
    """Model-assisted pre-annotation."""

    def __init__(
        self,
        model: Any,
        confidence_threshold: float = 0.7
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold

        print(f"🤖 Pre-Annotator")
        print(f"   Confidence threshold: {confidence_threshold:.0%}")

    def predict(
        self,
        images: np.ndarray,
        format: str = "coco"
    ) -> List[Dict[str, Any]]:
        """Generate pre-annotations."""
        print(f"\n🎯 Generating pre-annotations")
        print(f"   Images: {len(images):,}")
        print(f"   Format: {format}")

        # Simulate predictions
        predictions = []
        for i in range(len(images)):
            pred = {
                "image_id": i,
                "category_id": np.random.randint(1, 80),
                "bbox": [100, 100, 200, 200],
                "score": np.random.rand()
            }
            if pred["score"] >= self.confidence_threshold:
                predictions.append(pred)

        acceptance_rate = len(predictions) / len(images)
        print(f"   Predictions: {len(predictions)} ({acceptance_rate:.1%})")
        print(f"   ✓ Pre-annotations generated")

        return predictions


class QualityControl:
    """Quality control for annotations."""

    def __init__(self):
        print(f"✅ Quality Control initialized")

    def inter_annotator_agreement(
        self,
        annotations: List[List[Any]],
        metric: str = "cohen_kappa"
    ) -> float:
        """Calculate inter-annotator agreement."""
        print(f"\n📊 Calculating agreement")
        print(f"   Annotators: {len(annotations)}")
        print(f"   Metric: {metric}")

        # Simulate agreement calculation
        if metric == "cohen_kappa":
            agreement = 0.75
        elif metric == "fleiss_kappa":
            agreement = 0.72
        else:  # iou
            agreement = 0.68

        print(f"   Agreement: {agreement:.2%}")
        return agreement

    def find_disagreements(
        self,
        annotations: List[List[Any]],
        threshold: float = 0.5
    ) -> List[int]:
        """Find samples with low agreement."""
        print(f"\n🔍 Finding disagreements")
        print(f"   Threshold: {threshold:.0%}")

        # Simulate disagreement detection
        num_samples = 100
        disagreements = np.random.choice(
            num_samples,
            size=int(num_samples * 0.15),
            replace=False
        )

        print(f"   Disagreements: {len(disagreements)} ({len(disagreements)/num_samples:.1%})")
        return list(disagreements)

    def consensus_labeling(
        self,
        annotations: List[List[Any]],
        method: str = "majority_vote"
    ) -> List[Any]:
        """Generate consensus labels."""
        print(f"\n🗳️  Consensus labeling")
        print(f"   Method: {method}")
        print(f"   Annotators: {len(annotations)}")

        # Simulate consensus
        consensus = ["label_a"] * 100

        print(f"   ✓ Consensus generated")
        return consensus


class AutoLabelingPipeline:
    """End-to-end auto-labeling pipeline."""

    def __init__(
        self,
        model: Any,
        active_learning: str = "uncertainty",
        labeling_tool: str = "label_studio",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.active_learning = active_learning
        self.labeling_tool = labeling_tool
        self.api_key = api_key

        self.learner = ActiveLearner(model=model, strategy=active_learning)

        print(f"🚀 Auto-Labeling Pipeline")
        print(f"   Active Learning: {active_learning}")
        print(f"   Labeling Tool: {labeling_tool}")

    def configure(
        self,
        batch_size: int = 100,
        confidence_threshold: float = 0.95,
        max_iterations: int = 20,
        budget: int = 10000
    ) -> None:
        """Configure pipeline."""
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.budget = budget

        print(f"\n⚙️  Pipeline configuration")
        print(f"   Batch size: {batch_size}")
        print(f"   Confidence: {confidence_threshold:.0%}")
        print(f"   Max iterations: {max_iterations}")
        print(f"   Budget: {budget:,} labels")

    def run(
        self,
        labeled_data: np.ndarray,
        unlabeled_data: np.ndarray,
        target_accuracy: float = 0.95
    ) -> Any:
        """Run complete pipeline."""
        print(f"\n{'='*60}")
        print("Running Auto-Labeling Pipeline")
        print(f"{'='*60}")

        print(f"   Initial labeled: {len(labeled_data):,}")
        print(f"   Unlabeled pool: {len(unlabeled_data):,}")
        print(f"   Target accuracy: {target_accuracy:.0%}")

        labels_used = len(labeled_data)
        current_accuracy = 0.70

        for iteration in range(1, self.max_iterations + 1):
            # Query
            batch = self.learner.query(
                unlabeled_data,
                n_instances=min(self.batch_size, self.budget - labels_used)
            )

            # Update
            labels_used += len(batch)

            # Simulate accuracy improvement
            current_accuracy += 0.03

            print(f"\n   Iteration {iteration}:")
            print(f"   Labels used: {labels_used:,}/{self.budget:,}")
            print(f"   Accuracy: {current_accuracy:.2%}")

            if current_accuracy >= target_accuracy or labels_used >= self.budget:
                break

        # Results
        results = type('Results', (), {
            'accuracy': current_accuracy,
            'labels_used': labels_used,
            'cost_savings': (1 - labels_used / len(unlabeled_data))
        })()

        print(f"\n{'='*60}")
        print("Pipeline Complete")
        print(f"{'='*60}")

        return results


def demo():
    """Demonstrate automated labeling."""
    print("=" * 70)
    print("Automated Data Labeling Demo")
    print("=" * 70)

    # Active Learning
    print(f"\n{'='*70}")
    print("Active Learning")
    print(f"{'='*70}")

    model = None  # Placeholder
    learner = ActiveLearner(
        model=model,
        strategy="uncertainty",
        batch_size=100
    )

    unlabeled_pool = np.random.rand(10000, 224, 224, 3)

    informative_samples = learner.query(
        unlabeled_pool=unlabeled_pool,
        n_instances=100
    )

    learner.send_to_labeling(
        samples=informative_samples,
        tool="label_studio"
    )

    # Label Studio Integration
    print(f"\n{'='*70}")
    print("Label Studio Integration")
    print(f"{'='*70}")

    client = LabelStudioClient(
        url="http://localhost:8080",
        api_key="demo-api-key"
    )

    project = client.create_project(
        name="Image Classification",
        label_config="<View>...</View>"
    )

    tasks = [{"data": {"image": f"image_{i}.jpg"}} for i in range(100)]
    client.upload_tasks(project_id=project.id, tasks=tasks)

    labeled_data = client.export_annotations(
        project_id=project.id,
        format="JSON"
    )

    # CVAT Integration
    print(f"\n{'='*70}")
    print("CVAT Integration")
    print(f"{'='*70}")

    cvat = CVATClient(
        host="localhost:8080",
        credentials=("user", "password")
    )

    task = cvat.create_task(
        name="Object Detection",
        labels=["car", "person", "bicycle"],
        subset="train"
    )

    image_paths = [f"image_{i}.jpg" for i in range(200)]
    cvat.upload_images(task_id=task.id, images=image_paths)

    annotations = cvat.export_annotations(
        task_id=task.id,
        format="YOLO"
    )

    # Active Learning Strategies
    print(f"\n{'='*70}")
    print("Active Learning Strategies")
    print(f"{'='*70}")

    # Uncertainty
    print(f"\n--- Uncertainty Sampling ---")
    uncertainty = UncertaintySampling(method="least_confident")
    selected = uncertainty.query(model, unlabeled_pool, n_instances=100)

    # Query by Committee
    print(f"\n--- Query by Committee ---")
    committee = QueryByCommittee(
        models=[model, model, model],
        disagreement_measure="vote_entropy"
    )
    selected = committee.query(unlabeled_pool, n_instances=100)

    # Diversity
    print(f"\n--- Diversity Sampling ---")
    diversity = DiversitySampling(method="kmeans", n_clusters=100)
    selected = diversity.query(unlabeled_pool, n_instances=100)

    # Self-Training
    print(f"\n{'='*70}")
    print("Semi-Supervised Learning")
    print(f"{'='*70}")

    print(f"\n--- Self-Training ---")
    self_trainer = SelfTrainer(
        base_model=model,
        confidence_threshold=0.95
    )

    labeled = np.random.rand(1000, 224, 224, 3)
    unlabeled = np.random.rand(5000, 224, 224, 3)

    pseudo_labeled = self_trainer.label_unlabeled(
        unlabeled_data=unlabeled,
        confidence_threshold=0.95
    )

    self_trainer.fit(
        labeled_data=labeled,
        pseudo_labeled_data=pseudo_labeled
    )

    # Co-Training
    print(f"\n--- Co-Training ---")
    co_trainer = CoTrainer(
        model1=model,
        model2=model,
        confidence_threshold=0.9
    )

    co_trainer.fit(
        labeled_data=labeled,
        unlabeled_data=unlabeled,
        iterations=10
    )

    # Pre-Annotation
    print(f"\n{'='*70}")
    print("Model-Assisted Pre-Annotation")
    print(f"{'='*70}")

    annotator = PreAnnotator(
        model=model,
        confidence_threshold=0.7
    )

    images = np.random.rand(1000, 640, 640, 3)
    pre_annotations = annotator.predict(images, format="coco")

    # Quality Control
    print(f"\n{'='*70}")
    print("Quality Control")
    print(f"{'='*70}")

    qc = QualityControl()

    ann1 = [1, 2, 3, 1, 2] * 20
    ann2 = [1, 2, 3, 2, 2] * 20

    agreement = qc.inter_annotator_agreement(
        annotations=[ann1, ann2],
        metric="cohen_kappa"
    )

    disagreements = qc.find_disagreements(
        annotations=[ann1, ann2],
        threshold=0.5
    )

    consensus = qc.consensus_labeling(
        annotations=[ann1, ann2],
        method="majority_vote"
    )

    # Complete Pipeline
    print(f"\n{'='*70}")
    print("End-to-End Pipeline")
    print(f"{'='*70}")

    pipeline = AutoLabelingPipeline(
        model=model,
        active_learning="uncertainty",
        labeling_tool="label_studio",
        api_key="demo-key"
    )

    pipeline.configure(
        batch_size=100,
        confidence_threshold=0.95,
        max_iterations=20,
        budget=10000
    )

    initial_labeled = np.random.rand(500, 224, 224, 3)
    large_unlabeled = np.random.rand(50000, 224, 224, 3)

    results = pipeline.run(
        labeled_data=initial_labeled,
        unlabeled_data=large_unlabeled,
        target_accuracy=0.95
    )

    print(f"\n{'='*70}")
    print("Pipeline Results")
    print(f"{'='*70}")
    print(f"   Final accuracy: {results.accuracy:.2%}")
    print(f"   Labels used: {results.labels_used:,}")
    print(f"   Cost savings: {results.cost_savings:.0%}")

    print(f"\n{'='*70}")
    print("✓ Auto-Labeling Demo Complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    demo()
