# Google Cloud Platform (GCP) Portfolio

Comprehensive GCP cloud solutions covering compute, serverless, storage, data processing, messaging, AI/ML, and CI/CD.

## 📊 Projects Overview

### 1. BigQuery - Cloud Data Warehouse ⭐
**Description:** Advanced data warehousing with BigQuery ML, partitioning, and query optimization

**Features:**
- Dataset and table management with partitioning/clustering
- BigQuery ML for in-database machine learning
- Advanced analytics with window functions
- Query optimization and cost estimation
- Materialized views for performance
- Streaming inserts and batch loading

**Technologies:** BigQuery, BigQuery ML, SQL

**[View Project →](BigQuery/)**

---

### 2. Pub/Sub - Messaging Service ⭐
**Description:** Reliable asynchronous messaging for event-driven architectures

**Features:**
- Topic and subscription management (pull/push)
- Batch publishing with ordering guarantees
- Dead letter queues with retry policies
- Message filtering and schema validation
- Flow control and acknowledgment
- Exactly-once delivery support

**Technologies:** Cloud Pub/Sub, Event Streaming

**[View Project →](PubSub/)**

---

### 3. Cloud Logging - Centralized Logging ⭐
**Description:** Structured logging, log analytics, and monitoring integration

**Features:**
- Structured and batch logging
- Advanced log queries with filtering
- Log-based metrics and alerting
- Log sinks to BigQuery/Storage/Pub/Sub
- Log analytics and pattern detection
- Alert policies with notifications

**Technologies:** Cloud Logging, Cloud Monitoring

**[View Project →](CloudLogging/)**

---

### 4. Firestore - NoSQL Database ⭐
**Description:** Scalable NoSQL document database with real-time synchronization

**Features:**
- Document CRUD with batch operations
- Advanced queries (compound, range, pagination)
- Atomic transactions and field operations
- Composite indexes for complex queries
- Real-time listeners with filters
- Array operations (ArrayUnion/ArrayRemove)

**Technologies:** Cloud Firestore, NoSQL

**[View Project →](Firestore/)**

---

### 5. Cloud Run - Serverless Containers ⭐
**Description:** Fully managed serverless platform for containerized applications

**Features:**
- Container deployment with autoscaling
- Traffic splitting and canary deployments
- Resource configuration (CPU/memory)
- IAM and secrets integration
- Revision management and rollback
- Scale to zero capability

**Technologies:** Cloud Run, Docker, Kubernetes

**[View Project →](CloudRun/)**

---

### 6. Secret Manager - Secure Secret Storage ⭐
**Description:** Centralized secret management with automatic rotation

**Features:**
- Secret creation with versioning
- Secure secret access and retrieval
- Automatic rotation with Cloud Scheduler
- IAM-based access control
- Version lifecycle management
- Audit logging for compliance

**Technologies:** Secret Manager, Cloud Scheduler

**[View Project →](SecretManager/)**

---

### 7. Cloud Scheduler - Cron Job Service ⭐
**Description:** Managed cron job scheduling with multiple target types

**Features:**
- HTTP, Pub/Sub, and App Engine targets
- Flexible cron expressions with time zones
- Retry policies with exponential backoff
- Job pause/resume and manual execution
- Common schedule templates
- Job monitoring and management

**Technologies:** Cloud Scheduler, Cron

**[View Project →](CloudScheduler/)**

---

### 8. Cloud Tasks - Task Queue Service ⭐
**Description:** Distributed task queues with rate limiting and scheduling

**Features:**
- HTTP and App Engine task targets
- Queue rate limiting and concurrency control
- Task scheduling with delays
- Batch task creation
- Retry configuration with backoff
- Queue monitoring and purging

**Technologies:** Cloud Tasks, Task Queues

**[View Project →](CloudTasks/)**

---

### 9. Dataflow - Stream & Batch Processing ⭐
**Description:** Apache Beam pipelines for data processing at scale

**Features:**
- Batch and streaming pipelines
- ETL with BigQuery integration
- Windowing (fixed, sliding, session)
- Pipeline templates with parameters
- Late data handling and watermarks
- Job monitoring and cancellation

**Technologies:** Dataflow, Apache Beam, Python

**[View Project →](Dataflow/)**

---

### 10. Dataproc - Managed Spark & Hadoop ⭐
**Description:** Fully managed Apache Spark and Hadoop clusters

**Features:**
- Cluster creation with autoscaling (2-10 workers)
- Spark, PySpark, and Hive job submission
- Workflow templates for multi-job orchestration
- Lifecycle policies for cost optimization
- Initialization actions for custom setup
- Job monitoring and cluster management

**Technologies:** Dataproc, Apache Spark, Hadoop, Hive

**[View Project →](Dataproc/)**

---

### 11. Compute Engine
**Description:** VM instances and compute resources management

**Features:**
- VM instance management
- Instance templates
- Managed instance groups
- Autoscaling

**Technologies:** GCP Compute Engine

**[View Project →](ComputeEngine/)**

---

### 12. Cloud Functions
**Description:** Serverless functions on GCP

**Features:**
- HTTP triggers
- Pub/Sub triggers
- Storage triggers
- Function management

**Technologies:** GCP Cloud Functions

**[View Project →](CloudFunctions/)**

---

### 13. Cloud Storage
**Description:** Object storage and data management

**Features:**
- Bucket management
- Object operations
- Lifecycle policies
- Signed URLs
- Versioning

**Technologies:** GCP Cloud Storage

**[View Project →](CloudStorage/)**

---

### 14. Vertex AI
**Description:** Machine learning platform

**Features:**
- Dataset management
- Model training
- Model deployment
- Online predictions
- Endpoint management

**Technologies:** Vertex AI

**[View Project →](VertexAI/)**

---

### 15. Cloud Build
**Description:** CI/CD automation

**Features:**
- YAML build configuration
- GitHub triggers
- Multi-step builds
- Container builds
- Build history

**Technologies:** GCP Cloud Build

**[View Project →](CloudBuild/)**

---

## 🚀 Getting Started

Each project contains:
- Complete Python implementation with advanced features
- Comprehensive README with usage examples
- Requirements file
- Demo functions showcasing all capabilities

### Running Demos

```bash
cd ProjectName/
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
python project_file.py
```

## 🎯 Key GCP Services

### Data & Analytics
- **BigQuery**: Data warehouse with ML capabilities
- **Dataflow**: Stream and batch processing (Apache Beam)
- **Dataproc**: Managed Spark and Hadoop clusters

### Compute & Containers
- **Compute Engine**: VM instances and managed groups
- **Cloud Run**: Serverless containers
- **Cloud Functions**: Serverless functions

### Messaging & Events
- **Pub/Sub**: Asynchronous messaging
- **Cloud Tasks**: Distributed task queues
- **Cloud Scheduler**: Managed cron jobs

### Databases
- **Firestore**: NoSQL document database with real-time sync

### Operations & Security
- **Cloud Logging**: Centralized logging and monitoring
- **Secret Manager**: Secure secret storage with rotation

### Storage & AI/ML
- **Cloud Storage**: Object storage
- **Vertex AI**: Machine learning platform
- **Cloud Build**: CI/CD automation

## 📚 Technologies Used

- **Python 3.8+**: Primary programming language
- **Google Cloud SDK**: Cloud service integration
- **Apache Beam**: Data processing framework
- **Apache Spark**: Big data processing
- **Docker**: Container technology

## 💡 Use Cases

### Data Engineering
- Build ETL pipelines with Dataflow
- Run Spark jobs on Dataproc clusters
- Store and analyze data in BigQuery

### Microservices
- Deploy containerized apps on Cloud Run
- Implement event-driven architecture with Pub/Sub
- Manage async tasks with Cloud Tasks

### Security & Compliance
- Store secrets securely with Secret Manager
- Implement centralized logging with Cloud Logging
- Control access with IAM policies

### Real-Time Applications
- Process streaming data with Dataflow
- Use Firestore for real-time databases
- Implement pub/sub messaging patterns

## 📊 Project Statistics

- **Total Projects**: 15
- **Lines of Code**: 10,000+
- **Expanded Projects**: 10 (BigQuery, Pub/Sub, Cloud Logging, Firestore, Cloud Run, Secret Manager, Cloud Scheduler, Cloud Tasks, Dataflow, Dataproc)
- **Coverage**: Compute, Storage, Databases, Messaging, Data Processing, ML, CI/CD

## 📧 Contact

For questions or collaboration: [clientbrill@gmail.com](mailto:clientbrill@gmail.com)

---

**Author:** Brill Consulting
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)
