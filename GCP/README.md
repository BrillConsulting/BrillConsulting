# Google Cloud Platform (GCP) Portfolio

**Kompletne portfolio 15 zaawansowanych projektów GCP** obejmujące compute, serverless, storage, data processing, messaging, AI/ML i CI/CD. Wszystkie projekty z produkcyjnymi funkcjami i pełną dokumentacją.

[![GCP](https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Apache Beam](https://img.shields.io/badge/Apache%20Beam-FF6600?style=for-the-badge&logo=apache&logoColor=white)](https://beam.apache.org/)

## 📊 Szybki Przegląd

| # | Projekt | Technologia | Linie Kodu | Kluczowe Funkcje |
|---|---------|-------------|------------|------------------|
| 1 | **BigQuery** | Data Warehouse | 578 | ML, Partycjonowanie, Optymalizacja zapytań |
| 2 | **Pub/Sub** | Messaging | 666 | Dead Letter Queues, Ordering, Batch publishing |
| 3 | **Cloud Logging** | Logging | 717 | Log-based metrics, Sinks, Alerty |
| 4 | **Firestore** | NoSQL Database | 727 | Real-time, Transakcje, Composite indexes |
| 5 | **Cloud Run** | Serverless | 681 | Autoscaling, Traffic splitting, Canary |
| 6 | **Secret Manager** | Security | 637 | Wersjonowanie, Auto-rotacja, IAM |
| 7 | **Cloud Scheduler** | Cron Jobs | 571 | HTTP/Pub/Sub/App Engine, Retry policies |
| 8 | **Cloud Tasks** | Task Queues | 618 | Rate limiting, Scheduling, Batch creation |
| 9 | **Dataflow** | Data Processing | 658 | Apache Beam, Windowing, ETL |
| 10 | **Dataproc** | Big Data | 799 | Spark/Hadoop, Autoscaling, Workflows |
| 11 | **Compute Engine** | Virtual Machines | 786 | GPU, MIGs, Load balancing |
| 12 | **Cloud Functions** | Serverless Functions | 836 | Multi-trigger, Versioning, Monitoring |
| 13 | **Cloud Storage** | Object Storage | 635 | Signed URLs, Lifecycle, Notifications |
| 14 | **Vertex AI** | Machine Learning | 910 | AutoML, GPU/TPU, Feature Store |
| 15 | **Cloud Build** | CI/CD | 748 | GitHub triggers, Artifacts, Analytics |

**Łącznie:** 11,000+ linii kodu | 15/15 projektów w pełni rozbudowanych ⭐

---

## 🗂️ Projekty według Kategorii

### 💾 Data & Analytics (3 projekty)
<table>
<tr>
<td width="33%">

#### BigQuery ⭐
**Data Warehouse z ML**

- BigQuery ML (LOGISTIC_REG)
- Partycjonowanie/clustering
- Materialized views
- Optymalizacja kosztów

[📂 View Project](BigQuery/)

</td>
<td width="33%">

#### Dataflow ⭐
**Stream & Batch Processing**

- Apache Beam pipelines
- Windowing (fixed/sliding)
- ETL do BigQuery
- Late data handling

[📂 View Project](Dataflow/)

</td>
<td width="33%">

#### Dataproc ⭐
**Managed Spark & Hadoop**

- Autoscaling (2-10 workers)
- PySpark/Hive jobs
- Workflow templates
- Lifecycle policies

[📂 View Project](Dataproc/)

</td>
</tr>
</table>

### 🖥️ Compute & Containers (3 projekty)
<table>
<tr>
<td width="33%">

#### Compute Engine ⭐
**VM & Infrastructure**

- GPU instances
- MIGs + autoscaling
- Load balancing
- Snapshots & templates

[📂 View Project](ComputeEngine/)

</td>
<td width="33%">

#### Cloud Run ⭐
**Serverless Containers**

- Scale to zero
- Traffic splitting (90/10)
- Canary deployments
- Secrets integration

[📂 View Project](CloudRun/)

</td>
<td width="33%">

#### Cloud Functions ⭐
**Event-Driven Functions**

- HTTP/Pub/Sub triggers
- Versioning & rollback
- Memory 128MB-8GB
- IAM access control

[📂 View Project](CloudFunctions/)

</td>
</tr>
</table>

### 📨 Messaging & Events (3 projekty)
<table>
<tr>
<td width="33%">

#### Pub/Sub ⭐
**Asynchronous Messaging**

- Pull/Push subscriptions
- Dead letter queues
- Message ordering
- Exactly-once delivery

[📂 View Project](PubSub/)

</td>
<td width="33%">

#### Cloud Tasks ⭐
**Distributed Task Queues**

- Rate limiting
- Task scheduling
- HTTP/App Engine targets
- Batch operations

[📂 View Project](CloudTasks/)

</td>
<td width="33%">

#### Cloud Scheduler ⭐
**Managed Cron Jobs**

- Cron expressions
- Multiple targets
- Retry with backoff
- Time zone support

[📂 View Project](CloudScheduler/)

</td>
</tr>
</table>

### 🗄️ Storage & Databases (2 projekty)
<table>
<tr>
<td width="50%">

#### Cloud Storage ⭐
**Object Storage**

- Signed URLs (v4)
- Lifecycle policies
- Pub/Sub notifications
- 4 storage classes

[📂 View Project](CloudStorage/)

</td>
<td width="50%">

#### Firestore ⭐
**NoSQL Database**

- Real-time listeners
- Atomic transactions
- Composite indexes
- Array operations

[📂 View Project](Firestore/)

</td>
</tr>
</table>

### 🤖 AI/ML & DevOps (4 projekty)
<table>
<tr>
<td width="25%">

#### Vertex AI ⭐
**ML Platform**

- AutoML (Tables/Vision/NLP)
- GPU/TPU training
- Feature Store
- Batch predictions

[📂 View Project](VertexAI/)

</td>
<td width="25%">

#### Cloud Build ⭐
**CI/CD**

- GitHub triggers
- Multi-step builds
- Artifact publishing
- Build analytics

[📂 View Project](CloudBuild/)

</td>
<td width="25%">

#### Cloud Logging ⭐
**Centralized Logging**

- Log-based metrics
- Sinks (BQ/Storage)
- Alert policies
- Log analytics

[📂 View Project](CloudLogging/)

</td>
<td width="25%">

#### Secret Manager ⭐
**Secret Storage**

- Versioning
- Auto-rotation
- IAM policies
- Audit logging

[📂 View Project](SecretManager/)

</td>
</tr>
</table>

---

## 🚀 Szybki Start

### Wymagania
```bash
Python 3.8+
Google Cloud SDK
pip install google-cloud-* (per project)
```

### Konfiguracja
```bash
# 1. Skonfiguruj uwierzytelnianie
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"

# 2. Ustaw projekt GCP
gcloud config set project YOUR_PROJECT_ID

# 3. Uruchom dowolny projekt
cd BigQuery/
pip install -r requirements.txt
python bigquery_ml.py
```

### Przykład: BigQuery ML
```python
from bigquery_ml import BigQueryMLManager

mgr = BigQueryMLManager('my-project', 'my-dataset')

# Utwórz model ML
mgr.create_ml_model({
    'model_name': 'churn_predictor',
    'model_type': 'LOGISTIC_REG',
    'input_table': 'customers',
    'label_column': 'churned'
})

# Wykonaj predykcje
predictions = mgr.predict('churn_predictor', 'new_customers')
```

---

## 🏗️ Przykładowe Architektury

### 1. Data Pipeline (ETL)
```
Cloud Storage → Dataflow → BigQuery → Data Studio
     ↓              ↓
  Pub/Sub    Cloud Logging
```
**Projekty:** Cloud Storage, Dataflow, BigQuery, Pub/Sub, Cloud Logging

### 2. Serverless Web App
```
Load Balancer → Cloud Run → Firestore
                    ↓           ↓
              Cloud Functions  Pub/Sub
                    ↓
              Cloud Storage
```
**Projekty:** Cloud Run, Cloud Functions, Firestore, Cloud Storage, Pub/Sub

### 3. ML Pipeline
```
Cloud Storage → Vertex AI (Training) → Model Registry
                     ↓                        ↓
              Cloud Logging          Vertex AI (Serving)
                                            ↓
                                     Cloud Functions
```
**Projekty:** Vertex AI, Cloud Storage, Cloud Functions, Cloud Logging

### 4. CI/CD Pipeline
```
GitHub → Cloud Build → Container Registry → Cloud Run
           ↓                                    ↓
      Pub/Sub (notifications)          Cloud Logging
           ↓
     Cloud Functions (Slack alert)
```
**Projekty:** Cloud Build, Cloud Run, Cloud Functions, Pub/Sub

---

## 🎓 Ścieżka Nauki

### Poziom 1: Podstawy (Początkujący)
1. **Cloud Storage** - Zarządzanie plikami
2. **Cloud Functions** - Pierwsze funkcje serverless
3. **Cloud Logging** - Monitorowanie aplikacji

### Poziom 2: Pośredni
4. **Compute Engine** - Zarządzanie VM
5. **Cloud Run** - Kontenery serverless
6. **Pub/Sub** - Messaging asynchroniczny
7. **Firestore** - Bazy danych NoSQL

### Poziom 3: Zaawansowany
8. **BigQuery** - Data warehousing + ML
9. **Dataflow** - Przetwarzanie danych
10. **Vertex AI** - Machine learning
11. **Cloud Build** - CI/CD

### Poziom 4: Ekspert
12. **Dataproc** - Big data (Spark/Hadoop)
13. **Cloud Scheduler** - Orkiestracja
14. **Cloud Tasks** - Kolejki zadań
15. **Secret Manager** - Security

---

## 📋 Matryca Funkcji

| Funkcja | Projekty | Poziom |
|---------|----------|--------|
| **Autoscaling** | Compute Engine, Cloud Run, Dataproc, Vertex AI | ⭐⭐⭐ |
| **Real-time Processing** | Pub/Sub, Dataflow, Firestore | ⭐⭐⭐ |
| **Machine Learning** | BigQuery ML, Vertex AI | ⭐⭐⭐ |
| **CI/CD Integration** | Cloud Build, Cloud Functions | ⭐⭐⭐ |
| **Security & IAM** | Secret Manager, Cloud Functions, Cloud Storage | ⭐⭐⭐ |
| **Cost Optimization** | BigQuery, Cloud Run, Compute Engine | ⭐⭐ |
| **Monitoring** | Cloud Logging, Vertex AI, Cloud Build | ⭐⭐⭐ |
| **Event-Driven** | Cloud Functions, Pub/Sub, Cloud Tasks | ⭐⭐⭐ |

---

## 💡 Przykłady Użycia

### Use Case 1: E-commerce Platform
**Scenariusz:** Platforma e-commerce z real-time inventory i ML recommendations

**Rozwiązanie:**
- **Cloud Run** - API backend
- **Firestore** - Product catalog + inventory
- **Cloud Functions** - Order processing
- **Vertex AI** - Recommendation engine
- **Pub/Sub** - Order events
- **BigQuery** - Analytics
- **Cloud Storage** - Product images

### Use Case 2: Data Analytics Platform
**Scenariusz:** Przetwarzanie i analiza dużych zbiorów danych

**Rozwiązanie:**
- **Cloud Storage** - Data lake
- **Dataflow** - ETL pipelines
- **Dataproc** - Spark processing
- **BigQuery** - Data warehouse + ML
- **Cloud Logging** - Pipeline monitoring
- **Cloud Scheduler** - Scheduled jobs

### Use Case 3: IoT Data Processing
**Scenariusz:** Real-time processing danych z urządzeń IoT

**Rozwiązanie:**
- **Pub/Sub** - Device messages
- **Dataflow** - Stream processing
- **Firestore** - Device state
- **BigQuery** - Historical data
- **Cloud Functions** - Alerts
- **Vertex AI** - Anomaly detection

---

## 🛠️ Technologie

### Języki
- **Python 3.8+** - Główny język (wszystkie projekty)
- **SQL** - BigQuery queries
- **YAML** - Cloud Build config

### GCP SDK
- `google-cloud-bigquery` - BigQuery client
- `google-cloud-pubsub` - Pub/Sub messaging
- `google-cloud-firestore` - Firestore database
- `google-cloud-storage` - Cloud Storage
- `google-cloud-logging` - Cloud Logging
- `google-cloud-run` - Cloud Run
- `google-cloud-functions` - Cloud Functions
- `google-cloud-aiplatform` - Vertex AI
- `google-cloud-build` - Cloud Build

### Frameworks
- **Apache Beam** (Dataflow) - Data processing
- **Apache Spark** (Dataproc) - Big data
- **Docker** (Cloud Run, Cloud Build) - Containers

---

## 📊 Statystyki Projektu

### Ogólne
- **Łączna liczba projektów:** 15
- **Łączna liczba linii kodu:** 11,000+
- **Średni rozmiar projektu:** 700+ linii
- **Projekty w pełni rozbudowane:** 15/15 (100%) ⭐

### Breakdown według kategorii
- **Data & Analytics:** 3 projekty (2,035 linii)
- **Compute & Containers:** 3 projekty (2,303 linie)
- **Messaging & Events:** 3 projekty (1,855 linii)
- **Storage & Databases:** 2 projekty (1,362 linie)
- **AI/ML & DevOps:** 4 projekty (3,109 linii)

### Funkcje
- **Manager Classes:** 75+
- **Metody API:** 300+
- **Demo Functions:** 15
- **README Pages:** 15 (comprehensive)

---

## 🎯 Główne Cechy Wszystkich Projektów

### ✅ Wspólne Funkcje
- ✔️ **Pełna implementacja** z production-ready features
- ✔️ **Manager Classes** dla każdego serwisu
- ✔️ **Type Hints** i docstrings
- ✔️ **Comprehensive READMEs** z przykładami
- ✔️ **Demo Functions** pokazujące wszystkie możliwości
- ✔️ **Error Handling** i best practices
- ✔️ **Configuration Options** dla elastyczności
- ✔️ **Code Generation** (YAML, scripts)

### 🔥 Zaawansowane Funkcje
- ⚡ **Autoscaling** (Compute Engine, Cloud Run, Dataproc, Vertex AI)
- 🔄 **Versioning** (Cloud Functions, Vertex AI, Secret Manager)
- 📊 **Monitoring & Metrics** (wszystkie projekty)
- 🔐 **IAM & Security** (wszystkie projekty)
- 📈 **Cost Optimization** (BigQuery, Cloud Run, Compute Engine)
- 🚀 **Performance Tuning** (BigQuery, Dataflow, Dataproc)

---

## 📚 Dodatkowe Zasoby

### Dokumentacja GCP
- [BigQuery Documentation](https://cloud.google.com/bigquery/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)

### Best Practices
- [GCP Architecture Framework](https://cloud.google.com/architecture/framework)
- [GCP Security Best Practices](https://cloud.google.com/security/best-practices)
- [Cost Optimization](https://cloud.google.com/architecture/framework/cost-optimization)

### Certyfikacje
- **Associate Cloud Engineer**
- **Professional Cloud Architect**
- **Professional Data Engineer**
- **Professional Machine Learning Engineer**

---

## 🤝 Contributing

Projekty są częścią portfolio demonstracyjnego. Dla sugestii lub pytań:

**Email:** [clientbrill@gmail.com](mailto:clientbrill@gmail.com)
**LinkedIn:** [brillconsulting](https://www.linkedin.com/in/brillconsulting)

---

## 📄 Licencja

Portfolio demonstracyjne - Brill Consulting © 2024

---

## 🌟 Highlights

**Co wyróżnia to portfolio:**

1. 🎯 **Kompleksowe pokrycie** - 15 kluczowych serwisów GCP
2. 💪 **Production-ready** - Wszystkie projekty z zaawansowanymi funkcjami
3. 📖 **Szczegółowa dokumentacja** - README z przykładami dla każdego projektu
4. 🏗️ **Real-world patterns** - Architektury używane w produkcji
5. 🔧 **Best practices** - Zgodność z GCP guidelines
6. 🚀 **Skalowalne** - Autoscaling, load balancing, redundancy
7. 🔐 **Secure** - IAM, Secret Manager, proper access control
8. 📊 **Monitorowalne** - Logging, metrics, alerts w każdym projekcie

---

**Autor:** Brill Consulting | **Last Updated:** 2024 | **Status:** Complete ✅
