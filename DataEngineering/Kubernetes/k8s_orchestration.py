"""
Kubernetes Orchestration & Management
Complete Kubernetes cluster management and application deployment
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class KubernetesOrchestrator:
    """Comprehensive Kubernetes cluster management"""

    def __init__(self, cluster_name: str, namespace: str = 'default'):
        """
        Initialize Kubernetes orchestrator

        Args:
            cluster_name: Cluster name
            namespace: Default namespace
        """
        self.cluster_name = cluster_name
        self.namespace = namespace
        self.deployments = []
        self.services = []
        self.pods = []
        self.configmaps = []
        self.secrets = []
        self.ingresses = []
        self.persistent_volumes = []
        self.jobs = []

    def create_deployment(self, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Kubernetes deployment

        Args:
            deployment_config: Deployment configuration

        Returns:
            Deployment details
        """
        deployment = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': deployment_config.get('name', 'app-deployment'),
                'namespace': deployment_config.get('namespace', self.namespace),
                'labels': deployment_config.get('labels', {})
            },
            'spec': {
                'replicas': deployment_config.get('replicas', 3),
                'selector': deployment_config.get('selector', {}),
                'template': {
                    'metadata': {
                        'labels': deployment_config.get('pod_labels', {})
                    },
                    'spec': {
                        'containers': deployment_config.get('containers', []),
                        'volumes': deployment_config.get('volumes', [])
                    }
                },
                'strategy': deployment_config.get('strategy', {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxSurge': 1,
                        'maxUnavailable': 0
                    }
                })
            },
            'status': {
                'replicas': deployment_config.get('replicas', 3),
                'readyReplicas': deployment_config.get('replicas', 3),
                'availableReplicas': deployment_config.get('replicas', 3)
            },
            'created_at': datetime.now().isoformat()
        }

        self.deployments.append(deployment)
        print(f"✓ Deployment created: {deployment['metadata']['name']}")
        print(f"  Replicas: {deployment['spec']['replicas']}")
        return deployment

    def create_service(self, service_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Kubernetes service

        Args:
            service_config: Service configuration

        Returns:
            Service details
        """
        service = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': service_config.get('name', 'app-service'),
                'namespace': service_config.get('namespace', self.namespace)
            },
            'spec': {
                'type': service_config.get('type', 'ClusterIP'),
                'selector': service_config.get('selector', {}),
                'ports': service_config.get('ports', []),
                'sessionAffinity': service_config.get('sessionAffinity', 'None')
            },
            'created_at': datetime.now().isoformat()
        }

        if service['spec']['type'] == 'LoadBalancer':
            service['status'] = {
                'loadBalancer': {
                    'ingress': [{'ip': '203.0.113.50'}]
                }
            }

        self.services.append(service)
        print(f"✓ Service created: {service['metadata']['name']} (Type: {service['spec']['type']})")
        return service

    def create_configmap(self, configmap_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ConfigMap

        Args:
            configmap_config: ConfigMap configuration

        Returns:
            ConfigMap details
        """
        configmap = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': configmap_config.get('name', 'app-config'),
                'namespace': configmap_config.get('namespace', self.namespace)
            },
            'data': configmap_config.get('data', {}),
            'created_at': datetime.now().isoformat()
        }

        self.configmaps.append(configmap)
        print(f"✓ ConfigMap created: {configmap['metadata']['name']}")
        print(f"  Keys: {list(configmap['data'].keys())}")
        return configmap

    def create_secret(self, secret_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Secret

        Args:
            secret_config: Secret configuration

        Returns:
            Secret details
        """
        secret = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': secret_config.get('name', 'app-secret'),
                'namespace': secret_config.get('namespace', self.namespace)
            },
            'type': secret_config.get('type', 'Opaque'),
            'data': secret_config.get('data', {}),
            'created_at': datetime.now().isoformat()
        }

        self.secrets.append(secret)
        print(f"✓ Secret created: {secret['metadata']['name']}")
        return secret

    def create_ingress(self, ingress_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Ingress

        Args:
            ingress_config: Ingress configuration

        Returns:
            Ingress details
        """
        ingress = {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': ingress_config.get('name', 'app-ingress'),
                'namespace': ingress_config.get('namespace', self.namespace),
                'annotations': ingress_config.get('annotations', {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod'
                })
            },
            'spec': {
                'rules': ingress_config.get('rules', []),
                'tls': ingress_config.get('tls', [])
            },
            'created_at': datetime.now().isoformat()
        }

        self.ingresses.append(ingress)
        print(f"✓ Ingress created: {ingress['metadata']['name']}")
        return ingress

    def create_persistent_volume(self, pv_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create PersistentVolume

        Args:
            pv_config: PV configuration

        Returns:
            PV details
        """
        pv = {
            'apiVersion': 'v1',
            'kind': 'PersistentVolume',
            'metadata': {
                'name': pv_config.get('name', 'pv-data')
            },
            'spec': {
                'capacity': {
                    'storage': pv_config.get('storage', '10Gi')
                },
                'accessModes': pv_config.get('accessModes', ['ReadWriteOnce']),
                'persistentVolumeReclaimPolicy': pv_config.get('reclaimPolicy', 'Retain'),
                'storageClassName': pv_config.get('storageClassName', 'standard'),
                'hostPath': pv_config.get('hostPath', {'path': '/mnt/data'})
            },
            'created_at': datetime.now().isoformat()
        }

        self.persistent_volumes.append(pv)
        print(f"✓ PersistentVolume created: {pv['metadata']['name']}")
        print(f"  Storage: {pv['spec']['capacity']['storage']}")
        return pv

    def create_persistent_volume_claim(self, pvc_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create PersistentVolumeClaim

        Args:
            pvc_config: PVC configuration

        Returns:
            PVC details
        """
        pvc = {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': pvc_config.get('name', 'pvc-data'),
                'namespace': pvc_config.get('namespace', self.namespace)
            },
            'spec': {
                'accessModes': pvc_config.get('accessModes', ['ReadWriteOnce']),
                'resources': {
                    'requests': {
                        'storage': pvc_config.get('storage', '5Gi')
                    }
                },
                'storageClassName': pvc_config.get('storageClassName', 'standard')
            },
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ PersistentVolumeClaim created: {pvc['metadata']['name']}")
        return pvc

    def create_horizontal_pod_autoscaler(self, hpa_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create HorizontalPodAutoscaler

        Args:
            hpa_config: HPA configuration

        Returns:
            HPA details
        """
        hpa = {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': hpa_config.get('name', 'app-hpa'),
                'namespace': hpa_config.get('namespace', self.namespace)
            },
            'spec': {
                'scaleTargetRef': hpa_config.get('scaleTargetRef', {}),
                'minReplicas': hpa_config.get('minReplicas', 2),
                'maxReplicas': hpa_config.get('maxReplicas', 10),
                'metrics': hpa_config.get('metrics', [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    }
                ])
            },
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ HorizontalPodAutoscaler created: {hpa['metadata']['name']}")
        print(f"  Min/Max replicas: {hpa['spec']['minReplicas']}/{hpa['spec']['maxReplicas']}")
        return hpa

    def create_job(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create Kubernetes Job

        Args:
            job_config: Job configuration

        Returns:
            Job details
        """
        job = {
            'apiVersion': 'batch/v1',
            'kind': 'Job',
            'metadata': {
                'name': job_config.get('name', 'batch-job'),
                'namespace': job_config.get('namespace', self.namespace)
            },
            'spec': {
                'completions': job_config.get('completions', 1),
                'parallelism': job_config.get('parallelism', 1),
                'backoffLimit': job_config.get('backoffLimit', 3),
                'template': {
                    'spec': {
                        'containers': job_config.get('containers', []),
                        'restartPolicy': 'OnFailure'
                    }
                }
            },
            'status': {
                'succeeded': 1,
                'completionTime': datetime.now().isoformat()
            },
            'created_at': datetime.now().isoformat()
        }

        self.jobs.append(job)
        print(f"✓ Job created: {job['metadata']['name']}")
        return job

    def create_cronjob(self, cronjob_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create CronJob

        Args:
            cronjob_config: CronJob configuration

        Returns:
            CronJob details
        """
        cronjob = {
            'apiVersion': 'batch/v1',
            'kind': 'CronJob',
            'metadata': {
                'name': cronjob_config.get('name', 'scheduled-job'),
                'namespace': cronjob_config.get('namespace', self.namespace)
            },
            'spec': {
                'schedule': cronjob_config.get('schedule', '0 * * * *'),
                'jobTemplate': {
                    'spec': {
                        'template': {
                            'spec': {
                                'containers': cronjob_config.get('containers', []),
                                'restartPolicy': 'OnFailure'
                            }
                        }
                    }
                },
                'successfulJobsHistoryLimit': cronjob_config.get('successfulJobsHistoryLimit', 3),
                'failedJobsHistoryLimit': cronjob_config.get('failedJobsHistoryLimit', 1)
            },
            'created_at': datetime.now().isoformat()
        }

        print(f"✓ CronJob created: {cronjob['metadata']['name']}")
        print(f"  Schedule: {cronjob['spec']['schedule']}")
        return cronjob

    def scale_deployment(self, deployment_name: str, replicas: int) -> Dict[str, Any]:
        """Scale deployment"""
        deployment = next((d for d in self.deployments if d['metadata']['name'] == deployment_name), None)
        if deployment:
            deployment['spec']['replicas'] = replicas
            print(f"✓ Deployment scaled: {deployment_name} → {replicas} replicas")
            return deployment
        return {'error': 'Deployment not found'}

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster information"""
        return {
            'cluster_name': self.cluster_name,
            'namespace': self.namespace,
            'deployments': len(self.deployments),
            'services': len(self.services),
            'configmaps': len(self.configmaps),
            'secrets': len(self.secrets),
            'ingresses': len(self.ingresses),
            'persistent_volumes': len(self.persistent_volumes),
            'jobs': len(self.jobs),
            'timestamp': datetime.now().isoformat()
        }


def demo():
    """Demonstrate Kubernetes orchestration"""

    print("=" * 60)
    print("Kubernetes Orchestration Demo")
    print("=" * 60)

    # Initialize orchestrator
    k8s = KubernetesOrchestrator(
        cluster_name='production-cluster',
        namespace='production'
    )

    print("\n1. Creating ConfigMap...")
    configmap = k8s.create_configmap({
        'name': 'app-config',
        'data': {
            'DATABASE_HOST': 'postgres.production.svc.cluster.local',
            'DATABASE_PORT': '5432',
            'CACHE_TTL': '3600',
            'LOG_LEVEL': 'INFO'
        }
    })

    print("\n2. Creating Secret...")
    secret = k8s.create_secret({
        'name': 'app-secret',
        'type': 'Opaque',
        'data': {
            'DATABASE_PASSWORD': 'cGFzc3dvcmQxMjM=',
            'API_KEY': 'YXBpa2V5MTIz'
        }
    })

    print("\n3. Creating Deployment...")
    deployment = k8s.create_deployment({
        'name': 'web-app',
        'replicas': 3,
        'labels': {'app': 'web', 'tier': 'frontend'},
        'selector': {'matchLabels': {'app': 'web'}},
        'pod_labels': {'app': 'web', 'tier': 'frontend'},
        'containers': [
            {
                'name': 'web',
                'image': 'nginx:1.21',
                'ports': [{'containerPort': 80}],
                'resources': {
                    'requests': {'cpu': '100m', 'memory': '128Mi'},
                    'limits': {'cpu': '500m', 'memory': '512Mi'}
                },
                'envFrom': [
                    {'configMapRef': {'name': 'app-config'}},
                    {'secretRef': {'name': 'app-secret'}}
                ]
            }
        ]
    })

    print("\n4. Creating Service...")
    service = k8s.create_service({
        'name': 'web-service',
        'type': 'LoadBalancer',
        'selector': {'app': 'web'},
        'ports': [
            {'port': 80, 'targetPort': 80, 'protocol': 'TCP'}
        ]
    })

    print("\n5. Creating Ingress...")
    ingress = k8s.create_ingress({
        'name': 'web-ingress',
        'rules': [
            {
                'host': 'www.example.com',
                'http': {
                    'paths': [
                        {
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': 'web-service',
                                    'port': {'number': 80}
                                }
                            }
                        }
                    ]
                }
            }
        ],
        'tls': [
            {
                'hosts': ['www.example.com'],
                'secretName': 'tls-secret'
            }
        ]
    })

    print("\n6. Creating PersistentVolume...")
    pv = k8s.create_persistent_volume({
        'name': 'data-pv',
        'storage': '50Gi',
        'accessModes': ['ReadWriteOnce'],
        'storageClassName': 'fast-ssd'
    })

    print("\n7. Creating PersistentVolumeClaim...")
    pvc = k8s.create_persistent_volume_claim({
        'name': 'data-pvc',
        'storage': '20Gi',
        'storageClassName': 'fast-ssd'
    })

    print("\n8. Creating HorizontalPodAutoscaler...")
    hpa = k8s.create_horizontal_pod_autoscaler({
        'name': 'web-hpa',
        'scaleTargetRef': {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'name': 'web-app'
        },
        'minReplicas': 3,
        'maxReplicas': 10
    })

    print("\n9. Creating Job...")
    job = k8s.create_job({
        'name': 'data-migration',
        'completions': 1,
        'parallelism': 1,
        'containers': [
            {
                'name': 'migrator',
                'image': 'migration-tool:latest',
                'command': ['python', 'migrate.py']
            }
        ]
    })

    print("\n10. Creating CronJob...")
    cronjob = k8s.create_cronjob({
        'name': 'backup-job',
        'schedule': '0 2 * * *',
        'containers': [
            {
                'name': 'backup',
                'image': 'backup-tool:latest',
                'command': ['backup.sh']
            }
        ]
    })

    print("\n11. Scaling deployment...")
    k8s.scale_deployment('web-app', 5)

    print("\n12. Cluster summary:")
    info = k8s.get_cluster_info()
    print(f"  Cluster: {info['cluster_name']}")
    print(f"  Namespace: {info['namespace']}")
    print(f"  Deployments: {info['deployments']}")
    print(f"  Services: {info['services']}")
    print(f"  ConfigMaps: {info['configmaps']}")
    print(f"  Secrets: {info['secrets']}")
    print(f"  Ingresses: {info['ingresses']}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
