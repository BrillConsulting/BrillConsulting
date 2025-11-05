# AWS EC2 Management

**Advanced EC2 instance management system with auto-scaling, security groups, load balancing, and comprehensive monitoring.**

## 🎯 Overview

Production-ready EC2 management toolkit that provides:
- Complete instance lifecycle management (launch, stop, start, terminate)
- Security group creation and ingress/egress rule configuration
- Auto-scaling group setup with launch templates
- Load balancer integration
- Instance monitoring and health checks
- Resource tagging and organization

## ✨ Features

### Instance Management
- **Launch Instances**: Deploy EC2 instances with custom configurations
- **Start/Stop/Terminate**: Full lifecycle control
- **Instance Filtering**: Query instances by state, tags, or attributes
- **Tagging**: Organize resources with custom tags
- **Monitoring**: Track instance health and status

### Security Groups
- **Create Security Groups**: Define network access rules
- **Ingress Rules**: Configure inbound traffic (HTTP, HTTPS, SSH, custom ports)
- **Egress Rules**: Control outbound traffic
- **VPC Integration**: Support for VPC-based security groups

### Auto Scaling
- **Launch Templates**: Define instance configurations for scaling
- **Auto Scaling Groups**: Automatic capacity management
- **Scaling Policies**: Scale based on demand
- **Load Balancer Integration**: Distribute traffic across instances
- **Health Checks**: Automatic replacement of unhealthy instances

### Monitoring
- **Instance Status**: System and instance health checks
- **Resource Tracking**: Monitor CPU, memory, network
- **Summary Reports**: Aggregate statistics across resources

## 📋 Prerequisites

1. **AWS Account**: Active AWS account with appropriate permissions
2. **IAM Permissions**: Required permissions include:
   - `ec2:RunInstances`
   - `ec2:DescribeInstances`
   - `ec2:StartInstances`
   - `ec2:StopInstances`
   - `ec2:TerminateInstances`
   - `ec2:CreateSecurityGroup`
   - `ec2:AuthorizeSecurityGroupIngress`
   - `autoscaling:CreateAutoScalingGroup`
   - `autoscaling:DescribeAutoScalingGroups`
   - `elasticloadbalancing:*`

3. **AWS CLI**: Configured with credentials
4. **Python**: Version 3.8 or higher
5. **boto3**: AWS SDK for Python

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

**Option A: AWS CLI Configuration**
```bash
aws configure
# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region: us-east-1
# Default output format: json
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option C: IAM Role (for EC2 instances)**
- Attach an IAM role with appropriate permissions to your EC2 instance
- No manual credential configuration needed

### 3. Verify Setup

```bash
python aws_ec2.py
```

## 💻 Usage Examples

### Initialize EC2 Manager

```python
from aws_ec2 import EC2Manager

# Initialize with default region
ec2_manager = EC2Manager(region="us-east-1")

# Or use a specific AWS profile
ec2_manager = EC2Manager(region="us-west-2", profile="production")
```

### Launch EC2 Instances

```python
# Launch a single instance
instances = ec2_manager.launch_instance(
    image_id='ami-0c55b159cbfafe1f0',  # Amazon Linux 2 AMI
    instance_type='t2.micro',
    key_name='my-keypair',
    security_group_ids=['sg-0123456789abcdef0'],
    subnet_id='subnet-0123456789abcdef0',
    tags={
        'Name': 'WebServer',
        'Environment': 'Production',
        'Team': 'DevOps'
    }
)

print(f"Launched instance: {instances[0]['instance_id']}")
```

### Manage Instance Lifecycle

```python
# List all running instances
running_instances = ec2_manager.describe_instances(
    filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
)

for instance in running_instances:
    print(f"{instance['instance_id']}: {instance['public_ip']}")

# Stop instances
ec2_manager.stop_instances(['i-1234567890abcdef0'])

# Start instances
ec2_manager.start_instances(['i-1234567890abcdef0'])

# Terminate instances (permanent deletion)
ec2_manager.terminate_instances(['i-1234567890abcdef0'])
```

### Create Security Groups

```python
# Create security group
sg = ec2_manager.create_security_group(
    group_name='web-server-sg',
    description='Security group for web servers',
    vpc_id='vpc-0123456789abcdef0'
)

# Add HTTP rule
ec2_manager.add_ingress_rule(
    group_id=sg['group_id'],
    ip_protocol='tcp',
    from_port=80,
    to_port=80,
    cidr_ip='0.0.0.0/0'  # Allow from anywhere
)

# Add HTTPS rule
ec2_manager.add_ingress_rule(
    group_id=sg['group_id'],
    ip_protocol='tcp',
    from_port=443,
    to_port=443,
    cidr_ip='0.0.0.0/0'
)

# Add SSH rule (restricted to specific IP)
ec2_manager.add_ingress_rule(
    group_id=sg['group_id'],
    ip_protocol='tcp',
    from_port=22,
    to_port=22,
    cidr_ip='203.0.113.0/24'  # Your office IP range
)
```

### Setup Auto Scaling

```python
# Create launch template
template = ec2_manager.create_launch_template(
    template_name='web-server-template',
    image_id='ami-0c55b159cbfafe1f0',
    instance_type='t2.micro',
    key_name='my-keypair',
    security_group_ids=['sg-0123456789abcdef0'],
    user_data='''#!/bin/bash
    yum update -y
    yum install -y httpd
    systemctl start httpd
    systemctl enable httpd
    echo "Hello from $(hostname)" > /var/www/html/index.html
    '''
)

# Create auto scaling group
asg = ec2_manager.create_auto_scaling_group(
    group_name='web-server-asg',
    launch_template_id=template['template_id'],
    min_size=2,
    max_size=10,
    desired_capacity=3,
    vpc_zone_identifiers=[
        'subnet-0123456789abcdef0',
        'subnet-0123456789abcdef1'
    ]
)

print(f"Auto Scaling Group created: {asg['group_name']}")
```

### Monitor Instances

```python
# Get instance status
instance_ids = ['i-1234567890abcdef0', 'i-0987654321fedcba0']
monitoring = ec2_manager.get_instance_monitoring(instance_ids)

for status in monitoring:
    print(f"Instance: {status['instance_id']}")
    print(f"  State: {status['instance_state']}")
    print(f"  System Status: {status['system_status']}")
    print(f"  Instance Status: {status['instance_status']}")

# Get comprehensive summary
summary = ec2_manager.get_summary()
print(f"Total Instances: {summary['total_instances']}")
print(f"Instance States: {summary['instance_states']}")
print(f"Security Groups: {summary['security_groups']}")
```

## 🏗️ Architecture

```
EC2Manager
├── Instance Management
│   ├── launch_instance()      # Create new instances
│   ├── describe_instances()   # Query instance details
│   ├── start_instances()      # Start stopped instances
│   ├── stop_instances()       # Stop running instances
│   ├── terminate_instances()  # Permanently delete instances
│   └── tag_instance()         # Add resource tags
│
├── Security Groups
│   ├── create_security_group()    # Create new security group
│   ├── add_ingress_rule()         # Configure inbound traffic
│   └── describe_security_groups() # List security groups
│
├── Auto Scaling
│   ├── create_launch_template()      # Define instance template
│   ├── create_auto_scaling_group()   # Setup ASG
│   └── describe_auto_scaling_groups() # Query ASG details
│
└── Monitoring
    ├── get_instance_monitoring()  # Health checks
    └── get_summary()              # Aggregate statistics
```

## 🔒 Security Best Practices

1. **Least Privilege IAM**: Use minimal required permissions
2. **Security Groups**: Restrict ingress rules to specific IPs
3. **SSH Keys**: Use key-based authentication, rotate keys regularly
4. **Encryption**: Enable EBS volume encryption
5. **VPC**: Deploy instances in private subnets when possible
6. **Monitoring**: Enable CloudWatch detailed monitoring
7. **Tags**: Use tags for resource organization and cost tracking
8. **Backup**: Regular AMI snapshots for disaster recovery

## 📊 Common Use Cases

### Web Application Deployment
```python
# Launch web server with user data
instances = ec2_manager.launch_instance(
    image_id='ami-amazon-linux-2',
    instance_type='t3.medium',
    key_name='prod-key',
    security_group_ids=['sg-web-server'],
    user_data='''#!/bin/bash
    # Install and configure web server
    yum update -y
    yum install -y docker
    systemctl start docker
    docker run -d -p 80:80 myapp:latest
    ''',
    tags={'Name': 'WebApp', 'Env': 'Production'}
)
```

### Development Environment
```python
# Quick dev instance for testing
dev_instance = ec2_manager.launch_instance(
    image_id='ami-ubuntu-20-04',
    instance_type='t2.micro',
    key_name='dev-key',
    tags={'Name': 'DevBox', 'Team': 'Engineering'}
)
```

### Scheduled Scaling
```python
# Create auto-scaling for variable workloads
asg = ec2_manager.create_auto_scaling_group(
    group_name='batch-processing-asg',
    launch_template_id='lt-0123456789',
    min_size=0,      # Scale to zero during off-hours
    max_size=50,     # Burst capacity for peak loads
    desired_capacity=5
)
```

## 🐛 Troubleshooting

### Issue: "NoCredentialsError"
**Solution**: Configure AWS credentials using `aws configure` or environment variables

### Issue: "UnauthorizedOperation"
**Solution**: Ensure IAM user/role has required EC2 permissions

### Issue: "InvalidAMIID.NotFound"
**Solution**: Verify AMI ID exists in the target region

### Issue: "VcpuLimitExceeded"
**Solution**: Request vCPU limit increase in AWS Service Quotas

### Issue: "SecurityGroupLimitExceeded"
**Solution**: Delete unused security groups or request limit increase

## 📚 API Reference

### EC2Manager Class

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| `launch_instance()` | image_id, instance_type, ... | List[Dict] | Launch EC2 instances |
| `describe_instances()` | instance_ids, filters | List[Dict] | Query instance details |
| `stop_instances()` | instance_ids | Dict | Stop running instances |
| `start_instances()` | instance_ids | Dict | Start stopped instances |
| `terminate_instances()` | instance_ids | Dict | Terminate instances |
| `create_security_group()` | group_name, description, vpc_id | Dict | Create security group |
| `add_ingress_rule()` | group_id, protocol, ports, cidr | None | Add ingress rule |
| `create_auto_scaling_group()` | group_name, template_id, sizes | Dict | Create ASG |
| `get_summary()` | - | Dict | Get resource summary |

## 🔗 Related AWS Services

- **Amazon VPC**: Network isolation and routing
- **Elastic Load Balancing**: Traffic distribution
- **Amazon CloudWatch**: Monitoring and logging
- **AWS Auto Scaling**: Automatic capacity management
- **Amazon EBS**: Block storage volumes
- **AWS Systems Manager**: Instance management and patching

## 📞 Support

For questions or issues:
- **Email**: clientbrill@gmail.com
- **LinkedIn**: [brillconsulting](https://www.linkedin.com/in/brillconsulting)

## 📄 License

**Author**: Brill Consulting

---

**Last Updated**: November 2025
