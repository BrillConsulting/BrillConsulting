"""
AWS EC2 Management
==================

Comprehensive EC2 instance management with auto-scaling, security groups,
load balancing, and advanced monitoring capabilities.

Author: Brill Consulting
"""

import boto3
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from botocore.exceptions import ClientError, NoCredentialsError


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EC2Manager:
    """
    Advanced AWS EC2 Management System

    Provides comprehensive EC2 instance management including:
    - Instance lifecycle management (launch, stop, start, terminate)
    - Security group configuration with rule management
    - Auto-scaling group setup and policies
    - Elastic Load Balancer integration
    - Instance monitoring and tagging
    """

    def __init__(self, region: str = "us-east-1", profile: Optional[str] = None):
        """
        Initialize EC2 Manager.

        Args:
            region: AWS region (default: us-east-1)
            profile: AWS CLI profile name (optional)
        """
        try:
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.ec2_client = session.client('ec2', region_name=region)
            self.ec2_resource = session.resource('ec2', region_name=region)
            self.autoscaling_client = session.client('autoscaling', region_name=region)
            self.elb_client = session.client('elbv2', region_name=region)
            self.region = region
            logger.info(f"EC2 Manager initialized for region: {region}")
        except NoCredentialsError:
            logger.error("AWS credentials not found")
            raise
        except ClientError as e:
            logger.error(f"Error initializing EC2 Manager: {e}")
            raise

    # ==================== Instance Management ====================

    def launch_instance(
        self,
        image_id: str,
        instance_type: str = "t2.micro",
        key_name: Optional[str] = None,
        security_group_ids: Optional[List[str]] = None,
        subnet_id: Optional[str] = None,
        user_data: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        min_count: int = 1,
        max_count: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Launch EC2 instance(s).

        Args:
            image_id: AMI ID to launch
            instance_type: Instance type (default: t2.micro)
            key_name: SSH key pair name
            security_group_ids: List of security group IDs
            subnet_id: Subnet ID for VPC
            user_data: User data script
            tags: Tags to apply to instance
            min_count: Minimum instances to launch
            max_count: Maximum instances to launch

        Returns:
            List of launched instance details
        """
        try:
            logger.info(f"Launching {max_count} instance(s) with AMI: {image_id}")

            launch_params = {
                'ImageId': image_id,
                'InstanceType': instance_type,
                'MinCount': min_count,
                'MaxCount': max_count
            }

            if key_name:
                launch_params['KeyName'] = key_name
            if security_group_ids:
                launch_params['SecurityGroupIds'] = security_group_ids
            if subnet_id:
                launch_params['SubnetId'] = subnet_id
            if user_data:
                launch_params['UserData'] = user_data

            response = self.ec2_client.run_instances(**launch_params)

            instances = []
            for instance_data in response['Instances']:
                instance_id = instance_data['InstanceId']
                instances.append({
                    'instance_id': instance_id,
                    'instance_type': instance_data['InstanceType'],
                    'state': instance_data['State']['Name'],
                    'launch_time': instance_data['LaunchTime'].isoformat()
                })

                # Apply tags if provided
                if tags:
                    self.tag_instance(instance_id, tags)

                logger.info(f"✓ Instance launched: {instance_id}")

            return instances

        except ClientError as e:
            logger.error(f"Error launching instance: {e}")
            raise

    def describe_instances(
        self,
        instance_ids: Optional[List[str]] = None,
        filters: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Describe EC2 instances.

        Args:
            instance_ids: List of instance IDs to describe
            filters: Filters to apply (e.g., [{'Name': 'instance-state-name', 'Values': ['running']}])

        Returns:
            List of instance details
        """
        try:
            params = {}
            if instance_ids:
                params['InstanceIds'] = instance_ids
            if filters:
                params['Filters'] = filters

            response = self.ec2_client.describe_instances(**params)

            instances = []
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instances.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance['InstanceType'],
                        'state': instance['State']['Name'],
                        'public_ip': instance.get('PublicIpAddress', 'N/A'),
                        'private_ip': instance.get('PrivateIpAddress', 'N/A'),
                        'launch_time': instance['LaunchTime'].isoformat(),
                        'tags': {tag['Key']: tag['Value'] for tag in instance.get('Tags', [])}
                    })

            logger.info(f"Found {len(instances)} instance(s)")
            return instances

        except ClientError as e:
            logger.error(f"Error describing instances: {e}")
            raise

    def stop_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Stop EC2 instances."""
        try:
            logger.info(f"Stopping instances: {instance_ids}")
            response = self.ec2_client.stop_instances(InstanceIds=instance_ids)
            logger.info("✓ Instances stopped")
            return response
        except ClientError as e:
            logger.error(f"Error stopping instances: {e}")
            raise

    def start_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Start EC2 instances."""
        try:
            logger.info(f"Starting instances: {instance_ids}")
            response = self.ec2_client.start_instances(InstanceIds=instance_ids)
            logger.info("✓ Instances started")
            return response
        except ClientError as e:
            logger.error(f"Error starting instances: {e}")
            raise

    def terminate_instances(self, instance_ids: List[str]) -> Dict[str, Any]:
        """Terminate EC2 instances."""
        try:
            logger.info(f"Terminating instances: {instance_ids}")
            response = self.ec2_client.terminate_instances(InstanceIds=instance_ids)
            logger.info("✓ Instances terminated")
            return response
        except ClientError as e:
            logger.error(f"Error terminating instances: {e}")
            raise

    def tag_instance(self, instance_id: str, tags: Dict[str, str]) -> None:
        """
        Add tags to an instance.

        Args:
            instance_id: Instance ID
            tags: Dictionary of tags {key: value}
        """
        try:
            tag_list = [{'Key': k, 'Value': v} for k, v in tags.items()]
            self.ec2_client.create_tags(Resources=[instance_id], Tags=tag_list)
            logger.info(f"✓ Tags applied to instance: {instance_id}")
        except ClientError as e:
            logger.error(f"Error tagging instance: {e}")
            raise

    # ==================== Security Groups ====================

    def create_security_group(
        self,
        group_name: str,
        description: str,
        vpc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a security group.

        Args:
            group_name: Security group name
            description: Security group description
            vpc_id: VPC ID (optional)

        Returns:
            Security group details
        """
        try:
            logger.info(f"Creating security group: {group_name}")

            params = {
                'GroupName': group_name,
                'Description': description
            }
            if vpc_id:
                params['VpcId'] = vpc_id

            response = self.ec2_client.create_security_group(**params)
            group_id = response['GroupId']

            logger.info(f"✓ Security group created: {group_id}")
            return {'group_id': group_id, 'group_name': group_name}

        except ClientError as e:
            logger.error(f"Error creating security group: {e}")
            raise

    def add_ingress_rule(
        self,
        group_id: str,
        ip_protocol: str,
        from_port: int,
        to_port: int,
        cidr_ip: str = "0.0.0.0/0"
    ) -> None:
        """
        Add ingress rule to security group.

        Args:
            group_id: Security group ID
            ip_protocol: Protocol (tcp, udp, icmp, -1 for all)
            from_port: Start of port range
            to_port: End of port range
            cidr_ip: CIDR IP range (default: 0.0.0.0/0)
        """
        try:
            self.ec2_client.authorize_security_group_ingress(
                GroupId=group_id,
                IpPermissions=[{
                    'IpProtocol': ip_protocol,
                    'FromPort': from_port,
                    'ToPort': to_port,
                    'IpRanges': [{'CidrIp': cidr_ip}]
                }]
            )
            logger.info(f"✓ Ingress rule added: {ip_protocol}:{from_port}-{to_port}")
        except ClientError as e:
            logger.error(f"Error adding ingress rule: {e}")
            raise

    def describe_security_groups(
        self,
        group_ids: Optional[List[str]] = None,
        group_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Describe security groups."""
        try:
            params = {}
            if group_ids:
                params['GroupIds'] = group_ids
            if group_names:
                params['GroupNames'] = group_names

            response = self.ec2_client.describe_security_groups(**params)

            groups = []
            for sg in response['SecurityGroups']:
                groups.append({
                    'group_id': sg['GroupId'],
                    'group_name': sg['GroupName'],
                    'description': sg['Description'],
                    'vpc_id': sg.get('VpcId', 'N/A'),
                    'ingress_rules': len(sg.get('IpPermissions', [])),
                    'egress_rules': len(sg.get('IpPermissionsEgress', []))
                })

            logger.info(f"Found {len(groups)} security group(s)")
            return groups

        except ClientError as e:
            logger.error(f"Error describing security groups: {e}")
            raise

    # ==================== Auto Scaling ====================

    def create_launch_template(
        self,
        template_name: str,
        image_id: str,
        instance_type: str = "t2.micro",
        key_name: Optional[str] = None,
        security_group_ids: Optional[List[str]] = None,
        user_data: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create EC2 launch template for auto-scaling."""
        try:
            logger.info(f"Creating launch template: {template_name}")

            launch_template_data = {
                'ImageId': image_id,
                'InstanceType': instance_type
            }

            if key_name:
                launch_template_data['KeyName'] = key_name
            if security_group_ids:
                launch_template_data['SecurityGroupIds'] = security_group_ids
            if user_data:
                launch_template_data['UserData'] = user_data

            response = self.ec2_client.create_launch_template(
                LaunchTemplateName=template_name,
                LaunchTemplateData=launch_template_data
            )

            template_id = response['LaunchTemplate']['LaunchTemplateId']
            logger.info(f"✓ Launch template created: {template_id}")
            return {
                'template_id': template_id,
                'template_name': template_name
            }

        except ClientError as e:
            logger.error(f"Error creating launch template: {e}")
            raise

    def create_auto_scaling_group(
        self,
        group_name: str,
        launch_template_id: str,
        min_size: int,
        max_size: int,
        desired_capacity: int,
        vpc_zone_identifiers: Optional[List[str]] = None,
        target_group_arns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create Auto Scaling Group.

        Args:
            group_name: ASG name
            launch_template_id: Launch template ID
            min_size: Minimum instances
            max_size: Maximum instances
            desired_capacity: Desired number of instances
            vpc_zone_identifiers: List of subnet IDs
            target_group_arns: Load balancer target group ARNs
        """
        try:
            logger.info(f"Creating auto-scaling group: {group_name}")

            params = {
                'AutoScalingGroupName': group_name,
                'LaunchTemplate': {
                    'LaunchTemplateId': launch_template_id
                },
                'MinSize': min_size,
                'MaxSize': max_size,
                'DesiredCapacity': desired_capacity
            }

            if vpc_zone_identifiers:
                params['VPCZoneIdentifier'] = ','.join(vpc_zone_identifiers)
            if target_group_arns:
                params['TargetGroupARNs'] = target_group_arns

            self.autoscaling_client.create_auto_scaling_group(**params)

            logger.info(f"✓ Auto-scaling group created: {group_name}")
            return {
                'group_name': group_name,
                'min_size': min_size,
                'max_size': max_size,
                'desired_capacity': desired_capacity
            }

        except ClientError as e:
            logger.error(f"Error creating auto-scaling group: {e}")
            raise

    def describe_auto_scaling_groups(
        self,
        group_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Describe auto-scaling groups."""
        try:
            params = {}
            if group_names:
                params['AutoScalingGroupNames'] = group_names

            response = self.autoscaling_client.describe_auto_scaling_groups(**params)

            groups = []
            for asg in response['AutoScalingGroups']:
                groups.append({
                    'group_name': asg['AutoScalingGroupName'],
                    'min_size': asg['MinSize'],
                    'max_size': asg['MaxSize'],
                    'desired_capacity': asg['DesiredCapacity'],
                    'current_instances': len(asg['Instances']),
                    'availability_zones': asg['AvailabilityZones']
                })

            logger.info(f"Found {len(groups)} auto-scaling group(s)")
            return groups

        except ClientError as e:
            logger.error(f"Error describing auto-scaling groups: {e}")
            raise

    # ==================== Monitoring ====================

    def get_instance_monitoring(self, instance_ids: List[str]) -> List[Dict[str, Any]]:
        """Get instance monitoring status."""
        try:
            response = self.ec2_client.describe_instance_status(
                InstanceIds=instance_ids,
                IncludeAllInstances=True
            )

            monitoring = []
            for status in response['InstanceStatuses']:
                monitoring.append({
                    'instance_id': status['InstanceId'],
                    'instance_state': status['InstanceState']['Name'],
                    'system_status': status.get('SystemStatus', {}).get('Status', 'N/A'),
                    'instance_status': status.get('InstanceStatus', {}).get('Status', 'N/A'),
                    'availability_zone': status['AvailabilityZone']
                })

            return monitoring

        except ClientError as e:
            logger.error(f"Error getting instance monitoring: {e}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive EC2 summary."""
        try:
            instances = self.describe_instances()
            security_groups = self.describe_security_groups()

            # Count instances by state
            state_counts = {}
            for instance in instances:
                state = instance['state']
                state_counts[state] = state_counts.get(state, 0) + 1

            return {
                'region': self.region,
                'total_instances': len(instances),
                'instance_states': state_counts,
                'security_groups': len(security_groups),
                'timestamp': datetime.now().isoformat()
            }

        except ClientError as e:
            logger.error(f"Error getting summary: {e}")
            return {'error': str(e)}


def demo():
    """
    Demonstration of EC2 Manager capabilities.

    Note: This demo uses mock/placeholder values. Replace with actual
    AWS resources (AMI IDs, VPC IDs, etc.) for real deployments.
    """
    print("AWS EC2 Manager - Advanced Demo")
    print("=" * 70)

    # Initialize manager (uses default AWS credentials)
    # Uncomment to use: ec2_manager = EC2Manager(region="us-east-1")

    print("\n📋 DEMO MODE - Showing API Usage Examples")
    print("-" * 70)

    # Example 1: Launch instances
    print("\n1️⃣  Launch EC2 Instance:")
    print("""
    instances = ec2_manager.launch_instance(
        image_id='ami-0c55b159cbfafe1f0',  # Amazon Linux 2
        instance_type='t2.micro',
        key_name='my-key-pair',
        tags={'Name': 'WebServer', 'Environment': 'Production'}
    )
    """)

    # Example 2: Security groups
    print("\n2️⃣  Create Security Group:")
    print("""
    sg = ec2_manager.create_security_group(
        group_name='web-server-sg',
        description='Security group for web servers',
        vpc_id='vpc-12345678'
    )

    # Add HTTP rule
    ec2_manager.add_ingress_rule(
        group_id=sg['group_id'],
        ip_protocol='tcp',
        from_port=80,
        to_port=80,
        cidr_ip='0.0.0.0/0'
    )
    """)

    # Example 3: Auto Scaling
    print("\n3️⃣  Create Auto Scaling Group:")
    print("""
    # Create launch template
    template = ec2_manager.create_launch_template(
        template_name='web-server-template',
        image_id='ami-0c55b159cbfafe1f0',
        instance_type='t2.micro',
        security_group_ids=['sg-12345678']
    )

    # Create ASG
    asg = ec2_manager.create_auto_scaling_group(
        group_name='web-server-asg',
        launch_template_id=template['template_id'],
        min_size=2,
        max_size=10,
        desired_capacity=3,
        vpc_zone_identifiers=['subnet-1', 'subnet-2']
    )
    """)

    # Example 4: Instance management
    print("\n4️⃣  Instance Management:")
    print("""
    # List all running instances
    instances = ec2_manager.describe_instances(
        filters=[{'Name': 'instance-state-name', 'Values': ['running']}]
    )

    # Stop instances
    ec2_manager.stop_instances(['i-1234567890abcdef0'])

    # Start instances
    ec2_manager.start_instances(['i-1234567890abcdef0'])

    # Get monitoring data
    monitoring = ec2_manager.get_instance_monitoring(['i-1234567890abcdef0'])
    """)

    # Example 5: Summary
    print("\n5️⃣  Get EC2 Summary:")
    print("""
    summary = ec2_manager.get_summary()
    print(f"Total instances: {summary['total_instances']}")
    print(f"Instance states: {summary['instance_states']}")
    """)

    print("\n" + "=" * 70)
    print("✓ Demo Complete!")
    print("\n⚠️  Setup Instructions:")
    print("   1. Configure AWS credentials: aws configure")
    print("   2. Replace placeholder values with actual AWS resource IDs")
    print("   3. Ensure IAM permissions for EC2, Auto Scaling, and ELB")
    print("   4. Review security group rules before deployment")


if __name__ == '__main__':
    demo()
