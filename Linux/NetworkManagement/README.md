# Linux Network Management - Production-Ready v2.0.0

Enterprise-grade network configuration, routing, DNS, firewall management, and monitoring system for Linux servers.

## Overview

A comprehensive Python-based network management system designed for production Linux environments. Provides complete control over network interfaces, routing, DNS, firewalls, VPNs, and real-time monitoring with bandwidth tracking and port scanning capabilities.

## Features

### Core Network Management
- **Interface Configuration**: Configure static/DHCP network interfaces with netplan
- **Interface Discovery**: List and inspect all network interfaces with detailed statistics
- **MAC Address Management**: Track and manage MAC addresses
- **MTU Configuration**: Set custom MTU values per interface

### DNS and Routing
- **DNS Configuration**: Manage /etc/resolv.conf with nameservers, search domains, and options
- **Static Routes**: Add custom routing rules with metrics and routing tables
- **Policy-Based Routing**: Support for multiple routing tables
- **Route Inspection**: View and analyze routing tables

### Network Diagnostics
- **Ping Tests**: ICMP echo tests with packet loss and latency statistics
- **Traceroute**: Trace network paths with hop-by-hop analysis
- **Netstat/SS**: View network connections, listening ports, and statistics
- **Connection Tracking**: Monitor active TCP/UDP connections with process information
- **Service Detection**: Identify services running on open ports

### Bandwidth Monitoring
- **Real-time Monitoring**: Track RX/TX bytes, packets, and rates in Mbps
- **Historical Data**: Maintain bandwidth history for trend analysis
- **Error Tracking**: Monitor packet errors and drops
- **Interface Statistics**: Comprehensive network interface metrics

### Port Scanning
- **TCP Port Scanning**: Scan individual or multiple ports
- **Common Port Scanning**: Quick scan of well-known service ports
- **Service Detection**: Identify services on open ports
- **Scan Results Tracking**: Maintain history of port scan results

### VPN Configuration
- **WireGuard VPN**: Modern, fast VPN with peer configuration
- **OpenVPN**: Traditional VPN server setup
- **IPSec Support**: Framework for IPSec VPN configuration
- **Multi-peer Support**: Configure multiple VPN peers

### Firewall Management
- **iptables Integration**: Full iptables rule management
- **Rule Builder**: Intuitive firewall rule creation
- **Chain Management**: Support for INPUT, OUTPUT, FORWARD chains
- **State Tracking**: Connection state filtering (NEW, ESTABLISHED, RELATED)
- **NAT Configuration**: Network Address Translation setup
- **Port Forwarding**: Configure DNAT for port forwarding
- **Rule Listing**: View all active firewall rules

### Advanced Features
- **Network Bridges**: Create bridge interfaces for VMs and containers
- **VLAN Support**: 802.1Q VLAN tagging and configuration
- **Network Bonding**: Link aggregation (not in current implementation but supported)
- **Configuration Export**: Export network configuration to JSON
- **Dry Run Mode**: Test configurations without applying changes

## Architecture

### Core Components

```
NetworkManager
├── Interface Management
│   ├── configure_interface()
│   └── list_interfaces()
├── DNS & Routing
│   ├── configure_dns()
│   ├── add_route()
│   └── show_routing_table()
├── Network Diagnostics
│   ├── ping()
│   ├── traceroute()
│   ├── netstat()
│   └── get_active_connections()
├── Bandwidth Monitoring
│   ├── monitor_bandwidth()
│   └── get_bandwidth_history()
├── Port Scanning
│   ├── scan_port()
│   ├── scan_ports()
│   └── scan_common_ports()
├── VPN Configuration
│   ├── setup_vpn_wireguard()
│   └── setup_vpn_openvpn()
├── Firewall Management
│   ├── add_firewall_rule()
│   ├── list_firewall_rules()
│   ├── setup_nat()
│   └── setup_port_forwarding()
└── Advanced Features
    ├── setup_bridge()
    ├── setup_vlan()
    └── export_config()
```

### Data Structures

- **NetworkInterface**: Complete interface configuration with statistics
- **Route**: Static route configuration with metrics and tables
- **FirewallRule**: Comprehensive firewall rule with actions and states
- **Connection**: Active network connection with process information
- **BandwidthStats**: Real-time bandwidth statistics and rates

## Installation

### Prerequisites

- Python 3.8+
- Linux operating system (Ubuntu, Debian, CentOS, RHEL)
- Root/sudo access for network operations
- iproute2 package (ip command)
- iptables/nftables
- Optional: WireGuard, OpenVPN, traceroute

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3 iproute2 iptables traceroute wireguard-tools openvpn

# RHEL/CentOS
sudo yum install -y python3 iproute iptables traceroute wireguard-tools openvpn

# Arch Linux
sudo pacman -S python iproute2 iptables traceroute wireguard-tools openvpn
```

### Python Installation

```bash
# Install from repository
cd /home/user/BrillConsulting/Linux/NetworkManagement
pip install -r requirements.txt

# No external dependencies required for basic functionality
# All features use Python standard library
```

## Usage

### Basic Usage

```python
from network_manager import NetworkManager

# Initialize manager
manager = NetworkManager(hostname='prod-server-01')

# Configure network interface
interface = manager.configure_interface({
    'name': 'eth0',
    'ip_address': '192.168.1.100',
    'netmask': '255.255.255.0',
    'gateway': '192.168.1.1',
    'dns_servers': ['8.8.8.8', '1.1.1.1'],
    'mtu': 1500
})

# Add static route
route = manager.add_route({
    'destination': '10.0.0.0/8',
    'gateway': '192.168.1.254',
    'interface': 'eth0',
    'metric': 100
})

# Configure DNS
dns = manager.configure_dns({
    'nameservers': ['8.8.8.8', '8.8.4.4'],
    'search_domains': ['example.com'],
    'options': ['timeout:2']
})
```

### Network Diagnostics

```python
# Ping test
ping_result = manager.ping('8.8.8.8', count=4)
print(f"Packet loss: {ping_result['packet_loss_percent']}%")
print(f"Average latency: {ping_result['avg_ms']}ms")

# Traceroute
trace_result = manager.traceroute('google.com', max_hops=20)
print(f"Total hops: {trace_result['total_hops']}")

# View active connections
connections = manager.get_active_connections()
for conn in connections[:10]:
    print(f"{conn.protocol} {conn.local_address}:{conn.local_port} -> {conn.remote_address}:{conn.remote_port}")

# Network statistics
netstat_result = manager.netstat('-tuln')
print(f"Total connections: {netstat_result['total_connections']}")
```

### Bandwidth Monitoring

```python
# Monitor bandwidth for 10 seconds
stats = manager.monitor_bandwidth('eth0', duration=10)
print(f"RX Rate: {stats.rx_rate_mbps:.2f} Mbps")
print(f"TX Rate: {stats.tx_rate_mbps:.2f} Mbps")
print(f"RX Packets: {stats.rx_packets:,}")
print(f"TX Packets: {stats.tx_packets:,}")

# Get bandwidth history
history = manager.get_bandwidth_history('eth0')
for stat in history:
    print(f"{stat.timestamp}: RX={stat.rx_rate_mbps:.2f} Mbps, TX={stat.tx_rate_mbps:.2f} Mbps")
```

### Port Scanning

```python
# Scan common ports
scan_result = manager.scan_common_ports('192.168.1.1')
print(f"Open ports: {len(scan_result['open_ports'])}")
for port_info in scan_result['open_ports']:
    print(f"  Port {port_info['port']}: {port_info['service']}")

# Scan specific ports
custom_scan = manager.scan_ports('192.168.1.1', [80, 443, 8080, 8443])

# Scan single port
port_result = manager.scan_port('192.168.1.1', 22)
print(f"Port 22 is {port_result['state']}")
```

### Firewall Management

```python
# Add firewall rule - Allow SSH
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'tcp',
    'dport': 22,
    'state': 'NEW,ESTABLISHED',
    'comment': 'Allow SSH'
})

# Add rule - Allow HTTP/HTTPS
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'tcp',
    'dport': 80,
    'comment': 'Allow HTTP'
})

manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'tcp',
    'dport': 443,
    'comment': 'Allow HTTPS'
})

# Setup NAT
nat = manager.setup_nat({
    'internal_interface': 'eth0',
    'external_interface': 'eth1',
    'internal_network': '192.168.1.0/24'
})

# Setup port forwarding
port_forward = manager.setup_port_forwarding({
    'external_port': 8080,
    'internal_ip': '192.168.1.100',
    'internal_port': 80,
    'protocol': 'tcp'
})

# List all firewall rules
rules = manager.list_firewall_rules()
```

### VPN Configuration

```python
# Setup WireGuard VPN
wireguard = manager.setup_vpn_wireguard({
    'interface': 'wg0',
    'address': '10.0.0.1/24',
    'listen_port': 51820,
    'peers': [
        {
            'public_key': 'peer1_public_key_here',
            'allowed_ips': '10.0.0.2/32',
            'endpoint': '203.0.113.5:51820'
        }
    ],
    'dns': ['1.1.1.1', '1.0.0.1']
})

# Setup OpenVPN
openvpn = manager.setup_vpn_openvpn({
    'name': 'server',
    'protocol': 'udp',
    'port': 1194,
    'server_network': '10.8.0.0',
    'server_netmask': '255.255.255.0'
})
```

### Advanced Features

```python
# Create network bridge
bridge = manager.setup_bridge({
    'name': 'br0',
    'interfaces': ['eth0', 'eth1'],
    'ip_address': '192.168.10.1',
    'netmask': '255.255.255.0',
    'stp': True
})

# Setup VLAN
vlan = manager.setup_vlan({
    'parent_interface': 'eth0',
    'vlan_id': 100,
    'ip_address': '192.168.100.1',
    'netmask': '255.255.255.0'
})

# Export configuration
manager.export_config('/tmp/network_config.json')

# Get network summary
info = manager.get_network_info()
print(f"Hostname: {info['hostname']}")
print(f"Interfaces: {info['interfaces']}")
print(f"Firewall rules: {info['firewall_rules']}")
print(f"VPN connections: {info['vpn_connections']}")
```

### Dry Run Mode

```python
# Test configurations without applying
manager = NetworkManager(hostname='test-server', dry_run=True)

# All operations will be simulated
interface = manager.configure_interface({...})  # Won't actually configure
route = manager.add_route({...})  # Won't actually add route
```

## Command-Line Usage

```bash
# Run demo
python3 network_manager.py

# Run specific operations (requires root)
sudo python3 -c "
from network_manager import NetworkManager
manager = NetworkManager()
manager.configure_interface({
    'name': 'eth0',
    'ip_address': '192.168.1.100',
    'netmask': '255.255.255.0',
    'gateway': '192.168.1.1'
})
"
```

## Configuration Files

### Netplan Configuration
Generated netplan configurations are displayed but not automatically written. To apply:

```bash
# Review generated configuration
# Copy to /etc/netplan/01-netcfg.yaml
sudo nano /etc/netplan/01-netcfg.yaml

# Apply configuration
sudo netplan apply
```

### DNS Configuration
DNS settings generate /etc/resolv.conf content. To apply:

```bash
# Backup current configuration
sudo cp /etc/resolv.conf /etc/resolv.conf.backup

# Apply new configuration (copy generated content)
sudo nano /etc/resolv.conf
```

### Firewall Rules
Firewall rules are applied immediately with iptables. To persist:

```bash
# Save iptables rules
sudo iptables-save > /etc/iptables/rules.v4

# Restore on boot (Ubuntu/Debian)
sudo apt-get install iptables-persistent
```

## Security Considerations

### Permissions
- Requires root/sudo access for network operations
- Use `dry_run=True` for testing without privileges
- Validate all input configurations before applying

### Best Practices
1. **Test in dry-run mode first**
2. **Backup current network configuration** before changes
3. **Use firewall rules** to restrict access
4. **Monitor bandwidth** for unusual activity
5. **Secure VPN keys** and certificates
6. **Log all network changes** for audit trails
7. **Validate IP addresses** and CIDR notation
8. **Test connectivity** after configuration changes

### Firewall Security
```python
# Example: Secure server setup
manager = NetworkManager()

# Default deny policy
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'DROP',
    'protocol': 'all',
    'comment': 'Default deny'
})

# Allow established connections
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'all',
    'state': 'ESTABLISHED,RELATED',
    'comment': 'Allow established'
})

# Allow loopback
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'all',
    'interface_in': 'lo',
    'comment': 'Allow loopback'
})

# Allow SSH from specific network
manager.add_firewall_rule({
    'chain': 'INPUT',
    'action': 'ACCEPT',
    'protocol': 'tcp',
    'source': '192.168.1.0/24',
    'dport': 22,
    'comment': 'Allow SSH from LAN'
})
```

## Performance Considerations

- **Bandwidth monitoring**: Uses `/sys/class/net/` for efficient statistics
- **Port scanning**: Configurable timeout to balance speed vs accuracy
- **Connection tracking**: Efficient parsing of ss/netstat output
- **Memory usage**: Maintains history in memory (configurable limits recommended)

## Troubleshooting

### Interface Configuration Issues
```python
# List all interfaces to verify
interfaces = manager.list_interfaces()

# Check routing table
routes = manager.show_routing_table()

# Test connectivity
ping_result = manager.ping('8.8.8.8')
```

### Firewall Issues
```python
# List current rules
rules = manager.list_firewall_rules()

# Check if rules are applied
import subprocess
subprocess.run(['sudo', 'iptables', '-L', '-n', '-v'])
```

### VPN Issues
```python
# Verify WireGuard interface
subprocess.run(['sudo', 'wg', 'show'])

# Check if interface is up
subprocess.run(['ip', 'link', 'show', 'wg0'])
```

## API Reference

### NetworkManager Class

#### Constructor
```python
NetworkManager(hostname: str = 'localhost', dry_run: bool = False)
```

#### Interface Management
- `configure_interface(interface_config: Dict) -> NetworkInterface`
- `list_interfaces() -> List[Dict]`

#### DNS and Routing
- `add_route(route_config: Dict) -> Route`
- `configure_dns(dns_config: Dict) -> Dict`
- `show_routing_table() -> List[Dict]`

#### Network Diagnostics
- `ping(target: str, count: int = 4, timeout: int = 5) -> Dict`
- `traceroute(target: str, max_hops: int = 30) -> Dict`
- `netstat(options: str = '-tuln') -> Dict`
- `get_active_connections() -> List[Connection]`

#### Bandwidth Monitoring
- `monitor_bandwidth(interface: str, duration: int = 5) -> BandwidthStats`
- `get_bandwidth_history(interface: str) -> List[BandwidthStats]`

#### Port Scanning
- `scan_port(host: str, port: int, timeout: float = 1.0) -> Dict`
- `scan_ports(host: str, ports: List[int], timeout: float = 1.0) -> Dict`
- `scan_common_ports(host: str) -> Dict`

#### VPN Configuration
- `setup_vpn_wireguard(vpn_config: Dict) -> Dict`
- `setup_vpn_openvpn(vpn_config: Dict) -> Dict`

#### Firewall Management
- `add_firewall_rule(rule_config: Dict) -> FirewallRule`
- `list_firewall_rules() -> List[Dict]`
- `setup_nat(nat_config: Dict) -> Dict`
- `setup_port_forwarding(forward_config: Dict) -> Dict`

#### Advanced Features
- `setup_bridge(bridge_config: Dict) -> Dict`
- `setup_vlan(vlan_config: Dict) -> Dict`
- `get_network_info() -> Dict`
- `export_config(filepath: str) -> bool`

## Use Cases

### 1. Server Deployment
Automated network configuration for new server deployments with standardized settings.

### 2. Network Monitoring
Real-time monitoring of bandwidth usage and connection tracking for performance analysis.

### 3. Security Auditing
Port scanning and firewall rule management for security compliance.

### 4. VPN Gateway
Configure and manage VPN servers (WireGuard, OpenVPN) for remote access.

### 5. Network Troubleshooting
Diagnostic tools (ping, traceroute, netstat) for network issue resolution.

### 6. Container Networking
Bridge and VLAN configuration for Docker, Kubernetes, or LXC containers.

### 7. Load Balancer Setup
NAT and port forwarding configuration for load balancing scenarios.

## Technologies

- **Python 3.8+**: Core implementation
- **iproute2**: Modern Linux networking (ip command)
- **iptables/nftables**: Firewall management
- **netplan**: Network configuration (Ubuntu/Debian)
- **WireGuard**: Modern VPN protocol
- **OpenVPN**: Traditional VPN solution
- **Socket API**: Port scanning and service detection

## Contributing

Contributions are welcome! Areas for improvement:
- Additional VPN protocols (IPSec, L2TP)
- Network bonding implementation
- Advanced routing (BGP, OSPF)
- IPv6 support enhancement
- Network performance testing tools
- GUI/Web interface

## License

Copyright (c) 2024 BrillConsulting. All rights reserved.

## Support

For issues, questions, or feature requests:
- Review code documentation and examples
- Check system logs for network operation errors
- Verify permissions (root/sudo required)
- Test in dry-run mode first

## Version History

### v2.0.0 (2024-11)
- Complete rewrite with production-ready features
- Added bandwidth monitoring with real-time statistics
- Implemented comprehensive port scanning
- Enhanced firewall management with rule builder
- Added connection tracking and analysis
- Improved VPN configuration (WireGuard, OpenVPN)
- Added network diagnostics (ping, traceroute, netstat)
- Configuration export/import functionality
- Dry-run mode for safe testing
- Comprehensive error handling and logging

### v1.0.0 (Initial)
- Basic interface configuration
- Simple routing and DNS management
- Basic VPN support
- Minimal firewall integration
