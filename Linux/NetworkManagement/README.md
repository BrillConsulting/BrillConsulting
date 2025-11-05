# Linux Network Management

Complete network configuration, routing, DNS, and VPN management for Linux systems.

## Features

- **Interface Configuration**: Static/DHCP network interface setup with netplan
- **Routing**: Static routes, policy-based routing
- **DNS Management**: resolv.conf configuration, search domains
- **Network Bridges**: Bridge interfaces for VMs and containers
- **VLAN Support**: 802.1Q VLAN configuration
- **Bonding**: Link aggregation (active-backup, balance-rr, 802.3ad)
- **NAT**: Network Address Translation and IP forwarding
- **VPN**: WireGuard VPN configuration
- **Connectivity Testing**: ping, traceroute, mtr, netcat

## Technologies

- Netplan, NetworkManager
- iproute2 (ip command)
- iptables/nftables
- WireGuard VPN
- Bridge utils, VLAN tools
- Bonding drivers

## Use Cases

- Server network configuration
- Router and gateway setup
- VPN server deployment
- Network isolation with VLANs
- High availability with bonding
- Container/VM networking

## Implementation

Complete Python implementation with:
- Interface configuration generators
- Route management
- DNS configuration
- Bridge/VLAN/Bonding setup
- NAT and firewall rules
- VPN configuration
- Network diagnostics
