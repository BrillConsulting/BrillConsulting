/**
 * React Dashboard Application
 * Modern admin dashboard with real-time data visualization
 */

import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import './App.css';

// Types
interface MetricData {
  name: string;
  value: number;
  change: number;
}

interface ChartData {
  name: string;
  sales: number;
  revenue: number;
  users: number;
}

interface User {
  id: number;
  name: string;
  email: string;
  status: 'active' | 'inactive';
  lastLogin: string;
}

// Custom Hooks
const useWebSocket = (url: string) => {
  const [data, setData] = useState<any>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket(url);

    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => setData(JSON.parse(event.data));
    ws.onclose = () => setIsConnected(false);

    return () => ws.close();
  }, [url]);

  return { data, isConnected };
};

// Components
const MetricCard: React.FC<{ metric: MetricData }> = ({ metric }) => {
  const isPositive = metric.change >= 0;

  return (
    <div className="metric-card">
      <h3>{metric.name}</h3>
      <div className="metric-value">{metric.value.toLocaleString()}</div>
      <div className={`metric-change ${isPositive ? 'positive' : 'negative'}`}>
        {isPositive ? '↑' : '↓'} {Math.abs(metric.change)}%
      </div>
    </div>
  );
};

const SalesChart: React.FC<{ data: ChartData[] }> = ({ data }) => {
  return (
    <div className="chart-container">
      <h3>Sales Overview</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="sales" stroke="#8884d8" strokeWidth={2} />
          <Line type="monotone" dataKey="revenue" stroke="#82ca9d" strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

const RevenueChart: React.FC<{ data: ChartData[] }> = ({ data }) => {
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  return (
    <div className="chart-container">
      <h3>Revenue Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="revenue"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
};

const UserTable: React.FC<{ users: User[] }> = ({ users }) => {
  const [sortField, setSortField] = useState<keyof User>('name');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const sortedUsers = [...users].sort((a, b) => {
    const aVal = a[sortField];
    const bVal = b[sortField];

    if (aVal < bVal) return sortDirection === 'asc' ? -1 : 1;
    if (aVal > bVal) return sortDirection === 'asc' ? 1 : -1;
    return 0;
  });

  const handleSort = (field: keyof User) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  return (
    <div className="table-container">
      <h3>Recent Users</h3>
      <table className="user-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('name')}>Name {sortField === 'name' && (sortDirection === 'asc' ? '↑' : '↓')}</th>
            <th onClick={() => handleSort('email')}>Email {sortField === 'email' && (sortDirection === 'asc' ? '↑' : '↓')}</th>
            <th onClick={() => handleSort('status')}>Status {sortField === 'status' && (sortDirection === 'asc' ? '↑' : '↓')}</th>
            <th onClick={() => handleSort('lastLogin')}>Last Login {sortField === 'lastLogin' && (sortDirection === 'asc' ? '↑' : '↓')}</th>
          </tr>
        </thead>
        <tbody>
          {sortedUsers.map(user => (
            <tr key={user.id}>
              <td>{user.name}</td>
              <td>{user.email}</td>
              <td>
                <span className={`status-badge ${user.status}`}>
                  {user.status}
                </span>
              </td>
              <td>{new Date(user.lastLogin).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// Main App
const App: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricData[]>([
    { name: 'Total Sales', value: 125430, change: 12.5 },
    { name: 'Revenue', value: 89250, change: 8.3 },
    { name: 'Users', value: 5432, change: -2.1 },
    { name: 'Conversion', value: 3.2, change: 15.7 }
  ]);

  const [chartData, setChartData] = useState<ChartData[]>([
    { name: 'Jan', sales: 4000, revenue: 2400, users: 240 },
    { name: 'Feb', sales: 3000, revenue: 1398, users: 221 },
    { name: 'Mar', sales: 2000, revenue: 9800, users: 229 },
    { name: 'Apr', sales: 2780, revenue: 3908, users: 200 },
    { name: 'May', sales: 1890, revenue: 4800, users: 218 },
    { name: 'Jun', sales: 2390, revenue: 3800, users: 250 }
  ]);

  const [users, setUsers] = useState<User[]>([
    { id: 1, name: 'John Doe', email: 'john@example.com', status: 'active', lastLogin: '2024-01-15T10:30:00' },
    { id: 2, name: 'Jane Smith', email: 'jane@example.com', status: 'active', lastLogin: '2024-01-15T09:15:00' },
    { id: 3, name: 'Bob Johnson', email: 'bob@example.com', status: 'inactive', lastLogin: '2024-01-10T14:20:00' },
    { id: 4, name: 'Alice Williams', email: 'alice@example.com', status: 'active', lastLogin: '2024-01-15T11:45:00' }
  ]);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => prev.map(metric => ({
        ...metric,
        value: metric.value + Math.floor(Math.random() * 100 - 50),
        change: +(Math.random() * 20 - 10).toFixed(1)
      })));
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Analytics Dashboard</h1>
        <div className="header-actions">
          <button className="btn btn-primary">Export</button>
          <button className="btn btn-secondary">Settings</button>
        </div>
      </header>

      <div className="dashboard">
        <section className="metrics-section">
          {metrics.map((metric, index) => (
            <MetricCard key={index} metric={metric} />
          ))}
        </section>

        <section className="charts-section">
          <div className="chart-row">
            <SalesChart data={chartData} />
            <RevenueChart data={chartData} />
          </div>
        </section>

        <section className="users-section">
          <UserTable users={users} />
        </section>
      </div>
    </div>
  );
};

export default App;
