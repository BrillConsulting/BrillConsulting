# React Analytics Dashboard

Modern, responsive analytics dashboard built with React 18, TypeScript, and Recharts.

## Features

- **Real-time Metrics**: Live updating KPIs with trend indicators
- **Interactive Charts**: Line charts, pie charts, and bar charts using Recharts
- **Sortable Tables**: Click-to-sort user table with status badges
- **Responsive Design**: Mobile-first design that works on all devices
- **TypeScript**: Fully typed for better development experience
- **Modern UI**: Clean, professional interface with smooth animations

## Tech Stack

- **React 18** - Latest React with hooks and concurrent features
- **TypeScript** - Type-safe development
- **Recharts** - Composable charting library
- **Vite** - Lightning-fast build tool
- **CSS3** - Modern styling with CSS variables

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) to view in browser.

### Build

```bash
npm run build
```

### Test

```bash
npm test
```

## Project Structure

```
ReactDashboard/
├── App.tsx           # Main application component
├── App.css           # Styling
├── package.json      # Dependencies
└── README.md         # Documentation
```

## Components

### MetricCard
Displays KPI metrics with trend indicators.

### SalesChart
Line chart showing sales and revenue trends over time.

### RevenueChart
Pie chart displaying revenue distribution.

### UserTable
Sortable table with user information and status badges.

## Usage Example

```tsx
import App from './App';

function Dashboard() {
  return <App />;
}
```

## Features in Detail

### Real-time Updates
Data automatically refreshes every 5 seconds to simulate live dashboard.

### Responsive Charts
All charts are responsive and adapt to different screen sizes using ResponsiveContainer.

### Sortable Tables
Click column headers to sort by that field. Click again to reverse sort direction.

### Type Safety
Full TypeScript support ensures type safety throughout the application.

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

MIT
