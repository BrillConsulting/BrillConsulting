// Configuration
const colors = {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#10b981',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#3b82f6',
    light: '#f3f4f6'
};

const chartColors = [
    '#667eea', '#764ba2', '#f093fb', '#4facfe',
    '#43e97b', '#fa709a', '#fee140', '#30cfd0'
];

// Generate random data
function generateData(count, min = 0, max = 100) {
    return Array.from({length: count}, () => Math.floor(Math.random() * (max - min + 1)) + min);
}

// Line Chart
const lineCtx = document.getElementById('lineChart').getContext('2d');
const lineChart = new Chart(lineCtx, {
    type: 'line',
    data: {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
        datasets: [{
            label: 'Sales 2024',
            data: generateData(12, 30, 90),
            borderColor: colors.primary,
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            tension: 0.4,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7
        }, {
            label: 'Sales 2023',
            data: generateData(12, 20, 70),
            borderColor: colors.secondary,
            backgroundColor: 'rgba(118, 75, 162, 0.1)',
            tension: 0.4,
            fill: true,
            pointRadius: 5,
            pointHoverRadius: 7
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: true,
        plugins: {
            legend: {
                display: true,
                position: 'top'
            },
            title: {
                display: false
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Doughnut Chart
const doughnutCtx = document.getElementById('doughnutChart').getContext('2d');
const doughnutChart = new Chart(doughnutCtx, {
    type: 'doughnut',
    data: {
        labels: ['Electronics', 'Clothing', 'Food', 'Books', 'Sports'],
        datasets: [{
            data: generateData(5, 15, 40),
            backgroundColor: chartColors.slice(0, 5),
            borderWidth: 2,
            borderColor: '#fff'
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'right'
            }
        }
    }
});

// Bar Chart
const barCtx = document.getElementById('barChart').getContext('2d');
const barChart = new Chart(barCtx, {
    type: 'bar',
    data: {
        labels: ['Q1', 'Q2', 'Q3', 'Q4'],
        datasets: [{
            label: 'Revenue',
            data: generateData(4, 50, 100),
            backgroundColor: colors.primary,
            borderColor: colors.primary,
            borderWidth: 1
        }, {
            label: 'Costs',
            data: generateData(4, 30, 70),
            backgroundColor: colors.danger,
            borderColor: colors.danger,
            borderWidth: 1
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                display: true
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Radar Chart
const radarCtx = document.getElementById('radarChart').getContext('2d');
const radarChart = new Chart(radarCtx, {
    type: 'radar',
    data: {
        labels: ['Speed', 'Quality', 'Cost', 'Innovation', 'Support', 'Reliability'],
        datasets: [{
            label: 'Product A',
            data: generateData(6, 60, 95),
            borderColor: colors.primary,
            backgroundColor: 'rgba(102, 126, 234, 0.2)',
            pointBackgroundColor: colors.primary
        }, {
            label: 'Product B',
            data: generateData(6, 50, 90),
            borderColor: colors.secondary,
            backgroundColor: 'rgba(118, 75, 162, 0.2)',
            pointBackgroundColor: colors.secondary
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'top'
            }
        },
        scales: {
            r: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

// Polar Area Chart
const polarCtx = document.getElementById('polarChart').getContext('2d');
const polarChart = new Chart(polarCtx, {
    type: 'polarArea',
    data: {
        labels: ['North', 'South', 'East', 'West', 'Central'],
        datasets: [{
            data: generateData(5, 20, 80),
            backgroundColor: chartColors.slice(0, 5).map(color => color + '80')
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'right'
            }
        }
    }
});

// Pie Chart
const pieCtx = document.getElementById('pieChart').getContext('2d');
const pieChart = new Chart(pieCtx, {
    type: 'pie',
    data: {
        labels: ['USA', 'Europe', 'Asia', 'Africa', 'Australia'],
        datasets: [{
            data: generateData(5, 10, 50),
            backgroundColor: chartColors.slice(0, 5),
            borderWidth: 2,
            borderColor: '#fff'
        }]
    },
    options: {
        responsive: true,
        plugins: {
            legend: {
                position: 'bottom'
            }
        }
    }
});

// Refresh functionality
document.querySelectorAll('.refresh-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        const chartType = this.getAttribute('data-chart');
        let chart;

        switch(chartType) {
            case 'line':
                chart = lineChart;
                chart.data.datasets.forEach(dataset => {
                    dataset.data = generateData(12, 30, 90);
                });
                break;
            case 'doughnut':
                chart = doughnutChart;
                chart.data.datasets[0].data = generateData(5, 15, 40);
                break;
            case 'bar':
                chart = barChart;
                chart.data.datasets.forEach(dataset => {
                    dataset.data = generateData(4, 30, 100);
                });
                break;
            case 'radar':
                chart = radarChart;
                chart.data.datasets.forEach(dataset => {
                    dataset.data = generateData(6, 50, 95);
                });
                break;
            case 'polar':
                chart = polarChart;
                chart.data.datasets[0].data = generateData(5, 20, 80);
                break;
            case 'pie':
                chart = pieChart;
                chart.data.datasets[0].data = generateData(5, 10, 50);
                break;
        }

        if (chart) {
            chart.update('active');
        }
    });
});

// Auto-refresh every 5 seconds (optional - uncomment to enable)
// setInterval(() => {
//     document.querySelectorAll('.refresh-btn').forEach(btn => btn.click());
// }, 5000);
