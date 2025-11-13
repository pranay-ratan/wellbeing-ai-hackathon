# WellbeingAI Web Application

A modern, beautiful web interface for the WellbeingAI mental health crisis prevention system. Built with Flask, Bootstrap 5, and Chart.js for a responsive, user-friendly experience.

## 🚀 Features

- **Multi-Role Dashboards**: Separate interfaces for employees, managers, and administrators
- **Real-Time Analytics**: Live charts and insights powered by the AI agents
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Privacy-First**: Built with GDPR compliance and HIPAA readiness in mind
- **Modern UI**: Clean, professional interface using Bootstrap 5 and custom styling
- **Interactive Charts**: Beautiful visualizations using Chart.js
- **Real-Time Updates**: Live data updates and notifications

## 🏗️ Architecture

```
website/
├── app.py                 # Flask application with API endpoints
├── requirements.txt       # Python dependencies
├── templates/            # Jinja2 HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Landing page
│   ├── employee_dashboard.html
│   ├── manager_dashboard.html
│   └── admin_dashboard.html
├── static/               # Static assets
│   ├── css/
│   │   └── style.css     # Custom styling
│   ├── js/
│   │   └── main.js       # Frontend JavaScript
│   └── images/           # Images and icons
└── README.md
```

## 🛠️ Installation

1. **Install Dependencies**
   ```bash
   cd website
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   # Copy and edit configuration
   cp ../config/config.yaml.template ../config/config.yaml
   # Edit config.yaml with your Databricks credentials
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```

4. **Open in Browser**
   ```
   http://localhost:5000
   ```

## 🎨 Dashboard Elements

The application includes comprehensive dashboard components:

### Employee Dashboard
- **Personal Metrics Cards**: Mood, stress, risk score, check-in streak
- **90-Day Trends Chart**: Line chart showing wellbeing patterns
- **Risk Factors Panel**: Current risk indicators and contributing factors
- **Intervention Recommendations**: Personalized AI suggestions
- **Personal Insights**: AI-generated wellbeing insights
- **Recent Check-ins Table**: Historical check-in data

### Manager Dashboard
- **Team Overview Cards**: Team size, average mood, high-risk count, check-in rate
- **Team Mood Distribution**: Doughnut chart of team mood levels
- **Risk Level Breakdown**: Bar chart of risk distribution
- **Team Trends Chart**: 30-day team wellbeing trends
- **Team Insights Panel**: AI insights about team patterns
- **Team Member Table**: Individual team member overview with actions

### Key Dashboard Components

#### 1. Metrics Cards
```html
<div class="card h-100 border-left-primary">
    <div class="card-body">
        <div class="d-flex align-items-center">
            <div class="flex-grow-1">
                <div class="text-xs font-weight-bold text-primary text-uppercase mb-1">
                    Current Mood
                </div>
                <div class="h5 mb-0 font-weight-bold text-gray-800" id="currentMood">--</div>
                <div class="text-xs text-muted mt-1" id="moodChange">Loading...</div>
            </div>
            <div class="ms-3">
                <i class="fas fa-smile fa-2x text-primary"></i>
            </div>
        </div>
    </div>
</div>
```

#### 2. Interactive Charts
```javascript
const wellbeingChart = new Chart(ctx, {
    type: 'line',
    data: {
        labels: weeklyData.map(d => new Date(d.date).toLocaleDateString()),
        datasets: [{
            label: 'Mood (7-day avg)',
            data: weeklyData.map(d => d.mood_7day_avg),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            tension: 0.4
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: { beginAtZero: true, max: 10 }
        }
    }
});
```

#### 3. Risk Assessment Panel
```html
<div class="card shadow h-100">
    <div class="card-header py-3">
        <h6 class="m-0 font-weight-bold text-warning">
            <i class="fas fa-exclamation-triangle me-1"></i>Risk Factors
        </h6>
    </div>
    <div class="card-body">
        <div id="riskFactors" class="text-center py-4">
            <!-- Dynamic risk factors loaded here -->
        </div>
    </div>
</div>
```

#### 4. Intervention Recommendations
```html
<div class="card shadow h-100">
    <div class="card-header py-3">
        <h6 class="m-0 font-weight-bold text-success">
            <i class="fas fa-lightbulb me-1"></i>Recommended Interventions
        </h6>
    </div>
    <div class="card-body">
        <div id="interventionsList">
            <!-- AI-generated interventions -->
        </div>
    </div>
</div>
```

## 🔌 API Endpoints

The Flask application provides RESTful API endpoints:

- `GET /api/health` - Health check
- `POST /api/query` - Natural language queries
- `GET /api/dashboard/<role>` - Dashboard data for specific roles
- `POST /api/checkin/submit` - Submit daily check-ins
- `POST /api/risk/score` - Get risk scores
- `POST /api/interventions/recommend` - Get intervention recommendations
- `POST /api/privacy/report` - Generate privacy-preserving reports

## 🎯 Usage Examples

### Starting the Web Application
```python
from website.app import app

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### Using Dashboard Components
```javascript
// Load employee dashboard data
async function loadDashboardData() {
    const response = await fetch('/api/dashboard/employee');
    const data = await response.json();

    if (data.error) {
        showError('Failed to load dashboard data: ' + data.error);
        return;
    }

    updateDashboard(data);
}
```

## 📱 Responsive Design

The application is fully responsive and works on:
- **Desktop**: Full feature set with large charts and tables
- **Tablet**: Optimized layouts with touch-friendly controls
- **Mobile**: Streamlined interface with collapsible navigation

## 🔒 Security Features

- **CSRF Protection**: All forms protected against cross-site request forgery
- **Input Validation**: Client and server-side validation
- **Secure Headers**: Security headers configured
- **Session Management**: Secure session handling
- **Privacy Compliance**: GDPR and HIPAA-ready architecture

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
gunicorn --bind 0.0.0.0:8000 --workers 4 app:app
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License
