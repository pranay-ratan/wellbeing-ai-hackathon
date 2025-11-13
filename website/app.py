#!/usr/bin/env python3
"""
WellbeingAI Web Application

A modern web interface for the WellbeingAI mental health system,
featuring beautiful dashboards, real-time analytics, and user-friendly
interfaces for employees, managers, and administrators.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import json
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try to import AI agents, but provide fallbacks if not available
try:
    from src.agents.supervisor_agent import SupervisorAgent
    from src.agents.analytics_agent import AnalyticsAgent
    AGENTS_AVAILABLE = True
except ImportError:
    print("⚠️  AI agents not available - running in demo mode")
    AGENTS_AVAILABLE = False
    SupervisorAgent = None
    AnalyticsAgent = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__,
            static_folder='static',
            template_folder='templates')
CORS(app)
app.secret_key = 'wellbeing-ai-secret-key-2025'

# Initialize AI agents
supervisor_agent = None
analytics_agent = None

def initialize_agents():
    """Initialize the AI agents"""
    global supervisor_agent, analytics_agent
    try:
        if AGENTS_AVAILABLE and SupervisorAgent and AnalyticsAgent:
            supervisor_agent = SupervisorAgent()
            analytics_agent = AnalyticsAgent()
            logger.info("✅ AI agents initialized successfully")
            return True
        else:
            logger.info("ℹ️  Running in demo mode - AI agents not available")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to initialize agents: {e}")
        return False

@app.route('/')
def index():
    """Main landing page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Main dashboard - redirects based on user role"""
    user_role = session.get('user_role', 'employee')
    if user_role == 'employee':
        return redirect(url_for('employee_dashboard'))
    elif user_role == 'manager':
        return redirect(url_for('manager_dashboard'))
    elif user_role == 'admin':
        return redirect(url_for('admin_dashboard'))
    else:
        return redirect(url_for('employee_dashboard'))

@app.route('/employee/dashboard')
def employee_dashboard():
    """Employee personal dashboard"""
    session['user_role'] = 'employee'
    return render_template('employee_dashboard.html')

@app.route('/manager/dashboard')
def manager_dashboard():
    """Manager team dashboard"""
    session['user_role'] = 'manager'
    return render_template('manager_dashboard.html')

@app.route('/admin/dashboard')
def admin_dashboard():
    """Administrator organization dashboard"""
    session['user_role'] = 'admin'
    return render_template('admin_dashboard.html')

@app.route('/checkin')
def checkin():
    """Daily check-in page"""
    return render_template('checkin.html')

@app.route('/analytics')
def analytics():
    """Analytics and insights page"""
    return render_template('analytics.html')

# API Endpoints

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    agents_status = "operational" if supervisor_agent else "initializing"
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": agents_status,
        "version": "1.0.0"
    })

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language queries"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'web_user')
        user_role = data.get('user_role', 'employee')
        query = data.get('query', '')
        department = data.get('department')

        if not supervisor_agent:
            return jsonify({"error": "AI agents not initialized"}), 503

        response = supervisor_agent.process_query(
            user_id=user_id,
            user_role=user_role,
            query=query,
            department=department
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({"error": "Query processing failed", "message": str(e)}), 500

@app.route('/api/dashboard/<role>')
def get_dashboard_data(role):
    """Get dashboard data for specific role"""
    try:
        if not analytics_agent:
            # Return mock data for demo purposes
            return jsonify(get_mock_dashboard_data(role))

        user_id = f"dashboard_user_{role}"
        department = "Engineering" if role == "manager" else None

        dashboard_data = analytics_agent.get_dashboard_data(
            user_id=user_id,
            user_role=role,
            department=department
        )

        return jsonify(dashboard_data)

    except Exception as e:
        logger.error(f"Dashboard data error: {e}")
        return jsonify({"error": "Failed to load dashboard data"}), 500

@app.route('/api/checkin/submit', methods=['POST'])
def submit_checkin():
    """Submit daily check-in"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'web_user')

        # Import here to avoid circular imports
        from src.agents.checkin_agent import CheckInAgent
        checkin_agent = CheckInAgent()

        result = checkin_agent.submit_checkin(user_id, data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Check-in submission error: {e}")
        return jsonify({"error": "Check-in submission failed"}), 500

@app.route('/api/risk/score', methods=['POST'])
def get_risk_score():
    """Get risk score for user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'web_user')

        # Import here to avoid circular imports
        from src.agents.risk_scorer_agent import RiskScorerAgent
        risk_agent = RiskScorerAgent()

        risk_score = risk_agent.score_user_risk(user_id)

        return jsonify({
            "risk_score": risk_score.risk_score if hasattr(risk_score, 'risk_score') else risk_score.get('risk_score', 0.5),
            "risk_level": risk_score.risk_level if hasattr(risk_score, 'risk_level') else "moderate",
            "contributing_factors": risk_score.contributing_factors if hasattr(risk_score, 'contributing_factors') else [],
            "recommended_action": risk_score.recommended_action if hasattr(risk_score, 'recommended_action') else "Monitor regularly"
        })

    except Exception as e:
        logger.error(f"Risk score error: {e}")
        return jsonify({"error": "Risk assessment failed"}), 500

@app.route('/api/interventions/recommend', methods=['POST'])
def get_interventions():
    """Get personalized interventions"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'web_user')

        # Import here to avoid circular imports
        from src.agents.intervention_agent import InterventionAgent
        intervention_agent = InterventionAgent()

        # Get user's risk for personalization
        from src.agents.risk_scorer_agent import RiskScorerAgent
        risk_agent = RiskScorerAgent()
        risk_assessment = risk_agent.score_user_risk(user_id)

        risk_score = risk_assessment.risk_score if hasattr(risk_assessment, 'risk_score') else 0.5
        factors = risk_assessment.contributing_factors if hasattr(risk_assessment, 'contributing_factors') else []

        interventions = intervention_agent.recommend_interventions(
            user_id, risk_score, factors
        )

        return jsonify({"interventions": interventions})

    except Exception as e:
        logger.error(f"Intervention recommendation error: {e}")
        return jsonify({"error": "Intervention recommendation failed"}), 500

@app.route('/api/privacy/report', methods=['POST'])
def get_privacy_report():
    """Get privacy-preserving report"""
    try:
        data = request.get_json()
        user_role = data.get('user_role', 'employee')
        department = data.get('department')

        # Import here to avoid circular imports
        from src.agents.privacy_agent import PrivacyPreservingAgent
        privacy_agent = PrivacyPreservingAgent()

        report = privacy_agent.generate_safe_report(
            user_role=user_role,
            department=department
        )

        return jsonify(report)

    except Exception as e:
        logger.error(f"Privacy report error: {e}")
        return jsonify({"error": "Privacy report generation failed"}), 500

def get_mock_dashboard_data(role):
    """Generate mock dashboard data for demo purposes"""
    from datetime import datetime, timedelta
    import random

    base_date = datetime.now()

    if role == 'employee':
        return {
            "trend_data": {
                "current_values": {
                    "mood": random.randint(6, 9),
                    "stress": random.randint(2, 6),
                    "energy": random.randint(6, 9),
                    "social_contact": random.randint(3, 5),
                    "sleep": round(random.uniform(7.0, 9.0), 1),
                    "isolation": random.choice([True, False])
                },
                "trends": {
                    "mood": random.choice(['improving', 'stable', 'declining']),
                    "stress": random.choice(['improving', 'stable', 'declining'])
                },
                "weekly_averages": [
                    {
                        "date": (base_date - timedelta(days=i*7)).strftime('%Y-%m-%d'),
                        "mood_7day_avg": round(random.uniform(6.0, 8.5), 1),
                        "stress_7day_avg": round(random.uniform(2.5, 5.5), 1)
                    } for i in range(12, -1, -1)
                ]
            },
            "current_risk": {
                "risk_score": round(random.uniform(0.2, 0.7), 2),
                "risk_level": random.choice(['low_risk', 'moderate_risk', 'elevated_risk']),
                "contributing_factors": random.sample([
                    "Recent stress increase",
                    "Reduced social contact",
                    "Irregular sleep patterns",
                    "High workload perception"
                ], random.randint(1, 3)),
                "recommended_action": "Monitor regularly and consider stress management techniques"
            },
            "insights": [
                "Your mood has been stable this week - keep up the good work!",
                "Consider increasing social connections for better wellbeing",
                "Your check-in consistency is excellent"
            ]
        }
    elif role == 'manager':
        return {
            "team_stats": {
                "total_members": random.randint(8, 15),
                "active_today": random.randint(6, 12),
                "avg_mood": round(random.uniform(6.5, 8.0), 1),
                "mood_trend": random.choice(['improving', 'stable', 'declining']),
                "high_risk_count": random.randint(1, 4),
                "risk_change": random.randint(-2, 2),
                "checkin_rate": round(random.uniform(75, 95), 1),
                "participation_rate": random.randint(70, 90)
            },
            "mood_distribution": {
                "very_low": random.randint(0, 2),
                "low": random.randint(1, 3),
                "moderate": random.randint(2, 5),
                "high": random.randint(3, 6),
                "very_high": random.randint(1, 4)
            },
            "risk_distribution": {
                "low_risk": random.randint(5, 8),
                "moderate_risk": random.randint(2, 4),
                "elevated_risk": random.randint(1, 3),
                "high_risk": random.randint(0, 2)
            },
            "team_trends": [
                {
                    "date": (base_date - timedelta(days=i)).strftime('%Y-%m-%d'),
                    "avg_mood": round(random.uniform(6.5, 8.0), 1),
                    "avg_stress": round(random.uniform(3.0, 5.5), 1)
                } for i in range(29, -1, -1)
            ],
            "team_insights": [
                "Team mood has improved 15% this month",
                f"{random.randint(2, 4)} team members may need additional support",
                "Consider team-building activities to boost morale",
                f"Check-in participation is at {random.randint(75, 85)}% - good job!"
            ],
            "team_members": [
                {
                    "name": "Alice Johnson",
                    "department": "Engineering",
                    "current_mood": random.randint(6, 9),
                    "risk_level": random.choice(['low_risk', 'moderate_risk', 'elevated_risk', 'high_risk']),
                    "last_checkin": (base_date - timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d'),
                    "streak": random.randint(5, 15)
                },
                {
                    "name": "Bob Smith",
                    "department": "Engineering",
                    "current_mood": random.randint(5, 8),
                    "risk_level": random.choice(['low_risk', 'moderate_risk']),
                    "last_checkin": (base_date - timedelta(days=random.randint(0, 1))).strftime('%Y-%m-%d'),
                    "streak": random.randint(8, 12)
                },
                {
                    "name": "Carol Davis",
                    "department": "Design",
                    "current_mood": random.randint(4, 7),
                    "risk_level": random.choice(['moderate_risk', 'elevated_risk', 'high_risk']),
                    "last_checkin": (base_date - timedelta(days=random.randint(1, 3))).strftime('%Y-%m-%d'),
                    "streak": random.randint(3, 8)
                }
            ]
        }
    else:
        return {}

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("🚀 Starting WellbeingAI Web Application...")
    print("🤖 Initializing AI agents...")

    agents_initialized = initialize_agents()

    if agents_initialized:
        print("✅ AI agents ready!")
    else:
        print("ℹ️  Running in demo mode with mock data")
        print("💡 To enable full AI functionality, configure config/config.yaml")

    print("🌐 Starting web server on http://localhost:8000")
    print("📊 Dashboard available at http://localhost:8000/dashboard")
    print("🩺 Check-in available at http://localhost:8000/checkin")
    print("🏠 Landing page at http://localhost:8000/")

    app.run(debug=True, host='0.0.0.0', port=8000)
