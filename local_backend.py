#!/usr/bin/env python3
"""
Local Backend for WellbeingAI - Simplified for Hackathon Demo

This runs all AI agents locally with in-memory data for quick demo.
No Databricks required - everything works locally!
"""

import sys
import os
from pathlib import Path
from flask import Flask, request, jsonify, session, redirect, url_for, render_template
from flask_cors import CORS
import json
import random
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Mock AI agents for demo (since full agents need Databricks)
class MockSupervisorAgent:
    def process_query(self, user_id, user_role, query, department=None):
        return {
            "response": f"AI Analysis: Based on your query '{query}', I recommend monitoring your stress levels and considering a short break.",
            "confidence": 0.85,
            "recommendations": ["Take 5 deep breaths", "Go for a 10-minute walk", "Talk to a colleague"]
        }

class MockAnalyticsAgent:
    def get_employee_dashboard(self, user_id):
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
                        "date": (datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d'),
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

    def get_manager_dashboard(self, user_id, department):
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
                    "date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
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
                    "department": department or "Engineering",
                    "current_mood": random.randint(6, 9),
                    "risk_level": random.choice(['low_risk', 'moderate_risk', 'elevated_risk', 'high_risk']),
                    "last_checkin": (datetime.now() - timedelta(days=random.randint(0, 2))).strftime('%Y-%m-%d'),
                    "streak": random.randint(5, 15)
                },
                {
                    "name": "Bob Smith",
                    "department": department or "Engineering",
                    "current_mood": random.randint(5, 8),
                    "risk_level": random.choice(['low_risk', 'moderate_risk']),
                    "last_checkin": (datetime.now() - timedelta(days=random.randint(0, 1))).strftime('%Y-%m-%d'),
                    "streak": random.randint(8, 12)
                },
                {
                    "name": "Carol Davis",
                    "department": department or "Design",
                    "current_mood": random.randint(4, 7),
                    "risk_level": random.choice(['moderate_risk', 'elevated_risk', 'high_risk']),
                    "last_checkin": (datetime.now() - timedelta(days=random.randint(1, 3))).strftime('%Y-%m-%d'),
                    "streak": random.randint(3, 8)
                }
            ]
        }

# Initialize Flask app
app = Flask(__name__,
            static_folder='website/static',
            template_folder='website/templates')
CORS(app)
app.secret_key = 'wellbeing-ai-demo-secret-key'

# Add route for static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return app.send_static_file(filename)

# Initialize mock agents
supervisor_agent = MockSupervisorAgent()
analytics_agent = MockAnalyticsAgent()

# In-memory storage for demo
checkins_data = []

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

@app.route('/checkin')
def checkin():
    """Daily check-in page"""
    return render_template('checkin.html')

@app.route('/analytics')
def analytics():
    """Analytics page"""
    return render_template('analytics.html')

# API Endpoints

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents": "mock_agents_active",
        "version": "1.0.0-demo",
        "mode": "local_backend"
    })

@app.route('/api/dashboard/<role>')
def get_dashboard_data(role):
    """Get dashboard data for specific role"""
    try:
        if role == 'employee':
            user_id = request.args.get('user_id', 'demo_user')
            data = analytics_agent.get_employee_dashboard(user_id)
        elif role == 'manager':
            user_id = request.args.get('user_id', 'demo_manager')
            department = request.args.get('department', 'Engineering')
            data = analytics_agent.get_manager_dashboard(user_id, department)
        else:
            data = analytics_agent.get_employee_dashboard('admin_user')

        return jsonify(data)

    except Exception as e:
        print(f"Dashboard error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process natural language queries"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'demo_user')
        user_role = data.get('user_role', 'employee')
        query = data.get('query', '')
        department = data.get('department')

        response = supervisor_agent.process_query(
            user_id=user_id,
            user_role=user_role,
            query=query,
            department=department
        )

        return jsonify(response)

    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/checkin/submit', methods=['POST'])
def submit_checkin():
    """Submit daily check-in"""
    try:
        data = request.get_json()

        # Store checkin data
        checkin_record = {
            "id": len(checkins_data) + 1,
            "user_id": data.get('user_id', 'demo_user'),
            "timestamp": datetime.now().isoformat(),
            **data
        }
        checkins_data.append(checkin_record)

        print(f"✅ Check-in submitted: {checkin_record}")

        return jsonify({
            "success": True,
            "message": "Check-in submitted successfully!",
            "checkin_id": checkin_record["id"]
        })

    except Exception as e:
        print(f"Check-in error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/risk/score', methods=['POST'])
def get_risk_score():
    """Get risk score for user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'demo_user')

        # Mock risk assessment
        risk_score = round(random.uniform(0.1, 0.8), 2)
        risk_level = "low_risk" if risk_score < 0.3 else "moderate_risk" if risk_score < 0.6 else "high_risk"

        return jsonify({
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_factors": ["Recent check-in patterns", "Historical trends"],
            "recommended_action": "Continue monitoring and maintain regular check-ins"
        })

    except Exception as e:
        print(f"Risk score error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/interventions/recommend', methods=['POST'])
def get_interventions():
    """Get personalized interventions"""
    try:
        interventions = [
            {
                "type": "breathing_exercise",
                "description": "Practice the 4-7-8 breathing technique for 5 minutes daily",
                "success_rate": 0.78,
                "time_commitment": "5 minutes"
            },
            {
                "type": "social_connection",
                "description": "Reach out to a colleague or friend for meaningful conversation",
                "success_rate": 0.82,
                "time_commitment": "30 minutes"
            },
            {
                "type": "physical_activity",
                "description": "Take a 20-minute walk during lunch break",
                "success_rate": 0.75,
                "time_commitment": "20 minutes"
            }
        ]

        return jsonify({"interventions": interventions[:3]})

    except Exception as e:
        print(f"Interventions error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/privacy/report', methods=['POST'])
def get_privacy_report():
    """Get privacy-preserving report"""
    try:
        data = request.get_json()
        user_role = data.get('user_role', 'employee')
        department = data.get('department')

        # Mock privacy report
        report = {
            "total_users": random.randint(50, 200),
            "anonymized_insights": [
                f"{random.randint(60, 90)}% of users in {department or 'Engineering'} report good wellbeing",
                f"{random.randint(10, 30)}% show signs of elevated stress",
                "All data is k-anonymized with minimum group size of 5"
            ],
            "compliance_status": "GDPR Compliant • HIPAA Ready",
            "last_updated": datetime.now().isoformat()
        }

        return jsonify(report)

    except Exception as e:
        print(f"Privacy report error: {e}")
        return jsonify({"error": str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("🚀 Starting WellbeingAI Local Backend...")
    print("🤖 Using mock AI agents for demo")
    print("📊 In-memory data storage")
    print("🌐 Starting web server on http://localhost:8080")
    print("📱 Dashboard available at http://localhost:8080/dashboard")
    print("🩺 Check-in available at http://localhost:8080/checkin")
    print("🏠 Landing page at http://localhost:8080/")
    print("")
    print("🎯 READY FOR HACKATHON DEMO!")

    app.run(debug=True, host='0.0.0.0', port=8080)
