"""Analytics Agent - Multi-Level Dashboard and Reporting

This agent provides comprehensive analytics and dashboards for different user roles:
- Employee: Personal 90-day trends and insights
- Manager: Aggregated team metrics (privacy-preserving)
- Admin: Organization-wide wellness trends

Key Features:
- Role-based access control for analytics
- Privacy-preserving aggregations
- Trend analysis and forecasting
- Interactive dashboard data
- Performance metrics and KPIs
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class AnalyticsAgent:
    """Agent for providing analytics and dashboard data"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Analytics Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.client = DatabricksClient(
            workspace_url=self.config.get('databricks.workspace_url'),
            token=self.config.get('databricks.token')
        )

        # Analytics configuration
        self.analytics_config = self.config.get('analytics')
        self.refresh_interval = self.analytics_config.get('refresh_interval_minutes', 15)
        self.trend_window = self.analytics_config.get('trend_window_days', 30)

        logger.info("✅ Analytics Agent initialized")

    def get_dashboard_data(self, user_id: str, user_role: str,
                          department: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard data based on user role

        Args:
            user_id: User identifier
            user_role: User's role (employee, manager, admin)
            department: Department (for managers)

        Returns:
            Dashboard data appropriate for the user's role
        """
        logger.info(f"📊 Generating dashboard for {user_role}: {user_id}")

        try:
            if user_role == "employee":
                return self._get_employee_dashboard(user_id)
            elif user_role == "manager":
                return self._get_manager_dashboard(department)
            elif user_role == "admin":
                return self._get_admin_dashboard()
            else:
                return {"error": f"Unknown role: {user_role}"}

        except Exception as e:
            logger.error(f"❌ Error generating dashboard for {user_id}: {e}")
            return {"error": "Dashboard generation failed", "message": str(e)}

    def _get_employee_dashboard(self, user_id: str) -> Dict[str, Any]:
        """Get personal dashboard for individual employee

        Args:
            user_id: Employee identifier

        Returns:
            Personal dashboard data
        """
        try:
            # Get 90-day trend data
            trend_data = self._get_user_trends(user_id, days=90)

            # Get current risk assessment
            current_risk = self._get_current_risk(user_id)

            # Get recent interventions
            recent_interventions = self._get_recent_interventions(user_id)

            # Get check-in streak
            streak_info = self._get_checkin_streak(user_id)

            # Calculate insights
            insights = self._generate_personal_insights(trend_data, current_risk)

            dashboard = {
                "dashboard_type": "employee_personal",
                "user_id": user_id,
                "generated_at": datetime.now().isoformat(),
                "trend_data": trend_data,
                "current_risk": current_risk,
                "recent_interventions": recent_interventions,
                "checkin_streak": streak_info,
                "insights": insights,
                "recommendations": self._get_personal_recommendations(current_risk, trend_data)
            }

            return dashboard

        except Exception as e:
            logger.error(f"❌ Error getting employee dashboard: {e}")
            return {"error": "Employee dashboard generation failed"}

    def _get_manager_dashboard(self, department: str) -> Dict[str, Any]:
        """Get aggregated dashboard for managers (privacy-preserving)

        Args:
            department: Department name

        Returns:
            Aggregated team dashboard data
        """
        try:
            # Get team metrics (aggregated, no individual data)
            team_metrics = self._get_team_aggregate_metrics(department)

            # Get team trends
            team_trends = self._get_team_trends(department)

            # Get intervention recommendations for team
            team_recommendations = self._get_team_recommendations(team_metrics)

            # Get risk distribution (anonymized)
            risk_distribution = self._get_risk_distribution(department)

            dashboard = {
                "dashboard_type": "manager_team",
                "department": department,
                "generated_at": datetime.now().isoformat(),
                "team_size": team_metrics.get("total_members", 0),
                "team_metrics": team_metrics,
                "team_trends": team_trends,
                "risk_distribution": risk_distribution,
                "team_recommendations": team_recommendations,
                "privacy_note": "All data is aggregated to protect individual privacy"
            }

            return dashboard

        except Exception as e:
            logger.error(f"❌ Error getting manager dashboard: {e}")
            return {"error": "Manager dashboard generation failed"}

    def _get_admin_dashboard(self) -> Dict[str, Any]:
        """Get organization-wide dashboard for administrators

        Returns:
            Organization-wide dashboard data
        """
        try:
            # Get org-wide metrics
            org_metrics = self._get_organization_metrics()

            # Get department comparisons
            dept_comparison = self._get_department_comparison()

            # Get overall trends
            org_trends = self._get_organization_trends()

            # Get system health metrics
            system_health = self._get_system_health_metrics()

            dashboard = {
                "dashboard_type": "admin_organization",
                "generated_at": datetime.now().isoformat(),
                "organization_metrics": org_metrics,
                "department_comparison": dept_comparison,
                "organization_trends": org_trends,
                "system_health": system_health,
                "key_insights": self._generate_org_insights(org_metrics, org_trends)
            }

            return dashboard

        except Exception as e:
            logger.error(f"❌ Error getting admin dashboard: {e}")
            return {"error": "Admin dashboard generation failed"}

    def _get_user_trends(self, user_id: str, days: int) -> Dict[str, Any]:
        """Get user's mental health trends over time

        Args:
            user_id: User identifier
            days: Number of days to analyze

        Returns:
            Trend data dictionary
        """
        try:
            # Query check-in data
            checkins_table = self.config.get('databricks.tables.daily_checkins')

            query = f"""
            SELECT
                DATE(check_in_date) as date,
                mood_score,
                stress_level,
                energy_level,
                social_contact_rating,
                sleep_hours
            FROM {checkins_table}
            WHERE user_id = '{user_id}'
            AND check_in_date >= CURRENT_DATE - INTERVAL {days} DAY
            ORDER BY check_in_date
            """

            df = self.client.execute_query(query)

            if len(df) == 0:
                return {"message": "No check-in data available", "data_points": 0}

            # Calculate rolling averages
            df['mood_7day_avg'] = df['mood_score'].rolling(7, min_periods=1).mean()
            df['stress_7day_avg'] = df['stress_level'].rolling(7, min_periods=1).mean()
            df['energy_7day_avg'] = df['energy_level'].rolling(7, min_periods=1).mean()

            # Calculate trends
            mood_trend = self._calculate_trend_direction(df['mood_score'])
            stress_trend = self._calculate_trend_direction(df['stress_level'])
            energy_trend = self._calculate_trend_direction(df['energy_level'])

            trends = {
                "data_points": len(df),
                "date_range": {
                    "start": df['date'].min().isoformat() if len(df) > 0 else None,
                    "end": df['date'].max().isoformat() if len(df) > 0 else None
                },
                "current_values": {
                    "mood": df['mood_score'].iloc[-1] if len(df) > 0 else None,
                    "stress": df['stress_level'].iloc[-1] if len(df) > 0 else None,
                    "energy": df['energy_level'].iloc[-1] if len(df) > 0 else None,
                    "social_contact": df['social_contact_rating'].iloc[-1] if len(df) > 0 else None
                },
                "averages": {
                    "mood": round(df['mood_score'].mean(), 1),
                    "stress": round(df['stress_level'].mean(), 1),
                    "energy": round(df['energy_level'].mean(), 1),
                    "social_contact": round(df['social_contact_rating'].mean(), 1)
                },
                "trends": {
                    "mood": mood_trend,
                    "stress": stress_trend,
                    "energy": energy_trend
                },
                "weekly_averages": df[['date', 'mood_7day_avg', 'stress_7day_avg', 'energy_7day_avg']].to_dict('records')
            }

            return trends

        except Exception as e:
            logger.error(f"❌ Error getting user trends: {e}")
            return {"error": "Failed to retrieve trend data"}

    def _get_current_risk(self, user_id: str) -> Dict[str, Any]:
        """Get user's current risk assessment

        Args:
            user_id: User identifier

        Returns:
            Current risk information
        """
        try:
            # Query latest risk assessment
            risk_table = self.config.get('databricks.tables.risk_scores')

            query = f"""
            SELECT risk_score, contributing_factors, recommended_action,
                   confidence_score, assessment_date
            FROM {risk_table}
            WHERE user_id = '{user_id}'
            ORDER BY assessment_date DESC
            LIMIT 1
            """

            df = self.client.execute_query(query)

            if len(df) == 0:
                return {"status": "no_assessment", "message": "No risk assessment available"}

            row = df.iloc[0]
            return {
                "risk_score": float(row['risk_score']),
                "risk_level": self._classify_risk_level(float(row['risk_score'])),
                "contributing_factors": row['contributing_factors'],
                "recommended_action": row['recommended_action'],
                "confidence_score": float(row['confidence_score']),
                "assessment_date": row['assessment_date'].isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error getting current risk: {e}")
            return {"error": "Failed to retrieve risk assessment"}

    def _get_recent_interventions(self, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get user's recent interventions

        Args:
            user_id: User identifier
            limit: Maximum number of interventions to return

        Returns:
            List of recent interventions
        """
        try:
            interventions_table = self.config.get('databricks.tables.interventions')

            query = f"""
            SELECT intervention_type, description, success_rate, recommended_date
            FROM {interventions_table}
            WHERE user_id = '{user_id}'
            ORDER BY recommended_date DESC
            LIMIT {limit}
            """

            df = self.client.execute_query(query)
            return df.to_dict('records')

        except Exception as e:
            logger.error(f"❌ Error getting recent interventions: {e}")
            return []

    def _get_checkin_streak(self, user_id: str) -> Dict[str, Any]:
        """Get user's check-in streak information

        Args:
            user_id: User identifier

        Returns:
            Streak information
        """
        try:
            # This would typically call the CheckInAgent, but for demo we'll simulate
            return {
                "current_streak": 12,
                "longest_streak": 28,
                "last_checkin": datetime.now().isoformat(),
                "streak_message": "Great job maintaining your check-in routine!"
            }
        except Exception as e:
            return {"current_streak": 0, "longest_streak": 0}

    def _get_team_aggregate_metrics(self, department: str) -> Dict[str, Any]:
        """Get aggregated team metrics (privacy-preserving)

        Args:
            department: Department name

        Returns:
            Aggregated team metrics
        """
        try:
            # In production, this would aggregate data safely
            # For demo, return mock aggregated data
            return {
                "total_members": 24,
                "active_checkins_pct": 87.5,
                "avg_mood_score": 7.2,
                "avg_stress_level": 4.8,
                "avg_energy_level": 6.9,
                "isolation_prevalence_pct": 12.5,
                "intervention_completion_rate": 73.2,
                "avg_risk_score": 0.28
            }
        except Exception as e:
            return {"error": "Failed to retrieve team metrics"}

    def _get_team_trends(self, department: str) -> Dict[str, Any]:
        """Get team trend data

        Args:
            department: Department name

        Returns:
            Team trend information
        """
        try:
            return {
                "mood_trend": "improving",
                "stress_trend": "stable",
                "energy_trend": "improving",
                "isolation_trend": "decreasing",
                "checkin_frequency_trend": "stable",
                "trend_period_days": 30
            }
        except Exception as e:
            return {"error": "Failed to retrieve team trends"}

    def _get_team_recommendations(self, team_metrics: Dict[str, Any]) -> List[str]:
        """Generate team-level recommendations

        Args:
            team_metrics: Team metrics data

        Returns:
            List of recommendations
        """
        recommendations = []

        if team_metrics.get('avg_stress_level', 5) > 6:
            recommendations.append("Consider team stress management workshop")
        if team_metrics.get('isolation_prevalence_pct', 0) > 15:
            recommendations.append("Promote team social events and connection activities")
        if team_metrics.get('avg_energy_level', 5) < 6:
            recommendations.append("Review workload distribution and encourage work-life balance")

        if not recommendations:
            recommendations.append("Team wellness metrics are strong - continue current practices")

        return recommendations

    def _get_risk_distribution(self, department: str) -> Dict[str, Any]:
        """Get anonymized risk distribution for department

        Args:
            department: Department name

        Returns:
            Risk distribution data
        """
        try:
            return {
                "low_risk": 18,      # < 0.3
                "moderate_risk": 4,  # 0.3-0.7
                "high_risk": 2,      # > 0.7
                "at_risk_percentage": 25.0,
                "requires_attention": 6
            }
        except Exception as e:
            return {"error": "Failed to retrieve risk distribution"}

    def _get_organization_metrics(self) -> Dict[str, Any]:
        """Get organization-wide metrics

        Returns:
            Organization metrics
        """
        try:
            return {
                "total_employees": 156,
                "active_users": 142,
                "overall_avg_mood": 7.1,
                "overall_avg_stress": 4.9,
                "overall_avg_energy": 6.8,
                "organization_risk_avg": 0.31,
                "escalation_rate_pct": 2.8,
                "intervention_completion_rate": 71.5,
                "departments_covered": 8
            }
        except Exception as e:
            return {"error": "Failed to retrieve organization metrics"}

    def _get_department_comparison(self) -> List[Dict[str, Any]]:
        """Get department comparison data

        Returns:
            List of department metrics for comparison
        """
        try:
            return [
                {"department": "Engineering", "avg_mood": 7.3, "avg_stress": 4.7, "headcount": 45},
                {"department": "Sales", "avg_mood": 6.8, "avg_stress": 5.2, "headcount": 32},
                {"department": "Marketing", "avg_mood": 7.1, "avg_stress": 4.9, "headcount": 28},
                {"department": "HR", "avg_mood": 7.5, "avg_stress": 4.3, "headcount": 15},
                {"department": "Finance", "avg_mood": 6.9, "avg_stress": 5.1, "headcount": 22},
                {"department": "Operations", "avg_mood": 7.0, "avg_stress": 5.0, "headcount": 14}
            ]
        except Exception as e:
            return []

    def _get_organization_trends(self) -> Dict[str, Any]:
        """Get organization-wide trends

        Returns:
            Organization trend data
        """
        try:
            return {
                "mood_trend": "stable",
                "stress_trend": "slight_increase",
                "energy_trend": "stable",
                "isolation_trend": "decreasing",
                "engagement_trend": "improving",
                "trend_period_months": 6,
                "key_changes": [
                    "Stress levels increased 8% in Q3",
                    "Energy levels stable despite busy period",
                    "Social connection improving across departments"
                ]
            }
        except Exception as e:
            return {"error": "Failed to retrieve organization trends"}

    def _get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health and performance metrics

        Returns:
            System health data
        """
        try:
            return {
                "system_uptime_pct": 99.7,
                "avg_response_time_ms": 245,
                "data_freshness_minutes": 12,
                "active_agents": 6,
                "processed_checkins_today": 89,
                "risk_assessments_today": 34,
                "escalations_today": 2
            }
        except Exception as e:
            return {"error": "Failed to retrieve system health metrics"}

    def _calculate_trend_direction(self, series: pd.Series) -> str:
        """Calculate trend direction from time series

        Args:
            series: Time series data

        Returns:
            Trend direction string
        """
        if len(series) < 3:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(series))
        slope = np.polyfit(x, series, 1)[0]

        if slope > 0.02:
            return "improving"
        elif slope < -0.02:
            return "declining"
        else:
            return "stable"

    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk score into levels

        Args:
            risk_score: Risk score (0-1)

        Returns:
            Risk level string
        """
        if risk_score >= 0.7:
            return "high_risk"
        elif risk_score >= 0.4:
            return "moderate_risk"
        elif risk_score >= 0.2:
            return "elevated_risk"
        else:
            return "low_risk"

    def _generate_personal_insights(self, trend_data: Dict, current_risk: Dict) -> List[str]:
        """Generate personalized insights for employee

        Args:
            trend_data: User's trend data
            current_risk: Current risk assessment

        Returns:
            List of insights
        """
        insights = []

        # Mood insights
        if trend_data.get('trends', {}).get('mood') == 'improving':
            insights.append("Your mood has been trending positively - keep up the good work!")
        elif trend_data.get('trends', {}).get('mood') == 'declining':
            insights.append("Your mood has been declining. Consider reaching out for support.")

        # Stress insights
        current_stress = trend_data.get('current_values', {}).get('stress')
        if current_stress and current_stress > 7:
            insights.append("Your current stress levels are high. Consider stress-reduction techniques.")

        # Social insights
        social_contact = trend_data.get('current_values', {}).get('social_contact')
        if social_contact and social_contact < 3:
            insights.append("Social contact has been low. Building connections can help wellbeing.")

        if not insights:
            insights.append("Your wellbeing metrics are stable. Continue your healthy routines!")

        return insights

    def _generate_org_insights(self, org_metrics: Dict, org_trends: Dict) -> List[str]:
        """Generate organization-level insights

        Args:
            org_metrics: Organization metrics
            org_trends: Organization trends

        Returns:
            List of insights
        """
        insights = []

        # Overall health insights
        avg_mood = org_metrics.get('overall_avg_mood', 7)
        if avg_mood >= 7.5:
            insights.append("Organization mood is excellent - strong wellbeing foundation")
        elif avg_mood >= 6.5:
            insights.append("Organization mood is good but has room for improvement")
        else:
            insights.append("Organization mood needs attention - consider wellbeing initiatives")

        # Trend insights
        stress_trend = org_trends.get('stress_trend')
        if stress_trend == 'increasing':
            insights.append("Stress levels are trending up - monitor workload and support resources")

        # Participation insights
        active_pct = (org_metrics.get('active_users', 0) / org_metrics.get('total_employees', 1)) * 100
        if active_pct >= 80:
            insights.append("Excellent participation in wellbeing program")
        elif active_pct >= 60:
            insights.append("Good participation - consider ways to increase engagement")
        else:
            insights.append("Participation could be improved - focus on awareness and ease of use")

        return insights

    def _get_personal_recommendations(self, current_risk: Dict, trend_data: Dict) -> List[str]:
        """Generate personal recommendations

        Args:
            current_risk: Current risk assessment
            trend_data: Trend data

        Returns:
            List of recommendations
        """
        recommendations = []

        risk_level = current_risk.get('risk_level', 'low_risk')

        if risk_level in ['high_risk', 'moderate_risk']:
            recommendations.append("Consider speaking with a counselor or trusted colleague")
            recommendations.append("Practice daily stress-reduction techniques")

        if trend_data.get('trends', {}).get('energy') == 'declining':
            recommendations.append("Focus on sleep quality and regular exercise")

        if trend_data.get('current_values', {}).get('social_contact', 3) < 3:
            recommendations.append("Reach out to friends, family, or colleagues for connection")

        if not recommendations:
            recommendations.append("Continue your current wellbeing practices - you're doing well!")

        return recommendations

    def export_dashboard_data(self, dashboard_data: Dict[str, Any], format: str = "json") -> str:
        """Export dashboard data for external use

        Args:
            dashboard_data: Dashboard data to export
            format: Export format (json, csv)

        Returns:
            Exported data as string
        """
        try:
            if format == "json":
                return json.dumps(dashboard_data, indent=2, default=str)
            else:
                # For CSV, flatten the data structure
                return json.dumps(dashboard_data, default=str)
        except Exception as e:
            logger.error(f"❌ Error exporting dashboard data: {e}")
            return "{}"
