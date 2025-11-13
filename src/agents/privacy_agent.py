"""Privacy-Preserving Reporting Agent - The Game Changer

This agent implements k-anonymity and de-identification to provide safe management
reporting without compromising individual privacy. It aggregates mental health data
by team/department while ensuring NO individual names, moods, or personal concerns
are visible to management.

Key Features:
- K-anonymity with minimum group sizes
- Complete de-identification of personal data
- Role-based access control
- Audit trail for all data access
- Safe aggregate reporting for management decisions
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import uuid
import hashlib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np

from src.models.schemas import AuditLog
from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class PrivacyPreservingAgent:
    """Privacy-preserving agent for safe management reporting"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Privacy-Preserving Reporting Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.client = DatabricksClient(
            workspace_url=self.config.get('databricks.workspace_url'),
            token=self.config.get('databricks.token')
        )

        # Privacy configuration
        self.k_anonymity = self.config.get('privacy.k_anonymity')
        self.role_permissions = self.config.get('privacy.role_permissions')

        # Audit configuration
        self.audit_enabled = self.config.get('privacy.audit.enabled', True)

        logger.info("✅ Privacy-Preserving Agent initialized with k-anonymity protection")

    def generate_safe_report(self, user_role: str, department: Optional[str] = None,
                           time_range_days: int = 30) -> Dict[str, Any]:
        """Generate a privacy-safe report for the requesting user

        Args:
            user_role: Role of the requesting user (employee, manager, admin)
            department: Optional department filter
            time_range_days: Number of days to include in report

        Returns:
            Dictionary containing safe aggregate data
        """
        logger.info(f"🔒 Generating safe report for role: {user_role}, department: {department}")

        # Check permissions
        if not self._check_permissions(user_role, "generate_reports"):
            logger.warning(f"❌ Access denied for role: {user_role}")
            return {"error": "Access denied", "message": f"Role '{user_role}' cannot access reports"}

        try:
            # Log access for audit trail
            self._log_access(user_role, "generate_report", f"department={department},days={time_range_days}")

            # Get appropriate data based on role
            if user_role == "employee":
                return self._generate_employee_report()
            elif user_role == "manager":
                return self._generate_manager_report(department, time_range_days)
            elif user_role == "admin":
                return self._generate_admin_report(time_range_days)
            else:
                return {"error": "Invalid role", "message": f"Unknown role: {user_role}"}

        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")
            return {"error": "Report generation failed", "message": str(e)}

    def _generate_employee_report(self) -> Dict[str, Any]:
        """Generate report for individual employee (their own anonymized data)"""
        # Employees can only see their own data in aggregate form
        # This is mainly for consistency - they would typically use the main dashboard
        return {
            "report_type": "employee_self",
            "message": "Individual employee reports are handled by the main dashboard",
            "data": {}
        }

    def _generate_manager_report(self, department: Optional[str], time_range_days: int) -> Dict[str, Any]:
        """Generate privacy-safe manager report for their team/department

        Args:
            department: Department to report on
            time_range_days: Time range for the report

        Returns:
            Dictionary with aggregated team data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)

            # Get risk scores and check-in data for the department
            data = self._get_department_data(department, start_date, end_date)

            if len(data) < self.k_anonymity.get('min_group_size', 5):
                return {
                    "error": "Insufficient data",
                    "message": f"Department has fewer than {self.k_anonymity.get('min_group_size', 5)} members for privacy protection"
                }

            # Apply k-anonymity and aggregation
            safe_report = self._apply_k_anonymity_and_aggregate(data, "team")

            # Add department context
            safe_report.update({
                "report_type": "manager_team",
                "department": department,
                "time_range_days": time_range_days,
                "generated_at": datetime.now().isoformat(),
                "privacy_level": "k-anonymized",
                "k_value": self.k_anonymity.get('min_group_size', 5)
            })

            logger.info(f"✅ Generated manager report for department {department}: {len(data)} records aggregated")
            return safe_report

        except Exception as e:
            logger.error(f"❌ Error generating manager report: {e}")
            return {"error": "Manager report generation failed", "message": str(e)}

    def _generate_admin_report(self, time_range_days: int) -> Dict[str, Any]:
        """Generate organization-wide admin report

        Args:
            time_range_days: Time range for the report

        Returns:
            Dictionary with organization-wide aggregated data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)

            # Get organization-wide data
            data = self._get_organization_data(start_date, end_date)

            # Group by department for reporting
            department_reports = {}
            for dept in data['department'].unique():
                dept_data = data[data['department'] == dept]
                if len(dept_data) >= self.k_anonymity.get('min_group_size', 5):
                    department_reports[dept] = self._apply_k_anonymity_and_aggregate(dept_data, "department")

            # Organization summary
            org_summary = self._apply_k_anonymity_and_aggregate(data, "organization")

            report = {
                "report_type": "admin_organization",
                "time_range_days": time_range_days,
                "generated_at": datetime.now().isoformat(),
                "privacy_level": "k-anonymized",
                "k_value": self.k_anonymity.get('min_group_size', 5),
                "organization_summary": org_summary,
                "department_breakdown": department_reports,
                "total_departments": len(department_reports),
                "total_employees_covered": sum(len(dept_data) for dept_data in [data[data['department'] == dept]
                                               for dept in department_reports.keys()])
            }

            logger.info(f"✅ Generated admin report: {len(department_reports)} departments, {report['total_employees_covered']} employees")
            return report

        except Exception as e:
            logger.error(f"❌ Error generating admin report: {e}")
            return {"error": "Admin report generation failed", "message": str(e)}

    def _get_department_data(self, department: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get department data with privacy protections

        Args:
            department: Department name
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with department data (already de-identified)
        """
        try:
            # Query risk scores and check-ins for department
            # In production, this would join multiple Delta tables
            query = f"""
            SELECT
                r.user_id,
                r.risk_score,
                r.contributing_factors,
                c.mood_score,
                c.stress_level,
                c.energy_level,
                c.social_contact_rating,
                c.isolation_flag,
                c.check_in_date,
                u.department,
                u.job_role
            FROM wellbeing.risk_scores r
            JOIN wellbeing.daily_checkins c ON r.user_id = c.user_id
            JOIN wellbeing.user_profiles u ON r.user_id = u.user_id
            WHERE u.department = '{department}'
            AND c.check_in_date >= '{start_date.isoformat()}'
            AND c.check_in_date <= '{end_date.isoformat()}'
            """

            # For demo purposes, create mock data
            data = self._create_mock_department_data(department, start_date, end_date)
            return data

        except Exception as e:
            logger.error(f"❌ Error fetching department data: {e}")
            return pd.DataFrame()

    def _get_organization_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get organization-wide data with privacy protections

        Args:
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with organization data
        """
        try:
            # Query organization-wide data
            # For demo purposes, create comprehensive mock data
            data = self._create_mock_organization_data(start_date, end_date)
            return data

        except Exception as e:
            logger.error(f"❌ Error fetching organization data: {e}")
            return pd.DataFrame()

    def _apply_k_anonymity_and_aggregate(self, data: pd.DataFrame, aggregation_level: str) -> Dict[str, Any]:
        """Apply k-anonymity and create safe aggregations

        Args:
            data: Raw data to aggregate
            aggregation_level: Level of aggregation (team, department, organization)

        Returns:
            Dictionary with safe aggregated statistics
        """
        if len(data) < self.k_anonymity.get('min_group_size', 5):
            return {"error": "Insufficient data for k-anonymity"}

        try:
            # Remove any potentially identifying information
            safe_data = data.drop(columns=['user_id', 'job_role'], errors='ignore')

            # Calculate safe aggregations
            aggregations = {
                "total_members": len(data),
                "avg_risk_score": round(data['risk_score'].mean(), 3),
                "risk_distribution": {
                    "low_risk": len(data[data['risk_score'] < 0.3]),      # < 0.3
                    "moderate_risk": len(data[(data['risk_score'] >= 0.3) & (data['risk_score'] < 0.7)]),  # 0.3-0.7
                    "high_risk": len(data[data['risk_score'] >= 0.7])      # >= 0.7
                },
                "avg_mood_score": round(data['mood_score'].mean(), 1),
                "avg_stress_level": round(data['stress_level'].mean(), 1),
                "avg_energy_level": round(data['energy_level'].mean(), 1),
                "isolation_prevalence": round(data['isolation_flag'].mean() * 100, 1),  # Percentage
                "social_contact_avg": round(data['social_contact_rating'].mean(), 1),
                "estimated_at_risk_users": self._estimate_at_risk_count(data),
                "top_concerns_summary": self._summarize_concerns_anonymously(data),
                "trend_indicators": self._calculate_safe_trends(data)
            }

            # Add suppression for small groups
            suppression_threshold = self.k_anonymity.get('suppression_threshold', 3)
            if len(data) < suppression_threshold:
                aggregations = {"suppressed": True, "reason": "Group too small for privacy"}

            return aggregations

        except Exception as e:
            logger.error(f"❌ Error in k-anonymity aggregation: {e}")
            return {"error": "Aggregation failed", "message": str(e)}

    def _estimate_at_risk_count(self, data: pd.DataFrame) -> str:
        """Estimate number of at-risk users without revealing individual data

        Args:
            data: DataFrame with risk scores

        Returns:
            String estimate (e.g., "2-3", "5-8")
        """
        high_risk_count = len(data[data['risk_score'] >= 0.7])
        moderate_risk_count = len(data[(data['risk_score'] >= 0.4) & (data['risk_score'] < 0.7)])

        # Provide ranges to maintain privacy
        if high_risk_count == 0:
            high_range = "0"
        elif high_risk_count <= 2:
            high_range = "1-2"
        elif high_risk_count <= 5:
            high_range = "3-5"
        else:
            high_range = "5+"

        if moderate_risk_count <= 3:
            moderate_range = "few"
        elif moderate_risk_count <= 8:
            moderate_range = "several"
        else:
            moderate_range = "many"

        return f"{high_range} high-risk, {moderate_range} moderate-risk"

    def _summarize_concerns_anonymously(self, data: pd.DataFrame) -> Dict[str, int]:
        """Summarize concerns without revealing individual content

        Args:
            data: DataFrame with contributing factors

        Returns:
            Dictionary with concern categories and counts
        """
        concern_counts = {}

        for factors in data['contributing_factors'].dropna():
            if isinstance(factors, str):
                # Parse the factors (they're stored as lists in the data)
                try:
                    factor_list = eval(factors) if factors.startswith('[') else [factors]
                    for factor in factor_list:
                        # Categorize concerns
                        category = self._categorize_concern(factor)
                        concern_counts[category] = concern_counts.get(category, 0) + 1
                except:
                    continue

        return dict(sorted(concern_counts.items(), key=lambda x: x[1], reverse=True))

    def _categorize_concern(self, concern: str) -> str:
        """Categorize individual concerns into safe categories

        Args:
            concern: Individual concern text

        Returns:
            Categorized concern type
        """
        concern_lower = concern.lower()

        if any(word in concern_lower for word in ['mood', 'depression', 'sad', 'unhappy']):
            return "Mood-related concerns"
        elif any(word in concern_lower for word in ['stress', 'anxiety', 'overwhelm']):
            return "Stress & anxiety"
        elif any(word in concern_lower for word in ['sleep', 'tired', 'energy']):
            return "Sleep & energy issues"
        elif any(word in concern_lower for word in ['social', 'isolation', 'lonely']):
            return "Social connection issues"
        elif any(word in concern_lower for word in ['work', 'job', 'career']):
            return "Work-related concerns"
        else:
            return "Other concerns"

    def _calculate_safe_trends(self, data: pd.DataFrame) -> Dict[str, str]:
        """Calculate trend indicators that don't reveal individual trajectories

        Args:
            data: DataFrame with time series data

        Returns:
            Dictionary with trend summaries
        """
        try:
            # Group by date and calculate daily averages
            daily_stats = data.groupby(data['check_in_date'].dt.date).agg({
                'risk_score': 'mean',
                'mood_score': 'mean',
                'stress_level': 'mean'
            }).reset_index()

            if len(daily_stats) < 7:
                return {"insufficient_data": "Need at least 7 days of data"}

            # Calculate overall trends
            risk_trend = "stable"
            mood_trend = "stable"
            stress_trend = "stable"

            # Simple trend detection on aggregated data
            if len(daily_stats) >= 14:
                first_half = daily_stats.head(len(daily_stats)//2)
                second_half = daily_stats.tail(len(daily_stats)//2)

                if second_half['risk_score'].mean() > first_half['risk_score'].mean() * 1.1:
                    risk_trend = "increasing"
                elif second_half['risk_score'].mean() < first_half['risk_score'].mean() * 0.9:
                    risk_trend = "decreasing"

                if second_half['mood_score'].mean() < first_half['mood_score'].mean() * 0.95:
                    mood_trend = "declining"
                elif second_half['mood_score'].mean() > first_half['mood_score'].mean() * 1.05:
                    mood_trend = "improving"

                if second_half['stress_level'].mean() > first_half['stress_level'].mean() * 1.1:
                    stress_trend = "increasing"
                elif second_half['stress_level'].mean() < first_half['stress_level'].mean() * 0.9:
                    stress_trend = "decreasing"

            return {
                "risk_trend": risk_trend,
                "mood_trend": mood_trend,
                "stress_trend": stress_trend,
                "data_points": len(daily_stats)
            }

        except Exception as e:
            logger.error(f"❌ Error calculating trends: {e}")
            return {"error": "Trend calculation failed"}

    def _check_permissions(self, user_role: str, action: str) -> bool:
        """Check if user role has permission for action

        Args:
            user_role: User's role
            action: Action to check permission for

        Returns:
            Boolean indicating permission
        """
        role_perms = self.role_permissions.get(user_role, [])
        return action in role_perms

    def _log_access(self, user_id: str, action: str, resource: str) -> None:
        """Log data access for audit trail

        Args:
            user_id: User identifier
            action: Action performed
            resource: Resource accessed
        """
        if not self.audit_enabled:
            return

        try:
            audit_log = AuditLog(
                log_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                user_id=user_id,
                role="system",  # This would be passed from the calling context
                action=action,
                resource=resource,
                details={"privacy_level": "k-anonymized", "k_value": self.k_anonymity.get('min_group_size', 5)}
            )

            # Store audit log (in production, this would go to Delta table)
            logger.info(f"📊 Audit: {user_id} accessed {resource} for {action}")

        except Exception as e:
            logger.error(f"❌ Failed to log audit event: {e}")

    def _create_mock_department_data(self, department: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create mock department data for demonstration

        Args:
            department: Department name
            start_date: Start date
            end_date: End date

        Returns:
            Mock DataFrame with department data
        """
        # Create realistic mock data
        num_users = np.random.randint(8, 15)  # 8-15 users per department
        days = (end_date - start_date).days

        data = []
        for user_idx in range(num_users):
            user_id = f"user_{department.lower()}_{user_idx}"

            # Generate time series for each user
            for day in range(days):
                check_in_date = start_date + timedelta(days=day)

                # Base characteristics with some variation
                base_mood = np.random.normal(6.5, 1.5)
                base_stress = np.random.normal(4.5, 1.8)
                base_energy = np.random.normal(6.0, 1.2)

                # Add some trend (some users improving, some declining)
                trend_factor = np.random.choice([-0.02, 0, 0.02])
                day_factor = day * trend_factor

                # Add daily variation
                mood_score = np.clip(base_mood + day_factor + np.random.normal(0, 0.5), 1, 10)
                stress_level = np.clip(base_stress - day_factor + np.random.normal(0, 0.5), 1, 10)
                energy_level = np.clip(base_energy + day_factor + np.random.normal(0, 0.3), 1, 10)

                # Risk score based on mood/stress
                risk_score = min(1.0, max(0.0, (11 - mood_score) / 10 * 0.4 + stress_level / 10 * 0.6))

                # Other metrics
                social_contact = np.random.randint(1, 6)
                isolation_flag = social_contact < 3 and np.random.random() > 0.7

                # Contributing factors based on risk
                if risk_score > 0.7:
                    factors = ["Severely declining mood", "High stress levels", "Social isolation"]
                elif risk_score > 0.4:
                    factors = ["Elevated stress", "Reduced social contact"]
                else:
                    factors = ["Stable mood and stress levels"]

                data.append({
                    'user_id': user_id,
                    'risk_score': risk_score,
                    'contributing_factors': str(factors),
                    'mood_score': mood_score,
                    'stress_level': stress_level,
                    'energy_level': energy_level,
                    'social_contact_rating': social_contact,
                    'isolation_flag': isolation_flag,
                    'check_in_date': check_in_date,
                    'department': department,
                    'job_role': f"Role {user_idx % 3 + 1}"
                })

        return pd.DataFrame(data)

    def _create_mock_organization_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Create mock organization-wide data for demonstration

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            Mock DataFrame with organization data
        """
        departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations"]
        all_data = []

        for dept in departments:
            dept_data = self._create_mock_department_data(dept, start_date, end_date)
            all_data.append(dept_data)

        return pd.concat(all_data, ignore_index=True)

    def get_privacy_metrics(self) -> Dict[str, Any]:
        """Get privacy and compliance metrics

        Returns:
            Dictionary with privacy metrics
        """
        return {
            "k_anonymity_level": self.k_anonymity.get('min_group_size', 5),
            "suppression_threshold": self.k_anonymity.get('suppression_threshold', 3),
            "audit_trail_enabled": self.audit_enabled,
            "supported_roles": list(self.role_permissions.keys()),
            "privacy_features": [
                "K-anonymity aggregation",
                "Complete de-identification",
                "Role-based access control",
                "Audit trail logging",
                "Data suppression for small groups"
            ],
            "last_updated": datetime.now().isoformat()
        }
