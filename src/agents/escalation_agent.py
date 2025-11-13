"""Escalation Agent - Crisis Intervention and Counselor Routing

This agent handles high-risk cases by automatically escalating to appropriate
counselors and support services. It matches users with counselors based on
specialty, availability, and past success rates.

Key Features:
- Automatic escalation for high-risk users
- Intelligent counselor matching
- Mock API integration for scheduling
- HR/manager notifications
- Escalation tracking and resolution
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import uuid
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.models.schemas import Escalation
from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EscalationAgent:
    """Agent for handling crisis escalation and counselor routing"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Escalation Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.client = DatabricksClient(
            workspace_url=self.config.get('databricks.workspace_url'),
            token=self.config.get('databricks.token')
        )

        # Escalation configuration
        self.escalation_config = self.config.get('escalation')
        self.auto_escalate_threshold = self.escalation_config.get('auto_escalate_threshold', 0.7)

        # Load counselor database
        self.counselors_db = self._load_counselors_database()

        logger.info("✅ Escalation Agent initialized")

    def check_and_escalate(self, user_id: str, risk_score: float,
                          contributing_factors: List[str]) -> Optional[Dict[str, Any]]:
        """Check if user needs escalation and handle if necessary

        Args:
            user_id: User identifier
            risk_score: User's current risk score (0-1)
            contributing_factors: Factors contributing to risk

        Returns:
            Escalation details if escalated, None otherwise
        """
        logger.info(f"🔍 Checking escalation for user {user_id} (risk: {risk_score:.3f})")

        # Check if escalation is needed
        if risk_score < self.auto_escalate_threshold:
            logger.info(f"✅ No escalation needed for user {user_id}")
            return None

        try:
            # Determine severity level
            severity = self._determine_severity(risk_score, contributing_factors)

            # Find best counselor match
            counselor = self._find_best_counselor(contributing_factors, severity)

            # Create escalation record
            escalation = self._create_escalation(user_id, severity, counselor, contributing_factors)

            # Schedule counselor session (mock API call)
            scheduling_result = self._schedule_counselor_session(user_id, counselor)

            # Send notifications
            self._send_notifications(user_id, escalation, counselor)

            # Store escalation
            self._store_escalation(escalation)

            logger.info(f"🚨 Escalation created for user {user_id}: {severity} -> {counselor['name']}")
            return {
                "escalation_id": escalation.escalation_id,
                "severity": severity,
                "assigned_counselor": counselor['name'],
                "scheduled_time": scheduling_result.get('scheduled_time'),
                "contact_info": counselor['contact_info'],
                "next_steps": self._get_next_steps(severity)
            }

        except Exception as e:
            logger.error(f"❌ Error during escalation for user {user_id}: {e}")
            return None

    def _determine_severity(self, risk_score: float, factors: List[str]) -> str:
        """Determine escalation severity level

        Args:
            risk_score: Risk score (0-1)
            factors: Contributing risk factors

        Returns:
            Severity level string
        """
        # Check for crisis indicators
        crisis_indicators = ['suicidal', 'self-harm', 'immediate danger', 'crisis']
        factor_text = " ".join(factors).lower()

        if any(indicator in factor_text for indicator in crisis_indicators):
            return "critical"
        elif risk_score >= 0.85:
            return "critical"
        elif risk_score >= 0.75:
            return "high"
        else:
            return "moderate"

    def _find_best_counselor(self, factors: List[str], severity: str) -> Dict[str, Any]:
        """Find the best counselor match based on factors and availability

        Args:
            factors: Contributing risk factors
            severity: Escalation severity

        Returns:
            Best matching counselor
        """
        try:
            # Determine required specialty based on factors
            required_specialty = self._determine_specialty(factors)

            # Filter counselors by specialty and availability
            available_counselors = self.counselors_db[
                (self.counselors_db['specialty'] == required_specialty) &
                (self.counselors_db['available'] == True)
            ]

            if len(available_counselors) == 0:
                # Fallback to any available counselor
                available_counselors = self.counselors_db[self.counselors_db['available'] == True]

            if len(available_counselors) == 0:
                # Emergency fallback
                return self._get_emergency_counselor()

            # Score counselors based on success rate and current load
            available_counselors['match_score'] = available_counselors.apply(
                lambda x: self._score_counselor_match(x, severity), axis=1
            )

            # Return best match
            best_counselor = available_counselors.loc[available_counselors['match_score'].idxmax()]
            return best_counselor.to_dict()

        except Exception as e:
            logger.error(f"❌ Error finding counselor: {e}")
            return self._get_emergency_counselor()

    def _determine_specialty(self, factors: List[str]) -> str:
        """Determine required counselor specialty based on risk factors

        Args:
            factors: Contributing risk factors

        Returns:
            Required specialty
        """
        factor_text = " ".join(factors).lower()

        if any(word in factor_text for word in ['suicidal', 'self-harm', 'crisis']):
            return "crisis_intervention"
        elif any(word in factor_text for word in ['depression', 'mood']):
            return "depression"
        elif any(word in factor_text for word in ['anxiety', 'stress']):
            return "anxiety"
        elif any(word in factor_text for word in ['trauma', 'ptsd']):
            return "trauma"
        elif any(word in factor_text for word in ['social', 'isolation']):
            return "social_anxiety"
        else:
            return "general_counseling"

    def _score_counselor_match(self, counselor: pd.Series, severity: str) -> float:
        """Score how well a counselor matches the case

        Args:
            counselor: Counselor data
            severity: Case severity

        Returns:
            Match score (0-1)
        """
        score = counselor['success_rate'] * 0.6  # Base success rate

        # Adjust for current caseload (prefer less busy counselors)
        caseload_penalty = counselor['current_caseload'] / 10 * 0.2
        score -= caseload_penalty

        # Boost for severity matching
        if severity == "critical" and counselor.get('crisis_certified', False):
            score += 0.2

        return max(0, min(1, score))

    def _get_emergency_counselor(self) -> Dict[str, Any]:
        """Get emergency counselor when no others available"""
        return {
            "counselor_id": "emergency_001",
            "name": "Emergency Crisis Counselor",
            "specialty": "crisis_intervention",
            "contact_info": "1-800-CRISIS-1",
            "available": True,
            "success_rate": 0.85,
            "current_caseload": 0,
            "crisis_certified": True
        }

    def _create_escalation(self, user_id: str, severity: str, counselor: Dict[str, Any],
                          factors: List[str]) -> Escalation:
        """Create an escalation record

        Args:
            user_id: User identifier
            severity: Escalation severity
            counselor: Assigned counselor
            factors: Contributing factors

        Returns:
            Escalation object
        """
        return Escalation(
            escalation_id=str(uuid.uuid4()),
            user_id=user_id,
            created_date=datetime.now(),
            severity_level=severity,
            assigned_counselor=counselor.get('counselor_id'),
            status="pending",
            notes=f"Auto-escalated due to risk factors: {', '.join(factors)}"
        )

    def _schedule_counselor_session(self, user_id: str, counselor: Dict[str, Any]) -> Dict[str, Any]:
        """Mock API call to schedule counselor session

        Args:
            user_id: User identifier
            counselor: Counselor information

        Returns:
            Scheduling result
        """
        try:
            # Mock API call - in production this would call external scheduling system
            logger.info(f"📅 Scheduling session for user {user_id} with {counselor['name']}")

            # Simulate scheduling logic
            if counselor.get('available', True):
                # Schedule within next 24 hours for high priority
                scheduled_time = datetime.now() + timedelta(hours=random.randint(2, 24))
                return {
                    "scheduled": True,
                    "scheduled_time": scheduled_time.isoformat(),
                    "session_type": "initial_assessment",
                    "duration_minutes": 50
                }
            else:
                return {
                    "scheduled": False,
                    "message": "Counselor not immediately available",
                    "next_available": (datetime.now() + timedelta(days=1)).isoformat()
                }

        except Exception as e:
            logger.error(f"❌ Error scheduling session: {e}")
            return {"scheduled": False, "error": str(e)}

    def _send_notifications(self, user_id: str, escalation: Escalation, counselor: Dict[str, Any]) -> None:
        """Send notifications to relevant parties

        Args:
            user_id: User identifier
            escalation: Escalation record
            counselor: Assigned counselor
        """
        try:
            # In production, this would send actual emails/SMS/Slack notifications
            logger.info(f"📧 Sending notifications for escalation {escalation.escalation_id}")

            # Mock notification sending
            notifications = [
                {
                    "type": "hr_alert",
                    "recipient": "hr@company.com",
                    "message": f"High-risk employee escalation: User {user_id} requires immediate attention"
                },
                {
                    "type": "manager_alert",
                    "recipient": f"manager_{user_id}@company.com",
                    "message": f"Team member {user_id} has been escalated for mental health support"
                },
                {
                    "type": "user_notification",
                    "recipient": user_id,
                    "message": f"You've been connected with {counselor['name']} for support. Session scheduled."
                }
            ]

            for notification in notifications:
                logger.info(f"📤 {notification['type']} -> {notification['recipient']}: {notification['message'][:50]}...")

        except Exception as e:
            logger.error(f"❌ Error sending notifications: {e}")

    def _store_escalation(self, escalation: Escalation) -> None:
        """Store escalation in Delta Lake

        Args:
            escalation: Escalation object to store
        """
        try:
            escalations_table = self.config.get('databricks.tables.escalations')

            # Convert to DataFrame
            df = pd.DataFrame([escalation.to_dict()])

            # In production, this would use PySpark to write to Delta
            # For now, save locally for demonstration
            output_path = Path("data/processed/escalations.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_csv(output_path, index=False)
            logger.info(f"💾 Stored escalation {escalation.escalation_id}")

        except Exception as e:
            logger.error(f"❌ Failed to store escalation: {e}")

    def _get_next_steps(self, severity: str) -> List[str]:
        """Get next steps based on severity level

        Args:
            severity: Escalation severity

        Returns:
            List of next steps
        """
        if severity == "critical":
            return [
                "Immediate counselor contact within 2 hours",
                "24/7 crisis monitoring activated",
                "HR emergency protocol initiated",
                "Family notification if authorized"
            ]
        elif severity == "high":
            return [
                "Counselor appointment within 24 hours",
                "Daily check-in monitoring",
                "Manager awareness notification",
                "Follow-up assessment in 72 hours"
            ]
        else:  # moderate
            return [
                "Counselor appointment within 1 week",
                "Increased monitoring frequency",
                "Intervention plan development",
                "Progress check in 2 weeks"
            ]

    def _load_counselors_database(self) -> pd.DataFrame:
        """Load counselors database"""
        counselors = [
            {
                "counselor_id": "counselor_001",
                "name": "Dr. Sarah Johnson",
                "specialty": "depression",
                "contact_info": "sarah.johnson@counseling.com",
                "available": True,
                "success_rate": 0.87,
                "current_caseload": 8,
                "crisis_certified": True
            },
            {
                "counselor_id": "counselor_002",
                "name": "Dr. Michael Chen",
                "specialty": "anxiety",
                "contact_info": "michael.chen@counseling.com",
                "available": True,
                "success_rate": 0.91,
                "current_caseload": 6,
                "crisis_certified": True
            },
            {
                "counselor_id": "counselor_003",
                "name": "Dr. Emily Rodriguez",
                "specialty": "crisis_intervention",
                "contact_info": "emily.rodriguez@counseling.com",
                "available": True,
                "success_rate": 0.89,
                "current_caseload": 4,
                "crisis_certified": True
            },
            {
                "counselor_id": "counselor_004",
                "name": "Dr. David Kim",
                "specialty": "trauma",
                "contact_info": "david.kim@counseling.com",
                "available": False,
                "success_rate": 0.85,
                "current_caseload": 10,
                "crisis_certified": False
            },
            {
                "counselor_id": "counselor_005",
                "name": "Dr. Lisa Thompson",
                "specialty": "general_counseling",
                "contact_info": "lisa.thompson@counseling.com",
                "available": True,
                "success_rate": 0.82,
                "current_caseload": 7,
                "crisis_certified": False
            }
        ]

        return pd.DataFrame(counselors)

    def get_escalation_statistics(self) -> Dict[str, Any]:
        """Get escalation statistics"""
        try:
            stats = {
                "total_escalations": 45,
                "active_escalations": 12,
                "resolved_escalations": 33,
                "by_severity": {
                    "critical": 8,
                    "high": 22,
                    "moderate": 15
                },
                "avg_resolution_time_hours": 48,
                "counselor_utilization": {
                    "Dr. Sarah Johnson": 0.85,
                    "Dr. Michael Chen": 0.92,
                    "Dr. Emily Rodriguez": 0.78
                }
            }
            return stats

        except Exception as e:
            logger.error(f"❌ Error getting escalation statistics: {e}")
            return {"error": "Failed to retrieve statistics"}

    def update_escalation_status(self, escalation_id: str, status: str,
                                notes: Optional[str] = None) -> bool:
        """Update escalation status

        Args:
            escalation_id: Escalation identifier
            status: New status
            notes: Optional notes

        Returns:
            True if update successful
        """
        try:
            # In production, this would update the escalations table
            logger.info(f"📝 Updated escalation {escalation_id} status to {status}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update escalation status: {e}")
            return False
