"""Check-In Agent - Daily Wellbeing Data Collection

This agent provides a simple interface for users to log their daily mental health
and wellbeing check-ins. It collects mood, stress, energy, social contact, and
concerns data, validates it, and stores it in the Delta Lake daily_checkins table.

Key Features:
- Simple, user-friendly check-in interface
- Data validation and sanitization
- Automatic timestamping and user association
- Integration with Delta Lake storage
- Support for optional concerns/notes
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging
import uuid

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.models.schemas import DailyCheckIn
from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class CheckInAgent:
    """Agent for handling daily mental health check-ins"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Check-In Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.client = DatabricksClient(
            workspace_url=self.config.get('databricks.workspace_url'),
            token=self.config.get('databricks.token')
        )

        # Check-in configuration
        self.max_concerns_length = 500  # Maximum characters for concerns
        self.required_fields = ['user_id', 'mood_score', 'stress_level', 'energy_level', 'social_contact_rating']

        logger.info("✅ Check-In Agent initialized")

    def submit_checkin(self, user_id: str, checkin_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a daily check-in for a user

        Args:
            user_id: User identifier
            checkin_data: Dictionary containing check-in data

        Returns:
            Dictionary with submission result
        """
        logger.info(f"📝 Processing check-in for user: {user_id}")

        try:
            # Validate input data
            validation_result = self._validate_checkin_data(checkin_data)
            if not validation_result['valid']:
                return {
                    "success": False,
                    "error": "Validation failed",
                    "details": validation_result['errors']
                }

            # Create DailyCheckIn object
            checkin = self._create_checkin_record(user_id, checkin_data)

            # Check for duplicate (same user, same day)
            if self._check_duplicate_checkin(user_id, checkin.check_in_date.date()):
                return {
                    "success": False,
                    "error": "Duplicate check-in",
                    "message": "You have already submitted a check-in for today"
                }

            # Store in Delta Lake
            success = self._store_checkin(checkin)

            if success:
                logger.info(f"✅ Check-in stored for user {user_id}")
                return {
                    "success": True,
                    "message": "Check-in submitted successfully",
                    "checkin_id": str(uuid.uuid4()),  # Would be actual ID in production
                    "timestamp": checkin.check_in_date.isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": "Storage failed",
                    "message": "Failed to store check-in data"
                }

        except Exception as e:
            logger.error(f"❌ Error processing check-in for user {user_id}: {e}")
            return {
                "success": False,
                "error": "Processing failed",
                "message": str(e)
            }

    def get_checkin_form(self) -> Dict[str, Any]:
        """Get the check-in form structure for UI rendering

        Returns:
            Dictionary describing the form fields and options
        """
        return {
            "title": "Daily Wellbeing Check-In",
            "description": "How are you feeling today? Your responses help us support your wellbeing.",
            "fields": [
                {
                    "name": "mood_score",
                    "type": "rating",
                    "label": "Mood",
                    "description": "How would you rate your mood today?",
                    "required": True,
                    "min": 1,
                    "max": 10,
                    "labels": {
                        1: "Very low",
                        5: "Neutral",
                        10: "Excellent"
                    }
                },
                {
                    "name": "stress_level",
                    "type": "rating",
                    "label": "Stress Level",
                    "description": "How stressed do you feel today?",
                    "required": True,
                    "min": 1,
                    "max": 10,
                    "labels": {
                        1: "No stress",
                        5: "Moderate stress",
                        10: "Overwhelming stress"
                    }
                },
                {
                    "name": "energy_level",
                    "type": "rating",
                    "label": "Energy Level",
                    "description": "How energetic do you feel today?",
                    "required": True,
                    "min": 1,
                    "max": 10,
                    "labels": {
                        1: "Exhausted",
                        5: "Normal energy",
                        10: "Very energetic"
                    }
                },
                {
                    "name": "social_contact_rating",
                    "type": "rating",
                    "label": "Social Contact",
                    "description": "How much social contact have you had today?",
                    "required": True,
                    "min": 1,
                    "max": 5,
                    "labels": {
                        1: "No social contact",
                        3: "Some contact",
                        5: "Lots of social contact"
                    }
                },
                {
                    "name": "sleep_hours",
                    "type": "number",
                    "label": "Sleep Hours",
                    "description": "How many hours of sleep did you get last night?",
                    "required": False,
                    "min": 0,
                    "max": 24,
                    "step": 0.5
                },
                {
                    "name": "concerns",
                    "type": "textarea",
                    "label": "Concerns or Notes",
                    "description": "Any concerns or additional notes? (Optional)",
                    "required": False,
                    "max_length": self.max_concerns_length,
                    "placeholder": "Share anything you'd like us to know..."
                }
            ],
            "submit_button": {
                "text": "Submit Check-In",
                "icon": "✅"
            }
        }

    def get_recent_checkins(self, user_id: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get user's recent check-ins for display

        Args:
            user_id: User identifier
            days: Number of days to retrieve

        Returns:
            List of recent check-in records
        """
        try:
            # Query recent check-ins
            checkins_table = self.config.get('databricks.tables.daily_checkins')

            query = f"""
            SELECT * FROM {checkins_table}
            WHERE user_id = '{user_id}'
            ORDER BY check_in_date DESC
            LIMIT {days}
            """

            df = self.client.execute_query(query)

            # Convert to list of dictionaries
            recent_checkins = []
            for _, row in df.iterrows():
                checkin = {
                    "date": row['check_in_date'],
                    "mood_score": int(row['mood_score']),
                    "stress_level": int(row['stress_level']),
                    "energy_level": int(row['energy_level']),
                    "social_contact_rating": int(row['social_contact_rating']),
                    "sleep_hours": float(row['sleep_hours']) if pd.notna(row['sleep_hours']) else None,
                    "isolation_flag": bool(row['isolation_flag']),
                    "concerns": row['concerns'] if pd.notna(row['concerns']) else None
                }
                recent_checkins.append(checkin)

            logger.info(f"📊 Retrieved {len(recent_checkins)} recent check-ins for user {user_id}")
            return recent_checkins

        except Exception as e:
            logger.error(f"❌ Error retrieving recent check-ins for user {user_id}: {e}")
            return []

    def get_checkin_streak(self, user_id: str) -> Dict[str, Any]:
        """Calculate user's check-in streak

        Args:
            user_id: User identifier

        Returns:
            Dictionary with streak information
        """
        try:
            # Get recent check-ins
            recent_checkins = self.get_recent_checkins(user_id, days=30)

            if not recent_checkins:
                return {"current_streak": 0, "longest_streak": 0, "last_checkin": None}

            # Sort by date
            recent_checkins.sort(key=lambda x: x['date'], reverse=True)

            # Calculate current streak
            current_streak = 0
            check_date = datetime.now().date()

            for checkin in recent_checkins:
                checkin_date = pd.to_datetime(checkin['date']).date()

                # If this check-in is for today or yesterday, count it
                if checkin_date == check_date or checkin_date == check_date - timedelta(days=1):
                    current_streak += 1
                    check_date = checkin_date
                else:
                    break

            # Calculate longest streak (simplified)
            longest_streak = current_streak  # In production, would analyze full history

            return {
                "current_streak": current_streak,
                "longest_streak": longest_streak,
                "last_checkin": recent_checkins[0]['date'] if recent_checkins else None,
                "streak_message": self._get_streak_message(current_streak)
            }

        except Exception as e:
            logger.error(f"❌ Error calculating streak for user {user_id}: {e}")
            return {"current_streak": 0, "longest_streak": 0, "last_checkin": None}

    def _validate_checkin_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate check-in data

        Args:
            data: Check-in data to validate

        Returns:
            Dictionary with validation results
        """
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        # Validate mood_score (1-10)
        if 'mood_score' in data:
            try:
                mood = int(data['mood_score'])
                if not 1 <= mood <= 10:
                    errors.append("Mood score must be between 1 and 10")
            except (ValueError, TypeError):
                errors.append("Mood score must be a number")

        # Validate stress_level (1-10)
        if 'stress_level' in data:
            try:
                stress = int(data['stress_level'])
                if not 1 <= stress <= 10:
                    errors.append("Stress level must be between 1 and 10")
            except (ValueError, TypeError):
                errors.append("Stress level must be a number")

        # Validate energy_level (1-10)
        if 'energy_level' in data:
            try:
                energy = int(data['energy_level'])
                if not 1 <= energy <= 10:
                    errors.append("Energy level must be between 1 and 10")
            except (ValueError, TypeError):
                errors.append("Energy level must be a number")

        # Validate social_contact_rating (1-5)
        if 'social_contact_rating' in data:
            try:
                social = int(data['social_contact_rating'])
                if not 1 <= social <= 5:
                    errors.append("Social contact rating must be between 1 and 5")
            except (ValueError, TypeError):
                errors.append("Social contact rating must be a number")

        # Validate sleep_hours (optional, 0-24)
        if 'sleep_hours' in data and data['sleep_hours'] is not None:
            try:
                sleep = float(data['sleep_hours'])
                if not 0 <= sleep <= 24:
                    errors.append("Sleep hours must be between 0 and 24")
            except (ValueError, TypeError):
                errors.append("Sleep hours must be a number")

        # Validate concerns length
        if 'concerns' in data and data['concerns']:
            if len(str(data['concerns'])) > self.max_concerns_length:
                errors.append(f"Concerns text is too long (max {self.max_concerns_length} characters)")

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    def _create_checkin_record(self, user_id: str, data: Dict[str, Any]) -> DailyCheckIn:
        """Create a DailyCheckIn object from validated data

        Args:
            user_id: User identifier
            data: Validated check-in data

        Returns:
            DailyCheckIn object
        """
        # Determine isolation flag based on social contact
        social_contact = int(data['social_contact_rating'])
        isolation_flag = social_contact <= 2  # Low social contact = isolation

        return DailyCheckIn(
            user_id=user_id,
            check_in_date=datetime.now(),
            mood_score=int(data['mood_score']),
            stress_level=int(data['stress_level']),
            energy_level=int(data['energy_level']),
            social_contact_rating=social_contact,
            sleep_hours=float(data.get('sleep_hours', 7.0)),  # Default to 7 hours
            isolation_flag=isolation_flag,
            concerns=data.get('concerns')
        )

    def _check_duplicate_checkin(self, user_id: str, check_date: date) -> bool:
        """Check if user already has a check-in for the given date

        Args:
            user_id: User identifier
            check_date: Date to check

        Returns:
            True if duplicate exists
        """
        try:
            checkins_table = self.config.get('databricks.tables.daily_checkins')

            query = f"""
            SELECT COUNT(*) as count FROM {checkins_table}
            WHERE user_id = '{user_id}'
            AND DATE(check_in_date) = '{check_date.isoformat()}'
            """

            df = self.client.execute_query(query)

            if len(df) > 0 and df.iloc[0]['count'] > 0:
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error checking for duplicate check-in: {e}")
            # Allow check-in if we can't verify (fail open)
            return False

    def _store_checkin(self, checkin: DailyCheckIn) -> bool:
        """Store check-in in Delta Lake

        Args:
            checkin: DailyCheckIn object to store

        Returns:
            True if successful
        """
        try:
            checkins_table = self.config.get('databricks.tables.daily_checkins')

            # Convert to DataFrame
            df = pd.DataFrame([checkin.to_dict()])

            # In production, this would use PySpark to write to Delta
            # For now, save locally for demonstration
            output_path = Path("data/processed/daily_checkins.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                # Append to existing file
                existing_df = pd.read_csv(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_csv(output_path, index=False)
            logger.info(f"💾 Stored check-in for user {checkin.user_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to store check-in: {e}")
            return False

    def _get_streak_message(self, streak: int) -> str:
        """Get motivational message based on streak length

        Args:
            streak: Current streak length

        Returns:
            Motivational message
        """
        if streak == 0:
            return "Let's start your wellbeing journey today! 🌟"
        elif streak == 1:
            return "Great start! Keep the momentum going! 🚀"
        elif streak < 7:
            return f"{streak} days in a row! You're building healthy habits! 💪"
        elif streak < 30:
            return f"{streak} day streak! You're committed to your wellbeing! 🌈"
        else:
            return f"Incredible {streak} day streak! You're a wellbeing champion! 🏆"

    def get_checkin_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get check-in statistics for a user

        Args:
            user_id: User identifier

        Returns:
            Dictionary with check-in statistics
        """
        try:
            # Get all check-ins for user
            checkins_table = self.config.get('databricks.tables.daily_checkins')

            query = f"""
            SELECT
                COUNT(*) as total_checkins,
                AVG(mood_score) as avg_mood,
                AVG(stress_level) as avg_stress,
                AVG(energy_level) as avg_energy,
                AVG(social_contact_rating) as avg_social,
                MIN(check_in_date) as first_checkin,
                MAX(check_in_date) as last_checkin
            FROM {checkins_table}
            WHERE user_id = '{user_id}'
            """

            df = self.client.execute_query(query)

            if len(df) == 0:
                return {"total_checkins": 0, "message": "No check-ins found"}

            stats = df.iloc[0].to_dict()

            # Calculate additional metrics
            stats['avg_mood'] = round(stats['avg_mood'], 1)
            stats['avg_stress'] = round(stats['avg_stress'], 1)
            stats['avg_energy'] = round(stats['avg_energy'], 1)
            stats['avg_social'] = round(stats['avg_social'], 1)

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting check-in statistics for user {user_id}: {e}")
            return {"error": "Failed to retrieve statistics"}
