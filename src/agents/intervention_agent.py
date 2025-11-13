"""Intervention Agent - Personalized Intervention Recommendations

This agent uses Vector Search to match users with the most effective interventions
based on their risk profile and historical data. It recommends personalized actions
like breathing exercises, peer support groups, or counseling sessions.

Key Features:
- Vector Search for semantic matching of interventions
- Personalized recommendations based on risk profile
- Success rate tracking and optimization
- Multiple intervention types and categories
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import uuid
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

from src.models.schemas import Intervention
from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class InterventionAgent:
    """Agent for recommending personalized interventions using Vector Search"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the Intervention Agent

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        self.client = DatabricksClient(
            workspace_url=self.config.get('databricks.workspace_url'),
            token=self.config.get('databricks.token')
        )

        # Vector Search configuration
        self.vector_search_config = self.config.get('vector_search')
        self.intervention_types = self.config.get('intervention.types', [])

        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model()

        # Load intervention database
        self.interventions_db = self._load_interventions_database()

        logger.info("✅ Intervention Agent initialized with Vector Search")

    def _initialize_embedding_model(self) -> SentenceTransformer:
        """Initialize the sentence transformer for embeddings"""
        try:
            model_name = self.vector_search_config.get('embedding_model', 'all-MiniLM-L6-v2')
            model = SentenceTransformer(model_name)
            logger.info(f"✅ Embedding model loaded: {model_name}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            return None

    def _load_interventions_database(self) -> pd.DataFrame:
        """Load the interventions database with embeddings"""
        try:
            # In production, this would be a Delta table with pre-computed embeddings
            # For demo, create a comprehensive interventions database
            interventions = self._create_interventions_database()

            # Generate embeddings for semantic search
            if self.embedding_model:
                descriptions = interventions['description'].tolist()
                embeddings = self.embedding_model.encode(descriptions, convert_to_numpy=True)
                interventions['embedding'] = list(embeddings)

            logger.info(f"✅ Loaded {len(interventions)} interventions with embeddings")
            return interventions

        except Exception as e:
            logger.error(f"❌ Failed to load interventions database: {e}")
            return pd.DataFrame()

    def recommend_interventions(self, user_id: str, risk_score: float,
                              contributing_factors: List[str],
                              time_range_days: int = 30) -> List[Dict[str, Any]]:
        """Recommend personalized interventions for a user

        Args:
            user_id: User identifier
            risk_score: User's current risk score (0-1)
            contributing_factors: List of factors contributing to risk
            time_range_days: Time range to consider for historical data

        Returns:
            List of recommended interventions
        """
        logger.info(f"🎯 Generating interventions for user {user_id} (risk: {risk_score:.3f})")

        try:
            # Get user's historical data for context
            user_context = self._get_user_context(user_id, time_range_days)

            # Generate intervention recommendations
            recommendations = self._generate_recommendations(
                risk_score, contributing_factors, user_context
            )

            # Store recommendations
            self._store_interventions(user_id, recommendations)

            logger.info(f"✅ Generated {len(recommendations)} intervention recommendations")
            return recommendations

        except Exception as e:
            logger.error(f"❌ Error generating interventions for user {user_id}: {e}")
            return self._get_fallback_interventions(risk_score)

    def _get_user_context(self, user_id: str, time_range_days: int) -> Dict[str, Any]:
        """Get user's historical context for personalization

        Args:
            user_id: User identifier
            time_range_days: Time range for historical data

        Returns:
            Dictionary with user context information
        """
        try:
            # Get recent check-ins and past interventions
            end_date = datetime.now()
            start_date = end_date - timedelta(days=time_range_days)

            # Query recent check-ins
            checkins_query = f"""
            SELECT AVG(mood_score) as avg_mood, AVG(stress_level) as avg_stress,
                   AVG(energy_level) as avg_energy, AVG(social_contact_rating) as avg_social
            FROM wellbeing.daily_checkins
            WHERE user_id = '{user_id}'
            AND check_in_date >= '{start_date.isoformat()}'
            """

            checkins_df = self.client.execute_query(checkins_query)

            # Query past interventions and their success
            interventions_query = f"""
            SELECT intervention_type, success_rate, completed_date
            FROM wellbeing.interventions
            WHERE user_id = '{user_id}'
            ORDER BY recommended_date DESC
            LIMIT 10
            """

            interventions_df = self.client.execute_query(interventions_query)

            context = {
                "avg_mood": checkins_df.iloc[0]['avg_mood'] if len(checkins_df) > 0 else 5.0,
                "avg_stress": checkins_df.iloc[0]['avg_stress'] if len(checkins_df) > 0 else 5.0,
                "avg_energy": checkins_df.iloc[0]['avg_energy'] if len(checkins_df) > 0 else 5.0,
                "avg_social": checkins_df.iloc[0]['avg_social'] if len(checkins_df) > 0 else 3.0,
                "past_interventions": interventions_df.to_dict('records') if len(interventions_df) > 0 else []
            }

            return context

        except Exception as e:
            logger.error(f"❌ Error getting user context: {e}")
            return {
                "avg_mood": 5.0,
                "avg_stress": 5.0,
                "avg_energy": 5.0,
                "avg_social": 3.0,
                "past_interventions": []
            }

    def _generate_recommendations(self, risk_score: float, factors: List[str],
                                user_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized intervention recommendations

        Args:
            risk_score: User's risk score
            factors: Contributing risk factors
            user_context: User's historical context

        Returns:
            List of recommended interventions
        """
        recommendations = []

        # Determine intervention priority based on risk level
        if risk_score >= 0.7:
            # High risk - immediate, intensive interventions
            recommendations.extend(self._get_crisis_interventions(factors, user_context))
        elif risk_score >= 0.4:
            # Moderate risk - preventive interventions
            recommendations.extend(self._get_moderate_risk_interventions(factors, user_context))
        else:
            # Low risk - maintenance interventions
            recommendations.extend(self._get_maintenance_interventions(factors, user_context))

        # Ensure we have exactly 3 recommendations
        while len(recommendations) < 3:
            recommendations.append(self._get_default_intervention())

        return recommendations[:3]  # Limit to top 3

    def _get_crisis_interventions(self, factors: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get interventions for high-risk users"""
        interventions = []

        # Check for specific risk factors
        factor_text = " ".join(factors).lower()

        if "isolation" in factor_text or "social" in factor_text:
            interventions.append(self._find_best_intervention("peer_support_group"))
        if "stress" in factor_text or "anxiety" in factor_text:
            interventions.append(self._find_best_intervention("breathing_exercise"))
        if "mood" in factor_text or "depression" in factor_text:
            interventions.append(self._find_best_intervention("counselor_session"))

        # Add immediate professional support
        interventions.append(self._find_best_intervention("crisis_hotline"))

        return interventions

    def _get_moderate_risk_interventions(self, factors: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get interventions for moderate-risk users"""
        interventions = []

        # Check user context for personalization
        if context.get('avg_stress', 5) > 6:
            interventions.append(self._find_best_intervention("stress_management_course"))
        if context.get('avg_social', 3) < 3:
            interventions.append(self._find_best_intervention("social_connection"))
        if context.get('avg_energy', 5) < 5:
            interventions.append(self._find_best_intervention("physical_activity"))

        # Add wellness app for ongoing support
        interventions.append(self._find_best_intervention("meditation_app"))

        return interventions

    def _get_maintenance_interventions(self, factors: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get interventions for low-risk users (maintenance/prevention)"""
        interventions = []

        # Focus on prevention and wellness
        interventions.append(self._find_best_intervention("wellness_workshop"))
        interventions.append(self._find_best_intervention("meditation_app"))
        interventions.append(self._find_best_intervention("physical_activity"))

        return interventions

    def _find_best_intervention(self, intervention_type: str) -> Dict[str, Any]:
        """Find the best intervention of a specific type using vector search

        Args:
            intervention_type: Type of intervention to find

        Returns:
            Best matching intervention
        """
        try:
            # Filter interventions by type
            type_interventions = self.interventions_db[
                self.interventions_db['intervention_type'] == intervention_type
            ]

            if len(type_interventions) == 0:
                return self._get_default_intervention()

            # For demo, return the highest success rate intervention
            # In production, this would use vector similarity search
            best_intervention = type_interventions.loc[
                type_interventions['success_rate'].idxmax()
            ]

            return {
                "intervention_id": str(uuid.uuid4()),
                "intervention_type": best_intervention['intervention_type'],
                "description": best_intervention['description'],
                "success_rate": best_intervention['success_rate'],
                "estimated_benefit": self._estimate_benefit(best_intervention),
                "time_commitment": best_intervention.get('time_commitment', '15-30 minutes'),
                "difficulty": best_intervention.get('difficulty', 'Easy')
            }

        except Exception as e:
            logger.error(f"❌ Error finding intervention {intervention_type}: {e}")
            return self._get_default_intervention()

    def _estimate_benefit(self, intervention: pd.Series) -> str:
        """Estimate the benefit level of an intervention"""
        success_rate = intervention.get('success_rate', 0.5)

        if success_rate >= 0.8:
            return "High"
        elif success_rate >= 0.6:
            return "Moderate"
        else:
            return "Low"

    def _get_default_intervention(self) -> Dict[str, Any]:
        """Get a default intervention when others fail"""
        return {
            "intervention_id": str(uuid.uuid4()),
            "intervention_type": "breathing_exercise",
            "description": "Try the 4-7-8 breathing technique: Inhale for 4 seconds, hold for 7 seconds, exhale for 8 seconds. Repeat 4 times.",
            "success_rate": 0.75,
            "estimated_benefit": "Moderate",
            "time_commitment": "5 minutes",
            "difficulty": "Easy"
        }

    def _get_fallback_interventions(self, risk_score: float) -> List[Dict[str, Any]]:
        """Get fallback interventions when generation fails"""
        return [
            {
                "intervention_id": str(uuid.uuid4()),
                "intervention_type": "breathing_exercise",
                "description": "Practice deep breathing exercises for 5 minutes",
                "success_rate": 0.7,
                "estimated_benefit": "Moderate",
                "time_commitment": "5 minutes",
                "difficulty": "Easy"
            },
            {
                "intervention_id": str(uuid.uuid4()),
                "intervention_type": "peer_support_group",
                "description": "Join a peer support group for shared experiences",
                "success_rate": 0.65,
                "estimated_benefit": "High",
                "time_commitment": "1 hour weekly",
                "difficulty": "Medium"
            },
            {
                "intervention_id": str(uuid.uuid4()),
                "intervention_type": "counselor_session",
                "description": "Schedule a session with a mental health counselor",
                "success_rate": 0.8,
                "estimated_benefit": "High",
                "time_commitment": "50 minutes",
                "difficulty": "Medium"
            }
        ]

    def _store_interventions(self, user_id: str, recommendations: List[Dict[str, Any]]) -> None:
        """Store intervention recommendations in Delta Lake

        Args:
            user_id: User identifier
            recommendations: List of intervention recommendations
        """
        try:
            interventions_table = self.config.get('databricks.tables.interventions')

            intervention_records = []
            for rec in recommendations:
                intervention = Intervention(
                    intervention_id=rec['intervention_id'],
                    user_id=user_id,
                    recommended_date=datetime.now(),
                    intervention_type=rec['intervention_type'],
                    description=rec['description'],
                    success_rate=rec['success_rate']
                )
                intervention_records.append(intervention.to_dict())

            # Convert to DataFrame and save
            df = pd.DataFrame(intervention_records)

            # In production, this would use PySpark to write to Delta
            # For now, save locally for demonstration
            output_path = Path("data/processed/interventions.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.exists():
                existing_df = pd.read_csv(output_path)
                df = pd.concat([existing_df, df], ignore_index=True)

            df.to_csv(output_path, index=False)
            logger.info(f"💾 Stored {len(recommendations)} interventions for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Failed to store interventions: {e}")

    def _create_interventions_database(self) -> pd.DataFrame:
        """Create a comprehensive database of interventions"""
        interventions = [
            {
                "intervention_type": "breathing_exercise",
                "description": "4-7-8 Breathing: Inhale quietly through nose for 4 seconds, hold breath for 7 seconds, exhale through mouth for 8 seconds. Repeat 4 times.",
                "success_rate": 0.78,
                "time_commitment": "5 minutes",
                "difficulty": "Easy",
                "category": "Stress Relief"
            },
            {
                "intervention_type": "peer_support_group",
                "description": "Join a moderated peer support group where you can share experiences and learn from others facing similar challenges.",
                "success_rate": 0.82,
                "time_commitment": "1 hour weekly",
                "difficulty": "Medium",
                "category": "Social Support"
            },
            {
                "intervention_type": "counselor_session",
                "description": "Schedule a one-on-one session with a licensed mental health counselor for personalized support and guidance.",
                "success_rate": 0.85,
                "time_commitment": "50 minutes",
                "difficulty": "Medium",
                "category": "Professional Support"
            },
            {
                "intervention_type": "meditation_app",
                "description": "Use a guided meditation app like Headspace or Calm for daily mindfulness and stress reduction exercises.",
                "success_rate": 0.71,
                "time_commitment": "10-20 minutes daily",
                "difficulty": "Easy",
                "category": "Digital Wellness"
            },
            {
                "intervention_type": "physical_activity",
                "description": "Engage in moderate physical activity like walking, yoga, or swimming for at least 30 minutes, 3-5 times per week.",
                "success_rate": 0.76,
                "time_commitment": "30-60 minutes",
                "difficulty": "Medium",
                "category": "Physical Health"
            },
            {
                "intervention_type": "stress_management_course",
                "description": "Enroll in an online or in-person stress management course covering techniques like time management and relaxation.",
                "success_rate": 0.73,
                "time_commitment": "2 hours weekly for 6 weeks",
                "difficulty": "Medium",
                "category": "Education"
            },
            {
                "intervention_type": "social_connection",
                "description": "Reach out to friends or family for meaningful social interaction, or join a hobby group or club.",
                "success_rate": 0.79,
                "time_commitment": "1-2 hours",
                "difficulty": "Easy",
                "category": "Social Support"
            },
            {
                "intervention_type": "wellness_workshop",
                "description": "Attend a workplace wellness workshop on topics like work-life balance, mindfulness, or healthy communication.",
                "success_rate": 0.68,
                "time_commitment": "2 hours",
                "difficulty": "Easy",
                "category": "Education"
            },
            {
                "intervention_type": "crisis_hotline",
                "description": "Contact a crisis hotline like 988 Suicide & Crisis Lifeline for immediate support during difficult moments.",
                "success_rate": 0.88,
                "time_commitment": "As needed",
                "difficulty": "Easy",
                "category": "Crisis Support"
            },
            {
                "intervention_type": "sleep_hygiene",
                "description": "Improve sleep quality by maintaining consistent sleep schedule, creating a relaxing bedtime routine, and optimizing your sleep environment.",
                "success_rate": 0.74,
                "time_commitment": "Ongoing",
                "difficulty": "Medium",
                "category": "Sleep Health"
            }
        ]

        return pd.DataFrame(interventions)

    def get_intervention_statistics(self) -> Dict[str, Any]:
        """Get intervention effectiveness statistics"""
        try:
            if len(self.interventions_db) == 0:
                return {"error": "No intervention data available"}

            stats = {
                "total_interventions": len(self.interventions_db),
                "avg_success_rate": self.interventions_db['success_rate'].mean(),
                "intervention_types": self.interventions_db['intervention_type'].value_counts().to_dict(),
                "top_performing": self.interventions_db.nlargest(3, 'success_rate')[['intervention_type', 'success_rate']].to_dict('records'),
                "categories": self.interventions_db['category'].value_counts().to_dict()
            }

            return stats

        except Exception as e:
            logger.error(f"❌ Error getting intervention statistics: {e}")
            return {"error": "Failed to retrieve statistics"}

    def update_intervention_success(self, intervention_id: str, success_rating: float) -> bool:
        """Update the success rate of an intervention based on user feedback

        Args:
            intervention_id: Intervention identifier
            success_rating: User's rating of success (0-1)

        Returns:
            True if update successful
        """
        try:
            # In production, this would update the interventions table
            # For demo, just log the feedback
            logger.info(f"📊 Intervention {intervention_id} success rating: {success_rating}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to update intervention success: {e}")
            return False
