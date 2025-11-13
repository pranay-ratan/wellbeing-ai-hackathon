"""Process CSV files and prepare data for Delta Lake"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVProcessor:
    """Process and transform CSV data for WellbeingAI system"""
    
    def __init__(self):
        self.user_mapping = {}  # Map original IDs to anonymized UUIDs
    
    def process_post_pandemic_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process post-pandemic remote work health impact data
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (user_profiles, daily_checkins) DataFrames
        """
        logger.info(f"Processing post-pandemic data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded {len(df)} records from post-pandemic dataset")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Create user profiles
        user_profiles = self._create_user_profiles_from_pandemic(df)
        
        # Create daily check-ins from the data
        daily_checkins = self._create_checkins_from_pandemic(df)
        
        logger.info(f"Created {len(user_profiles)} user profiles")
        logger.info(f"Created {len(daily_checkins)} daily check-ins")
        
        return user_profiles, daily_checkins
    
    def process_survey_data(self, csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process mental health survey data
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Tuple of (user_profiles, survey_responses) DataFrames
        """
        logger.info(f"Processing survey data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        logger.info(f"Loaded {len(df)} records from survey dataset")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Create user profiles from survey
        user_profiles = self._create_user_profiles_from_survey(df)
        
        # Store survey responses
        survey_responses = self._process_survey_responses(df)
        
        logger.info(f"Created {len(user_profiles)} user profiles from survey")
        logger.info(f"Processed {len(survey_responses)} survey responses")
        
        return user_profiles, survey_responses
    
    def _create_user_profiles_from_pandemic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user profiles from post-pandemic data"""
        # Group by unique identifiers to create user profiles
        # Since there's no explicit user ID, create based on combinations
        df['temp_user_key'] = (
            df['Age'].astype(str) + '_' + 
            df['Gender'].astype(str) + '_' + 
            df['Region'].astype(str) + '_' +
            df['Industry'].astype(str) + '_' +
            df['Job_Role'].astype(str)
        )
        
        user_groups = df.groupby('temp_user_key').first().reset_index()
        
        profiles = []
        for idx, row in user_groups.iterrows():
            user_id = str(uuid.uuid4())
            self.user_mapping[row['temp_user_key']] = user_id
            
            profile = {
                'user_id': user_id,
                'age': int(row['Age']) if pd.notna(row['Age']) else None,
                'gender': str(row['Gender']) if pd.notna(row['Gender']) else 'Unknown',
                'region': str(row['Region']) if pd.notna(row['Region']) else 'Unknown',
                'department': str(row['Industry']) if pd.notna(row['Industry']) else 'Unknown',
                'job_role': str(row['Job_Role']) if pd.notna(row['Job_Role']) else 'Unknown',
                'work_arrangement': str(row['Work_Arrangement']) if pd.notna(row['Work_Arrangement']) else 'Unknown',
                'salary_range': str(row['Salary_Range']) if pd.notna(row['Salary_Range']) else 'Unknown',
                'created_at': datetime.now().isoformat()
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _create_checkins_from_pandemic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create daily check-ins from post-pandemic data"""
        # Add temp user key if not exists
        if 'temp_user_key' not in df.columns:
            df['temp_user_key'] = (
                df['Age'].astype(str) + '_' + 
                df['Gender'].astype(str) + '_' + 
                df['Region'].astype(str) + '_' +
                df['Industry'].astype(str) + '_' +
                df['Job_Role'].astype(str)
            )
        
        checkins = []
        for idx, row in df.iterrows():
            user_id = self.user_mapping.get(row['temp_user_key'])
            if not user_id:
                continue
            
            # Parse survey date
            check_in_date = pd.to_datetime(row['Survey_Date']) if pd.notna(row['Survey_Date']) else datetime.now()
            
            # Map mental health status and burnout to scores
            mood_score = self._map_mental_health_to_score(row.get('Mental_Health_Status', 'Unknown'))
            stress_level = self._map_burnout_to_score(row.get('Burnout_Level', 'Low'))
            energy_level = 10 - stress_level  # Inverse relationship
            social_contact = 5 - int(row.get('Social_Isolation_Score', 3))
            sleep_hours = 7.0 + np.random.uniform(-1, 1)  # Estimate
            
            # Check isolation flag
            isolation_flag = int(row.get('Social_Isolation_Score', 0)) >= 4
            
            checkin = {
                'user_id': user_id,
                'check_in_date': check_in_date.isoformat(),
                'mood_score': max(1, min(10, mood_score)),
                'stress_level': max(1, min(10, stress_level)),
                'energy_level': max(1, min(10, energy_level)),
                'social_contact_rating': max(1, min(5, social_contact)),
                'sleep_hours': max(4.0, min(9.0, sleep_hours)),
                'isolation_flag': isolation_flag,
                'work_life_balance_score': int(row.get('Work_Life_Balance_Score', 5)),
                'concerns': str(row.get('Physical_Health_Issues', '')) if pd.notna(row.get('Physical_Health_Issues')) else None
            }
            checkins.append(checkin)
        
        return pd.DataFrame(checkins)
    
    def _create_user_profiles_from_survey(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user profiles from survey data"""
        profiles = []
        
        for idx, row in df.iterrows():
            user_id = str(uuid.uuid4())
            
            profile = {
                'user_id': user_id,
                'age': int(row['Age']) if pd.notna(row['Age']) else None,
                'gender': str(row['Gender']) if pd.notna(row['Gender']) else 'Unknown',
                'country': str(row['Country']) if pd.notna(row['Country']) else 'Unknown',
                'state': str(row['state']) if pd.notna(row['state']) else 'Unknown',
                'self_employed': str(row['self_employed']) if pd.notna(row['self_employed']) else 'Unknown',
                'tech_company': str(row['tech_company']) if pd.notna(row['tech_company']) else 'Unknown',
                'remote_work': str(row['remote_work']) if pd.notna(row['remote_work']) else 'Unknown',
                'family_history': str(row['family_history']) if pd.notna(row['family_history']) else 'Unknown',
                'treatment': str(row['treatment']) if pd.notna(row['treatment']) else 'Unknown',
                'created_at': datetime.now().isoformat()
            }
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def _process_survey_responses(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process full survey responses for storage"""
        survey_data = []
        
        for idx, row in df.iterrows():
            user_id = str(uuid.uuid4())  # Should map to profile created above
            
            response = {
                'response_id': str(uuid.uuid4()),
                'user_id': user_id,
                'timestamp': pd.to_datetime(row['Timestamp']) if pd.notna(row['Timestamp']) else datetime.now(),
                'work_interfere': str(row['work_interfere']) if pd.notna(row['work_interfere']) else None,
                'benefits': str(row['benefits']) if pd.notna(row['benefits']) else None,
                'care_options': str(row['care_options']) if pd.notna(row['care_options']) else None,
                'wellness_program': str(row['wellness_program']) if pd.notna(row['wellness_program']) else None,
                'seek_help': str(row['seek_help']) if pd.notna(row['seek_help']) else None,
                'anonymity': str(row['anonymity']) if pd.notna(row['anonymity']) else None,
                'leave': str(row['leave']) if pd.notna(row['leave']) else None,
                'mental_health_consequence': str(row['mental_health_consequence']) if pd.notna(row['mental_health_consequence']) else None,
                'coworkers': str(row['coworkers']) if pd.notna(row['coworkers']) else None,
                'supervisor': str(row['supervisor']) if pd.notna(row['supervisor']) else None,
                'comments': str(row['comments']) if pd.notna(row['comments']) else None
            }
            survey_data.append(response)
        
        df_survey = pd.DataFrame(survey_data)
        df_survey['timestamp'] = pd.to_datetime(df_survey['timestamp'])
        return df_survey
    
    def _map_mental_health_to_score(self, status: str) -> int:
        """Map mental health status to mood score (1-10)"""
        mapping = {
            'Healthy': 8,
            'Stress Disorder': 4,
            'Anxiety': 5,
            'Depression': 3,
            'ADHD': 6,
            'Unknown': 7
        }
        return mapping.get(str(status), 7)
    
    def _map_burnout_to_score(self, burnout: str) -> int:
        """Map burnout level to stress score (1-10)"""
        mapping = {
            'Low': 3,
            'Medium': 6,
            'High': 9
        }
        return mapping.get(str(burnout), 5)
    
    def generate_time_series_data(self, user_profiles: pd.DataFrame, days: int = 90) -> pd.DataFrame:
        """Generate 90-day time series check-ins for each user
        
        Args:
            user_profiles: DataFrame with user profiles
            days: Number of days to generate
            
        Returns:
            DataFrame with daily check-ins
        """
        logger.info(f"Generating {days} days of time series data for {len(user_profiles)} users")
        
        all_checkins = []
        start_date = datetime.now() - timedelta(days=days)
        
        for _, user in user_profiles.iterrows():
            # Determine user's trend (declining, stable, improving)
            trend = np.random.choice(['declining', 'stable', 'improving'], p=[0.15, 0.65, 0.20])
            
            base_mood = np.random.randint(6, 9)
            base_stress = np.random.randint(3, 7)
            base_energy = np.random.randint(5, 8)
            
            for day in range(days):
                current_date = start_date + timedelta(days=day)
                
                # Apply trend
                if trend == 'declining':
                    mood_delta = -day * 0.03
                    stress_delta = day * 0.02
                    energy_delta = -day * 0.02
                elif trend == 'improving':
                    mood_delta = day * 0.02
                    stress_delta = -day * 0.02
                    energy_delta = day * 0.02
                else:  # stable
                    mood_delta = 0
                    stress_delta = 0
                    energy_delta = 0
                
                # Add random daily variation
                mood_score = int(np.clip(base_mood + mood_delta + np.random.uniform(-1, 1), 1, 10))
                stress_level = int(np.clip(base_stress + stress_delta + np.random.uniform(-1, 1), 1, 10))
                energy_level = int(np.clip(base_energy + energy_delta + np.random.uniform(-1, 1), 1, 10))
                
                checkin = {
                    'user_id': user['user_id'],
                    'check_in_date': current_date.isoformat(),
                    'mood_score': mood_score,
                    'stress_level': stress_level,
                    'energy_level': energy_level,
                    'social_contact_rating': np.random.randint(1, 6),
                    'sleep_hours': np.clip(np.random.normal(7, 1), 4, 9),
                    'isolation_flag': stress_level > 7 and np.random.random() > 0.5,
                    'concerns': None
                }
                all_checkins.append(checkin)
        
        logger.info(f"Generated {len(all_checkins)} check-in records")
        return pd.DataFrame(all_checkins)
