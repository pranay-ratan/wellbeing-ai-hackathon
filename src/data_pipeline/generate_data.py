"""Generate synthetic mental health check-in data"""

import random
import uuid
from datetime import datetime, timedelta
from typing import List
import pandas as pd
import numpy as np
from pathlib import Path


class SyntheticDataGenerator:
    """Generate realistic synthetic mental health data"""
    
    def __init__(self, num_users: int = 100, num_days: int = 90, start_date: str = "2024-08-01"):
        self.num_users = num_users
        self.num_days = num_days
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.user_ids = [f"user_{str(uuid.uuid4())[:8]}" for _ in range(num_users)]
        
        # Assign user trajectories
        self.declining_users = set(random.sample(self.user_ids, int(num_users * 0.15)))
        self.improving_users = set(random.sample(
            [u for u in self.user_ids if u not in self.declining_users],
            int(num_users * 0.20)
        ))
        self.stable_users = set(u for u in self.user_ids 
                               if u not in self.declining_users and u not in self.improving_users)
    
    def generate_check_in(self, user_id: str, day: int) -> dict:
        """Generate a single check-in record
        
        Args:
            user_id: User identifier
            day: Day number (0 to num_days-1)
        
        Returns:
            Check-in record as dictionary
        """
        check_in_date = self.start_date + timedelta(days=day)
        
        # Determine trajectory
        if user_id in self.declining_users:
            # Declining mental health
            trend_factor = -day / self.num_days  # Gets more negative
            base_mood = 7
            base_stress = 4
            base_energy = 7
        elif user_id in self.improving_users:
            # Improving mental health
            trend_factor = day / self.num_days  # Gets more positive
            base_mood = 5
            base_stress = 7
            base_energy = 5
        else:
            # Stable
            trend_factor = 0
            base_mood = 6.5
            base_stress = 5
            base_energy = 6.5
        
        # Add some randomness
        noise = random.gauss(0, 0.5)
        
        # Calculate scores
        mood_score = max(1, min(10, int(base_mood + trend_factor * 5 + noise)))
        stress_level = max(1, min(10, int(base_stress - trend_factor * 5 + noise)))
        energy_level = max(1, min(10, int(base_energy + trend_factor * 4 + noise)))
        social_contact = max(1, min(5, int(3 + trend_factor * 2 + noise * 0.5)))
        sleep_hours = max(4, min(9, 7 + trend_factor * 2 + random.gauss(0, 0.8)))
        
        # Isolation flag for declining users
        isolation_flag = user_id in self.declining_users and day > self.num_days * 0.5 and random.random() < 0.3
        
        # Generate concerns for at-risk users
        concerns = None
        if mood_score <= 4 or stress_level >= 8:
            concern_options = [
                "Feeling overwhelmed with work",
                "Having trouble sleeping",
                "Feeling disconnected from team",
                "Difficulty concentrating",
                "Feeling anxious about deadlines",
                "Missing social interactions",
                None  # Sometimes no specific concern mentioned
            ]
            concerns = random.choice(concern_options)
        
        return {
            'user_id': user_id,
            'check_in_date': check_in_date.strftime('%Y-%m-%d'),
            'mood_score': mood_score,
            'stress_level': stress_level,
            'energy_level': energy_level,
            'social_contact_rating': social_contact,
            'sleep_hours': round(sleep_hours, 1),
            'isolation_flag': isolation_flag,
            'concerns': concerns
        }
    
    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete dataset of check-ins
        
        Returns:
            DataFrame with all check-in records
        """
        records = []
        
        for day in range(self.num_days):
            for user_id in self.user_ids:
                # Not everyone checks in every day (85% compliance rate)
                if random.random() < 0.85:
                    records.append(self.generate_check_in(user_id, day))
        
        df = pd.DataFrame(records)
        return df
    
    def save_to_csv(self, output_path: str = "data/synthetic_checkins.csv"):
        """Generate and save dataset to CSV
        
        Args:
            output_path: Path to save CSV file
        """
        df = self.generate_dataset()
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} check-in records")
        print(f"Saved to {output_path}")
        print(f"\nDataset summary:")
        print(f"  Users: {df['user_id'].nunique()}")
        print(f"  Date range: {df['check_in_date'].min()} to {df['check_in_date'].max()}")
        print(f"  Declining users: {len(self.declining_users)}")
        print(f"  Improving users: {len(self.improving_users)}")
        print(f"  Stable users: {len(self.stable_users)}")
        
        return df


if __name__ == "__main__":
    generator = SyntheticDataGenerator(num_users=100, num_days=90)
    df = generator.save_to_csv()
    
    print("\nSample records:")
    print(df.head())
