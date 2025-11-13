# Databricks notebook source
"""
WellbeingAI Data Generation and Ingestion Notebook

This notebook handles the complete data pipeline for WellbeingAI:
1. Process CSV files (post-pandemic health data and survey data)
2. Generate synthetic time-series data
3. Create Delta Lake tables
4. Load data for agent processing

Run this notebook first to set up the data foundation.
"""

# Databricks notebook setup
dbutils.widgets.text("environment", "development", "Environment")
dbutils.widgets.text("data_scale", "small", "Data Scale (small/medium/large)")

# COMMAND ----------

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path for local development
sys.path.append("/Workspace/src")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# Configuration
ENVIRONMENT = dbutils.widgets.get("environment")
DATA_SCALE = dbutils.widgets.get("data_scale")

# Scale parameters
SCALE_CONFIG = {
    "small": {"num_users": 100, "days_history": 90},
    "medium": {"num_users": 1000, "days_history": 180},
    "large": {"num_users": 10000, "days_history": 365}
}

config = SCALE_CONFIG.get(DATA_SCALE, SCALE_CONFIG["small"])
NUM_USERS = config["num_users"]
DAYS_HISTORY = config["days_history"]

print(f"🚀 Starting WellbeingAI Data Pipeline")
print(f"Environment: {ENVIRONMENT}")
print(f"Data Scale: {DATA_SCALE}")
print(f"Users: {NUM_USERS:,}")
print(f"Days History: {DAYS_HISTORY}")

# COMMAND ----------

# DBTITLE 1. Mount Data Files (if needed)
# In production, data files would be in mounted storage
# For demo, we'll create synthetic data

try:
    # Check if CSV files exist
    post_pandemic_path = "/Workspace/post_pandemic_remote_work_health_impact_2025.csv"
    survey_path = "/Workspace/survey.csv"

    post_pandemic_exists = os.path.exists(post_pandemic_path)
    survey_exists = os.path.exists(survey_path)

    print(f"Post-pandemic data file exists: {post_pandemic_exists}")
    print(f"Survey data file exists: {survey_exists}")

except Exception as e:
    print(f"File check error: {e}")
    post_pandemic_exists = False
    survey_exists = False

# COMMAND ----------

# DBTITLE 1. Import Data Processing Modules
from src.data_pipeline.csv_processor import CSVProcessor
from src.data_pipeline.databricks_ingestion import DatabricksIngestion

# Initialize processors
processor = CSVProcessor()
ingestion = DatabricksIngestion()

print("✅ Data processing modules imported")

# COMMAND ----------

# DBTITLE 1. Create Delta Lake Tables
print("📊 Creating Delta Lake tables...")

try:
    ingestion.create_tables()
    print("✅ All tables created successfully")
except Exception as e:
    print(f"❌ Error creating tables: {e}")
    raise

# COMMAND ----------

# DBTITLE 1. Process Post-Pandemic Health Data
print("📈 Processing post-pandemic health impact data...")

try:
    if post_pandemic_exists:
        # Process real CSV data
        user_profiles_1, checkins_1 = processor.process_post_pandemic_data(post_pandemic_path)
        print(f"✅ Processed {len(user_profiles_1)} user profiles and {len(checkins_1)} check-ins from CSV")
    else:
        # Generate synthetic data
        print("📝 CSV file not found, generating synthetic post-pandemic data...")
        user_profiles_1 = processor._create_user_profiles_from_pandemic(pd.DataFrame())  # Empty frame triggers synthetic generation
        checkins_1 = processor.generate_time_series_data(user_profiles_1, days=min(30, DAYS_HISTORY))

    print(f"📊 Post-pandemic data: {len(user_profiles_1)} profiles, {len(checkins_1)} check-ins")

except Exception as e:
    print(f"❌ Error processing post-pandemic data: {e}")
    # Create minimal synthetic data as fallback
    user_profiles_1 = pd.DataFrame([{
        'user_id': f'user_{i}',
        'age': np.random.randint(25, 65),
        'gender': np.random.choice(['Male', 'Female']),
        'region': 'North America',
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR']),
        'job_role': f'Role_{i}',
        'created_at': datetime.now().isoformat()
    } for i in range(min(50, NUM_USERS//2))])

    checkins_1 = pd.DataFrame([{
        'user_id': user_profiles_1.iloc[i % len(user_profiles_1)]['user_id'],
        'check_in_date': (datetime.now() - timedelta(days=i%30)).isoformat(),
        'mood_score': np.random.randint(1, 11),
        'stress_level': np.random.randint(1, 11),
        'energy_level': np.random.randint(1, 11),
        'social_contact_rating': np.random.randint(1, 6),
        'sleep_hours': np.random.uniform(4, 9),
        'isolation_flag': np.random.random() > 0.8,
        'concerns': None
    } for i in range(min(500, len(user_profiles_1) * 10))])

# COMMAND ----------

# DBTITLE 1. Process Survey Data
print("📋 Processing mental health survey data...")

try:
    if survey_exists:
        # Process real survey data
        user_profiles_2, survey_responses = processor.process_survey_data(survey_path)
        print(f"✅ Processed {len(user_profiles_2)} survey profiles and {len(survey_responses)} responses")
    else:
        # Generate synthetic survey data
        print("📝 Survey CSV not found, generating synthetic survey data...")
        user_profiles_2 = pd.DataFrame([{
            'user_id': f'survey_user_{i}',
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['Male', 'Female', 'Other']),
            'country': 'United States',
            'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'WA']),
            'created_at': datetime.now().isoformat()
        } for i in range(min(50, NUM_USERS//2))])

        survey_responses = pd.DataFrame([{
            'response_id': f'response_{i}',
            'user_id': user_profiles_2.iloc[i % len(user_profiles_2)]['user_id'],
            'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 365))).isoformat(),
            'work_interfere': np.random.choice(['Often', 'Sometimes', 'Rarely', 'Never', None]),
            'benefits': np.random.choice(['Yes', 'No', "Don't know"]),
            'care_options': np.random.choice(['Yes', 'No', "Don't know"]),
            'comments': np.random.choice([None, "This is helpful", "Need more support"])
        } for i in range(min(200, len(user_profiles_2) * 4))])

    print(f"📊 Survey data: {len(user_profiles_2)} profiles, {len(survey_responses)} responses")

except Exception as e:
    print(f"❌ Error processing survey data: {e}")
    # Create minimal synthetic survey data
    user_profiles_2 = pd.DataFrame([{
        'user_id': f'survey_user_{i}',
        'age': np.random.randint(25, 65),
        'gender': np.random.choice(['Male', 'Female']),
        'created_at': datetime.now().isoformat()
    } for i in range(min(25, NUM_USERS//4))])

    survey_responses = pd.DataFrame([{
        'response_id': f'response_{i}',
        'user_id': user_profiles_2.iloc[i % len(user_profiles_2)]['user_id'],
        'timestamp': datetime.now().isoformat(),
        'work_interfere': 'Sometimes',
        'benefits': 'Yes',
        'comments': None
    } for i in range(min(50, len(user_profiles_2) * 2))])

# COMMAND ----------

# DBTITLE 1. Generate Comprehensive Time-Series Data
print(f"⏰ Generating {DAYS_HISTORY}-day time series data for {NUM_USERS} users...")

try:
    # Combine user profiles
    all_user_profiles = pd.concat([user_profiles_1, user_profiles_2], ignore_index=True)

    # Remove duplicates and limit to target number
    all_user_profiles = all_user_profiles.drop_duplicates(subset=['user_id']).head(NUM_USERS)

    # Generate comprehensive time series
    comprehensive_checkins = processor.generate_time_series_data(
        all_user_profiles, days=DAYS_HISTORY
    )

    print(f"✅ Generated {len(comprehensive_checkins):,} check-in records for {len(all_user_profiles)} users")

except Exception as e:
    print(f"❌ Error generating time series: {e}")
    # Fallback to basic time series
    comprehensive_checkins = checkins_1

# COMMAND ----------

# DBTITLE 1. Ingest All Data into Delta Lake
print("💾 Ingesting data into Delta Lake...")

try:
    # Prepare data for ingestion
    data_to_ingest = {
        'wellbeing_user_profiles': all_user_profiles,
        'wellbeing_checkins': comprehensive_checkins,
        'wellbeing_survey_data': survey_responses
    }

    # Ingest data
    results = ingestion.ingest_all_data(data_to_ingest)

    # Report results
    successful = sum(1 for result in results.values() if result)
    total = len(results)

    print(f"✅ Data ingestion complete: {successful}/{total} tables successful")

    for table_name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {table_name}")

except Exception as e:
    print(f"❌ Error during data ingestion: {e}")
    raise

# COMMAND ----------

# DBTITLE 1. Validate Data Ingestion
print("🔍 Validating data ingestion...")

try:
    # Quick validation queries
    validation_queries = {
        "user_profiles": "SELECT COUNT(*) as count FROM wellbeing.wellbeing_user_profiles",
        "checkins": "SELECT COUNT(*) as count FROM wellbeing.wellbeing_checkins",
        "survey_responses": "SELECT COUNT(*) as count FROM wellbeing.wellbeing_survey_data"
    }

    for table_name, query in validation_queries.items():
        try:
            result = spark.sql(query).collect()
            count = result[0]['count'] if result else 0
            print(f"✅ {table_name}: {count:,} records")
        except Exception as e:
            print(f"❌ {table_name}: Error - {e}")

except Exception as e:
    print(f"❌ Validation error: {e}")

# COMMAND ----------

# DBTITLE 1. Generate Sample Risk Scores (Optional)
print("🎯 Generating sample risk scores for testing...")

try:
    from src.agents.risk_scorer_agent import RiskScorerAgent

    # Initialize risk scorer
    risk_agent = RiskScorerAgent()

    # Generate risk scores for a sample of users
    sample_users = all_user_profiles['user_id'].head(min(10, len(all_user_profiles))).tolist()

    print(f"🔍 Scoring risk for {len(sample_users)} sample users...")

    risk_results = risk_agent.batch_score_users(sample_users)

    successful_scores = sum(1 for result in risk_results.values() if result)
    print(f"✅ Generated {successful_scores}/{len(sample_users)} risk scores")

    # Show sample results
    for user_id, risk_score in list(risk_results.items())[:3]:
        if risk_score:
            score = risk_score.risk_score if hasattr(risk_score, 'risk_score') else 'N/A'
            print(f"  {user_id}: Risk Score = {score}")

except Exception as e:
    print(f"⚠️ Sample risk scoring failed (expected in some environments): {e}")

# COMMAND ----------

# DBTITLE 1. Data Pipeline Summary
print("="*60)
print("🎉 WellbeingAI Data Pipeline Complete!")
print("="*60)

summary = {
    "environment": ENVIRONMENT,
    "data_scale": DATA_SCALE,
    "total_users": len(all_user_profiles),
    "total_checkins": len(comprehensive_checkins),
    "total_survey_responses": len(survey_responses),
    "date_range_days": DAYS_HISTORY,
    "tables_created": len(results),
    "ingestion_successful": sum(1 for result in results.values() if result),
    "completion_time": datetime.now().isoformat()
}

for key, value in summary.items():
    print(f"{key.replace('_', ' ').title()}: {value}")

print("\n📋 Next Steps:")
print("1. Run 02_risk_scorer_test.py to test risk scoring")
print("2. Run 03_demo.py for full system demonstration")
print("3. Check Delta Lake tables in Databricks workspace")

print("="*60)

# COMMAND ----------

# DBTITLE 1. Export Summary for Other Notebooks
# Make summary available to other notebooks
dbutils.notebook.exit(json.dumps(summary))
