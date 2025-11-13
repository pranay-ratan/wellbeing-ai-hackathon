#!/usr/bin/env python3
"""Main script to process CSVs and ingest into Databricks"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from src.data_pipeline.csv_processor import CSVProcessor
from src.data_pipeline.databricks_ingestion import DatabricksIngestion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main ingestion pipeline"""
    logger.info("="*60)
    logger.info("🚀 Starting WellbeingAI Data Ingestion Pipeline")
    logger.info("="*60)
    
    # Initialize components
    processor = CSVProcessor()
    ingestion = DatabricksIngestion(config_path="config/config.yaml")
    
    # Step 1: Create tables
    logger.info("\n📊 Step 1: Creating Delta Lake tables...")
    ingestion.create_tables()
    
    # Step 2: Process post-pandemic data
    logger.info("\n📈 Step 2: Processing post-pandemic health impact data...")
    pandemic_csv = "/app/post_pandemic_remote_work_health_impact_2025.csv"
    user_profiles_1, checkins_1 = processor.process_post_pandemic_data(pandemic_csv)
    
    # Step 3: Generate 90-day time series data
    logger.info("\n⏰ Step 3: Generating 90-day time series check-ins...")
    checkins_timeseries = processor.generate_time_series_data(user_profiles_1, days=90)
    
    # Step 4: Process survey data
    logger.info("\n📋 Step 4: Processing mental health survey data...")
    survey_csv = "/app/survey.csv"
    user_profiles_2, survey_responses = processor.process_survey_data(survey_csv)
    
    # Step 5: Combine data
    logger.info("\n🔗 Step 5: Combining datasets...")
    
    # Combine user profiles (keep unique)
    import pandas as pd
    all_user_profiles = pd.concat([user_profiles_1, user_profiles_2], ignore_index=True)
    logger.info(f"Total user profiles: {len(all_user_profiles)}")
    
    # Combine check-ins
    all_checkins = pd.concat([checkins_1, checkins_timeseries], ignore_index=True)
    logger.info(f"Total check-ins: {len(all_checkins)}")
    
    # Step 6: Ingest all data
    logger.info("\n💾 Step 6: Ingesting data into Databricks Delta Lake...")
    
    data_to_ingest = {
        'wellbeing_user_profiles': all_user_profiles,
        'wellbeing_checkins': all_checkins,
        'wellbeing_survey_data': survey_responses
    }
    
    results = ingestion.ingest_all_data(data_to_ingest)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("✅ Data Ingestion Pipeline Complete!")
    logger.info("="*60)
    logger.info(f"📊 User Profiles: {len(all_user_profiles):,} records")
    logger.info(f"📈 Daily Check-ins: {len(all_checkins):,} records")
    logger.info(f"📋 Survey Responses: {len(survey_responses):,} records")
    logger.info("\n🎯 Next Steps:")
    logger.info("  1. Verify data in Databricks workspace")
    logger.info("  2. Run agents to generate risk scores")
    logger.info("  3. Test intervention recommendations")
    logger.info("="*60)


if __name__ == "__main__":
    main()
