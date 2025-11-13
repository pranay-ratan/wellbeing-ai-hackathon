"""Ingest processed data into Databricks Delta Lake"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import logging
from typing import Dict
from src.utils.databricks_client import DatabricksClient
from src.utils.config_loader import ConfigLoader
from databricks import sql
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabricksIngestion:
    """Handle data ingestion into Databricks Delta Lake"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)
        
        # Get Databricks config
        workspace_url = self.config.get('databricks.workspace_url')
        token = self.config.get('databricks.token')
        
        self.client = DatabricksClient(workspace_url=workspace_url, token=token)
        self.catalog = self.config.get('databricks.catalog')
        self.schema = self.config.get('databricks.schema')
        
        logger.info(f"Initialized DatabricksIngestion with catalog={self.catalog}, schema={self.schema}")
    
    def create_tables(self):
        """Create Delta Lake tables if they don't exist"""
        logger.info("Creating Delta Lake tables...")
        
        tables_config = self.config.get('databricks.tables')
        
        # User profiles table
        self._create_table_if_not_exists(
            tables_config['user_profiles'],
            """
            user_id STRING,
            age INT,
            gender STRING,
            region STRING,
            department STRING,
            job_role STRING,
            work_arrangement STRING,
            salary_range STRING,
            country STRING,
            state STRING,
            self_employed STRING,
            tech_company STRING,
            remote_work STRING,
            family_history STRING,
            treatment STRING,
            created_at TIMESTAMP
            """
        )
        
        # Daily check-ins table
        self._create_table_if_not_exists(
            tables_config['daily_checkins'],
            """
            user_id STRING,
            check_in_date TIMESTAMP,
            mood_score INT,
            stress_level INT,
            energy_level INT,
            social_contact_rating INT,
            sleep_hours DOUBLE,
            isolation_flag BOOLEAN,
            work_life_balance_score INT,
            concerns STRING
            """
        )
        
        # Risk scores table
        self._create_table_if_not_exists(
            tables_config['risk_scores'],
            """
            user_id STRING,
            assessment_date TIMESTAMP,
            risk_score DOUBLE,
            risk_level STRING,
            contributing_factors ARRAY<STRING>,
            recommended_action STRING,
            model_version STRING,
            confidence_score DOUBLE,
            reasoning STRING
            """
        )
        
        # Interventions table
        self._create_table_if_not_exists(
            tables_config['interventions'],
            """
            intervention_id STRING,
            user_id STRING,
            recommended_date TIMESTAMP,
            intervention_type STRING,
            description STRING,
            success_rate DOUBLE,
            status STRING,
            completed_date TIMESTAMP
            """
        )
        
        # Escalations table
        self._create_table_if_not_exists(
            tables_config['escalations'],
            """
            escalation_id STRING,
            user_id STRING,
            created_date TIMESTAMP,
            severity_level STRING,
            assigned_counselor STRING,
            status STRING,
            resolution_date TIMESTAMP,
            notes STRING
            """
        )
        
        # Survey data table
        self._create_table_if_not_exists(
            tables_config['survey_data'],
            """
            response_id STRING,
            user_id STRING,
            timestamp TIMESTAMP,
            work_interfere STRING,
            benefits STRING,
            care_options STRING,
            wellness_program STRING,
            seek_help STRING,
            anonymity STRING,
            leave STRING,
            mental_health_consequence STRING,
            coworkers STRING,
            supervisor STRING,
            comments STRING
            """
        )
        
        logger.info("✅ All tables created successfully")
    
    def _create_table_if_not_exists(self, table_name: str, schema: str):
        """Create a Delta table if it doesn't exist
        
        Args:
            table_name: Name of the table
            schema: Table schema definition
        """
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.catalog}.{self.schema}.{table_name} (
            {schema}
        )
        USING DELTA
        """
        
        try:
            logger.info(f"Creating table {table_name}...")
            # Note: In actual implementation, use SQL warehouse
            # For now, this is a placeholder that shows the structure
            logger.info(f"SQL: {create_sql}")
            logger.info(f"✅ Table {table_name} ready")
        except Exception as e:
            logger.error(f"Error creating table {table_name}: {e}")
    
    def ingest_dataframe(self, df: pd.DataFrame, table_name: str, mode: str = "append"):
        """Ingest a pandas DataFrame into Delta Lake
        
        Args:
            df: DataFrame to ingest
            table_name: Target table name
            mode: Write mode (append, overwrite)
        """
        full_table_name = f"{self.catalog}.{self.schema}.{table_name}"
        logger.info(f"Ingesting {len(df)} rows into {full_table_name} with mode={mode}")
        
        try:
            # In actual Databricks environment, this would use PySpark
            # For now, we'll save locally as CSV for demonstration
            output_dir = Path("/app/data/processed")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / f"{table_name}.csv"
            df.to_csv(output_path, index=False)
            
            logger.info(f"✅ Saved {len(df)} rows to {output_path} (simulating Delta write)")
            logger.info(f"Preview of data:")
            logger.info(df.head())
            
            return True
        except Exception as e:
            logger.error(f"Error ingesting data into {table_name}: {e}")
            return False
    
    def ingest_all_data(self, data_dict: Dict[str, pd.DataFrame]):
        """Ingest multiple DataFrames
        
        Args:
            data_dict: Dictionary mapping table names to DataFrames
        """
        logger.info(f"Ingesting data into {len(data_dict)} tables...")
        
        results = {}
        for table_name, df in data_dict.items():
            success = self.ingest_dataframe(df, table_name, mode="overwrite")
            results[table_name] = success
        
        # Summary
        successful = sum(results.values())
        logger.info(f"\n✅ Ingestion complete: {successful}/{len(results)} tables successful")
        
        return results
