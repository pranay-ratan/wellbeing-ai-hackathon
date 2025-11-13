"""Databricks client wrapper"""

import os
from typing import Any, Dict, List, Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
import pandas as pd


class DatabricksClient:
    """Wrapper for Databricks SDK operations"""
    
    def __init__(self, workspace_url: str = None, token: str = None):
        """Initialize Databricks client
        
        Args:
            workspace_url: Databricks workspace URL
            token: Databricks access token
        """
        self.workspace_url = workspace_url or os.getenv('DATABRICKS_HOST')
        self.token = token or os.getenv('DATABRICKS_TOKEN')
        
        if not self.workspace_url or not self.token:
            raise ValueError("Databricks workspace_url and token must be provided")
        
        self.client = WorkspaceClient(
            host=self.workspace_url,
            token=self.token
        )
    
    def execute_sql(self, query: str, warehouse_id: str = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query to execute
            warehouse_id: SQL warehouse ID
        
        Returns:
            Query results as pandas DataFrame
        """
        warehouse_id = warehouse_id or os.getenv('DATABRICKS_WAREHOUSE_ID')
        
        if not warehouse_id:
            raise ValueError("warehouse_id must be provided or set in DATABRICKS_WAREHOUSE_ID")
        
        # Execute query
        response = self.client.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            statement=query,
            wait_timeout="30s"
        )
        
        # Convert to DataFrame
        if response.result and response.result.data_array:
            columns = [col.name for col in response.manifest.schema.columns]
            data = response.result.data_array
            return pd.DataFrame(data, columns=columns)
        
        return pd.DataFrame()
    
    def write_delta_table(self, df: pd.DataFrame, table_name: str, mode: str = "append"):
        """Write DataFrame to Delta table
        
        Args:
            df: DataFrame to write
            table_name: Full table name (catalog.schema.table)
            mode: Write mode (append, overwrite, etc.)
        """
        # This would use PySpark in actual Databricks environment
        # For now, this is a placeholder
        print(f"Writing {len(df)} rows to {table_name} with mode={mode}")
    
    def read_delta_table(self, table_name: str, filter_condition: str = None) -> pd.DataFrame:
        """Read Delta table into DataFrame
        
        Args:
            table_name: Full table name (catalog.schema.table)
            filter_condition: Optional WHERE clause
        
        Returns:
            DataFrame with table contents
        """
        query = f"SELECT * FROM {table_name}"
        if filter_condition:
            query += f" WHERE {filter_condition}"
        
        return self.execute_sql(query)
    
    def invoke_llm_endpoint(self, endpoint: str, prompt: str, **kwargs) -> str:
        """Invoke Databricks LLM serving endpoint
        
        Args:
            endpoint: Model serving endpoint name
            prompt: Prompt text
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Model response text
        """
        # Use serving endpoints API
        response = self.client.serving_endpoints.query(
            name=endpoint,
            inputs=[{"prompt": prompt}],
            **kwargs
        )
        
        # Extract response text
        if response.predictions:
            return response.predictions[0].get('candidates', [{}])[0].get('text', '')
        
        return ""
