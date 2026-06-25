"""
Parquet Data Processor Script
Loads, checks, and processes Parquet files for ML model training
Designed for retail sales prediction with KNN
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ParquetProcessor:
    """Process and prepare Parquet data for machine learning"""
    
    def __init__(self, file_path: str):
        """
        Initialize the processor with a Parquet file path
        
        Args:
            file_path: Path to the .parquet file
        """
        self.file_path = Path(file_path)
        self.df = None
        self.df_processed = None
        self.processing_report = {}
        
    # ==================== LOAD DATA ====================
    
    def load_data(self) -> pd.DataFrame:
        """Load Parquet file into pandas DataFrame"""
        try:
            logger.info(f"Loading Parquet file: {self.file_path}")
            
            # Check if file exists
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            
            # Load parquet file
            self.df = pd.read_parquet(self.file_path)
            logger.info(f"✓ Successfully loaded {len(self.df)} rows")
            return self.df
            
        except Exception as e:
            logger.error(f"✗ Failed to load Parquet file: {str(e)}")
            raise
    
    # ==================== CHECK DATA ====================
    
    def check_data(self) -> Dict:
        """
        Comprehensive data quality check
        
        Returns:
            Dictionary with data inspection results
        """
        if self.df is None:
            logger.warning("No data loaded. Call load_data() first.")
            return {}
        
        logger.info("\n" + "="*60)
        logger.info("DATA INSPECTION REPORT")
        logger.info("="*60)
        
        report = {}
        
        # 1. Basic Info
        logger.info("\n[1] DATASET SHAPE")
        logger.info(f"   Rows: {self.df.shape[0]}")
        logger.info(f"   Columns: {self.df.shape[1]}")
        report['shape'] = self.df.shape
        
        # 2. Column Info
        logger.info("\n[2] COLUMN INFORMATION")
        logger.info(f"{'Column':<20} {'Type':<15} {'Non-Null':<12} {'Null %':<10}")
        logger.info("-" * 60)
        
        col_info = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            non_null = self.df[col].notna().sum()
            null_pct = (self.df[col].isna().sum() / len(self.df)) * 100
            
            logger.info(f"{col:<20} {dtype:<15} {non_null:<12} {null_pct:.2f}%")
            col_info.append({
                'column': col,
                'dtype': dtype,
                'non_null': non_null,
                'null_percent': null_pct
            })
        
        report['columns'] = col_info
        
        # 3. Missing Values Analysis
        logger.info("\n[3] MISSING VALUES ANALYSIS")
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            logger.info(f"   Total missing values: {missing.sum()}")
            for col in missing[missing > 0].index:
                logger.info(f"   - {col}: {missing[col]} ({(missing[col]/len(self.df))*100:.2f}%)")
        else:
            logger.info("   ✓ No missing values found")
        
        report['missing_values'] = missing.to_dict()
        
        # 4. Duplicate Rows
        logger.info("\n[4] DUPLICATE ANALYSIS")
        duplicates = self.df.duplicated().sum()
        logger.info(f"   Total duplicate rows: {duplicates}")
        if duplicates > 0:
            logger.warning(f"   ⚠ Found {duplicates} duplicate rows")
        
        report['duplicates'] = duplicates
        
        # 5. Data Types Summary
        logger.info("\n[5] DATA TYPE SUMMARY")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            logger.info(f"   {dtype}: {count} columns")
        
        report['dtype_summary'] = dtype_counts.to_dict()
        
        # 6. Numeric Columns Statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info("\n[6] NUMERIC COLUMNS STATISTICS")
            logger.info(f"{'Column':<20} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
            logger.info("-" * 60)
            
            for col in numeric_cols:
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                mean_val = self.df[col].mean()
                std_val = self.df[col].std()
                
                logger.info(f"{col:<20} {min_val:<12.2f} {max_val:<12.2f} {mean_val:<12.2f} {std_val:<12.2f}")
            
            report['numeric_stats'] = self.df[numeric_cols].describe().to_dict()
        
        # 7. Categorical Columns Summary
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            logger.info("\n[7] CATEGORICAL COLUMNS SUMMARY")
            for col in categorical_cols:
                logger.info(f"\n   {col} (Unique: {self.df[col].nunique()})")
                value_counts = self.df[col].value_counts().head(5)
                for val, count in value_counts.items():
                    logger.info(f"      - {val}: {count} ({(count/len(self.df))*100:.2f}%)")
        
        # 8. Data Quality Score
        logger.info("\n[8] DATA QUALITY SCORE")
        quality_score = self._calculate_quality_score()
        logger.info(f"   Quality Score: {quality_score:.2f}%")
        report['quality_score'] = quality_score
        
        logger.info("\n" + "="*60 + "\n")
        
        self.processing_report['check_report'] = report
        return report
    
    def _calculate_quality_score(self) -> float:
        """Calculate overall data quality score (0-100)"""
        scores = []
        
        # Missing values score
        missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        missing_score = max(0, 100 - missing_pct)
        scores.append(missing_score)
        
        # Duplicate score
        dup_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        dup_score = max(0, 100 - dup_pct)
        scores.append(dup_score)
        
        return np.mean(scores)
    
    # ==================== PROCESS DATA ====================
    
    def process_data(self, 
                    drop_duplicates: bool = True,
                    handle_missing: str = 'drop',
                    encode_categorical: bool = True,
                    scale_numeric: bool = False) -> pd.DataFrame:
        """
        Process and clean the data
        
        Args:
            drop_duplicates: Remove duplicate rows
            handle_missing: 'drop' or 'mean'/'median' for numeric columns
            encode_categorical: Encode categorical variables
            scale_numeric: Scale numeric features (0-1)
        
        Returns:
            Processed DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("DATA PROCESSING")
        logger.info("="*60)
        
        # Start with a copy
        df = self.df.copy()
        logger.info(f"\nStarting with: {len(df)} rows, {len(df.columns)} columns")
        
        # Step 1: Remove Duplicates
        if drop_duplicates:
            logger.info("\n[STEP 1] Removing Duplicates")
            initial_rows = len(df)
            df = df.drop_duplicates()
            removed = initial_rows - len(df)
            if removed > 0:
                logger.info(f"   ✓ Removed {removed} duplicate rows")
            else:
                logger.info(f"   ✓ No duplicates found")
        
        # Step 2: Handle Missing Values
        logger.info(f"\n[STEP 2] Handling Missing Values (method: {handle_missing})")
        missing_before = df.isnull().sum().sum()
        
        if handle_missing == 'drop':
            df = df.dropna()
            logger.info(f"   ✓ Dropped rows with missing values")
        
        elif handle_missing in ['mean', 'median']:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    if handle_missing == 'mean':
                        fill_value = df[col].mean()
                    else:
                        fill_value = df[col].median()
                    df[col].fillna(fill_value, inplace=True)
                    logger.info(f"   ✓ Filled {col} with {handle_missing}")
            
            # Drop rows with missing categorical values
            df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        logger.info(f"   Missing values before: {missing_before}, after: {missing_after}")
        
        # Step 3: Handle Categorical Variables
        logger.info(f"\n[STEP 3] Encoding Categorical Variables")
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if encode_categorical and len(categorical_cols) > 0:
            for col in categorical_cols:
                unique_vals = df[col].nunique()
                
                if unique_vals <= 10:  # One-hot encode if <= 10 categories
                    logger.info(f"   ✓ One-hot encoding: {col} ({unique_vals} categories)")
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
                else:  # Label encode if > 10 categories
                    logger.info(f"   ✓ Label encoding: {col} ({unique_vals} categories)")
                    df[col] = pd.factorize(df[col])[0]
        
        # Step 4: Scale Numeric Features
        if scale_numeric:
            logger.info(f"\n[STEP 4] Scaling Numeric Features")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                
                if max_val - min_val != 0:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                    logger.info(f"   ✓ Scaled: {col}")
        
        # Step 5: Final Statistics
        logger.info(f"\n[STEP 5] Final Dataset")
        logger.info(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"   Data types: {df.dtypes.value_counts().to_dict()}")
        logger.info(f"   Missing values: {df.isnull().sum().sum()}")
        
        self.df_processed = df
        logger.info("\n" + "="*60 + "\n")
        
        return df
    
    # ==================== SAVE & EXPORT ====================
    
    def save_processed_data(self, output_path: str = None) -> str:
        """
        Save processed data to CSV and Parquet formats
        
        Args:
            output_path: Path to save (without extension)
        
        Returns:
            Path to saved files
        """
        if self.df_processed is None:
            logger.warning("No processed data. Run process_data() first.")
            return None
        
        if output_path is None:
            output_path = self.file_path.stem + "_processed"
        
        output_path = Path(output_path)
        
        try:
            # Save as CSV
            csv_path = output_path.with_suffix('.csv')
            self.df_processed.to_csv(csv_path, index=False)
            logger.info(f"✓ Saved to CSV: {csv_path}")
            
            # Save as Parquet
            parquet_path = output_path.with_suffix('.parquet')
            self.df_processed.to_parquet(parquet_path, index=False)
            logger.info(f"✓ Saved to Parquet: {parquet_path}")
            
            return str(output_path)
        
        except Exception as e:
            logger.error(f"✗ Failed to save: {str(e)}")
            raise
    
    # ==================== GET SUMMARY ====================
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        return {
            'original_shape': self.df.shape if self.df is not None else None,
            'processed_shape': self.df_processed.shape if self.df_processed is not None else None,
            'report': self.processing_report
        }
    
    def display_first_rows(self, n: int = 5) -> None:
        """Display first N rows"""
        if self.df is None:
            logger.warning("No data loaded")
            return
        
        logger.info(f"\nFirst {n} rows:")
        logger.info(self.df.head(n).to_string())


# ==================== MAIN EXECUTION ====================

def main():
    """
    Example usage of the ParquetProcessor
    """
    
    # ===== CONFIGURE THIS SECTION =====
    # File is in the same directory as this script
    PARQUET_FILE_PATH = "train-00000-of-00001-a5a7c6e4bb30b016.parquet"
    # ==================================
    
    try:
        # Initialize processor
        processor = ParquetProcessor(PARQUET_FILE_PATH)
        
        # 1. LOAD DATA
        processor.load_data()
        processor.display_first_rows(n=5)
        
        # 2. CHECK DATA
        processor.check_data()
        
        # 3. PROCESS DATA
        processed_df = processor.process_data(
            drop_duplicates=True,
            handle_missing='drop',      # Options: 'drop', 'mean', 'median'
            encode_categorical=True,
            scale_numeric=False          # Set to True for KNN (recommended)
        )
        
        # 4. SAVE PROCESSED DATA
        processor.save_processed_data(output_path="./processed_data")
        
        # 5. GET SUMMARY
        summary = processor.get_summary()
        logger.info("\n" + "="*60)
        logger.info("PROCESSING COMPLETE")
        logger.info("="*60)
        logger.info(f"Original shape: {summary['original_shape']}")
        logger.info(f"Processed shape: {summary['processed_shape']}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()