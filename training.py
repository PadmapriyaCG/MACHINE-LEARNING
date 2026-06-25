"""
Complete KNN Classification Model Script
=========================================
Load preprocessed data → Train KNN → Save Model → Visualize Results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import pickle
import joblib
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class KNNClassificationModel:
    """Complete KNN Classification Pipeline"""
    
    def __init__(self, model_name: str = "knn_model"):
        """
        Initialize the KNN model
        
        Args:
            model_name: Name for saving model and artifacts
        """
        self.model_name = model_name
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.df = None
        self.target_column = None
        self.feature_columns = None
        self.label_encoder = None
        self.metrics = {}
        self.predictions = None
        self.y_test_pred = None
        
        # Create output directory
        self.output_dir = Path(f"./{model_name}_output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"✓ KNN Model Initialized: {model_name}")
    
    # ==================== LOAD DATA ====================
    
    def load_data(self, file_path: str, target_column: str = None) -> pd.DataFrame:
        """
        Load preprocessed data from parquet or CSV
        
        Args:
            file_path: Path to data file (.parquet or .csv)
            target_column: Name of target column (auto-detect if None)
        
        Returns:
            Loaded DataFrame
        """
        try:
            logger.info(f"\n{'='*70}")
            logger.info("PHASE 1: LOADING DATA")
            logger.info(f"{'='*70}")
            
            file_path = Path(file_path)
            
            # Load based on file extension
            if file_path.suffix == '.parquet':
                logger.info(f"Loading parquet file: {file_path.name}")
                self.df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                logger.info(f"Loading CSV file: {file_path.name}")
                self.df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"✓ Data loaded successfully")
            logger.info(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
            
            return self.df
        
        except Exception as e:
            logger.error(f"✗ Failed to load data: {str(e)}")
            raise
    
    # ==================== UNDERSTAND DATA ====================
    
    def understand_data(self, target_column: str = None) -> Dict:
        """
        Analyze and understand the dataset structure
        
        Args:
            target_column: Name of target column
                          If None, assumes last column is target
        
        Returns:
            Dictionary with data insights
        """
        if self.df is None:
            logger.error("No data loaded. Call load_data() first.")
            return {}
        
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 2: UNDERSTANDING DATA")
        logger.info(f"{'='*70}")
        
        # Auto-detect target column if not provided
        if target_column is None:
            target_column = self.df.columns[-1]
            logger.info(f"⚠ Target column not specified. Using last column: '{target_column}'")
        
        self.target_column = target_column
        self.feature_columns = [col for col in self.df.columns if col != target_column]
        
        # Display headers
        logger.info(f"\n[1] COLUMN HEADERS ({len(self.df.columns)} columns)")
        logger.info("-" * 70)
        for i, col in enumerate(self.df.columns, 1):
            dtype = str(self.df[col].dtype)
            if col == target_column:
                logger.info(f"  {i:2}. {col:<30} [{dtype:<10}] ← TARGET")
            else:
                logger.info(f"  {i:2}. {col:<30} [{dtype:<10}]")
        
        # Data types summary
        logger.info(f"\n[2] DATA TYPE SUMMARY")
        logger.info("-" * 70)
        for dtype, count in self.df.dtypes.value_counts().items():
            logger.info(f"  {dtype}: {count} columns")
        
        # Target variable analysis
        logger.info(f"\n[3] TARGET VARIABLE: '{target_column}'")
        logger.info("-" * 70)
        logger.info(f"  Data type: {self.df[target_column].dtype}")
        logger.info(f"  Unique values: {self.df[target_column].nunique()}")
        
        target_dist = self.df[target_column].value_counts()
        for value, count in target_dist.items():
            percentage = (count / len(self.df)) * 100
            logger.info(f"    - {value}: {count} ({percentage:.2f}%)")
        
        # Feature analysis
        logger.info(f"\n[4] FEATURE VARIABLES ({len(self.feature_columns)} features)")
        logger.info("-" * 70)
        
        numeric_features = self.df[self.feature_columns].select_dtypes(include=[np.number]).columns
        categorical_features = self.df[self.feature_columns].select_dtypes(include=['object']).columns
        
        logger.info(f"  Numeric features: {len(numeric_features)}")
        for col in numeric_features:
            logger.info(f"    - {col}: range [{self.df[col].min():.2f}, {self.df[col].max():.2f}]")
        
        logger.info(f"  Categorical features: {len(categorical_features)}")
        for col in categorical_features:
            logger.info(f"    - {col}: {self.df[col].nunique()} unique values")
        
        # Missing values
        logger.info(f"\n[5] DATA QUALITY")
        logger.info("-" * 70)
        missing = self.df.isnull().sum().sum()
        logger.info(f"  Missing values: {missing}")
        logger.info(f"  Duplicates: {self.df.duplicated().sum()}")
        
        logger.info(f"\n[6] DATASET PREVIEW")
        logger.info("-" * 70)
        logger.info(f"\n{self.df.head(3).to_string()}")
        
        return {
            'shape': self.df.shape,
            'target': target_column,
            'features': self.feature_columns,
            'numeric_features': list(numeric_features),
            'categorical_features': list(categorical_features)
        }
    
    # ==================== PREPARE DATA ====================
    
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training (split into train/test)
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 3: PREPARING DATA")
        logger.info(f"{'='*70}")
        
        # Prepare features and target
        X = self.df[self.feature_columns].copy()
        y = self.df[self.target_column].copy()
        
        # Handle categorical target if needed
        if y.dtype == 'object':
            logger.info(f"\nEncoding target variable: {self.target_column}")
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            logger.info(f"  Classes: {self.label_encoder.classes_}")
        
        # Split data
        logger.info(f"\nSplitting data: {(1-test_size)*100:.0f}% train, {test_size*100:.0f}% test")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"  Training set: {self.X_train.shape[0]} samples")
        logger.info(f"  Test set: {self.X_test.shape[0]} samples")
        logger.info(f"  Features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # ==================== FIND OPTIMAL K ====================
    
    def find_optimal_k(self, k_range: range = range(3, 31)) -> Dict:
        """
        Find optimal K value by testing multiple neighbors
        
        Args:
            k_range: Range of K values to test
        
        Returns:
            Dictionary with accuracy scores for each K
        """
        logger.info(f"\n{'='*70}")
        logger.info("FINDING OPTIMAL K VALUE")
        logger.info(f"{'='*70}")
        
        k_scores = {}
        best_k = None
        best_accuracy = 0
        
        logger.info(f"\nTesting K values: {k_range.start} to {k_range.stop-1}\n")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.X_train, self.y_train)
            accuracy = knn.score(self.X_test, self.y_test)
            k_scores[k] = accuracy
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
            
            if k % 5 == 0 or k == k_range.start:
                logger.info(f"  K={k:2d}: Accuracy = {accuracy:.4f}")
        
        logger.info(f"\n✓ Optimal K found: {best_k} with accuracy {best_accuracy:.4f}")
        
        return k_scores
    
    # ==================== TRAIN MODEL ====================
    
    def train_model(self, n_neighbors: int = 5, weights: str = 'uniform') -> Any:
        """
        Train the KNN classification model
        
        Args:
            n_neighbors: Number of neighbors to consider
            weights: 'uniform' or 'distance'
        
        Returns:
            Trained KNN model
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 4: TRAINING KNN MODEL")
        logger.info(f"{'='*70}")
        
        logger.info(f"\nModel Configuration:")
        logger.info(f"  Algorithm: K-Nearest Neighbors (KNN)")
        logger.info(f"  Number of neighbors (K): {n_neighbors}")
        logger.info(f"  Weights: {weights}")
        logger.info(f"  Training samples: {len(self.X_train)}")
        
        # Create and train model
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='auto',
            leaf_size=30,
            p=2
        )
        
        logger.info("\nTraining in progress...")
        self.model.fit(self.X_train, self.y_train)
        logger.info("✓ Model training completed")
        
        # Make predictions
        self.y_test_pred = self.model.predict(self.X_test)
        
        return self.model
    
    # ==================== EVALUATE MODEL ====================
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate model performance on test set
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 5: MODEL EVALUATION")
        logger.info(f"{'='*70}")
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.y_test_pred)
        precision = precision_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, self.y_test_pred, average='weighted', zero_division=0)
        
        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # Log metrics
        logger.info(f"\n[1] OVERALL METRICS")
        logger.info("-" * 70)
        logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1-Score:  {f1:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        logger.info(f"\n[2] CONFUSION MATRIX")
        logger.info("-" * 70)
        logger.info(f"\n{cm}")
        
        # Classification Report
        logger.info(f"\n[3] CLASSIFICATION REPORT")
        logger.info("-" * 70)
        class_report = classification_report(
            self.y_test, self.y_test_pred,
            target_names=[str(i) for i in range(len(np.unique(self.y_test)))]
        )
        logger.info(f"\n{class_report}")
        
        return self.metrics
    
    # ==================== SAVE MODEL ====================
    
    def save_model(self) -> str:
        """
        Save trained model and label encoder
        
        Returns:
            Path to saved model
        """
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 6: SAVING MODEL")
        logger.info(f"{'='*70}")
        
        try:
            # Save model using joblib
            model_path = self.output_dir / f"{self.model_name}.pkl"
            joblib.dump(self.model, model_path)
            logger.info(f"✓ Model saved: {model_path}")
            
            # Save label encoder if used
            if self.label_encoder is not None:
                encoder_path = self.output_dir / f"{self.model_name}_encoder.pkl"
                joblib.dump(self.label_encoder, encoder_path)
                logger.info(f"✓ Label encoder saved: {encoder_path}")
            
            # Save metadata
            metadata = {
                'model_name': self.model_name,
                'n_neighbors': self.model.n_neighbors,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'metrics': self.metrics,
                'classes': self.label_encoder.classes_.tolist() if self.label_encoder else None
            }
            
            metadata_path = self.output_dir / f"{self.model_name}_metadata.txt"
            with open(metadata_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            logger.info(f"✓ Metadata saved: {metadata_path}")
            
            return str(model_path)
        
        except Exception as e:
            logger.error(f"✗ Failed to save model: {str(e)}")
            raise
    
    # ==================== PREDICTIONS ====================
    
    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X_new: New feature data
        
        Returns:
            Predictions
        """
        if self.model is None:
            logger.error("Model not trained. Call train_model() first.")
            return None
        
        predictions = self.model.predict(X_new)
        
        # Decode if label encoder was used
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    # ==================== VISUALIZATIONS ====================
    
    def create_visualizations(self) -> None:
        """Create comprehensive visualizations of model results"""
        
        logger.info(f"\n{'='*70}")
        logger.info("PHASE 7: CREATING VISUALIZATIONS")
        logger.info(f"{'='*70}")
        
        # 1. Confusion Matrix Heatmap
        self._plot_confusion_matrix()
        
        # 2. Classification Metrics Bar Chart
        self._plot_metrics()
        
        # 3. Feature Importance (top features)
        self._plot_feature_importance()
        
        # 4. Model Performance Summary
        self._plot_performance_summary()
        
        logger.info(f"\n✓ All visualizations saved to: {self.output_dir}")
    
    def _plot_confusion_matrix(self) -> None:
        """Plot confusion matrix heatmap"""
        cm = confusion_matrix(self.y_test, self.y_test_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                    xticklabels=range(cm.shape[0]),
                    yticklabels=range(cm.shape[0]))
        plt.title('Confusion Matrix - KNN Classification', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / "01_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path.name}")
        plt.close()
    
    def _plot_metrics(self) -> None:
        """Plot classification metrics"""
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metrics_values = [
            self.metrics['accuracy'],
            self.metrics['precision'],
            self.metrics['recall'],
            self.metrics['f1_score']
        ]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title('Classification Metrics - KNN Model', fontsize=14, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "02_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path.name}")
        plt.close()
    
    def _plot_feature_importance(self) -> None:
        """Plot feature importance based on prediction distance"""
        # For KNN, calculate feature importance based on distance weights
        test_samples = self.X_test.iloc[:min(100, len(self.X_test))]
        
        # Get distances to nearest neighbors
        distances, indices = self.model.kneighbors(test_samples)
        
        # Calculate feature variance as importance metric
        feature_importance = self.X_train.std() * self.X_train.mean()
        feature_importance = feature_importance.sort_values(ascending=False)
        
        top_n = min(10, len(feature_importance))
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(top_features)), top_features.values, color='steelblue')
        plt.yticks(range(len(top_features)), top_features.index)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'Top {top_n} Important Features', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "03_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path.name}")
        plt.close()
    
    def _plot_performance_summary(self) -> None:
        """Plot overall performance summary"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Accuracy pie chart
        ax = axes[0, 0]
        accuracy = self.metrics['accuracy']
        ax.pie([accuracy, 1-accuracy], labels=['Correct', 'Incorrect'],
               autopct='%1.1f%%', colors=['#2ca02c', '#d62728'])
        ax.set_title('Prediction Accuracy', fontweight='bold')
        
        # 2. Metrics comparison
        ax = axes[0, 1]
        metrics_names = ['Acc', 'Prec', 'Rec', 'F1']
        metrics_values = [self.metrics['accuracy'], self.metrics['precision'],
                         self.metrics['recall'], self.metrics['f1_score']]
        ax.plot(metrics_names, metrics_values, marker='o', linewidth=2, markersize=8)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.set_title('Metrics Comparison', fontweight='bold')
        
        # 3. Prediction distribution
        ax = axes[1, 0]
        unique, counts = np.unique(self.y_test_pred, return_counts=True)
        ax.bar(unique, counts, color='skyblue')
        ax.set_xlabel('Predicted Class')
        ax.set_ylabel('Count')
        ax.set_title('Prediction Distribution', fontweight='bold')
        
        # 4. True vs Predicted
        ax = axes[1, 1]
        ax.scatter(self.y_test, self.y_test_pred, alpha=0.5, s=30)
        min_val = min(self.y_test.min(), self.y_test_pred.min())
        max_val = max(self.y_test.max(), self.y_test_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('True Label')
        ax.set_ylabel('Predicted Label')
        ax.set_title('True vs Predicted Labels', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / "04_performance_summary.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {save_path.name}")
        plt.close()
    
    # ==================== RUN COMPLETE PIPELINE ====================
    
    def run_pipeline(self, data_path: str, target_column: str = None, 
                    n_neighbors: int = 5, test_size: float = 0.2) -> Dict:
        """
        Run complete KNN pipeline
        
        Args:
            data_path: Path to preprocessed data file
            target_column: Name of target column
            n_neighbors: K value for KNN
            test_size: Test data fraction
        
        Returns:
            Dictionary with results
        """
        try:
            # 1. Load data
            self.load_data(data_path)
            
            # 2. Understand data
            self.understand_data(target_column)
            
            # 3. Prepare data
            self.prepare_data(test_size=test_size)
            
            # 4. Find optimal K (optional)
            logger.info("\nFinding optimal K value...")
            k_scores = self.find_optimal_k(k_range=range(3, 21))
            best_k = max(k_scores, key=k_scores.get)
            logger.info(f"Recommended K: {best_k}")
            
            # 5. Train model
            self.train_model(n_neighbors=best_k)
            
            # 6. Evaluate model
            self.evaluate_model()
            
            # 7. Save model
            self.save_model()
            
            # 8. Create visualizations
            self.create_visualizations()
            
            # Final summary
            logger.info(f"\n{'='*70}")
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"{'='*70}")
            logger.info(f"\nModel: {self.model_name}")
            logger.info(f"Best K value: {best_k}")
            logger.info(f"Accuracy: {self.metrics['accuracy']:.4f}")
            logger.info(f"F1-Score: {self.metrics['f1_score']:.4f}")
            logger.info(f"\nOutput directory: {self.output_dir}")
            logger.info(f"{'='*70}\n")
            
            return {
                'model': self.model,
                'metrics': self.metrics,
                'best_k': best_k,
                'output_dir': str(self.output_dir)
            }
        
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    DATA_FILE = "processed_data.parquet"  # Your data file path
    TARGET_COLUMN = None  # Set to None to use last column, or specify column name
    N_NEIGHBORS = 5  # Initial K value (will be optimized)
    TEST_SIZE = 0.2  # 20% for testing
    MODEL_NAME = "knn_retail_classifier"
    # =========================
    
    try:
        # Create and run pipeline
        knn_pipeline = KNNClassificationModel(model_name=MODEL_NAME)
        
        results = knn_pipeline.run_pipeline(
            data_path=DATA_FILE,
            target_column=TARGET_COLUMN,
            n_neighbors=N_NEIGHBORS,
            test_size=TEST_SIZE
        )
        
        # Make a sample prediction
        logger.info("\n" + "="*70)
        logger.info("SAMPLE PREDICTION")
        logger.info("="*70)
        
        sample = knn_pipeline.X_test.iloc[[0]]
        prediction = knn_pipeline.predict(sample)
        logger.info(f"\nSample features: {sample.values[0][:5]}...")
        logger.info(f"Predicted class: {prediction[0]}")
        logger.info("="*70 + "\n")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()