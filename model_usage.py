"""
KNN Model CLI Interface
======================
Real-time prediction interface for KNN retail classification model
Load model → Get user input → Make predictions → Display results
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Any
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KNNModelCLI:
    """CLI Interface for KNN Model Predictions"""
    
    def __init__(self, model_path: str, encoder_path: str = None, metadata_path: str = None):
        """
        Initialize CLI with trained model
        
        Args:
            model_path: Path to .pkl model file
            encoder_path: Path to label encoder (optional)
            metadata_path: Path to metadata file (optional)
        """
        self.model_path = Path(model_path)
        self.encoder_path = Path(encoder_path) if encoder_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        self.model = None
        self.encoder = None
        self.metadata = {}
        self.feature_columns = None
        self.n_features = None
        
        self._load_model()
        self._load_encoder()
        self._load_metadata()
    
    def _load_model(self) -> None:
        """Load the trained KNN model"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.n_features = self.model.n_features_in_
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"  Model type: {type(self.model).__name__}")
            logger.info(f"  Number of features: {self.n_features}")
            logger.info(f"  Number of neighbors (K): {self.model.n_neighbors}")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {str(e)}")
            raise
    
    def _load_encoder(self) -> None:
        """Load the label encoder if available"""
        if self.encoder_path and self.encoder_path.exists():
            try:
                logger.info(f"Loading label encoder from: {self.encoder_path}")
                self.encoder = joblib.load(self.encoder_path)
                logger.info(f"✓ Label encoder loaded")
            except Exception as e:
                logger.warning(f"⚠ Could not load encoder: {str(e)}")
        else:
            logger.info("⚠ Label encoder not found (optional)")
    
    def _load_metadata(self) -> None:
        """Load model metadata"""
        if self.metadata_path and self.metadata_path.exists():
            try:
                logger.info(f"Loading metadata from: {self.metadata_path}")
                with open(self.metadata_path, 'r') as f:
                    for line in f:
                        if ':' in line:
                            key, value = line.strip().split(':', 1)
                            self.metadata[key.strip()] = value.strip()
                
                # Parse feature columns from metadata
                if 'feature_columns' in self.metadata:
                    features_str = self.metadata['feature_columns']
                    # Handle list-like string
                    if '[' in features_str:
                        features_str = features_str.replace("'", "").replace("[", "").replace("]", "")
                        self.feature_columns = [f.strip() for f in features_str.split(',')]
                
                logger.info(f"✓ Metadata loaded")
                if self.feature_columns:
                    logger.info(f"  Features: {self.feature_columns}")
            except Exception as e:
                logger.warning(f"⚠ Could not load metadata: {str(e)}")
    
    def display_welcome(self) -> None:
        """Display welcome message and model info"""
        print("\n" + "="*80)
        print("🤖 KNN RETAIL PRODUCT CLASSIFICATION MODEL".center(80))
        print("="*80)
        print(f"\nModel Information:")
        print(f"  📊 Algorithm: K-Nearest Neighbors (KNN)")
        print(f"  🔢 Number of Features: {self.n_features}")
        print(f"  👥 K Value (Neighbors): {self.model.n_neighbors}")
        print(f"  ⏰ Loaded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.metadata.get('classes'):
            print(f"  🏷️  Classes: {self.metadata['classes']}")
        
        if self.metadata.get('accuracy'):
            print(f"  ✅ Model Accuracy: {self.metadata['accuracy']}")
        
        print("\n" + "="*80 + "\n")
    
    def show_example(self) -> None:
        """Display example input format"""
        print("\n" + "-"*80)
        print("📋 INPUT FORMAT EXAMPLE".center(80))
        print("-"*80)
        
        print("\nYou need to provide {n} feature values (comma-separated):".format(n=self.n_features))
        
        # Generate example data
        example_data = np.random.rand(self.n_features) * 100
        example_str = ", ".join([f"{val:.2f}" for val in example_data])
        
        print(f"\n📝 Example Input:")
        print(f"   {example_str}")
        
        print(f"\n💡 Tips:")
        print(f"   • Enter {self.n_features} numeric values separated by commas")
        print(f"   • Values can be floats or integers")
        print(f"   • Use comma (,) as separator")
        print(f"   • Press Enter to submit")
        
        if self.feature_columns:
            print(f"\n🏷️  Features (in order):")
            for i, feature in enumerate(self.feature_columns, 1):
                print(f"   {i:2d}. {feature}")
        
        print("\n" + "-"*80 + "\n")
    
    def get_user_input(self) -> List[float]:
        """
        Get feature values from user input
        
        Returns:
            List of feature values
        """
        while True:
            try:
                print(f"Enter {self.n_features} feature values (comma-separated):")
                print("Type 'example' to see an example, 'quit' to exit, 'clear' to clear screen")
                print(">>> ", end="", flush=True)
                
                user_input = input().strip()
                
                # Handle special commands
                if user_input.lower() == 'quit':
                    return None
                elif user_input.lower() == 'example':
                    self.show_example()
                    continue
                elif user_input.lower() == 'clear':
                    self._clear_screen()
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                # Parse input
                if not user_input:
                    print("❌ Error: Please enter values\n")
                    continue
                
                # Split by comma
                values_str = user_input.split(',')
                
                if len(values_str) != self.n_features:
                    print(f"❌ Error: Expected {self.n_features} values, got {len(values_str)}\n")
                    continue
                
                # Convert to float
                values = []
                for val in values_str:
                    try:
                        values.append(float(val.strip()))
                    except ValueError:
                        print(f"❌ Error: '{val.strip()}' is not a valid number\n")
                        raise ValueError(f"Invalid value: {val}")
                
                return values
            
            except ValueError as e:
                print(f"❌ Error: {str(e)}\n")
                continue
            except KeyboardInterrupt:
                print("\n\n⚠️  User interrupted")
                return None
            except Exception as e:
                print(f"❌ Error: {str(e)}\n")
                continue
    
    def make_prediction(self, features: List[float]) -> Tuple[Any, float]:
        """
        Make prediction using the model
        
        Args:
            features: List of feature values
        
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Convert to numpy array and reshape
            X = np.array(features).reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Get prediction probabilities (using neighbors)
            distances, indices = self.model.kneighbors(X)
            
            # Calculate confidence as inverse of average distance
            avg_distance = distances[0].mean()
            confidence = 1.0 / (1.0 + avg_distance)
            
            # Decode prediction if encoder available
            if self.encoder:
                prediction_label = self.encoder.inverse_transform([prediction])[0]
            else:
                prediction_label = prediction
            
            return prediction_label, confidence
        
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def display_prediction(self, features: List[float], prediction: Any, confidence: float) -> None:
        """
        Display prediction results beautifully
        
        Args:
            features: Input features
            prediction: Model prediction
            confidence: Confidence score
        """
        print("\n" + "="*80)
        print("🎯 PREDICTION RESULT".center(80))
        print("="*80)
        
        # Input features
        print("\n📊 Input Features:")
        print("-" * 40)
        for i, val in enumerate(features, 1):
            if self.feature_columns and i <= len(self.feature_columns):
                feature_name = self.feature_columns[i-1]
                print(f"  {i:2d}. {feature_name:<30s}: {val:>10.4f}")
            else:
                print(f"  {i:2d}. Feature {i:<22s}: {val:>10.4f}")
        
        # Prediction
        print("\n🔮 Classification Result:")
        print("-" * 40)
        print(f"  Predicted Class: {prediction}")
        print(f"  Confidence:      {confidence:.2%}")
        
        # Confidence indicator
        confidence_bar = self._get_confidence_bar(confidence)
        print(f"  [{confidence_bar}]")
        
        # Recommendation
        print("\n💡 Recommendation:")
        print("-" * 40)
        if confidence > 0.8:
            print(f"  ✅ HIGH CONFIDENCE prediction")
        elif confidence > 0.6:
            print(f"  ⚠️  MEDIUM CONFIDENCE prediction")
        else:
            print(f"  ⚠️  LOW CONFIDENCE prediction")
        
        print("\n" + "="*80 + "\n")
    
    def _get_confidence_bar(self, confidence: float) -> str:
        """Generate ASCII confidence bar"""
        bar_length = 30
        filled = int(bar_length * confidence)
        empty = bar_length - filled
        
        if confidence > 0.8:
            color = "✓" * filled + "░" * empty
        elif confidence > 0.6:
            color = "•" * filled + "░" * empty
        else:
            color = "○" * filled + "░" * empty
        
        return color
    
    def _clear_screen(self) -> None:
        """Clear terminal screen"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_help(self) -> None:
        """Display help information"""
        print("\n" + "="*80)
        print("📚 HELP & COMMANDS".center(80))
        print("="*80)
        print("""
Commands:
  example     - Show example input format
  help        - Display this help message
  clear       - Clear the screen
  quit        - Exit the program
  
Input Format:
  - Enter {n} numeric values separated by commas
  - Example: 10.5, 20.3, 15.7, ...
  
Keyboard Shortcuts:
  Ctrl+C      - Exit the program
  
Questions?
  Check the example format or type 'example' at the prompt.
        """.format(n=self.n_features))
        print("="*80 + "\n")
    
    def save_prediction(self, features: List[float], prediction: Any, confidence: float) -> None:
        """Save prediction to log file"""
        try:
            log_file = Path("predictions_log.json")
            
            prediction_record = {
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'prediction': str(prediction),
                'confidence': float(confidence)
            }
            
            # Load existing log or create new
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(prediction_record)
            
            # Save updated log
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"✓ Prediction saved to {log_file}")
        
        except Exception as e:
            logger.warning(f"⚠ Could not save prediction log: {str(e)}")
    
    def run_interactive(self) -> None:
        """Run interactive prediction loop"""
        self.display_welcome()
        self.show_example()
        
        prediction_count = 0
        
        while True:
            try:
                # Get user input
                features = self.get_user_input()
                
                if features is None:
                    break
                
                # Make prediction
                prediction, confidence = self.make_prediction(features)
                
                # Display result
                self.display_prediction(features, prediction, confidence)
                
                # Save prediction
                self.save_prediction(features, prediction, confidence)
                
                prediction_count += 1
                
                # Ask for next prediction
                print("Press Enter for next prediction, or type 'quit' to exit...")
                user_input = input(">>> ").strip().lower()
                
                if user_input in ['quit', 'q', 'exit']:
                    break
                elif user_input == 'clear':
                    self._clear_screen()
                    self.display_welcome()
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"❌ Error: {str(e)}\n")
        
        # Display summary
        self._display_exit_summary(prediction_count)
    
    def _display_exit_summary(self, prediction_count: int) -> None:
        """Display exit summary"""
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY".center(80))
        print("="*80)
        print(f"Total predictions made: {prediction_count}")
        print(f"Session ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Predictions saved to: predictions_log.json")
        print("\n✅ Thank you for using KNN Retail Classifier!")
        print("="*80 + "\n")
    
    def batch_predict(self, csv_file: str, output_file: str = None) -> None:
        """
        Make predictions on batch of data from CSV
        
        Args:
            csv_file: Path to CSV with features
            output_file: Path to save predictions
        """
        try:
            logger.info(f"Loading batch data from: {csv_file}")
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                logger.error("CSV file is empty")
                return
            
            if df.shape[1] != self.n_features:
                logger.error(f"Expected {self.n_features} features, got {df.shape[1]}")
                return
            
            logger.info(f"Making predictions on {len(df)} samples...")
            
            predictions = []
            confidences = []
            
            for idx, row in df.iterrows():
                features = row.values.tolist()
                pred, conf = self.make_prediction(features)
                predictions.append(pred)
                confidences.append(conf)
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} samples...")
            
            # Create results dataframe
            results_df = df.copy()
            results_df['prediction'] = predictions
            results_df['confidence'] = confidences
            
            # Save results
            if output_file is None:
                output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            results_df.to_csv(output_file, index=False)
            logger.info(f"✓ Predictions saved to: {output_file}")
            
            print(f"\n✅ Batch prediction complete!")
            print(f"   Input: {len(df)} samples")
            print(f"   Output: {output_file}")
        
        except Exception as e:
            logger.error(f"Error in batch prediction: {str(e)}")


# ==================== MAIN EXECUTION ====================

def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("🚀 KNN MODEL CLI INTERFACE".center(80))
    print("="*80 + "\n")
    
    # ===== CONFIGURATION =====
    MODEL_FILE = "knn_retail_classifier.pkl"
    ENCODER_FILE = "knn_retail_classifier_encoder.pkl"
    METADATA_FILE = "knn_retail_classifier_metadata.txt"
    # =========================
    
    try:
        # Initialize CLI
        cli = KNNModelCLI(
            model_path=MODEL_FILE,
            encoder_path=ENCODER_FILE,
            metadata_path=METADATA_FILE
        )
        
        # Run interactive mode
        cli.run_interactive()
    
    except FileNotFoundError as e:
        logger.error(f"❌ Model file not found: {str(e)}")
        print(f"\nPlease ensure these files exist:")
        print(f"  - {MODEL_FILE}")
        print(f"  - {ENCODER_FILE} (optional)")
        print(f"  - {METADATA_FILE} (optional)")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()