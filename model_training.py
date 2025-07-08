# model_training.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class ReadmissionModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and compare performance"""
        
        # Define models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42,
                class_weight='balanced'
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        print("Training models...")
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'auc_score': auc_score,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': classification_report(y_val, y_pred),
                'confusion_matrix': confusion_matrix(y_val, y_pred)
            }
            
            print(f"AUC Score: {auc_score:.4f}")
        
        # Select best model
        self.best_model_name = max(self.results.keys(), key=lambda k: self.results[k]['auc_score'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest model: {self.best_model_name} (AUC: {self.results[self.best_model_name]['auc_score']:.4f})")
        
        return self.results
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate best model on test set"""
        if self.best_model is None:
            raise ValueError("No trained model found. Run train_models first.")
        
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        test_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nTest Set Performance ({self.best_model_name}):")
        print(f"AUC Score: {test_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'auc_score': test_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def plot_roc_curves(self, y_val):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_val, result['y_pred_proba'])
            plt.plot(fpr, tpr, label=f"{name} (AUC = {result['auc_score']:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, y_val):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, len(self.results), figsize=(15, 5))
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
            axes[i].set_title(f'{name}\nAUC: {result["auc_score"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_feature_importance(self, preprocessor):
        """Get feature importance from best model"""
        if self.best_model is None:
            raise ValueError("No trained model found.")
        
        feature_names = preprocessor.feature_names
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_[0])
        else:
            return None
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, preprocessor, top_n=15):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(preprocessor)
        
        if importance_df is None:
            print("Feature importance not available for this model.")
            return
        
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def save_model(self, filepath):
        """Save the best model"""
        if self.best_model is None:
            raise ValueError("No trained model found.")
        
        joblib.dump({
            'model': self.best_model,
            'model_name': self.best_model_name,
            'results': self.results
        }, filepath)
        
        print(f"Model saved to {filepath}")

def train_readmission_model():
    """Main training function"""
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train = np.load('X_train.npy')
    X_val = np.load('X_val.npy')
    X_test = np.load('X_test.npy')
    y_train = np.load('y_train.npy')
    y_val = np.load('y_val.npy')
    y_test = np.load('y_test.npy')
    
    # Load preprocessor
    from data_preprocessing import ReadmissionPreprocessor
    preprocessor = ReadmissionPreprocessor.load('preprocessor.pkl')
    
    # Train models
    trainer = ReadmissionModelTrainer()
    results = trainer.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_results = trainer.evaluate_model(X_test, y_test)
    
    # Plot results
    trainer.plot_roc_curves(y_val)
    trainer.plot_confusion_matrices(y_val)
    importance_df = trainer.plot_feature_importance(preprocessor)
    
    # Save model
    trainer.save_model('readmission_model.pkl')
    
    return trainer, importance_df

if __name__ == "__main__":
    trainer, importance_df = train_readmission_model()
