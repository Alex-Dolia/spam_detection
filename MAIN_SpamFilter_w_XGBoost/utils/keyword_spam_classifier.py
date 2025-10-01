"""
Improved Keyword Spam Classification System

This module addresses the critical issues in the original notebook:
1. Fixes data leakage by proper train/test separation
2. Implements proper feature engineering
3. Adds comprehensive model validation
4. Uses better preprocessing and feature extraction
"""
import os
import re
import warnings
from typing import Tuple, List, Dict, Any
from itertools import chain

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import spacy
import unidecode
from collections import Counter
# Bayesian Optimization imports
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

class KeywordSpamClassifier:
    """
    A comprehensive keyword spam classifier that addresses the issues in the original approach.
    """
    
    def __init__(self,  isBayesOpt = False, use_smote = False, nlp_model: str = "en_core_web_sm"):
        """Initialize the classifier with proper preprocessing components."""
        self.isBayesOpt = isBayesOpt
        self.use_smote  = use_smote 

        self.nlp = None
        self.tfidf_vectorizer = None
        self.entity_vectorizer = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
        self.random_state = 11
        # Try to load spaCy model
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            print(f"Warning: {nlp_model} not found. Install with: python -m spacy download {nlp_model}")
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text while preserving important information.
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Normalize non-ASCII characters
        text = unidecode.unidecode(text)
        
        # Keep emojis, symbols, hashtags, mentions, currency, etc.
        text = re.sub(r'[\n\t\r]+', ' ', text)  # Normalize newlines/tabs
        text = re.sub(r' +', ' ', text)         # Normalize spaces

        ### Replace special characters with spaces (but preserve hashtags and @ symbols)
        ### text = re.sub(r'[,;@#!\?\+\*\n\-: /]', ' ', text)
        
        ### Keep alphanumeric characters, spaces, hashtags, and @ symbols
        ### text = re.sub(r'[^A-Za-z0-9& #@]', '', text)
        
        # Clean up multiple spaces
        text = ' '.join(text.split())
        
        return text
    
    def extract_basic_features(self, descriptions: List[str]) -> pd.DataFrame:
        """
        Extract basic statistical features from descriptions.
        
        Args:
            descriptions: List of product descriptions
            
        Returns:
            DataFrame with basic features
        """
        features = []
        
        for desc in descriptions:
            desc_clean = self.clean_text(desc)
            words = desc_clean.split()
            
            # Basic text statistics
            word_count = len(words)
            char_count = len(desc_clean)
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            
            # Character-level features
            upper_ratio = sum(1 for c in desc if c.isupper()) / len(desc) if desc else 0
            digit_ratio = sum(1 for c in desc if c.isdigit()) / len(desc) if desc else 0
            special_char_ratio = sum(1 for c in desc if not c.isalnum() and not c.isspace()) / len(desc) if desc else 0
            
            # Hashtag and mention features
            hashtag_count = desc.count('#')
            mention_count = desc.count('@')
            
            # Spam indicators
            exclamation_ratio = desc.count('!') / len(desc) if desc else 0
            caps_words_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / word_count if word_count > 0 else 0
            
            # Brand-related features (common fashion brands that might be spammed)
            fashion_brands = ['nike', 'adidas', 'supreme', 'gucci', 'lv', 'louis', 'vuitton', 'chanel', 
                            'prada', 'versace', 'dolce', 'gabbana', 'balenciaga', 'off', 'white',
                            'yeezy', 'jordan', 'levi', 'calvin', 'klein', 'tommy', 'hilfiger']
            
            brand_count = sum(1 for brand in fashion_brands if brand in desc_clean)
            brand_ratio = brand_count / word_count if word_count > 0 else 0
            
            features.append({
                'word_count': word_count,
                'char_count': char_count,
                'avg_word_length': avg_word_length,
                'upper_ratio': upper_ratio,
                'digit_ratio': digit_ratio,
                'special_char_ratio': special_char_ratio,
                'hashtag_count': hashtag_count,
                'mention_count': mention_count,
                'exclamation_ratio': exclamation_ratio,
                'caps_words_ratio': caps_words_ratio,
                'brand_count': brand_count,
                'brand_ratio': brand_ratio,
            })
        
        return pd.DataFrame(features)
    
    def extract_named_entities(self, descriptions: List[str]) -> List[str]:
        """
        Extract named entities from descriptions using spaCy.
        
        Args:
            descriptions: List of product descriptions
            
        Returns:
            List of entity strings for each description
        """
        if not self.nlp:
            return [''] * len(descriptions)
        
        entity_strings = []
        
        for desc in descriptions:
            # Apply true case for better NER
            try:
                import truecase
                desc_truecase = truecase.get_true_case(desc)
            except:
                desc_truecase = desc
            
            doc = self.nlp(desc_truecase)
            
            # Extract entities and combine
            entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT']]
            entity_strings.append(' '.join(entities))
        
        return entity_strings
    
    def extract_keyword_features(self, descriptions: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract TF-IDF features from descriptions and named entities.
        
        Args:
            descriptions: List of product descriptions
            
        Returns:
            Tuple of (description_features, entity_features)
        """
        # Clean descriptions
        cleaned_descriptions = [self.clean_text(desc) for desc in descriptions]
        
        # TF-IDF for main descriptions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),  # Include bigrams
            stop_words='english'
        )
        desc_features = self.tfidf_vectorizer.fit_transform(cleaned_descriptions)
        
        # Named entity features
        entity_strings = self.extract_named_entities(descriptions)
        self.entity_vectorizer = TfidfVectorizer(
            max_features=200,
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        entity_features = self.entity_vectorizer.fit_transform(entity_strings)
        
        return desc_features, entity_features

    def calculate_auc_cv(self, X_features, y_true, params):
        """Calculate AUC using stratified cross-validation"""

        # Convert parameters to appropriate types
        int_params = ['max_depth', 'n_estimators', 'min_child_weight']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        params['random_state'] = self.random_state
        params['use_label_encoder'] = False
        #params['eval_metric'] = 'logloss'
        
        xgb_model = xgb.XGBClassifier(**params)
        
        # Use Stratified K-Fold for robust validation
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        auc_scores = []
        
        for train_idx, val_idx in skf.split(X_features, y_true):
            X_train = X_features.iloc[train_idx]
            y_train = y_true.iloc[train_idx]
            X_val = X_features.iloc[val_idx]
            y_val = y_true.iloc[val_idx]
            
            # use SMOTE is required
            X_train_features, y_train_labels = self.extract_training_features(X_train, y_train)
            
            # Train model
            xgb_model.fit(X_train_features, y_train_labels)
            
            # Predict probabilities
            X_val_features = self.extract_test_features(X_val)
            y_pred_proba = xgb_model.predict_proba(X_val_features)[:, 1]
            
            # Calculate ROC AUC
            fold_auc = roc_auc_score(y_val, y_pred_proba)
            auc_scores.append(fold_auc)
        
        score = np.mean(auc_scores)
        out = params.copy()
        out["AUC"] = score
        self.all_parametes.append(out)
        return score
    
    def objective(self, params):
        """Objective function for Bayesian Optimization"""
        try:
            auc_score = self.calculate_auc_cv(self.X_train_processed, self.y_train, params)
            # We want to maximize AUC, so loss is negative AUC
            loss = -auc_score
            return {'loss': loss, 'status': STATUS_OK, 'auc_score': auc_score}
        except Exception as e:
            # Return a poor score if there's an error
            print(f"!!!!!!!!!!!!!!!!!!!! except Exception as e: {e}!!!!!!!!!!!!!!!!!!!!!!")  
            return {'loss': 1.0, 'status': STATUS_OK, 'auc_score': 0.0}

    def bayesian_optimization(self, X_train, y_train, xgb_space = None):
        """Perform Bayesian Optimization to find best hyperparameters"""
        self.all_parametes = []

        print("Starting Bayesian Optimization...")
        
        # Calculate scale_pos_weight for imbalanced data
        counter = Counter(y_train)
        scale_pos_weight_value = counter[0] / counter[1]
        
        # Define search space for XGBoost
        # I comment out some parameters because I want to check if I optimize the same parameters that I define wo using Bayesian Optimization
        if xgb_space is None:
           xgb_space = {
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'learning_rate': hp.uniform('learning_rate', 0.075, 0.25),
            'n_estimators': hp.quniform('n_estimators', 50, 110, 1),
            #'gamma': hp.uniform('gamma', 0, 5),
            #'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            #'reg_alpha': hp.uniform('reg_alpha', 0, 10),
            #'reg_lambda': hp.uniform('reg_lambda', 0, 10),
            'scale_pos_weight': hp.choice('scale_pos_weight', [scale_pos_weight_value]),
            'objective': hp.choice('objective', ['binary:logistic']),
            'eval_metric': hp.choice('eval_metric', ['logloss']),
           }
        
        # Store data for objective function
        self.X_train_processed = X_train
        self.y_train = y_train
        
        # Run Bayesian Optimization
        trials = Trials()
        
        # Fix for RandomState compatibility
        import random
        random_seed = self.random_state
        
        best = fmin(
         fn=self.objective,
         space=xgb_space,
         algo=tpe.suggest,
         max_evals=self.max_evals,
         trials=trials,
         rstate=np.random.default_rng(self.random_state),  # <- FIXED
         show_progressbar=True
        )

        # Get best parameters
        best_params = {}
        for key, value in best.items():
            if key in ['max_depth', 'min_child_weight', 'n_estimators']:
                best_params[key] = int(value)
            elif key == 'scale_pos_weight':
                best_params[key] = scale_pos_weight_value
            else:
                best_params[key] = value
        
        # Add fixed parameters
        best_params.update({
            'random_state': self.random_state,
            'use_label_encoder': False,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        })
        


        # Find best AUC from trials
        best_auc = max([trial['result']['auc_score'] for trial in trials.trials])
        
        print(f"Bayesian Optimization completed!")
        print(f"Best AUC: {best_auc:.4f}")
        
        return best_params, trials, best_auc 

    def extract_training_features(self, X: pd.DataFrame, y: pd.Series):
        descriptions = X['description'].tolist()
        
        # Extract basic features
        basic_features = self.extract_basic_features(descriptions)
        
        # Extract TF-IDF features
        desc_tfidf, entity_tfidf = self.extract_keyword_features(descriptions)
        
        # Combine all features
        all_features = np.hstack([
            basic_features.values,
            desc_tfidf.toarray(),
            entity_tfidf.toarray()
        ])
        
        # Scale features
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        # Create feature names for interpretability
        basic_feature_names = list(basic_features.columns)
        desc_feature_names = [f"desc_{feat}" for feat in self.tfidf_vectorizer.get_feature_names_out()]
        entity_feature_names = [f"entity_{feat}" for feat in self.entity_vectorizer.get_feature_names_out()]
        self.feature_names = basic_feature_names + desc_feature_names + entity_feature_names

        if self.use_smote:
           smote = SMOTE(random_state=42)
           all_features_scaled_out, y_out = smote.fit_resample(all_features_scaled, y)
        else:
           all_features_scaled_out, y_out = all_features_scaled, y

        return all_features_scaled_out, y_out
    
    def extract_test_features(self, X: pd.DataFrame):
        descriptions = X['description'].tolist()
        
        # Extract features using same pipeline
        basic_features = self.extract_basic_features(descriptions)
        
        # Transform using fitted vectorizers
        cleaned_descriptions = [self.clean_text(desc) for desc in descriptions]
        desc_tfidf = self.tfidf_vectorizer.transform(cleaned_descriptions)
        
        entity_strings = self.extract_named_entities(descriptions)
        entity_tfidf = self.entity_vectorizer.transform(entity_strings)
        
        # Combine features
        all_features = np.hstack([
            basic_features.values,
            desc_tfidf.toarray(),
            entity_tfidf.toarray()
        ])
        
        # Scale features
        all_features_scaled = self.scaler.transform(all_features)
        return all_features_scaled
    
    def fit(self, X: pd.DataFrame, y: pd.Series, params = None, max_evals = None, cv_folds = None) -> 'KeywordSpamClassifier':
        """
        Fit the classifier on training data.
        
        Args:
            X: Training features (should contain 'description' column)
            y: Training labels
            
        Returns:
            Self
        """
       
        # extract all features
        all_features_scaled, y_out = self.extract_training_features(X, y)
        
        # Train XGBoost model with proper parameters
        if not self.isBayesOpt:
           if params is not None:
              self.model = xgb.XGBClassifier(**params) 
           else: 
              self.model = xgb.XGBClassifier(
               n_estimators=100,  
               max_depth=6,
               learning_rate=0.1,
               subsample=0.8,
               colsample_bytree=0.8,
               random_state=42,
               eval_metric='logloss'
              )
        else:
           self.max_evals = 30 if max_evals is None else max_evals 
           self.cv_folds = 3 if cv_folds is None else cv_folds
           self.best_params, self.trials, self.best_auc = self.bayesian_optimization(X, y, xgb_space = params)
           self.model = xgb.XGBClassifier(**self.best_params)
            
        self.model.fit(all_features_scaled, y_out)

        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict labels for new data.
        
        Args:
            X: Features (should contain 'description' column)
            
        Returns:
            Predicted labels
        """
        all_features_scaled = self.extract_test_features(X)
        
        return self.model.predict(all_features_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for new data.
        
        Args:
            X: Features (should contain 'description' column)
            
        Returns:
            Predicted probabilities
        """
        all_features_scaled = self.extract_test_features(X)
        
        return self.model.predict_proba(all_features_scaled)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def load_data(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare the dataset.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')
    
    # Select relevant columns
    train_df = train_df[['product_id', 'description', 'label']].copy()
    test_df = test_df[['product_id', 'description', 'label']].copy()
    
    return train_df, test_df


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """

    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
    
    return metrics


def main():
    """
    Main function to run the improved keyword spam classifier.
    """
    print("Loading data...")
    train_df, test_df = load_data('data/train_set.tsv', 'data/test_set.tsv')
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Train spam rate: {train_df['label'].mean():.3f}")
    print(f"Test spam rate: {test_df['label'].mean():.3f}")
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = KeywordSpamClassifier()
    
    # Train the model
    print("Training model...")
    classifier.fit(train_df, train_df['label'])
    
    # Make predictions
    print("Making predictions...")
    y_pred = classifier.predict(test_df)
    y_proba = classifier.predict_proba(test_df)
    
    # Evaluate performance
    print("\n=== MODEL PERFORMANCE ===")
    metrics = evaluate_model(test_df['label'], y_pred, y_proba)
    
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    
    # Show classification report
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(test_df['label'], y_pred, target_names=['Non-Spam', 'Spam']))
    
    # Show confusion matrix
    print("\n=== CONFUSION MATRIX ===")
    cm = confusion_matrix(test_df['label'], y_pred)
    print("Confusion Matrix:")
    print("                Predicted")
    print("Actual     Non-Spam  Spam")
    print(f"Non-Spam      {cm[0,0]:4d}    {cm[0,1]:4d}")
    print(f"Spam          {cm[1,0]:4d}    {cm[1,1]:4d}")
    
    # Show top features
    print("\n=== TOP 20 MOST IMPORTANT FEATURES ===")
    feature_importance = classifier.get_feature_importance()
    print(feature_importance.head(20).to_string(index=False))
    
    return classifier, metrics


if __name__ == "__main__":
   #classifier, metrics = main()
   print("KeywordSpamClassifier, Version 1") 

