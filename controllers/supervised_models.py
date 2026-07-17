"""
Modelos Supervisionados Simplificados

SVM Linear, LightGBM e Regressão Logística.
Interface simples: fit(), predict(), evaluate()

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report
)
import logging

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class SVMClassifier:
    """
    SVM Linear com TF-IDF.
    
    Conforme metodologia (Lorena e Carvalho, 2007):
    - Kernel linear para dados TF-IDF esparsos
    - One-vs-Rest para multiclasse
    """
    
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=42,
            dual='auto'
        )
        self.training_time = 0
        
    def fit(self, X_train, y_train):
        """Treina o modelo."""
        logger.info("Treinando SVM Linear...")
        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start
        logger.info(f"  Tempo de treino: {self.training_time:.2f}s")
        
    def predict(self, X):
        """Faz predições."""
        return self.model.predict(X)
    
    def evaluate(self, X, y, dataset_name='test'):
        """
        Avalia o modelo.
        
        Returns:
            Dict com métricas
        """
        logger.info(f"\nAvaliando SVM em {dataset_name}...")
        
        start = time.time()
        y_pred = self.predict(X)
        pred_time = time.time() - start
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'training_time': self.training_time,
            'prediction_time': pred_time
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        
        return metrics


class LightGBMClassifier:
    """
    LightGBM - Gradient Boosting eficiente.
    
    Conforme metodologia (Ke et al., 2017):
    - GOSS (Gradient-based One-Side Sampling)
    - EFB (Exclusive Feature Bundling)
    """
    
    def __init__(self, num_leaves=31, learning_rate=0.1, n_estimators=100):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM não instalado. Execute: pip install lightgbm")
        
        self.model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=42,
            verbose=-1
        )
        self.training_time = 0
        
    def fit(self, X_train, y_train):
        """Treina o modelo."""
        logger.info("Treinando LightGBM...")
        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start
        logger.info(f"  Tempo de treino: {self.training_time:.2f}s")
        
    def predict(self, X):
        """Faz predições."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Probabilidades."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, dataset_name='test'):
        """Avalia o modelo."""
        logger.info(f"\nAvaliando LightGBM em {dataset_name}...")
        
        start = time.time()
        y_pred = self.predict(X)
        pred_time = time.time() - start
        
        # Top-3 Accuracy
        y_proba = self.predict_proba(X)
        top_3_preds = np.argsort(y_proba, axis=1)[:, -3:]
        top_3_acc = np.mean([y[i] in top_3_preds[i] for i in range(len(y))])
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'top_3_accuracy': top_3_acc,
            'training_time': self.training_time,
            'prediction_time': pred_time
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Top-3 Acc: {metrics['top_3_accuracy']:.4f}")
        
        return metrics


class LogisticClassifier:
    """
    Regressão Logística Multinomial.
    
    Conforme metodologia (Faceli et al., 2011):
    - Softmax para multiclasse
    - Regularização L2
    """
    
    def __init__(self, C=1.0, max_iter=1000):
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            multi_class='multinomial',
            solver='lbfgs',
            random_state=42
        )
        self.training_time = 0
        
    def fit(self, X_train, y_train):
        """Treina o modelo."""
        logger.info("Treinando Regressão Logística...")
        start = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start
        logger.info(f"  Tempo de treino: {self.training_time:.2f}s")
        
    def predict(self, X):
        """Faz predições."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Probabilidades."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y, dataset_name='test'):
        """Avalia o modelo."""
        logger.info(f"\nAvaliando Regressão Logística em {dataset_name}...")
        
        start = time.time()
        y_pred = self.predict(X)
        pred_time = time.time() - start
        
        # Top-3 Accuracy
        y_proba = self.predict_proba(X)
        top_3_preds = np.argsort(y_proba, axis=1)[:, -3:]
        top_3_acc = np.mean([y[i] in top_3_preds[i] for i in range(len(y))])
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
            'precision': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y, y_pred, average='macro', zero_division=0),
            'top_3_accuracy': top_3_acc,
            'training_time': self.training_time,
            'prediction_time': pred_time
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Top-3 Acc: {metrics['top_3_accuracy']:.4f}")
        
        return metrics