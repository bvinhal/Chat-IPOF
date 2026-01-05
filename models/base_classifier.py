"""
Modelo Base de Classificador de Natureza de Despesa

Este módulo fornece uma classe base para os classificadores de natureza de despesa,
integrando com o pipeline de pré-processamento.

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import os
import pickle
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)

# Importar pipeline de pré-processamento
from utils.natureza_preprocessing import NaturezaPreprocessingPipeline

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaseNaturezaClassifier(ABC):
    """
    Classe base abstrata para classificadores de natureza de despesa.
    
    Todos os modelos (SVM, LightGBM, Regressão Logística, etc.) devem herdar desta classe.
    """
    
    def __init__(self, model_name: str, project_root: str = None):
        """
        Inicializa o classificador base.
        
        Args:
            model_name: Nome do modelo (ex: 'svm_linear', 'lightgbm', etc.)
            project_root: Diretório raiz do projeto
        """
        self.model_name = model_name
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models_dir = os.path.join(self.project_root, 'data', 'models', model_name)
        
        # Pipeline de pré-processamento
        self.pipeline = NaturezaPreprocessingPipeline(project_root=project_root)
        
        # Modelo (a ser definido nas subclasses)
        self.model = None
        
        # Dados de treinamento
        self.data = None
        
        # Métricas de avaliação
        self.metrics = {}
        
        logger.info(f"Classificador '{model_name}' inicializado")
    
    @abstractmethod
    def build_model(self, **kwargs):
        """
        Constrói o modelo específico.
        
        Este método deve ser implementado pelas subclasses.
        """
        pass
    
    def preprocess_data(self, filename: str = 'natureza_despesa_final.xlsx',
                       test_size: float = 0.15, val_size: float = 0.15,
                       random_state: int = 42, min_samples_per_class: int = 3) -> Dict:
        """
        Executa o pipeline de pré-processamento.
        
        Args:
            filename: Nome do arquivo de dados
            test_size: Proporção do conjunto de teste
            val_size: Proporção do conjunto de validação
            random_state: Seed para reprodutibilidade
            min_samples_per_class: Mínimo de amostras por classe
            
        Returns:
            Dicionário com dados processados
        """
        logger.info(f"Executando pré-processamento para {self.model_name}...")
        
        self.data = self.pipeline.run(
            filename=filename,
            test_size=test_size,
            val_size=val_size,
            random_state=random_state,
            min_samples_per_class=min_samples_per_class
        )
        
        return self.data
    
    def train(self, **kwargs):
        """
        Treina o modelo.
        
        Args:
            **kwargs: Parâmetros específicos do modelo
        """
        if self.data is None:
            raise ValueError("Dados não processados. Execute preprocess_data() primeiro.")
        
        if self.model is None:
            self.build_model(**kwargs)
        
        logger.info(f"Treinando modelo {self.model_name}...")
        
        X_train = self.data['X_train_vec']
        y_train = self.data['y_train_encoded']
        
        import time
        start_time = time.time()
        
        self.model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {training_time:.2f}s")
        
        self.metrics['training_time'] = training_time
    
    def evaluate(self, dataset: str = 'test') -> Dict[str, float]:
        """
        Avalia o modelo em um conjunto de dados.
        
        Args:
            dataset: 'train', 'val' ou 'test'
            
        Returns:
            Dicionário com métricas de avaliação
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        if self.data is None:
            raise ValueError("Dados não carregados.")
        
        # Selecionar conjunto de dados
        if dataset == 'train':
            X = self.data['X_train_vec']
            y = self.data['y_train_encoded']
        elif dataset == 'val':
            X = self.data['X_val_vec']
            y = self.data['y_val_encoded']
        elif dataset == 'test':
            X = self.data['X_test_vec']
            y = self.data['y_test_encoded']
        else:
            raise ValueError(f"Dataset inválido: {dataset}")
        
        logger.info(f"Avaliando modelo em conjunto de {dataset}...")
        
        import time
        start_time = time.time()
        
        y_pred = self.model.predict(X)
        
        prediction_time = time.time() - start_time
        
        # Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'f1_macro': f1_score(y, y_pred, average='macro'),
            'f1_weighted': f1_score(y, y_pred, average='weighted'),
            'precision_macro': precision_score(y, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y, y_pred, average='macro', zero_division=0),
            'prediction_time': prediction_time,
            'samples': len(y)
        }
        
        # Calcular Top-3 Accuracy se o modelo suporta predict_proba
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
            top_3_preds = np.argsort(y_proba, axis=1)[:, -3:]
            top_3_accuracy = np.mean([y[i] in top_3_preds[i] for i in range(len(y))])
            metrics['top_3_accuracy'] = top_3_accuracy
        
        # Salvar métricas
        self.metrics[f'{dataset}_metrics'] = metrics
        
        # Log das métricas
        logger.info(f"Métricas ({dataset}):")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1-Score: {metrics['f1_macro']:.4f}")
        if 'top_3_accuracy' in metrics:
            logger.info(f"  Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Tempo de predição: {metrics['prediction_time']:.4f}s")
        
        return metrics
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Faz predição para um texto.
        
        Args:
            text: Descrição da despesa
            
        Returns:
            Tupla (natureza prevista, confiança)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar texto
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        
        # Vetorizar
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Predição
        pred_encoded = self.model.predict(text_vec)[0]
        natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Confiança (se disponível)
        confidence = 1.0
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(text_vec)[0]
            confidence = float(proba[pred_encoded])
        
        return natureza, confidence
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retorna as top-k predições mais prováveis.
        
        Args:
            text: Descrição da despesa
            k: Número de predições a retornar
            
        Returns:
            Lista de tuplas (natureza, probabilidade)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        if not hasattr(self.model, 'predict_proba'):
            # Se o modelo não tem predict_proba, retorna apenas a predição principal
            natureza, conf = self.predict(text)
            return [(natureza, conf)]
        
        # Pré-processar e vetorizar
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Probabilidades
        proba = self.model.predict_proba(text_vec)[0]
        
        # Top-k índices
        top_k_indices = np.argsort(proba)[-k:][::-1]
        
        # Converter para naturezas
        top_k_naturezas = self.pipeline.label_encoder.inverse_transform(top_k_indices)
        top_k_probas = proba[top_k_indices]
        
        return list(zip(top_k_naturezas, top_k_probas))
    
    def save_model(self):
        """Salva o modelo e artefatos."""
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Salvar modelo
        model_path = os.path.join(self.models_dir, 'model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Salvar métricas
        metrics_path = os.path.join(self.models_dir, 'metrics.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(self.metrics, f)
        
        # Salvar artefatos do pipeline
        self.pipeline.save_artifacts(model_name=self.model_name)
        
        logger.info(f"Modelo salvo em: {self.models_dir}")
    
    def load_model(self):
        """Carrega o modelo e artefatos."""
        model_path = os.path.join(self.models_dir, 'model.pkl')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        # Carregar modelo
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Carregar métricas (se existir)
        metrics_path = os.path.join(self.models_dir, 'metrics.pkl')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                self.metrics = pickle.load(f)
        
        # Carregar artefatos do pipeline
        preprocessor, vectorizer, label_encoder = NaturezaPreprocessingPipeline.load_artifacts(
            model_name=self.model_name,
            project_root=self.project_root
        )
        
        self.pipeline.preprocessor = preprocessor
        self.pipeline.vectorizer = vectorizer
        self.pipeline.label_encoder = label_encoder
        
        logger.info(f"Modelo carregado de: {self.models_dir}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o modelo."""
        info = {
            'model_name': self.model_name,
            'model_type': self.__class__.__name__,
            'is_trained': self.model is not None,
            'metrics': self.metrics
        }
        
        if self.data is not None:
            info['data_info'] = {
                'n_samples_train': self.data['X_train_vec'].shape[0],
                'n_samples_val': self.data['X_val_vec'].shape[0],
                'n_samples_test': self.data['X_test_vec'].shape[0],
                'n_features': self.data['X_train_vec'].shape[1],
                'n_classes': len(self.pipeline.label_encoder.classes_)
            }
        
        return info