"""
Modelo Neural Simplificado - MLP + Sentence-BERT

MLP (Multi-Layer Perceptron) com embeddings S-BERT.
Interface simples: fit(), predict(), evaluate()

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import numpy as np
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score
)
import logging

logger = logging.getLogger(__name__)

# Importações opcionais
SBERT_AVAILABLE = False
SBERT_ERROR = None

try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError as e:
    SBERT_ERROR = str(e)
except Exception as e:
    SBERT_ERROR = f"Erro ao importar: {str(e)}"

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class MLPSBERTClassifier:
    """
    MLP + Sentence-BERT para classificação.
    
    Conforme metodologia (Reimers e Gurevych, 2019; Haykin, 2001):
    - Embeddings semânticos pré-treinados (S-BERT)
    - Rede neural MLP para classificação
    - Modelos em português: BERTimbau ou Legal-BERTimbau
    """
    
    def __init__(
        self, 
        model_name='neuralmind/bert-base-portuguese-cased',
        hidden_layer_sizes=(512, 256),
        learning_rate_init=0.001,
        max_iter=200
    ):
        """
        Inicializa o classificador.
        
        Args:
            model_name: Modelo S-BERT a usar:
                - 'neuralmind/bert-base-portuguese-cased' (BERTimbau)
                - 'rufimelo/Legal-BERTimbau' (Legal)
                - 'paraphrase-multilingual-MiniLM-L12-v2' (Multilingual)
            hidden_layer_sizes: Tamanho das camadas ocultas
            learning_rate_init: Taxa de aprendizado inicial
            max_iter: Máximo de iterações
        """
        if not SBERT_AVAILABLE:
            error_msg = "sentence-transformers não está disponível.\n\n"
            
            if SBERT_ERROR:
                error_msg += f"Erro detectado: {SBERT_ERROR}\n\n"
            
            error_msg += "Soluções:\n"
            error_msg += "1. Instale as dependências:\n"
            error_msg += "   pip install sentence-transformers torch\n\n"
            error_msg += "2. Se já instalou, verifique o ambiente Python:\n"
            error_msg += "   python -c \"import sentence_transformers; print('OK')\"\n\n"
            error_msg += "3. Se estiver em ambiente virtual, ative-o antes de instalar:\n"
            error_msg += "   source .venv/bin/activate  # Linux/Mac\n"
            error_msg += "   .venv\\Scripts\\activate     # Windows\n"
            
            raise ImportError(error_msg)
        
        self.model_name = model_name
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        
        # Componentes
        self.sbert_model = None
        self.mlp_model = None
        self.training_time = 0
        self.embedding_time = 0
        
        # Cache de embeddings
        self.X_train_embeddings = None
        
        logger.info(f"MLP + S-BERT inicializado (modelo: {model_name})")
    
    def _load_sbert_model(self):
        """Carrega modelo S-BERT (lazy loading)."""
        if self.sbert_model is None:
            logger.info(f"Carregando modelo S-BERT: {self.model_name}")
            start = time.time()
            self.sbert_model = SentenceTransformer(self.model_name)
            load_time = time.time() - start
            logger.info(f"  Modelo S-BERT carregado em {load_time:.2f}s")
    
    def _generate_embeddings(self, texts, batch_size=32):
        """
        Gera embeddings para textos.
        
        Args:
            texts: Lista ou array de textos
            batch_size: Tamanho do batch
            
        Returns:
            Array de embeddings
        """
        self._load_sbert_model()
        
        logger.info(f"Gerando embeddings para {len(texts)} textos...")
        start = time.time()
        
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        embedding_time = time.time() - start
        logger.info(f"  Embeddings gerados em {embedding_time:.2f}s")
        
        return embeddings, embedding_time
    
    def fit(self, X_train_texts, y_train):
        """
        Treina o modelo.
        
        Args:
            X_train_texts: Textos de treino (lista ou array)
            y_train: Labels de treino
        """
        logger.info("Treinando MLP + S-BERT...")
        
        total_start = time.time()
        
        # 1. Gerar embeddings
        self.X_train_embeddings, self.embedding_time = self._generate_embeddings(
            X_train_texts
        )
        
        # 2. Treinar MLP
        logger.info(f"Treinando MLP (camadas: {self.hidden_layer_sizes})...")
        mlp_start = time.time()
        
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False
        )
        
        self.mlp_model.fit(self.X_train_embeddings, y_train)
        
        mlp_time = time.time() - mlp_start
        self.training_time = time.time() - total_start
        
        logger.info(f"  Tempo de embeddings: {self.embedding_time:.2f}s")
        logger.info(f"  Tempo de treino MLP: {mlp_time:.2f}s")
        logger.info(f"  Tempo total: {self.training_time:.2f}s")
    
    def predict(self, X_texts):
        """
        Faz predições.
        
        Args:
            X_texts: Textos para predição (lista ou array)
            
        Returns:
            Array de predições
        """
        if self.mlp_model is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        # Gerar embeddings
        embeddings, _ = self._generate_embeddings(X_texts)
        
        # Predizer
        return self.mlp_model.predict(embeddings)
    
    def predict_proba(self, X_texts):
        """
        Probabilidades das predições.
        
        Args:
            X_texts: Textos para predição
            
        Returns:
            Array de probabilidades
        """
        if self.mlp_model is None:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")
        
        # Gerar embeddings
        embeddings, _ = self._generate_embeddings(X_texts)
        
        # Probabilidades
        return self.mlp_model.predict_proba(embeddings)
    
    def evaluate(self, X_texts, y, dataset_name='test'):
        """
        Avalia o modelo.
        
        Args:
            X_texts: Textos de teste
            y: Labels verdadeiros
            dataset_name: Nome do conjunto
            
        Returns:
            Dict com métricas
        """
        logger.info(f"\nAvaliando MLP + S-BERT em {dataset_name}...")
        
        start = time.time()
        y_pred = self.predict(X_texts)
        pred_time = time.time() - start
        
        # Top-3 Accuracy
        y_proba = self.predict_proba(X_texts)
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
            'embedding_time': self.embedding_time,
            'prediction_time': pred_time
        }
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Macro: {metrics['f1_macro']:.4f}")
        logger.info(f"  Top-3 Acc: {metrics['top_3_accuracy']:.4f}")
        logger.info(f"  Tempo total: {metrics['training_time']:.2f}s")
        
        return metrics


def prepare_texts_from_tfidf(data_processor, X_indices):
    """
    Converte índices de volta para textos originais.
    
    Args:
        data_processor: Instância do DataProcessor
        X_indices: Índices das amostras
        
    Returns:
        Lista de textos
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas não disponível")
    
    # Assumindo que data_processor tem os dados originais
    if hasattr(data_processor, 'data') and data_processor.data is not None:
        df = data_processor.data
        
        # Pegar textos limpos correspondentes
        texts = df.iloc[X_indices]['texto_limpo'].tolist()
        return texts
    else:
        raise ValueError(
            "DataProcessor não tem dados originais. "
            "Execute load_and_process() primeiro."
        )