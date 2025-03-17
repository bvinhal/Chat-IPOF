# models/claude_embedding_model.py

import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from config import active_config
from models.embedding_model_base import EmbeddingModelBase

class ClaudeEmbeddingModel(EmbeddingModelBase):
    """
    Implementação do modelo de embeddings usando a API do Claude (Anthropic).
    
    Nota: A API do Claude atualmente não oferece um endpoint específico para embeddings.
    Esta implementação usa um modelo alternativo e pode ser atualizada quando a Anthropic
    disponibilizar suporte nativo para embeddings.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo de embeddings alternativo para Claude.
        
        Args:
            model_name: Nome do modelo de embeddings (usado apenas para registro)
        """
        model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        super().__init__(model_name)
        self.api_key = active_config.CLAUDE_API_KEY  # Mantido para consistência
        self.embedding_dim = 384  # Dimensão dos embeddings do MiniLM-L6-v2
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo de embeddings local.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        try:
            # Como Claude não tem API de embeddings, usamos um modelo local
            self.client = SentenceTransformer(self.model_name)
            
            self.logger.info(f"Modelo de embeddings {self.model_name} inicializado com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelo de embeddings: {str(e)}")
            return False
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Gera embeddings para os textos fornecidos usando o modelo local.
        
        Args:
            texts: Lista de textos para gerar embeddings
            batch_size: Tamanho do lote para processamento em batch
            
        Returns:
            np.ndarray: Matriz de embeddings com shape (n_texts, embedding_dim)
        """
        if self.client is None and not self.initialize():
            self.logger.error("Modelo de embeddings não inicializado")
            raise ValueError("Modelo de embeddings não inicializado")
        
        try:
            # O modelo SentenceTransformer já implementa processamento em batch
            embeddings = self.client.encode(
                texts, 
                batch_size=batch_size, 
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            self.logger.info(f"Gerados {len(texts)} embeddings com dimensão {self.embedding_dim}")
            return embeddings
        
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings: {str(e)}")
            raise