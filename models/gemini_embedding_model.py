# models/gemini_embedding_model.py

import numpy as np
from typing import List
import time
from tqdm import tqdm
from config import active_config
from models.embedding_model_base import EmbeddingModelBase
from sentence_transformers import SentenceTransformer

class GeminiEmbeddingModel(EmbeddingModelBase):
    """
    Implementação do modelo de embeddings usando a API do Google Gemini.
    
    Observação: Se o Gemini não oferecer um endpoint específico para embeddings,
    esta implementação usará um modelo alternativo.
    """
    
    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Inicializa o modelo de embeddings do Gemini.
        
        Args:
            model_name: Nome do modelo de embeddings do Gemini
        """
        super().__init__(model_name)
        self.api_key = active_config.GEMINI_API_KEY
        self.embedding_dim = 768  # Dimensão aproximada dos embeddings do Gemini
    
    def initialize(self) -> bool:
        """
        Inicializa o cliente do Google para Gemini.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        if not self.validate_api_key():
            self.logger.error("Chave de API do Google Gemini não configurada")
            return False
        
        try:
            import google.generativeai as genai
            
            # Configura o cliente
            genai.configure(api_key=self.api_key)
            self.client = genai
            
            self.logger.info(f"Cliente Google Gemini inicializado para modelo {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar cliente Google Gemini: {str(e)}")
            return False
    
    def get_embeddings(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        Gera embeddings para os textos fornecidos usando a API do Gemini.
        
        Args:
            texts: Lista de textos para gerar embeddings
            batch_size: Tamanho do lote para processamento em batch
            
        Returns:
            np.ndarray: Matriz de embeddings com shape (n_texts, embedding_dim)
        """
        if self.client is None and not self.initialize():
            self.logger.error("Cliente Google Gemini não inicializado")
            raise ValueError("Cliente Google Gemini não inicializado. Verifique a chave de API.")
        
        try:
            all_embeddings = []
            
            # Processa em lotes para evitar limites de requisição
            for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings Gemini"):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = []
                
                # Processa cada texto no lote
                for text in batch_texts:
                    result = self.client.embed_content(
                        model=self.model_name,
                        content=text,
                        task_type="retrieval_query"
                    )
                    
                    # Extrai o embedding
                    embedding = result["embedding"]
                    batch_embeddings.append(embedding)
                
                all_embeddings.extend(batch_embeddings)
                
                # Pausa para respeitar limites de taxa
                if i + batch_size < len(texts):
                    time.sleep(0.5)
            
            # Converte para array numpy
            embeddings_array = np.array(all_embeddings)
            
            self.logger.info(f"Gerados {len(all_embeddings)} embeddings com dimensão {embeddings_array.shape[1]}")
            return embeddings_array
        
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings Gemini: {str(e)}")
            
            # Fallback para modelo local se a API do Gemini falhar
            self.logger.info("Tentando fallback para modelo local de embeddings")
            
            try:
                # Inicializa modelo local de embeddings
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                
                self.logger.info(f"Gerados {len(texts)} embeddings (fallback) com dimensão {embeddings.shape[1]}")
                return embeddings
                
            except Exception as fallback_error:
                self.logger.error(f"Erro ao usar fallback para embeddings: {str(fallback_error)}")
                raise