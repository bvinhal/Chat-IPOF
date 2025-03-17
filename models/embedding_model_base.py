# models/embedding_model_base.py

from abc import ABC, abstractmethod
import os
import numpy as np
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm
import time
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EmbeddingModelBase(ABC):
    """
    Classe base abstrata para modelos de embeddings.
    Define a interface comum para todos os modelos de embeddings suportados.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo de embeddings.
        
        Args:
            model_name: Nome do modelo específico a ser utilizado
        """
        self.model_name = model_name
        self.api_key = None
        self.client = None
        self.embedding_dim = None  # Será definido nas classes concretas
        self.embeddings_path = os.path.join(active_config.DATA_DIR, 'embeddings')
        os.makedirs(self.embeddings_path, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inicializa o modelo de embeddings e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        pass
    
    @abstractmethod
    def get_embeddings(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        Gera embeddings para os textos fornecidos.
        
        Args:
            texts: Lista de textos para gerar embeddings
            batch_size: Tamanho do lote para processamento em batch
            
        Returns:
            np.ndarray: Matriz de embeddings com shape (n_texts, embedding_dim)
        """
        pass
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key necessária está configurada.
        
        Returns:
            bool: True se a API key está configurada, False caso contrário
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def save_embeddings(self, embeddings: np.ndarray, texts: List[str], metadata: Dict[str, Any] = None) -> str:
        """
        Salva os embeddings gerados para uso posterior.
        
        Args:
            embeddings: Matriz de embeddings
            texts: Lista de textos correspondentes
            metadata: Metadados adicionais (opcional)
            
        Returns:
            str: Caminho onde os embeddings foram salvos
        """
        try:
            # Cria um timestamp para identificação única
            timestamp = int(time.time())
            save_id = f"{self.__class__.__name__}_{timestamp}"
            
            # Cria diretório específico para estes embeddings
            save_dir = os.path.join(self.embeddings_path, save_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # Salva os embeddings como array numpy
            embeddings_path = os.path.join(save_dir, "embeddings.npy")
            np.save(embeddings_path, embeddings)
            
            # Salva os textos correspondentes
            texts_path = os.path.join(save_dir, "texts.json")
            with open(texts_path, 'w', encoding='utf-8') as f:
                json.dump(texts, f, ensure_ascii=False, indent=2)
            
            # Prepara e salva metadados
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'model_name': self.model_name,
                'model_type': self.__class__.__name__,
                'timestamp': timestamp,
                'num_texts': len(texts),
                'embedding_dim': embeddings.shape[1],
                'embeddings_path': embeddings_path,
                'texts_path': texts_path
            })
            
            metadata_path = os.path.join(save_dir, "metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Embeddings salvos com sucesso em {save_dir}")
            return save_dir
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, embeddings_path: str = None) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[Dict[str, Any]]]:
        """
        Carrega embeddings salvos anteriormente.
        
        Args:
            embeddings_path: Caminho para o diretório de embeddings.
                            Se None, carrega os mais recentes.
            
        Returns:
            Tuple: (embeddings, texts, metadata) ou (None, None, None) se não encontrado
        """
        try:
            # Se não for especificado um caminho, encontra os embeddings mais recentes
            if embeddings_path is None:
                # Lista todos os diretórios de embeddings deste modelo
                model_dirs = [d for d in os.listdir(self.embeddings_path) 
                            if os.path.isdir(os.path.join(self.embeddings_path, d)) 
                            and d.startswith(self.__class__.__name__)]
                
                if not model_dirs:
                    self.logger.warning(f"Nenhum diretório de embeddings encontrado para {self.__class__.__name__}")
                    return None, None, None
                
                # Ordena por data de modificação (mais recente primeiro)
                model_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(self.embeddings_path, d)), reverse=True)
                embeddings_path = os.path.join(self.embeddings_path, model_dirs[0])
            
            # Verifica caminhos dos arquivos
            embeddings_file = os.path.join(embeddings_path, "embeddings.npy")
            texts_file = os.path.join(embeddings_path, "texts.json")
            metadata_file = os.path.join(embeddings_path, "metadata.json")
            
            if not os.path.exists(embeddings_file) or not os.path.exists(texts_file):
                self.logger.error(f"Arquivos de embeddings ou textos não encontrados em {embeddings_path}")
                return None, None, None
            
            # Carrega os dados
            embeddings = np.load(embeddings_file)
            
            with open(texts_file, 'r', encoding='utf-8') as f:
                texts = json.load(f)
            
            metadata = None
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            self.logger.info(f"Embeddings carregados com sucesso de {embeddings_path}")
            return embeddings, texts, metadata
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar embeddings: {str(e)}")
            return None, None, None