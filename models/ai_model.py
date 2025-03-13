from abc import ABC, abstractmethod
import os
import pickle
from typing import List, Dict, Any, Optional
from config import active_config
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIModel(ABC):
    """
    Classe base abstrata para modelos de IA.
    Define a interface comum para todos os modelos suportados.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo de IA.
        
        Args:
            model_name: Nome do modelo específico a ser utilizado
        """
        self.model_name = model_name
        self.is_trained = False
        self.model_path = None
        self.vectorstore = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Inicializa o modelo e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        pass
    
    @abstractmethod
    def train(self, documents_path: str) -> bool:
        """
        Treina o modelo com os documentos fornecidos.
        
        Args:
            documents_path: Caminho para os documentos de treinamento
            
        Returns:
            bool: True se o treinamento foi bem-sucedido, False caso contrário
        """
        pass
    
    @abstractmethod
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Gera uma resposta para a consulta do usuário.
        
        Args:
            query: Consulta do usuário
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        pass
    
    def save_model(self, model_name: str = None) -> bool:
        """
        Salva o modelo treinado para uso posterior.
        
        Args:
            model_name: Nome para salvar o modelo (opcional)
            
        Returns:
            bool: True se o salvamento foi bem-sucedido, False caso contrário
        """
        if not self.is_trained or self.vectorstore is None:
            self.logger.error("Tentativa de salvar modelo não treinado")
            return False
        
        try:
            if model_name is None:
                model_name = f"{self.__class__.__name__}_{self.model_name}"
            
            # Cria o diretório do modelo se não existir
            model_dir = os.path.join(active_config.MODELS_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Para FAISS vectorstore, salva os componentes essenciais separadamente
            if hasattr(self.vectorstore, 'serialize_to_bytes'):
                # Usa o método integrado do FAISS para serializar
                faiss_data = self.vectorstore.serialize_to_bytes()
                faiss_path = os.path.join(model_dir, "faiss_index.bin")
                with open(faiss_path, 'wb') as f:
                    f.write(faiss_data)
                    
                # Salva metadados adicionais necessários para reconstruir o vectorstore
                if hasattr(self.vectorstore, 'docstore'):
                    docstore_path = os.path.join(model_dir, "docstore.pkl")
                    with open(docstore_path, 'wb') as f:
                        pickle.dump(self.vectorstore.docstore, f)
                        
                # Salva informações do embedding
                if hasattr(self.vectorstore, '_embeddings'):
                    embeddings_info = {
                        'class_name': self.vectorstore._embeddings.__class__.__name__,
                        'model_name': getattr(self.vectorstore._embeddings, 'model_name', None)
                    }
                    embeddings_path = os.path.join(model_dir, "embeddings_info.pkl")
                    with open(embeddings_path, 'wb') as f:
                        pickle.dump(embeddings_info, f)
            else:
                # Tenta o método tradicional com pickle
                vectorstore_path = os.path.join(model_dir, "vectorstore.pkl")
                with open(vectorstore_path, 'wb') as f:
                    pickle.dump(self.vectorstore, f)
            
            # Salva metadados
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            metadata = {
                'model_name': self.model_name,
                'model_type': self.__class__.__name__,
                'is_trained': self.is_trained
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.model_path = model_dir
            self.logger.info(f"Modelo salvo com sucesso em {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            return False
            
    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo treinado anteriormente.
        
        Args:
            model_path: Caminho para o modelo salvo
            
        Returns:
            bool: True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            # Verifica se o diretório existe
            if not os.path.exists(model_path):
                self.logger.error(f"Caminho do modelo não existe: {model_path}")
                return False
            
            # Carrega metadados
            metadata_path = os.path.join(model_path, "metadata.pkl")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verifica se o tipo de modelo é compatível
            if metadata['model_type'] != self.__class__.__name__:
                self.logger.error(f"Tipo de modelo incompatível: {metadata['model_type']}")
                return False
            
            # Tenta carregar o vectorstore.pkl primeiro
            vectorstore_path = os.path.join(model_path, "vectorstore.pkl")
            if os.path.exists(vectorstore_path):
                with open(vectorstore_path, 'rb') as f:
                    self.vectorstore = pickle.load(f)
            else:
                # Se não encontrar, tenta carregar o docstore.pkl
                docstore_path = os.path.join(model_path, "docstore.pkl")
                if os.path.exists(docstore_path):
                    with open(docstore_path, 'rb') as f:
                        self.vectorstore = pickle.load(f)
                else:
                    self.logger.error(f"Nem vectorstore.pkl nem docstore.pkl encontrados em {model_path}")
                    return False
            
            # Atualiza propriedades do modelo
            self.model_name = metadata['model_name']
            self.is_trained = metadata['is_trained']
            self.model_path = model_path
            
            self.logger.info(f"Modelo carregado com sucesso de {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return False
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key necessária está configurada.
        
        Returns:
            bool: True se a API key está configurada, False caso contrário
        """
        return True  # Implementado nas classes derivadas
