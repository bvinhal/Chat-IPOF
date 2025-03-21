import os
import glob
import logging
from typing import List, Dict, Any
import json

from models import get_model_class
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatController:
    """
    Controlador para gerenciar a interação entre o chat e os modelos de IA.
    """
    
    def __init__(self):
        """Inicializa o controlador de chat"""
        self.current_model = None
        self.current_model_type = active_config.DEFAULT_MODEL
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tenta carregar o modelo padrão
        self._load_model(self.current_model_type)
    
    def _load_model(self, model_type: str) -> bool:
        """
        Carrega um modelo específico.
        
        Args:
            model_type: Tipo do modelo ('claude', 'openai', 'gemini')
            
        Returns:
            bool: True se o modelo foi carregado com sucesso, False caso contrário
        """
        # Obtém a classe do modelo
        model_class = get_model_class(model_type)
        if model_class is None:
            self.logger.error(f"Tipo de modelo '{model_type}' não suportado")
            return False
        
        # Verifica se existe um modelo treinado
        models_dir = active_config.MODELS_DIR
        model_dirs = glob.glob(os.path.join(models_dir, f"{model_class.__name__}_*"))
        
        # Ordena por data de modificação (mais recente primeiro)
        model_dirs.sort(key=os.path.getmtime, reverse=True)
        
        for model_dir in model_dirs:
            try:
                # Cria e carrega o modelo
                model = model_class()
                if model.load_model(model_dir):
                    self.current_model = model
                    self.current_model_type = model_type
                    self.logger.info(f"Modelo '{model_dir}' carregado com sucesso")
                    return True
            except Exception as e:
                self.logger.warning(f"Erro ao carregar modelo de {model_dir}: {str(e)}")
        
        self.logger.warning(f"Nenhum modelo treinado do tipo '{model_type}' encontrado")
        return False
    
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Processa uma mensagem do usuário e retorna a resposta.
        
        Args:
            message: Mensagem do usuário
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        if not message:
            return "Por favor, digite uma mensagem."
        
        if self.current_model is None:
            return (
                "Nenhum modelo está disponível no momento. "
                "Por favor, realize o treinamento primeiro ou escolha outro modelo."
            )
        
        try:
            response = self.current_model.generate_response(message, chat_history)
            return response
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem: {str(e)}")
            return (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                f"Detalhes do erro: {str(e)}"
            )
    
    def change_model(self, model_type: str) -> Dict[str, Any]:
        """
        Altera o modelo atual para o tipo especificado.
        """
        if model_type.lower() == self.current_model_type.lower() and self.current_model is not None:
            return {
                'success': True, 
                'message': f"Modelo já está definido para {model_type}"
            }
        
        # Tenta carregar o novo modelo
        if self._load_model(model_type):
            # Se a classe derivada implementa update_natureza_handler, chama o método
            if hasattr(self, 'update_natureza_handler'):
                self.update_natureza_handler()
                
            return {
                'success': True, 
                'message': f"Modelo alterado com sucesso para {model_type}"
            }
        else:
            return {
                'success': False, 
                'message': f"Falha ao alterar para modelo {model_type}. Verifique se ele está treinado."
            }