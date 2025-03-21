# controllers/enhanced_chat_controller.py

import os
import logging
from typing import List, Dict, Any
from controllers.chat_controller import ChatController
from models.chat_natureza_handler import ChatNaturezaHandler
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedChatController(ChatController):
    """
    Controlador de chat aprimorado que adiciona classificação de natureza de despesa.
    """
    def __init__(self):
        """Inicializa o controlador de chat aprimorado."""
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Inicializa o manipulador de natureza de despesa com o modelo atual
        self.natureza_handler = ChatNaturezaHandler(self.current_model_type)
        self.logger.info(f"Manipulador de natureza inicializado com provedor {self.current_model_type}")

    def update_natureza_handler(self):
        """Atualiza o manipulador de natureza quando o modelo é alterado."""
        if hasattr(self, 'natureza_handler'):
            previous_provider = self.natureza_handler.integrator.embedding_provider
            self.natureza_handler = ChatNaturezaHandler(self.current_model_type)
            self.logger.info(f"Manipulador de natureza atualizado de {previous_provider} para {self.current_model_type}")
            
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Processa uma mensagem do usuário com classificação de natureza de despesa.
        
        Args:
            message: Mensagem do usuário
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta aprimorada gerada pelo modelo
        """
        if not message:
            return "Por favor, digite uma mensagem."
        
        if self.current_model is None:
            return (
                "Nenhum modelo está disponível no momento. "
                "Por favor, realize o treinamento primeiro ou escolha outro modelo."
            )
        
        try:
            # Primeiro, obtém a resposta original do modelo de IA
            original_response = super().process_message(message, chat_history)
            
            # Em seguida, aprimora a resposta com informações de natureza de despesa
            enhanced_response = self.natureza_handler.enhance_response(
                message, 
                self.current_model, 
                original_response
            )
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem aprimorada: {str(e)}")
            return (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                f"Detalhes do erro: {str(e)}"
            )