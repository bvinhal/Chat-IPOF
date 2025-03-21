# controllers/enhanced_ipof_chat_controller.py

import os
import logging
from typing import List, Dict, Any
import re

from controllers.enhanced_chat_controller import EnhancedChatController
from controllers.ipof_controller import IPOFController
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedIPOFChatController(EnhancedChatController):
    """
    Controlador de chat aprimorado com suporte a IPOF.
    """
    
    def __init__(self):
        """Inicializa o controlador de chat com suporte a IPOF."""
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Inicializa o controlador de IPOF
        self.ipof_controller = IPOFController(self)
        
        # Diretório para armazenar arquivos temporários
        self.temp_dir = os.path.join(active_config.DATA_DIR, 'temp')
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def change_model(self, model_type: str) -> Dict[str, Any]:
        """
        Sobrescreve o método para garantir que o manipulador de natureza seja atualizado.
        """
        result = super().change_model(model_type)
        
        # Se houve sucesso, atualiza o manipulador de natureza
        if result.get('success', False):
            self.update_natureza_handler()
            
        return result
        
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Processa uma mensagem do usuário com suporte a IPOF.
        
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
            # Verifica se o controlador de IPOF está ativo
            if self.ipof_controller.esta_ativo():
                # Se está ativo, processa a mensagem pelo controlador de IPOF
                return self.ipof_controller.processar_mensagem(message, self.current_model)
            
            # Verifica se a mensagem indica o desejo de iniciar a criação de um IPOF
            if self.ipof_controller.verificar_inicio_ipof(message):
                # Inicia a criação de um IPOF
                return self.ipof_controller.iniciar_criacao_ipof()
            
            # Se não está relacionado a IPOF, processa normalmente
            return super().process_message(message, chat_history)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem: {str(e)}")
            return (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                f"Detalhes do erro: {str(e)}"
            )
    
    def process_file(self, file_path: str, file_type: str, message: str = "", chat_history: List[Dict[str, str]] = None) -> str:
        """
        Processa um arquivo enviado pelo usuário.
        
        Args:
            file_path: Caminho para o arquivo
            file_type: Tipo do arquivo
            message: Mensagem adicional do usuário (opcional)
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        try:
            # Verifica se o controlador de IPOF está ativo e no estado de receber arquivo
            if self.ipof_controller.esta_ativo() and self.ipof_controller.estado_atual == self.ipof_controller.ESTADOS['DESCRICAO']:
                # Processa o arquivo pelo controlador de IPOF
                return self.ipof_controller.processar_arquivo(file_path, file_type, self.current_model)
            
            # Se não está relacionado a IPOF, pode implementar processamento padrão de arquivo
            # Por enquanto, apenas retorna uma mensagem informativa
            return (
                f"Recebi seu arquivo do tipo {file_type}. "
                "No momento, apenas arquivos para a criação de IPOF são suportados. "
                "Se deseja criar um IPOF, digite algo como 'criar IPOF'."
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo: {str(e)}")
            return (
                "Desculpe, ocorreu um erro ao processar seu arquivo. "
                f"Detalhes do erro: {str(e)}"
            )