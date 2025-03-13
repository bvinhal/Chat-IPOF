import os
import glob
import logging
from datetime import datetime
from typing import List, Dict, Any

from models import get_model_class
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingController:
    """
    Controlador para gerenciar o treinamento dos modelos de IA.
    """
    
    def __init__(self):
        """Inicializa o controlador de treinamento"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def train_model(self, model_type: str) -> Dict[str, Any]:
        """
        Treina um modelo específico.
        
        Args:
            model_type: Tipo do modelo ('claude', 'openai', 'gemini')
            
        Returns:
            Dict[str, Any]: Resultado do treinamento
        """
        # Obtém a classe do modelo
        model_class = get_model_class(model_type)
        if model_class is None:
            error_msg = f"Tipo de modelo '{model_type}' não suportado"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
        
        try:
            # Cria e inicializa o modelo
            model_instance = model_class()
            if not model_instance.initialize():
                error_msg = f"Falha ao inicializar modelo {model_type}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Verifica se o diretório de treinamento existe
            if not os.path.exists(active_config.TRAINING_DATA_DIR):
                error_msg = f"Diretório de treinamento não encontrado: {active_config.TRAINING_DATA_DIR}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Treina o modelo
            self.logger.info(f"Iniciando treinamento do modelo {model_type}...")
            if not model_instance.train(active_config.TRAINING_DATA_DIR):
                error_msg = f"Falha no treinamento do modelo {model_type}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            # Gera um nome para o modelo baseado na data e hora
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_class.__name__}_{timestamp}"
            
            # Salva o modelo treinado
            if not model_instance.save_model(model_name):
                error_msg = f"Falha ao salvar modelo {model_type}"
                self.logger.error(error_msg)
                return {'success': False, 'message': error_msg}
            
            success_msg = f"Modelo {model_type} treinado e salvo com sucesso como {model_name}"
            self.logger.info(success_msg)
            return {'success': True, 'message': success_msg, 'model_name': model_name}
            
        except Exception as e:
            error_msg = f"Erro durante o treinamento do modelo {model_type}: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Lista todos os modelos treinados disponíveis.
        
        Returns:
            List[Dict[str, Any]]: Lista de modelos disponíveis
        """
        models_dir = active_config.MODELS_DIR
        available_models = []
        
        # Verifica se o diretório existe
        if not os.path.exists(models_dir):
            self.logger.warning(f"Diretório de modelos não encontrado: {models_dir}")
            return available_models
        
        # Busca por diretórios de modelos para cada tipo
        for model_type, model_class in [
            ('claude', 'ClaudeModel'), 
            ('openai', 'OpenAIModel'), 
            ('gemini', 'GeminiModel')
        ]:
            model_dirs = glob.glob(os.path.join(models_dir, f"{model_class}_*"))
            
            for model_dir in model_dirs:
                model_name = os.path.basename(model_dir)
                
                # Tenta ler os metadados do modelo
                metadata_path = os.path.join(model_dir, "metadata.pkl")
                model_details = {
                    'name': model_name,
                    'type': model_type,
                    'path': model_dir,
                    'created': datetime.fromtimestamp(os.path.getctime(model_dir)).strftime("%Y-%m-%d %H:%M:%S")
                }
                
                available_models.append(model_details)
        
        # Ordena por data de criação (mais recente primeiro)
        available_models.sort(key=lambda x: x['created'], reverse=True)
        
        return available_models
    
    def delete_model(self, model_path: str) -> Dict[str, Any]:
        """
        Exclui um modelo treinado.
        
        Args:
            model_path: Caminho para o modelo a ser excluído
            
        Returns:
            Dict[str, Any]: Resultado da operação
        """
        if not os.path.exists(model_path):
            error_msg = f"Modelo não encontrado: {model_path}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
        
        try:
            # Exclui todos os arquivos dentro do diretório
            for file_path in glob.glob(os.path.join(model_path, "*")):
                os.remove(file_path)
            
            # Exclui o diretório
            os.rmdir(model_path)
            
            success_msg = f"Modelo excluído com sucesso: {os.path.basename(model_path)}"
            self.logger.info(success_msg)
            return {'success': True, 'message': success_msg}
        except Exception as e:
            error_msg = f"Erro ao excluir modelo {model_path}: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'message': error_msg}
