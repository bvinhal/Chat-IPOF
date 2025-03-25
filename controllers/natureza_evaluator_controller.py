# controllers/natureza_evaluator_controller.py

import os
import logging
import glob
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.natureza_evaluator import NaturezaEvaluator
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaEvaluatorController:
    """
    Controlador para gerenciar modelos de avaliação de natureza de despesa.
    """
    
    def __init__(self):
        """Inicializa o controlador de avaliação de natureza"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.active_evaluator = None
        self.active_provider = active_config.DEFAULT_MODEL
        
        # Tenta carregar o modelo padrão
        if not self._load_evaluator(self.active_provider):
            self.logger.warning(f"Não foi possível carregar avaliador para {self.active_provider}, tentando alternativas...")
            
            # Tenta outros provedores em sequência
            for provider in ['claude', 'openai', 'gemini']:
                if provider != self.active_provider and self._load_evaluator(provider):
                    self.logger.info(f"Avaliador carregado para provedor alternativo: {provider}")
                    break
    
    def _load_evaluator(self, provider: str) -> bool:
        """
        Carrega um avaliador para o provedor especificado.
        
        Args:
            provider: Provedor de embeddings ('openai', 'claude', 'gemini')
            
        Returns:
            bool: True se carregado com sucesso, False caso contrário
        """
        try:
            self.logger.info(f"Tentando carregar avaliador para {provider}...")
            evaluator = NaturezaEvaluator(provider)
            if evaluator.load_model():
                self.active_evaluator = evaluator
                self.active_provider = provider
                self.logger.info(f"Avaliador de natureza carregado para provedor {provider}")
                return True
            else:
                self.logger.warning(f"Falha ao carregar avaliador para {provider}")
                return False
        except Exception as e:
            self.logger.error(f"Erro ao carregar avaliador para {provider}: {str(e)}")
            return False
    
    def train_evaluator(self, provider: str, mcasp_path: str = None, force_rebuild: bool = False) -> Dict[str, Any]:
        """
        Treina um avaliador de natureza para o provedor especificado.
        
        Args:
            provider: Provedor de embeddings ('openai', 'claude', 'gemini')
            mcasp_path: Caminho para o arquivo PDF do MCASP (opcional)
            force_rebuild: Se True, força o retreinamento mesmo se já existir
            
        Returns:
            Dict[str, Any]: Resultado do treinamento
        """
        try:
            self.logger.info(f"Iniciando treinamento de avaliador para {provider}")
            
            # Cria o avaliador
            evaluator = NaturezaEvaluator(provider)
            
            # Verifica se já está treinado (se não forçar reconstrução)
            if not force_rebuild and evaluator.load_model():
                self.logger.info(f"Avaliador para {provider} já existe e foi carregado")
                
                # Atualiza o avaliador ativo
                self.active_evaluator = evaluator
                self.active_provider = provider
                
                return {
                    'success': True,
                    'message': f"Avaliador para {provider} já existe e foi carregado",
                    'natureza_count': len(evaluator.natureza_map)
                }
            
            # Define caminho padrão do MCASP se não for fornecido
            if mcasp_path is None:
                mcasp_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp', 'mcasp.pdf')
                
                # Verifica se o arquivo existe
                if not os.path.exists(mcasp_path):
                    self.logger.error(f"Arquivo MCASP não encontrado em {mcasp_path}")
                    return {
                        'success': False,
                        'message': f"Arquivo MCASP não encontrado em {mcasp_path}"
                    }
                
                # Verifica tamanho do arquivo
                file_size = os.path.getsize(mcasp_path)
                if file_size == 0:
                    self.logger.error(f"Arquivo MCASP está vazio (0 bytes)")
                    return {
                        'success': False,
                        'message': f"Arquivo MCASP está vazio (0 bytes)"
                    }
                
                self.logger.info(f"Arquivo MCASP encontrado em {mcasp_path}, tamanho: {file_size/1024:.2f} KB")
            
            # Treina o modelo
            if evaluator.train(mcasp_path, force_rebuild):
                self.logger.info(f"Avaliador para {provider} treinado com sucesso")
                
                # Atualiza o avaliador ativo
                self.active_evaluator = evaluator
                self.active_provider = provider
                
                return {
                    'success': True,
                    'message': f"Avaliador para {provider} treinado com sucesso",
                    'natureza_count': len(evaluator.natureza_map)
                }
            else:
                self.logger.error(f"Falha ao treinar avaliador para {provider}")
                
                # Se o treinamento falhar com o provedor solicitado, tente um provedor alternativo
                fallback_provider = 'claude' if provider != 'claude' else 'openai'
                self.logger.warning(f"Tentando treinar com provedor alternativo: {fallback_provider}")
                
                fallback_evaluator = NaturezaEvaluator(fallback_provider)
                if fallback_evaluator.train(mcasp_path, force_rebuild):
                    self.logger.info(f"Avaliador treinado com provedor alternativo {fallback_provider}")
                    
                    # Atualiza o avaliador ativo
                    self.active_evaluator = fallback_evaluator
                    self.active_provider = fallback_provider
                    
                    return {
                        'success': True,
                        'message': f"Avaliador treinado com provedor alternativo {fallback_provider}",
                        'natureza_count': len(fallback_evaluator.natureza_map),
                        'used_fallback': True,
                        'original_provider': provider,
                        'fallback_provider': fallback_provider
                    }
                
                return {
                    'success': False,
                    'message': f"Falha ao treinar avaliador para {provider} e para o fallback"
                }
        except Exception as e:
            self.logger.error(f"Erro ao treinar avaliador: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Erro ao treinar avaliador: {str(e)}"
            }
    
    def evaluate_natureza(self, descricao: str, natureza_codigo: str) -> Dict[str, Any]:
        """
        Avalia se uma natureza de despesa é adequada para uma descrição.
        
        Args:
            descricao: Descrição da despesa
            natureza_codigo: Código da natureza a ser avaliada
            
        Returns:
            Dict[str, Any]: Resultado da avaliação
        """
        if self.active_evaluator is None:
            self.logger.warning("Nenhum avaliador de natureza está ativo. Tentando carregar ou treinar...")
            
            # Tenta carregar um avaliador para qualquer provedor
            for provider in [self.active_provider, 'claude', 'openai', 'gemini']:
                if self._load_evaluator(provider):
                    self.logger.info(f"Avaliador carregado para {provider}")
                    break
            
            # Se ainda não tiver um avaliador, tenta treinar
            if self.active_evaluator is None:
                self.logger.info("Tentando treinar um avaliador...")
                result = self.train_evaluator('claude')  # Usa claude como fallback seguro
                
                if not result['success']:
                    return {
                        'success': False,
                        'message': "Nenhum avaliador de natureza está ativo e não foi possível treinar um novo."
                    }
        
        try:
            result = self.active_evaluator.evaluate(descricao, natureza_codigo)
            return {
                'success': True,
                'result': result,
                'provider': self.active_provider
            }
        except Exception as e:
            self.logger.error(f"Erro ao avaliar natureza: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            
            # Tenta avaliação com outro provedor se o atual falhar
            if self.active_provider != 'claude':
                self.logger.info("Tentando avaliação com provedor alternativo (claude)...")
                try:
                    # Tenta carregar avaliador alternativo
                    alt_evaluator = NaturezaEvaluator('claude')
                    if alt_evaluator.load_model() or alt_evaluator.train():
                        alt_result = alt_evaluator.evaluate(descricao, natureza_codigo)
                        return {
                            'success': True,
                            'result': alt_result,
                            'provider': 'claude',
                            'note': 'Avaliação realizada com provedor alternativo'
                        }
                except Exception as alt_e:
                    self.logger.error(f"Erro na avaliação alternativa: {str(alt_e)}")
            
            return {
                'success': False,
                'message': f"Erro ao avaliar natureza: {str(e)}"
            }
    
    def change_provider(self, provider: str) -> Dict[str, Any]:
        """
        Altera o provedor de embeddings ativo.
        
        Args:
            provider: Provedor de embeddings ('openai', 'claude', 'gemini')
            
        Returns:
            Dict[str, Any]: Resultado da operação
        """
        if provider == self.active_provider and self.active_evaluator is not None:
            return {
                'success': True,
                'message': f"Avaliador já está utilizando provedor {provider}"
            }
        
        if self._load_evaluator(provider):
            return {
                'success': True,
                'message': f"Avaliador alterado para provedor {provider}"
            }
        else:
            # Se falhar ao carregar, tenta treinar
            self.logger.info(f"Tentando treinar novo avaliador para {provider}")
            result = self.train_evaluator(provider)
            
            if result['success']:
                return {
                    'success': True,
                    'message': f"Avaliador treinado e ativado para provedor {provider}",
                    'used_fallback': result.get('used_fallback', False),
                    'fallback_provider': result.get('fallback_provider')
                }
            else:
                # Se também falhar treinar, tenta qualquer outro provedor como fallback
                for fallback in ['claude', 'openai', 'gemini']:
                    if fallback != provider and self._load_evaluator(fallback):
                        return {
                            'success': True,
                            'message': f"Avaliador carregado com provedor alternativo {fallback} (solicitado: {provider})",
                            'used_fallback': True,
                            'fallback_provider': fallback
                        }
                
                return {
                    'success': False,
                    'message': f"Falha ao carregar ou treinar avaliador para {provider} ou qualquer alternativa"
                }
    
    def list_evaluators(self) -> List[Dict[str, Any]]:
        """
        Lista todos os avaliadores de natureza disponíveis.
        
        Returns:
            List[Dict[str, Any]]: Lista de avaliadores disponíveis
        """
        evaluators = []
        
        # Procura por diretórios de modelos de avaliador
        pattern = os.path.join(active_config.MODELS_DIR, 'natureza_evaluator_*')
        model_dirs = glob.glob(pattern)
        
        for model_dir in model_dirs:
            try:
                # Extrai o provedor do nome do diretório
                provider = os.path.basename(model_dir).replace('natureza_evaluator_', '')
                
                # Verifica se existe arquivo de informações
                info_path = os.path.join(model_dir, 'model_info.json')
                
                if os.path.exists(info_path):
                    import json
                    with open(info_path, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        
                    evaluators.append({
                        'provider': provider,
                        'natureza_count': info.get('natureza_count', 0),
                        'created_at': info.get('created_at', ''),
                        'is_active': provider == self.active_provider
                    })
                else:
                    # Se não tiver o arquivo de info, apenas lista o diretório
                    evaluators.append({
                        'provider': provider,
                        'natureza_count': 0,
                        'created_at': datetime.fromtimestamp(os.path.getctime(model_dir)).isoformat(),
                        'is_active': provider == self.active_provider
                    })
            except Exception as e:
                self.logger.error(f"Erro ao processar avaliador {model_dir}: {str(e)}")
        
        return evaluators
    
    def delete_evaluator(self, provider: str) -> Dict[str, Any]:
        """
        Exclui um avaliador de natureza.
        
        Args:
            provider: Provedor de embeddings do avaliador a ser excluído
            
        Returns:
            Dict[str, Any]: Resultado da operação
        """
        try:
            # Verifica se está tentando excluir o avaliador ativo
            if provider == self.active_provider:
                self.active_evaluator = None
                self.active_provider = None
            
            # Monta o caminho do diretório
            model_dir = os.path.join(active_config.MODELS_DIR, f'natureza_evaluator_{provider}')
            
            # Verifica se o diretório existe
            if not os.path.exists(model_dir):
                return {
                    'success': False,
                    'message': f"Avaliador para provedor {provider} não encontrado"
                }
            
            # Remove o diretório
            import shutil
            shutil.rmtree(model_dir)
            
            self.logger.info(f"Avaliador para {provider} excluído com sucesso")
            return {
                'success': True,
                'message': f"Avaliador para {provider} excluído com sucesso"
            }
        except Exception as e:
            self.logger.error(f"Erro ao excluir avaliador: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return {
                'success': False,
                'message': f"Erro ao excluir avaliador: {str(e)}"
            }