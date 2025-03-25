# models/natureza_evaluator_integration.py

import logging
from typing import Dict, Any, Optional

from models.natureza_classifier import NaturezaClassifier
from controllers.natureza_evaluator_controller import NaturezaEvaluatorController

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaEvaluatorIntegration:
    """
    Classe que integra o classificador e o avaliador de natureza no fluxo de chat.
    Permite validar se a natureza sugerida pelo classificador é adequada.
    """
    
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa a integração.
        
        Args:
            embedding_provider: Provedor de embeddings a usar
        """
        self.embedding_provider = embedding_provider
        self.classifier = NaturezaClassifier(embedding_provider)
        self.evaluator_controller = NaturezaEvaluatorController()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tenta carregar os modelos
        self.classifier.load_model()
        # O controlador de avaliador já tenta carregar no construtor
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Processa uma mensagem para classificar e avaliar a natureza de despesa.
        
        Args:
            message: Mensagem do usuário
            
        Returns:
            Dict[str, Any]: Resultado do processamento
        """
        try:
            # Verifica se parece ser uma consulta sobre natureza
            if not self.classifier.integrator.is_natureza_query(message):
                return {
                    'is_natureza_query': False,
                    'message': 'Não é uma consulta sobre natureza de despesa'
                }
            
            # 1. Classifica a natureza
            classifications = self.classifier.predict(message, top_k=3)
            
            if not classifications:
                return {
                    'is_natureza_query': True,
                    'has_classifications': False,
                    'message': 'Não foi possível classificar a natureza de despesa'
                }
            
            # 2. Avalia a natureza recomendada
            top_classification = classifications[0]
            evaluation = self.evaluator_controller.evaluate_natureza(
                message, 
                top_classification['codigo']
            )
            
            # 3. Retorna o resultado combinado
            result = {
                'is_natureza_query': True,
                'has_classifications': True,
                'classifications': classifications,
                'evaluation': evaluation.get('result') if evaluation.get('success', False) else None,
                'top_classification': top_classification
            }
            
            # Adiciona uma mensagem baseada no resultado
            if evaluation.get('success', False) and evaluation.get('result'):
                eval_result = evaluation['result']
                if eval_result.get('is_valid', False):
                    result['message'] = f"A natureza {top_classification['codigo']} foi classificada como adequada."
                else:
                    # Se a natureza não foi considerada adequada, verifica a sugestão do avaliador
                    if eval_result.get('best_match'):
                        best_match = eval_result['best_match']
                        result['message'] = f"A natureza {top_classification['codigo']} pode não ser a mais adequada. Considere {best_match['codigo']}."
                    else:
                        result['message'] = f"A natureza {top_classification['codigo']} pode não ser a mais adequada para esta descrição."
            else:
                result['message'] = f"Classificado como {top_classification['codigo']} (Não foi possível avaliar a adequação)."
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem: {str(e)}")
            return {
                'is_natureza_query': False,
                'message': f"Erro ao processar mensagem: {str(e)}"
            }
    
    def update_provider(self, provider: str) -> bool:
        """
        Atualiza o provedor de embeddings.
        
        Args:
            provider: Novo provedor
            
        Returns:
            bool: True se atualizado com sucesso
        """
        try:
            # Atualiza o classificador
            self.embedding_provider = provider
            self.classifier = NaturezaClassifier(provider)
            self.classifier.load_model()
            
            # Atualiza o avaliador
            self.evaluator_controller.change_provider(provider)
            
            return True
        except Exception as e:
            self.logger.error(f"Erro ao atualizar provedor: {str(e)}")
            return False
    
    def format_response(self, result: Dict[str, Any]) -> str:
        """
        Formata o resultado em texto para resposta ao usuário.
        
        Args:
            result: Resultado do processamento
            
        Returns:
            str: Mensagem formatada para o usuário
        """
        if not result.get('is_natureza_query', False):
            return ""  # Não é uma consulta sobre natureza
        
        if not result.get('has_classifications', False):
            return "Não foi possível classificar a natureza de despesa para sua consulta."
        
        # Obtém classificações e avaliação
        classifications = result.get('classifications', [])
        evaluation = result.get('evaluation')
        
        # Formata a resposta
        response = "**Análise de Natureza de Despesa:**\n\n"
        
        # Adiciona a classificação principal
        top = classifications[0]
        response += f"A natureza de despesa sugerida é **{top['codigo']} - {top['nome']}**\n"
        response += f"(Confiança: {top['confianca']:.2%})\n\n"
        
        # Adiciona a avaliação, se disponível
        if evaluation:
            response += f"**Avaliação da adequação:**\n"
            response += f"{evaluation['justificativa']}\n\n"
            
            # Se houver uma alternativa melhor sugerida pelo avaliador
            if not evaluation['is_valid'] and evaluation.get('best_match'):
                best = evaluation['best_match']
                response += f"**Sugestão alternativa:** {best['codigo']}"
                if best['info'] and 'nome' in best['info']:
                    response += f" - {best['info']['nome']}"
                response += f" (Similaridade: {best['similarity']:.2%})\n\n"
        
        # Adiciona outras classificações
        if len(classifications) > 1:
            response += "**Outras classificações possíveis:**\n\n"
            for i, classification in enumerate(classifications[1:], 2):
                response += f"{i}. **{classification['codigo']} - {classification['nome']}**\n"
                response += f"   Confiança: {classification['confianca']:.2%}\n"
        
        response += "\nEsta análise combina classificação e avaliação da natureza de despesa com base no MCASP."
        
        return response