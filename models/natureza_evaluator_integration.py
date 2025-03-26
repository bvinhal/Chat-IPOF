# models/natureza_evaluator_integration.py

import logging
from typing import Dict, Any, Optional, List

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
    Permite validar as naturezas sugeridas pelo classificador, ignorando os dois últimos dígitos.
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
        Versão melhorada que avalia múltiplos candidatos.
        
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
            
            # 1. Classifica a natureza (obtém até 5 classificações)
            classifications = self.classifier.predict(message, top_k=5)
            
            if not classifications:
                return {
                    'is_natureza_query': True,
                    'has_classifications': False,
                    'message': 'Não foi possível classificar a natureza de despesa'
                }
            
            # 2. Avalia as classificações com o avaliador
            evaluation = None
            try:
                evaluation = self.evaluator_controller.evaluate_candidates(message, classifications)
            except Exception as e:
                self.logger.error(f"Erro na avaliação dos candidatos: {str(e)}")
                # Continua o processamento mesmo com erro na avaliação
            
            # 3. Retorna o resultado combinado
            result = {
                'is_natureza_query': True,
                'has_classifications': True,
                'classifications': classifications,
                'evaluation': evaluation.get('result') if evaluation and evaluation.get('success', False) else None,
                'recommended': None
            }
            
            # Define a recomendação baseada na avaliação ou nas classificações originais
            if evaluation and evaluation.get('success', False) and evaluation.get('result'):
                eval_result = evaluation['result']
                if eval_result.get('best_alternative'):
                    result['recommended'] = eval_result['best_alternative']
                    result['message'] = f"Nenhuma das naturezas pré-classificadas parece adequada. Sugestão: {eval_result['best_alternative']['codigo']}"
                elif eval_result.get('recommended'):
                    result['recommended'] = eval_result['recommended']
                    result['message'] = f"A natureza mais adequada é {eval_result['recommended']['codigo']} (score: {eval_result['recommended'].get('score', 0):.2%})"
                else:
                    result['recommended'] = classifications[0]
                    result['message'] = f"Avaliador não conseguiu determinar uma recomendação. Usando classificação de maior confiança: {classifications[0]['codigo']}"
            else:
                # Se não conseguiu avaliar, usa a classificação de maior confiança
                result['recommended'] = classifications[0]
                result['message'] = f"Classificado como {classifications[0]['codigo']} (Não foi possível avaliar a adequação)"
            
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
        Versão melhorada que apresenta avaliação de múltiplos candidatos.
        
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
        recommended = result.get('recommended')
        
        # Formata a resposta
        response = "**Análise de Natureza de Despesa:**\n\n"
        
        # Adiciona a natureza recomendada
        if recommended:
            codigo = recommended.get('codigo', '')
            nome = recommended.get('nome', '')
            
            if not nome and 'codigo' in recommended:
                # Tenta encontrar o nome em outras estruturas
                for c in classifications:
                    if c.get('codigo') == codigo:
                        nome = c.get('nome', '')
                        break
            
            response += f"A natureza de despesa recomendada é **{codigo}"
            if nome:
                response += f" - {nome}"
            response += "**\n\n"
            
            # Adiciona informação sobre o score se disponível
            if 'score' in recommended:
                response += f"Score: {recommended['score']:.2%}\n\n"
        
        # Adiciona a justificativa, se disponível
        if evaluation and evaluation.get('justificativa'):
            response += f"**Justificativa:**\n{evaluation['justificativa']}\n\n"
        
        # Adiciona o ranking de naturezas, se disponível
        if evaluation and evaluation.get('ranking'):
            response += "**Ranking das naturezas candidatas:**\n\n"
            for i, item in enumerate(evaluation['ranking'], 1):
                codigo = item.get('codigo', '')
                nome = item.get('nome', '')
                score = item.get('score', 0)
                
                response += f"{i}. **{codigo}"
                if nome:
                    response += f" - {nome}"
                response += f"** (Score: {score:.2%})\n"
            
            response += "\n"
        
        # Se não houver avaliação mas tiver classificações, mostra as classificações originais
        elif classifications and len(classifications) > 1:
            response += "**Naturezas classificadas (sem avaliação):**\n\n"
            for i, classification in enumerate(classifications, 1):
                response += f"{i}. **{classification['codigo']} - {classification['nome']}**\n"
                response += f"   Confiança: {classification['confianca']:.2%}\n"
            
            response += "\n"
        
        response += "Esta análise combina classificação automática e avaliação da natureza de despesa com base no MCASP, considerando apenas os níveis 'c.g.mm.ee' e ignorando os desdobramentos."
        
        return response
        
