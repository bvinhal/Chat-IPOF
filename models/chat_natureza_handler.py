# models/chat_natureza_handler.py

import logging
from typing import Dict, List, Any, Optional
from config import active_config
from models.natureza_integrator import NaturezaIntegrator
from models.natureza_evaluator_integration import NaturezaEvaluatorIntegration

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatNaturezaHandler:
    """
    Manipulador para integrar o classificador de natureza com o chat existente.
    """
    
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa o manipulador de chat com natureza.
        
        Args:
            embedding_provider: Provedor de embeddings a usar
        """
        self.integrator = NaturezaIntegrator(embedding_provider)
        # Adiciona o integrador de avaliação
        self.evaluator_integration = NaturezaEvaluatorIntegration(embedding_provider)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def enhance_response(self, query: str, ai_model, original_response: str) -> str:
        """
        Método mantido para compatibilidade, mas agora aproveita o avaliador de natureza.
        """
        # Verifica se a consulta parece ser sobre natureza de despesa
        if not self.integrator.is_natureza_query(query):
            return original_response
        
        try:
            # Processa a consulta com o integrador, passando o modelo atual para validação
            integrator_result = self.integrator.process_query(query, ai_model)
            
            # Processa com o avaliador de natureza integrado
            evaluator_result = self.evaluator_integration.process_message(query)
            
            # Decide qual resposta usar com base nos resultados
            if evaluator_result.get('has_classifications', False) and evaluator_result.get('evaluation'):
                # Se temos classificação e avaliação, usa a resposta do avaliador integrado
                enhanced = original_response + "\n\n---\n\n"
                enhanced += self.evaluator_integration.format_response(evaluator_result)
                return enhanced
            elif integrator_result.get('predictions'):
                # Se não temos avaliação mas temos previsões do integrador, usa o método original
                enhanced = self._format_enhanced_response(original_response, integrator_result)
                return enhanced
            else:
                # Nenhum resultado válido, retorna a resposta original
                return original_response
            
        except Exception as e:
            self.logger.error(f"Erro ao aprimorar resposta: {str(e)}")
            # Em caso de erro, retorna a resposta original para garantir que o usuário receba algo
            return original_response
    
    def _format_enhanced_response(self, original_response: str, natureza_result: Dict[str, Any]) -> str:
        """
        Formata uma resposta aprimorada com informações de natureza.
        Mantido para compatibilidade com o fluxo anterior.
        
        Args:
            original_response: Resposta original do modelo
            natureza_result: Resultado do processamento de natureza
            
        Returns:
            str: Resposta formatada com informações de natureza
        """
        predictions = natureza_result['predictions']
        recommended = natureza_result['recommended']
        
        # Se temos uma validação, usamos a resposta da validação
        if natureza_result.get('validation') and natureza_result['validation'].get('response'):
            validation_response = natureza_result['validation']['response']
            
            # Adiciona a recomendação formatada ao final da resposta original
            enhanced = original_response + "\n\n---\n\n"
            enhanced += "**Classificação de Natureza de Despesa:**\n\n"
            enhanced += validation_response
            
            return enhanced
        
        # Se não temos validação, criamos uma resposta formatada com as previsões
        enhanced = original_response + "\n\n---\n\n"
        enhanced += "**Classificação de Natureza de Despesa:**\n\n"
        
        if recommended:
            enhanced += f"Com base na sua consulta, a natureza de despesa mais adequada parece ser:\n\n"
            enhanced += f"**{recommended['codigo']} - {recommended['nome']}**\n"
            enhanced += f"(Confiança: {recommended['confianca']:.2%})\n\n"
        
        if len(predictions) > 1:
            enhanced += "Outras possíveis classificações de natureza de despesa incluem:\n\n"
            
            for i, pred in enumerate(predictions[1:], 2):
                enhanced += f"{i}. **{pred['codigo']} - {pred['nome']}**\n"
                enhanced += f"   Confiança: {pred['confianca']:.2%}\n"
            
            enhanced += "\nEssas classificações são baseadas na análise do texto fornecido e podem precisar de validação adicional dependendo do contexto específico da despesa."
        
        return enhanced