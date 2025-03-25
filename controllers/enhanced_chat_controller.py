# controllers/enhanced_chat_controller.py

import os
import logging
from typing import List, Dict, Any
from controllers.chat_controller import ChatController
from models.chat_natureza_handler import ChatNaturezaHandler
from controllers.natureza_evaluator_controller import NaturezaEvaluatorController
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
        
        # Inicializa o controlador de avaliação de natureza
        self.natureza_evaluator_controller = NaturezaEvaluatorController()
        self.logger.info(f"Controlador de avaliação de natureza inicializado")

    def update_natureza_handler(self):
        """Atualiza o manipulador de natureza quando o modelo é alterado."""
        if hasattr(self, 'natureza_handler'):
            previous_provider = self.natureza_handler.integrator.embedding_provider
            self.natureza_handler = ChatNaturezaHandler(self.current_model_type)
            self.logger.info(f"Manipulador de natureza atualizado de {previous_provider} para {self.current_model_type}")
        
        # Atualiza também o avaliador de natureza
        if hasattr(self, 'natureza_evaluator_controller'):
            self.natureza_evaluator_controller.change_provider(self.current_model_type)
            self.logger.info(f"Avaliador de natureza atualizado para {self.current_model_type}")

    def change_model(self, model_type: str) -> Dict[str, Any]:
        """
        Altera o modelo atual para o tipo especificado e atualiza os componentes relacionados.
        """
        result = super().change_model(model_type)
        
        # Se houve sucesso, atualiza o manipulador de natureza e o avaliador
        if result.get('success', False):
            self.update_natureza_handler()
                
        return result

    def evaluate_natureza(self, descricao: str, natureza_codigo: str) -> Dict[str, Any]:
        """
        Avalia se uma natureza de despesa é adequada para uma descrição.
        
        Args:
            descricao: Descrição da despesa
            natureza_codigo: Código da natureza a ser avaliada
            
        Returns:
            Dict[str, Any]: Resultado da avaliação
        """
        if not hasattr(self, 'natureza_evaluator_controller') or self.natureza_evaluator_controller is None:
            return {
                'success': False,
                'message': "Avaliador de natureza não disponível"
            }
        
        return self.natureza_evaluator_controller.evaluate_natureza(descricao, natureza_codigo)
        
    def _format_response_with_natureza(self, original_response: str, natureza_result: Dict[str, Any]) -> str:
        """
        Formata a resposta para incluir a seção de classificação de natureza.
        Replica a formatação original do ChatNaturezaHandler.
        
        Args:
            original_response: Resposta original do modelo
            natureza_result: Resultado do processamento de natureza
            
        Returns:
            str: Resposta formatada com seção de natureza
        """
        predictions = natureza_result['predictions']
        recommended = natureza_result.get('recommended', predictions[0] if predictions else None)
        
        # Adiciona o separador e o título da seção
        formatted_response = original_response + "\n\n---\n\n"
        formatted_response += "**Classificação de Natureza de Despesa:**\n\n"
        
        if recommended:
            formatted_response += (
                f"Com base na sua consulta, a natureza de despesa mais adequada parece ser:\n\n"
                f"**{recommended['codigo']} - {recommended['nome']}**\n"
                f"(Confiança: {recommended['confianca']:.2%})\n\n"
            )
        
        if len(predictions) > 1:
            formatted_response += "Outras possíveis classificações de natureza de despesa incluem:\n\n"
            
            for i, pred in enumerate(predictions[1:], 2):
                formatted_response += f"{i}. **{pred['codigo']} - {pred['nome']}**\n"
                formatted_response += f"   Confiança: {pred['confianca']:.2%}\n"
            
            formatted_response += "\nEssas classificações são baseadas na análise do texto fornecido e podem precisar de validação adicional dependendo do contexto específico da despesa."
        
        return formatted_response
            
    def process_message(self, message: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Processa uma mensagem do usuário com classificação de natureza de despesa.
        Fluxo invertido: primeiro analisa com o classificador de natureza, depois consulta o modelo geral.
        
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
            # Verifica se a consulta parece ser sobre natureza de despesa
            is_natureza_query = self.natureza_handler.integrator.is_natureza_query(message)
            
            if is_natureza_query:
                # 1. Primeira etapa: Consulta o classificador especializado
                natureza_result = self.natureza_handler.integrator.process_query(message)
                
                # Se temos previsões, incorporamos essa informação na consulta ao modelo geral
                if natureza_result['predictions']:
                    # Prepara a consulta enriquecida para o modelo geral
                    enhanced_query = self._prepare_enhanced_query(message, natureza_result)
                    
                    # 2. Segunda etapa: Consulta o modelo geral com a consulta enriquecida
                    response = super().process_message(enhanced_query, chat_history)
                    
                    # Verifica se a resposta já contém a seção formatada
                    if '---' not in response and '**Classificação de Natureza de Despesa:**' not in response:
                        # Se não tiver, adiciona a formatação manualmente
                        response = self._format_response_with_natureza(response, natureza_result)
                    
                    return response
            
            # Se não for uma consulta de natureza ou não houver previsões,
            # processa normalmente com o modelo geral
            return super().process_message(message, chat_history)
            
        except Exception as e:
            self.logger.error(f"Erro ao processar mensagem aprimorada: {str(e)}")
            return (
                "Desculpe, ocorreu um erro ao processar sua pergunta. "
                f"Detalhes do erro: {str(e)}"
            )
    
    def _prepare_enhanced_query(self, original_query: str, natureza_result: Dict[str, Any]) -> str:
        """
        Prepara uma consulta enriquecida com os resultados do classificador de natureza.
        
        Args:
            original_query: Consulta original do usuário
            natureza_result: Resultado do processamento de natureza
            
        Returns:
            str: Consulta enriquecida para enviar ao modelo geral
        """
        predictions = natureza_result['predictions']
        recommended = natureza_result.get('recommended', predictions[0] if predictions else None)
        
        enhanced_query = (
            f"O usuário perguntou: '{original_query}'\n\n"
            f"Após análise com o classificador especializado, encontrei as seguintes possíveis "
            f"naturezas de despesa:\n\n"
        )
        
        # Adiciona a natureza recomendada
        if recommended:
            enhanced_query += (
                f"Natureza mais provável: {recommended['codigo']} - {recommended['nome']}\n"
                f"Confiança: {recommended['confianca']:.2%}\n"
                f"Referência: {recommended['texto_referencia']}\n\n"
            )
        
        # Adiciona outras naturezas sugeridas
        if len(predictions) > 1:
            enhanced_query += "Outras naturezas possíveis:\n"
            for i, pred in enumerate(predictions[1:], 1):
                enhanced_query += (
                    f"{i}. {pred['codigo']} - {pred['nome']} "
                    f"(Confiança: {pred['confianca']:.2%})\n"
                )
            enhanced_query += "\n"
        
        enhanced_query += (
            "Por favor, responda à consulta do usuário sobre natureza de despesa, "
            "levando em consideração estas informações do classificador. "
            "Explique qual é a natureza mais apropriada e por quê, ou sugira uma alternativa "
            "se nenhuma das naturezas identificadas for adequada.\n\n"
            "IMPORTANTE: Formate sua resposta de modo que primeiro venha sua explicação normal, "
            "e somente no final adicione um separador '---' seguido por uma seção intitulada "
            "'**Classificação de Natureza de Despesa:**' contendo suas recomendações. "
            "Essa formatação deve ser idêntica ao formato usado anteriormente pelo sistema."
        )
        
        return enhanced_query