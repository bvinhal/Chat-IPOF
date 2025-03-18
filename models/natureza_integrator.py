# models/natureza_integrator.py

import os
import logging
from typing import Dict, List, Any, Optional
from config import active_config
from models.natureza_classifier import NaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaIntegrator:
    """
    Integrador para combinar o classificador de natureza com o sistema de chat existente.
    """
    
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa o integrador.
        
        Args:
            embedding_provider: Provedor de embeddings a usar ('openai', 'claude', 'gemini')
        """
        self.classifier = NaturezaClassifier(embedding_provider)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tenta carregar o modelo classificador
        self._ensure_classifier_loaded()
    
    def _ensure_classifier_loaded(self) -> bool:
        """
        Garante que o classificador esteja carregado.
        
        Returns:
            bool: True se o classificador está pronto para uso
        """
        if not self.classifier.is_trained:
            return self.classifier.load_model()
        return True
    
    def is_natureza_query(self, text: str) -> bool:
        """
        Verifica se a consulta é relacionada a natureza de despesa.
        
        Args:
            text: Texto da consulta
            
        Returns:
            bool: True se a consulta parece ser sobre natureza de despesa
        """
        text_lower = text.lower()
        
        # Verificar primeiro se é uma pergunta definitória/conceitual
        # Perguntas de definição não precisam de classificação de natureza
        definition_patterns = [
            "o que é", "o que são", "defina", "definição de", "conceito de",
            "explique o que é", "significado de", "pode me explicar o que é"
        ]
        
        for pattern in definition_patterns:
            if pattern in text_lower:
                # É uma pergunta de definição, não uma solicitação de classificação
                return False
        
        # Combinações específicas que indicam consulta sobre classificação de natureza
        specific_patterns = [
            "natureza de despesa", "elemento de despesa", "classificar despesa",
            "qual natureza", "qual a natureza", "qual o código da natureza",
            "classificação de natureza", "classificação da despesa",
            "código de natureza", "que natureza", "classificar como"
        ]
        
        for pattern in specific_patterns:
            if pattern in text_lower:
                return True
        
        # Palavras individuais com peso maior
        strong_keywords = ["natureza", "classificação", "classificar", "elemento de despesa", "rubrica"]
        for keyword in strong_keywords:
            if keyword.lower() in text_lower:
                # Procura por palavras de contexto que indicam classificação
                context_words = ["qual", "adequada", "correta", "apropriada", "sugerir", "indicar", "para", "dessa", "desta", "do"]
                for context in context_words:
                    if context in text_lower:
                        return True
        
        # Palavras-chave que sozinhas são insuficientes, mas podem indicar em conjunto
        weak_keywords = ["despesa", "orçamento", "empenho", "dotação", "orçamentária"]
        keyword_count = 0
        
        for keyword in weak_keywords:
            if keyword.lower() in text_lower:
                keyword_count += 1
        
        # Se duas ou mais palavras frágeis estiverem presentes junto com indicadores de pergunta de classificação,
        # consideramos que é uma consulta sobre natureza
        if keyword_count >= 2:
            classification_indicators = ["qual", "como", "classific", "adequado", "correto", "apropriado"]
            for indicator in classification_indicators:
                if indicator in text_lower:
                    return True
        
        return False
    
    def process_query(self, text: str, ai_model=None) -> Dict[str, Any]:
        """
        Processa uma consulta sobre natureza de despesa.
        
        Args:
            text: Texto da consulta
            ai_model: Modelo de IA atual para validação (opcional)
            
        Returns:
            Dict[str, Any]: Resultado do processamento contendo:
                - is_natureza_query: Se a consulta é sobre natureza
                - predictions: Previsões do classificador
                - validation: Validação do modelo de IA (se disponível)
                - recommended: Natureza recomendada
        """
        result = {
            'is_natureza_query': self.is_natureza_query(text),
            'predictions': [],
            'validation': None,
            'recommended': None
        }
        
        # Se não parece ser uma consulta sobre natureza, retorna resultado vazio
        if not result['is_natureza_query']:
            return result
        
        # Verifica se o classificador está carregado
        if not self._ensure_classifier_loaded():
            self.logger.error("Classificador não está disponível")
            return result
        
        try:
            # Obtém previsões do classificador
            predictions = self.classifier.predict(text, top_k=3)
            result['predictions'] = predictions
            
            if not predictions:
                return result
            
            # A natureza mais provável é a primeira previsão
            most_likely = predictions[0]
            result['recommended'] = most_likely
            
            # Se um modelo de IA foi fornecido, solicita validação
            if ai_model and hasattr(ai_model, 'generate_response'):
                validation_query = self._create_validation_query(text, predictions)
                validation_response = ai_model.generate_response(validation_query)
                
                result['validation'] = {
                    'query': validation_query,
                    'response': validation_response
                }
                
                # Tenta extrair uma recomendação específica da resposta
                recommended_code = self._extract_recommendation(validation_response, [p['codigo'] for p in predictions])
                if recommended_code:
                    # Encontra a previsão correspondente ao código recomendado
                    for pred in predictions:
                        if pred['codigo'] == recommended_code:
                            result['recommended'] = pred
                            break
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao processar consulta: {str(e)}")
            return result
    
    def _create_validation_query(self, original_query: str, predictions: List[Dict[str, Any]]) -> str:
        """
        Cria uma consulta para validação com o modelo de IA.
        
        Args:
            original_query: Consulta original do usuário
            predictions: Previsões do classificador
            
        Returns:
            str: Consulta para validação
        """
        query = (
            f"Com base na consulta do usuário: '{original_query}', "
            f"o sistema identificou as seguintes possíveis naturezas de despesa:\n\n"
        )
        
        for i, pred in enumerate(predictions, 1):
            query += (
                f"{i}. Código: {pred['codigo']} - {pred['nome']}\n"
                f"   Confiança: {pred['confianca']:.2%}\n"
                f"   Baseado em: '{pred['texto_referencia']}'\n\n"
            )
        
        query += (
            "Por favor, analise estas opções e determine qual é a classificação de natureza de despesa "
            "mais adequada para a consulta do usuário. Se nenhuma das opções for adequada, "
            "explique o motivo e sugira a classificação correta. "
            "Ao final, indique claramente o código da natureza mais adequada."
        )
        
        return query
    
    def _extract_recommendation(self, validation_response: str, candidate_codes: List[str]) -> Optional[str]:
        """
        Tenta extrair um código de natureza recomendado da resposta de validação.
        
        Args:
            validation_response: Resposta de validação do modelo
            candidate_codes: Códigos candidatos a buscar
            
        Returns:
            Optional[str]: Código recomendado ou None se não identificado
        """
        # Primeiro procura por padrões explícitos como "código recomendado: XXXXX"
        response_lower = validation_response.lower()
        
        explicit_patterns = [
            "código recomendado", "código sugerido", "código mais adequado",
            "classificação adequada", "natureza recomendada", "recomendo o código",
            "sugiro o código", "código correto", "melhor classificação"
        ]
        
        # Verifica se algum dos padrões explícitos está presente
        for pattern in explicit_patterns:
            if pattern in response_lower:
                # Busca a posição do padrão e procura por códigos candidatos nas proximidades
                pattern_pos = response_lower.find(pattern)
                context = response_lower[pattern_pos:pattern_pos + 100]  # Busca nos próximos 100 caracteres
                
                for code in candidate_codes:
                    if code in context:
                        return code
        
        # Se não encontrou com padrões explícitos, busca qualquer ocorrência dos códigos candidatos
        for code in candidate_codes:
            if code in validation_response:
                return code
        
        return None