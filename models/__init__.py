from models.claude_model import ClaudeModel
from models.openai_model import OpenAIModel
from models.gemini_model import GeminiModel
from models.ai_model import AIModel

# Dicionário para mapear nomes de modelos para suas classes
MODEL_TYPES = {
    'claude': ClaudeModel,
    'openai': OpenAIModel,
    'gemini': GeminiModel
}

def get_model_class(model_type: str) -> type:
    """
    Retorna a classe do modelo correspondente ao tipo especificado.
    
    Args:
        model_type: Tipo do modelo ('claude', 'openai', 'gemini')
        
    Returns:
        type: Classe do modelo solicitado ou None se não existir
    """
    return MODEL_TYPES.get(model_type.lower())
