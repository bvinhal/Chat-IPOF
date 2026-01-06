from models.claude_model import ClaudeModel
from models.openai_model import OpenAIModel
from models.gemini_model import GeminiModel
from models.ai_model import AIModel

from .base_classifier import BaseNaturezaClassifier
from .svm_classifier import SVMNaturezaClassifier
from .logistic_classifier import LogisticRegressionNaturezaClassifier
from .similarity_classifier import SimilarityNaturezaClassifier
from .kmeans_classifier import KMeansNaturezaClassifier

    
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

# LightGBM (opcional)
try:
    from .lightgbm_classifier import LightGBMNaturezaClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LightGBMNaturezaClassifier = None

# MLP + S-BERT (opcional)
try:
    from .mlp_sbert_classifier import MLPSBERTNaturezaClassifier
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    MLPSBERTNaturezaClassifier = None

def get_available_models():
    """
    Retorna lista de modelos disponíveis.
    
    Returns:
        Lista de tuplas (nome, classe)
    """
    models = [
        ('SVM Linear', SVMNaturezaClassifier),
        ('Regressão Logística', LogisticRegressionNaturezaClassifier),
        ('Similaridade', SimilarityNaturezaClassifier),
        ('K-Means', KMeansNaturezaClassifier),
    ]
    
    if LIGHTGBM_AVAILABLE:
        models.append(('LightGBM', LightGBMNaturezaClassifier))
    
    if SBERT_AVAILABLE:
        models.append(('MLP + S-BERT', MLPSBERTNaturezaClassifier))
    
    return models


def create_model(model_name: str, project_root: str = None):
    """
    Factory method para criar modelos.
    
    Args:
        model_name: Nome do modelo ('svm', 'lightgbm', 'logistic', etc.)
        project_root: Diretório raiz do projeto
        
    Returns:
        Instância do modelo
        
    Raises:
        ValueError: Se o modelo não for encontrado
    """
    models_map = {
        'svm': SVMNaturezaClassifier,
        'svm_linear': SVMNaturezaClassifier,
        'logistic': LogisticRegressionNaturezaClassifier,
        'logistic_regression': LogisticRegressionNaturezaClassifier,
        'similarity': SimilarityNaturezaClassifier,
        'kmeans': KMeansNaturezaClassifier,
        'k_means': KMeansNaturezaClassifier,
    }
    
    if LIGHTGBM_AVAILABLE:
        models_map['lightgbm'] = LightGBMNaturezaClassifier
        models_map['lgbm'] = LightGBMNaturezaClassifier
    
    if SBERT_AVAILABLE:
        models_map['mlp_sbert'] = MLPSBERTNaturezaClassifier
        models_map['sbert'] = MLPSBERTNaturezaClassifier
    
    model_name = model_name.lower()
    
    if model_name not in models_map:
        raise ValueError(
            f"Modelo '{model_name}' não encontrado. "
            f"Modelos disponíveis: {list(models_map.keys())}"
        )
    
    return models_map[model_name](project_root=project_root)


# Definir o que será exportado quando fizer "from models import *"
__all__ = [
    'BaseNaturezaClassifier',
    'SVMNaturezaClassifier',
    'LogisticRegressionNaturezaClassifier',
    'SimilarityNaturezaClassifier',
    'KMeansNaturezaClassifier',
    'get_available_models',
    'create_model',
]

if LIGHTGBM_AVAILABLE:
    __all__.append('LightGBMNaturezaClassifier')

if SBERT_AVAILABLE:
    __all__.append('MLPSBERTNaturezaClassifier')