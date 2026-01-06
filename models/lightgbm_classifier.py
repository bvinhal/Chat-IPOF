"""
Classificador LightGBM para Natureza de Despesa

Implementação do LightGBM conforme descrito em Materiais e Métodos (Ke et al., 2017).
O LightGBM representa a evolução dos métodos ensemble baseados em gradient boosting,
utilizando GOSS e EFB para maior eficiência.

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
Referência: Ke et al. (2017) - LightGBM: A highly efficient gradient boosting decision tree
"""

import logging
import numpy as np
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM não instalado. Execute: pip install lightgbm")

from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LightGBMNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando LightGBM.
    
    Conforme metodologia, o LightGBM constrói sequencialmente árvores de decisão,
    onde cada nova árvore é treinada para corrigir os erros das anteriores,
    minimizando uma função de perda por gradiente descendente.
    
    Inovações técnicas (Ke et al., 2017):
    - GOSS (Gradient-based One-Side Sampling): Reduz exemplos mantendo gradientes maiores
    - EFB (Exclusive Feature Bundling): Agrupa características mutuamente exclusivas
    
    Vantagens:
    - Velocidade de treinamento significativamente superior
    - Suporte nativo a classificação multiclasse
    - Fornece importância de características
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o classificador LightGBM.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError(
                "LightGBM não está instalado. "
                "Instale com: pip install lightgbm"
            )
        
        super().__init__(model_name='lightgbm', project_root=project_root)
        logger.info("LightGBM Classifier inicializado")
    
    def build_model(self, num_leaves: int = 31, learning_rate: float = 0.1,
                   n_estimators: int = 100, max_depth: int = -1,
                   min_child_samples: int = 20, subsample: float = 1.0,
                   colsample_bytree: float = 1.0, reg_alpha: float = 0.0,
                   reg_lambda: float = 0.0, **kwargs):
        """
        Constrói o modelo LightGBM.
        
        Parâmetros conforme Ke et al. (2017):
        
        Args:
            num_leaves: Número máximo de folhas em uma árvore (padrão: 31)
            learning_rate: Taxa de aprendizado (padrão: 0.1)
            n_estimators: Número de árvores de boosting (padrão: 100)
            max_depth: Profundidade máxima da árvore (-1 = sem limite)
            min_child_samples: Número mínimo de amostras em um nó folha
            subsample: Fração de amostras para treinar cada árvore
            colsample_bytree: Fração de features para treinar cada árvore
            reg_alpha: Regularização L1
            reg_lambda: Regularização L2
            **kwargs: Outros parâmetros do LGBMClassifier
        """
        logger.info(
            f"Construindo modelo LightGBM "
            f"(n_estimators={n_estimators}, num_leaves={num_leaves}, "
            f"learning_rate={learning_rate})"
        )
        
        self.model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1,  # Usar todos os cores
            importance_type='gain',  # Para análise de importância
            verbose=-1,  # Silencioso
            **kwargs
        )
        
        logger.info("Modelo LightGBM construído com sucesso")
    
    def get_feature_importance(self, top_n: int = 20):
        """
        Retorna a importância das features.
        
        Conforme metodologia, o LightGBM fornece importância de características,
        identificando quais termos são mais relevantes.
        
        Args:
            top_n: Número de top features a retornar
            
        Returns:
            Lista de tuplas (feature, importância)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Modelo não possui feature_importances_")
        
        # Obter importâncias
        importances = self.model.feature_importances_
        
        # Obter nomes das features
        feature_names = self.pipeline.vectorizer.get_feature_names_out()
        
        # Criar lista de tuplas (feature, importância)
        feature_importance = list(zip(feature_names, importances))
        
        # Ordenar por importância
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        return feature_importance[:top_n]
    
    def print_feature_importance(self, top_n: int = 20):
        """
        Imprime a importância das features.
        
        Args:
            top_n: Número de top features a mostrar
        """
        logger.info(f"\nTop {top_n} Features Mais Importantes:")
        logger.info("="*60)
        
        feature_importance = self.get_feature_importance(top_n)
        
        for i, (feature, importance) in enumerate(feature_importance, 1):
            logger.info(f"{i:2d}. {feature:30s} - {importance:.4f}")
        
        logger.info("="*60)


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: LightGBM Classifier")
    print("="*80)
    
    if not LIGHTGBM_AVAILABLE:
        print("\n❌ LightGBM não está instalado!")
        print("Instale com: pip install lightgbm")
        exit(1)
    
    # 1. Criar classificador
    classifier = LightGBMNaturezaClassifier()
    
    # 2. Pré-processar dados
    print("\n[1/5] Pré-processando dados...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. Treinar modelo
    print("\n[2/5] Treinando modelo LightGBM...")
    classifier.train(
        num_leaves=31,
        learning_rate=0.1,
        n_estimators=100,
        max_depth=-1
    )
    
    # 4. Avaliar
    print("\n[3/5] Avaliando modelo...")
    test_metrics = classifier.evaluate(dataset='test')
    
    # 5. Importância de features
    print("\n[4/5] Analisando importância de features...")
    classifier.print_feature_importance(top_n=10)
    
    # 6. Salvar
    print("\n[5/5] Salvando modelo...")
    classifier.save_model()
    
    # Informações do modelo
    print("\n" + "="*80)
    print("INFORMAÇÕES DO MODELO")
    print("="*80)
    info = classifier.get_model_info()
    print(f"Modelo: {info['model_type']}")
    print(f"Nome: {info['model_name']}")
    print(f"Amostras de treino: {info['data_info']['n_samples_train']}")
    print(f"Features: {info['data_info']['n_features']}")
    print(f"Classes: {info['data_info']['n_classes']}")
    print(f"\nMétrica Principal (Macro F1-Score): {test_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    if 'top_3_accuracy' in test_metrics:
        print(f"Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
    
    # Teste de predição
    print("\n" + "="*80)
    print("TESTE DE PREDIÇÃO")
    print("="*80)
    texto_teste = "Aquisição de materiais de escritório para a secretaria"
    print(f"Texto: {texto_teste}")
    
    natureza, conf = classifier.predict(texto_teste)
    print(f"\nPredição: {natureza}")
    print(f"Confiança: {conf:.4f}")
    
    print("\nTop-3 predições:")
    top_3 = classifier.predict_top_k(texto_teste, k=3)
    for i, (nat, prob) in enumerate(top_3, 1):
        print(f"  {i}. {nat}: {prob:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)