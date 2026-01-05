"""
Classificador SVM Linear para Natureza de Despesa

Implementação do modelo SVM Linear conforme descrito em Materiais e Métodos.
O SVM representa a abordagem geométrica de classificação baseada em margem máxima.

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import logging
from sklearn.svm import LinearSVC
from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SVMNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando SVM Linear.
    
    Conforme metodologia descrita, o SVM linear foi considerado estado-da-arte
    em classificação de texto antes do advento do deep learning.
    
    Utiliza estratégia One-vs-Rest (OvR) para lidar com múltiplas classes.
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o classificador SVM.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        super().__init__(model_name='svm_linear', project_root=project_root)
        logger.info("SVM Linear Classifier inicializado")
    
    def build_model(self, C: float = 1.0, max_iter: int = 1000, **kwargs):
        """
        Constrói o modelo SVM Linear.
        
        Args:
            C: Parâmetro de regularização (padrão: 1.0)
            max_iter: Número máximo de iterações (padrão: 1000)
            **kwargs: Outros parâmetros do LinearSVC
        """
        logger.info(f"Construindo modelo SVM (C={C}, max_iter={max_iter})")
        
        self.model = LinearSVC(
            C=C,
            max_iter=max_iter,
            random_state=42,
            dual=False,  # dual=False é mais eficiente para n_samples > n_features
            **kwargs
        )
        
        logger.info("Modelo SVM construído com sucesso")


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: SVM Linear Classifier")
    print("="*80)
    
    # 1. Criar classificador
    classifier = SVMNaturezaClassifier()
    
    # 2. Pré-processar dados
    print("\n[1/4] Pré-processando dados...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. Treinar modelo
    print("\n[2/4] Treinando modelo...")
    classifier.train(C=1.0, max_iter=1000)
    
    # 4. Avaliar
    print("\n[3/4] Avaliando modelo...")
    test_metrics = classifier.evaluate(dataset='test')
    
    # 5. Salvar
    print("\n[4/4] Salvando modelo...")
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