"""
Classificador de Regressão Logística Multinomial para Natureza de Despesa

Implementação conforme descrito em Materiais e Métodos (Faceli et al., 2011).
A Regressão Logística multinomial representa o modelo probabilístico linear
mais utilizado em classificação de texto.

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
Referência: Faceli et al. (2011), Monard e Baranauskas (2003)
"""

import logging
import numpy as np
from sklearn.linear_model import LogisticRegression
from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LogisticRegressionNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando Regressão Logística Multinomial.
    
    Conforme metodologia, baseada na função softmax, estima a probabilidade
    de cada classe através de uma combinação linear das características,
    transformada pela função logística.
    
    Para k classes, o modelo calcula:
    P(y=k|x) = exp(w_k^T x) / Σ_j exp(w_j^T x)
    
    onde w_k são os pesos aprendidos para a classe k.
    
    Vantagens (Faceli et al., 2011):
    - Simplicidade teórica
    - Máxima interpretabilidade
    - Baseline essencial em trabalhos de classificação
    - Coeficientes diretamente interpretáveis
    - Probabilidades calibradas para quantificar incerteza
    - Convergência rápida mesmo para multiclasse de alta dimensionalidade
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o classificador de Regressão Logística.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        super().__init__(model_name='logistic_regression', project_root=project_root)
        logger.info("Logistic Regression Classifier inicializado")
    
    def build_model(self, C: float = 1.0, penalty: str = 'l2',
                   solver: str = 'lbfgs', max_iter: int = 1000,
                   multi_class: str = 'multinomial', **kwargs):
        """
        Constrói o modelo de Regressão Logística Multinomial.
        
        Conforme metodologia, a regularização L2 previne overfitting,
        controlando a magnitude dos pesos.
        
        Args:
            C: Inverso da força de regularização (padrão: 1.0)
               - Valores menores = regularização mais forte
            penalty: Norma da penalidade ('l1', 'l2', 'elasticnet', 'none')
            solver: Algoritmo de otimização
                   - 'lbfgs': Ótimo para multiclasse (padrão)
                   - 'saga': Suporta todas as penalidades
            max_iter: Número máximo de iterações (padrão: 1000)
            multi_class: Estratégia multiclasse
                        - 'multinomial': Softmax (recomendado)
                        - 'ovr': One-vs-Rest
            **kwargs: Outros parâmetros do LogisticRegression
        """
        logger.info(
            f"Construindo modelo Logistic Regression "
            f"(C={C}, penalty={penalty}, solver={solver}, "
            f"multi_class={multi_class})"
        )
        
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            random_state=42,
            n_jobs=-1,  # Usar todos os cores
            verbose=0,
            **kwargs
        )
        
        logger.info("Modelo Logistic Regression construído com sucesso")
    
    def get_feature_coefficients(self, class_index: int, top_n: int = 20):
        """
        Retorna os coeficientes das features para uma classe específica.
        
        Conforme metodologia, os coeficientes são diretamente interpretáveis,
        permitindo identificar quais termos aumentam ou diminuem a probabilidade
        de cada natureza de despesa.
        
        Args:
            class_index: Índice da classe
            top_n: Número de top features positivas e negativas
            
        Returns:
            Tupla (top_positivos, top_negativos)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Modelo não possui coeficientes")
        
        # Obter coeficientes para a classe
        coefficients = self.model.coef_[class_index]
        
        # Obter nomes das features
        feature_names = self.pipeline.vectorizer.get_feature_names_out()
        
        # Criar lista de tuplas (feature, coeficiente)
        feature_coef = list(zip(feature_names, coefficients))
        
        # Ordenar por coeficiente
        feature_coef.sort(key=lambda x: x[1], reverse=True)
        
        # Top positivos e negativos
        top_positive = feature_coef[:top_n]
        top_negative = feature_coef[-top_n:][::-1]  # Reverter para ordem decrescente
        
        return top_positive, top_negative
    
    def print_feature_coefficients(self, class_name: str, top_n: int = 10):
        """
        Imprime os coeficientes das features para uma classe.
        
        Args:
            class_name: Nome da classe (natureza)
            top_n: Número de features a mostrar
        """
        # Encontrar índice da classe
        try:
            class_index = np.where(self.pipeline.label_encoder.classes_ == class_name)[0][0]
        except IndexError:
            logger.error(f"Classe '{class_name}' não encontrada")
            return
        
        logger.info(f"\nCoeficientes para classe: {class_name}")
        logger.info("="*70)
        
        top_pos, top_neg = self.get_feature_coefficients(class_index, top_n)
        
        logger.info(f"\nTop {top_n} features com coeficientes POSITIVOS:")
        logger.info("(aumentam a probabilidade da classe)")
        for i, (feature, coef) in enumerate(top_pos, 1):
            logger.info(f"{i:2d}. {feature:30s} + {coef:8.4f}")
        
        logger.info(f"\nTop {top_n} features com coeficientes NEGATIVOS:")
        logger.info("(diminuem a probabilidade da classe)")
        for i, (feature, coef) in enumerate(top_neg, 1):
            logger.info(f"{i:2d}. {feature:30s} - {abs(coef):8.4f}")
        
        logger.info("="*70)
    
    def analyze_prediction(self, text: str, top_k: int = 5):
        """
        Analisa uma predição mostrando as probabilidades das classes.
        
        Conforme metodologia, oferece probabilidades calibradas,
        úteis para quantificar a incerteza das predições.
        
        Args:
            text: Texto a analisar
            top_k: Número de top classes a mostrar
        """
        # Fazer predição
        natureza, conf = self.predict(text)
        top_k_preds = self.predict_top_k(text, k=top_k)
        
        logger.info(f"\nAnálise de Predição:")
        logger.info("="*70)
        logger.info(f"Texto: {text}")
        logger.info(f"\nPredição Principal: {natureza} (probabilidade: {conf:.4f})")
        logger.info(f"\nTop-{top_k} Classes Mais Prováveis:")
        
        for i, (nat, prob) in enumerate(top_k_preds, 1):
            bar_length = int(prob * 50)
            bar = '█' * bar_length + '░' * (50 - bar_length)
            logger.info(f"{i}. {nat:20s} {bar} {prob:.4f}")
        
        logger.info("="*70)


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: Logistic Regression Classifier")
    print("="*80)
    
    # 1. Criar classificador
    classifier = LogisticRegressionNaturezaClassifier()
    
    # 2. Pré-processar dados
    print("\n[1/5] Pré-processando dados...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. Treinar modelo
    print("\n[2/5] Treinando modelo Logistic Regression...")
    classifier.train(
        C=1.0,
        penalty='l2',
        solver='lbfgs',
        max_iter=1000,
        multi_class='multinomial'
    )
    
    # 4. Avaliar
    print("\n[3/5] Avaliando modelo...")
    test_metrics = classifier.evaluate(dataset='test')
    
    # 5. Analisar coeficientes (para uma classe específica)
    print("\n[4/5] Analisando coeficientes de features...")
    # Pegar uma classe do conjunto de dados
    sample_class = classifier.pipeline.label_encoder.classes_[0]
    classifier.print_feature_coefficients(sample_class, top_n=5)
    
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
    
    # Teste de predição com análise
    print("\n" + "="*80)
    print("TESTE DE PREDIÇÃO COM ANÁLISE DE PROBABILIDADES")
    print("="*80)
    texto_teste = "Contratação de serviços de consultoria especializada"
    classifier.analyze_prediction(texto_teste, top_k=5)
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)
    print("\n💡 DICA: A Regressão Logística é o baseline ideal para validar")
    print("   se a complexidade de outros modelos realmente se justifica!")