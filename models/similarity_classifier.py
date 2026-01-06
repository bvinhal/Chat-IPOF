"""
Classificador por Similaridade com TF-IDF e Cosine Similarity

Implementação conforme descrito em Materiais e Métodos (Lorena e Carvalho, 2007).
Este método representa o aprendizado baseado em instâncias, onde a classificação
se baseia na similaridade com exemplos já conhecidos.

A escolha deste modelo tem uma razão estratégica importante: permite quantificar
a contribuição específica do LLM no método RAG (Lewis et al., 2020).

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
Referências: Lorena e Carvalho (2007), Faceli et al. (2011), Lewis et al. (2020)
"""

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimilarityNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando Busca por Similaridade.
    
    Conforme metodologia, este método representa o aprendizado baseado em instâncias.
    A abordagem não envolve treinamento no sentido tradicional, mas sim
    armazena os dados de treino e realiza buscas por similaridade no momento
    da classificação.
    
    Para cada nova descrição:
    1. Vetoriza usando TF-IDF
    2. Calcula similaridade de cosseno com todas as descrições de treino
    3. Retorna a natureza da descrição mais similar
    
    Razão Estratégica (Metodologia):
    O método RAG (Retrieval-Augmented Generation) combina recuperação de informações
    e geração de resposta por LLM. A busca por similaridade isola apenas o componente
    de recuperação, permitindo quantificar: quanto valor a geração de linguagem adiciona
    além da simples recuperação por similaridade?
    
    Vantagens:
    - Não requer treinamento (lazy learning)
    - Interpretável (pode mostrar exemplos similares)
    - Baseline para métodos RAG
    - Funciona bem quando há exemplos muito similares
    
    Desvantagens:
    - Lento em tempo de predição (O(n) para n exemplos)
    - Memória proporcional ao tamanho do conjunto de treino
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o classificador por similaridade.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        super().__init__(model_name='similarity', project_root=project_root)
        
        # Armazena dados de treino
        self.X_train_vec = None
        self.y_train = None
        
        # Modelo "dummy" para compatibilidade com BaseClassifier
        self.model = self  # Self-reference para indicar "treinado"
        
        logger.info("Similarity Classifier inicializado")
    
    def build_model(self, **kwargs):
        """
        Não há modelo a construir para similaridade.
        
        Este método é implementado para manter compatibilidade com
        a interface BaseNaturezaClassifier.
        """
        logger.info("Similarity Classifier não requer construção de modelo")
        self.model = self  # Self-reference
    
    def train(self, **kwargs):
        """
        "Treina" o modelo armazenando os dados de treino.
        
        Conforme metodologia, este é um método lazy learning - não há
        treinamento tradicional, apenas armazenamento dos exemplos.
        """
        if self.data is None:
            raise ValueError("Dados não processados. Execute preprocess_data() primeiro.")
        
        logger.info("Armazenando dados de treino para Similarity Classifier...")
        
        import time
        start_time = time.time()
        
        # Armazenar dados de treino
        self.X_train_vec = self.data['X_train_vec']
        self.y_train = self.data['y_train_encoded']
        
        training_time = time.time() - start_time
        logger.info(f"Dados armazenados em {training_time:.4f}s")
        logger.info(f"Total de exemplos: {len(self.y_train)}")
        
        self.metrics['training_time'] = training_time
        self.model = self  # Marcar como "treinado"
    
    def predict(self, text: str, return_similarity: bool = False) -> Tuple:
        """
        Faz predição baseada em similaridade de cosseno.
        
        Args:
            text: Descrição da despesa
            return_similarity: Se True, retorna também o score de similaridade
            
        Returns:
            Tupla (natureza prevista, confiança/similaridade)
        """
        if self.X_train_vec is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        # Pré-processar e vetorizar texto
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Calcular similaridade com todos os exemplos de treino
        similarities = cosine_similarity(text_vec, self.X_train_vec)[0]
        
        # Encontrar exemplo mais similar
        most_similar_idx = np.argmax(similarities)
        similarity_score = similarities[most_similar_idx]
        
        # Obter natureza correspondente
        pred_encoded = self.y_train[most_similar_idx]
        natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
        
        if return_similarity:
            return natureza, similarity_score, most_similar_idx
        
        return natureza, similarity_score
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retorna as top-k predições mais similares.
        
        Args:
            text: Descrição da despesa
            k: Número de predições a retornar
            
        Returns:
            Lista de tuplas (natureza, similaridade)
        """
        if self.X_train_vec is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar e vetorizar
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Calcular similaridades
        similarities = cosine_similarity(text_vec, self.X_train_vec)[0]
        
        # Top-k índices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Obter naturezas e similaridades
        results = []
        for idx in top_k_indices:
            pred_encoded = self.y_train[idx]
            natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
            similarity = similarities[idx]
            results.append((natureza, similarity))
        
        return results
    
    def find_similar_examples(self, text: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """
        Encontra os k exemplos mais similares no conjunto de treino.
        
        Útil para interpretabilidade: mostra quais exemplos o modelo usou
        para fazer a decisão.
        
        Args:
            text: Descrição da despesa
            k: Número de exemplos a retornar
            
        Returns:
            Lista de tuplas (texto_similar, natureza, similaridade)
        """
        if self.X_train_vec is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar e vetorizar
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Calcular similaridades
        similarities = cosine_similarity(text_vec, self.X_train_vec)[0]
        
        # Top-k índices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Obter exemplos
        X_train_raw = self.data['X_train_raw']
        results = []
        
        for idx in top_k_indices:
            texto_similar = X_train_raw[idx]
            pred_encoded = self.y_train[idx]
            natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
            similarity = similarities[idx]
            results.append((texto_similar, natureza, similarity))
        
        return results
    
    def analyze_prediction_with_examples(self, text: str, k: int = 3):
        """
        Analisa uma predição mostrando exemplos similares.
        
        Args:
            text: Texto a analisar
            k: Número de exemplos similares a mostrar
        """
        logger.info("\nAnálise de Predição com Exemplos Similares:")
        logger.info("="*80)
        logger.info(f"Texto: {text}")
        
        # Predição
        natureza, similarity, idx = self.predict(text, return_similarity=True)
        logger.info(f"\nPredição: {natureza}")
        logger.info(f"Similaridade: {similarity:.4f}")
        
        # Exemplos similares
        similar_examples = self.find_similar_examples(text, k=k)
        
        logger.info(f"\n{k} Exemplos Mais Similares:")
        logger.info("-"*80)
        
        for i, (texto_sim, nat_sim, sim) in enumerate(similar_examples, 1):
            logger.info(f"\n{i}. Similaridade: {sim:.4f}")
            logger.info(f"   Natureza: {nat_sim}")
            logger.info(f"   Texto: {texto_sim[:100]}...")
        
        logger.info("="*80)


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: Similarity Classifier")
    print("="*80)
    
    # 1. Criar classificador
    classifier = SimilarityNaturezaClassifier()
    
    # 2. Pré-processar dados
    print("\n[1/4] Pré-processando dados...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. "Treinar" (armazenar dados)
    print("\n[2/4] Armazenando dados de treino...")
    classifier.train()
    
    # 4. Avaliar
    print("\n[3/4] Avaliando modelo...")
    print("⚠️  Aviso: A predição pode ser mais lenta (O(n) por predição)")
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
    print(f"Exemplos armazenados: {info['data_info']['n_samples_train']}")
    print(f"Features: {info['data_info']['n_features']}")
    print(f"Classes: {info['data_info']['n_classes']}")
    print(f"\nMétrica Principal (Macro F1-Score): {test_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Teste de predição com análise
    print("\n" + "="*80)
    print("TESTE DE PREDIÇÃO COM EXEMPLOS SIMILARES")
    print("="*80)
    texto_teste = "Aquisição de equipamentos de informática"
    classifier.analyze_prediction_with_examples(texto_teste, k=3)
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)
    print("\n💡 DICA: Este modelo é ideal para entender o baseline de")
    print("   recuperação por similaridade antes de adicionar LLMs (RAG)!")