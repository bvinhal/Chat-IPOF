"""
Classificador K-Means Clustering para Natureza de Despesa

Implementação conforme descrito em Materiais e Métodos (Faceli et al., 2011).
O K-Means é o algoritmo de clustering particional mais conhecido e utilizado,
agrupando observações em k clusters minimizando a soma das distâncias quadráticas.

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
Referência: Faceli et al. (2011)
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple
from collections import Counter
from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KMeansNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando K-Means Clustering.
    
    Conforme metodologia, o algoritmo agrupa as observações em k clusters,
    minimizando: Σ_i Σ_(x∈C_i) ||x - μ_i||^2
    onde μ_i é o centróide do cluster C_i.
    
    O procedimento é iterativo, alternando entre:
    1. Atribuir cada ponto ao centróide mais próximo
    2. Recalcular os centróides como a média dos pontos atribuídos
    
    Objetivos conforme metodologia:
    1. Descobrir se existem agrupamentos naturais nas naturezas de despesa
    2. Validar se as 1.600 naturezas são todas necessárias ou se há redundâncias
    3. Representar aprendizado não supervisionado iterativo
    
    A análise de pureza dos clusters (proporção da classe majoritária em cada cluster)
    e do índice silhueta (qualidade da clusterização) fornece insights sobre a
    estrutura dos dados.
    
    Os centróides podem ser interpretados como "protótipos" de naturezas de despesa.
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o classificador K-Means.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        super().__init__(model_name='kmeans', project_root=project_root)
        
        # Mapeamento de clusters para naturezas (após treinamento)
        self.cluster_to_natureza = {}
        self.cluster_purities = {}
        self.silhouette_avg = None
        
        logger.info("K-Means Classifier inicializado")
    
    def build_model(self, n_clusters: int = None, init: str = 'k-means++',
                   n_init: int = 10, max_iter: int = 300, **kwargs):
        """
        Constrói o modelo K-Means.
        
        Conforme metodologia, a convergência é garantida, embora para um ótimo local,
        sendo o resultado sensível à inicialização dos centróides.
        
        Args:
            n_clusters: Número de clusters (padrão: número de classes únicas)
            init: Método de inicialização
                  - 'k-means++': Inicialização inteligente (padrão)
                  - 'random': Inicialização aleatória
            n_init: Número de vezes que o algoritmo será executado (melhor é mantido)
            max_iter: Número máximo de iterações
            **kwargs: Outros parâmetros do KMeans
        """
        # Se n_clusters não especificado, usar número de classes
        if n_clusters is None and self.data is not None:
            n_clusters = len(np.unique(self.data['y_train_encoded']))
            logger.info(f"n_clusters não especificado, usando número de classes: {n_clusters}")
        elif n_clusters is None:
            raise ValueError("n_clusters deve ser especificado ou dados devem estar carregados")
        
        logger.info(
            f"Construindo modelo K-Means "
            f"(n_clusters={n_clusters}, init={init}, n_init={n_init})"
        )
        
        self.model = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42,
            verbose=0,
            **kwargs
        )
        
        logger.info("Modelo K-Means construído com sucesso")
    
    def train(self, **kwargs):
        """
        Treina o modelo K-Means e calcula mapeamento cluster->natureza.
        
        Após o clustering, mapeia cada cluster para a natureza mais frequente
        naquele cluster (classe majoritária).
        """
        if self.data is None:
            raise ValueError("Dados não processados. Execute preprocess_data() primeiro.")
        
        if self.model is None:
            # Se modelo não foi construído, construir com n_clusters = n_classes
            n_clusters = len(np.unique(self.data['y_train_encoded']))
            self.build_model(n_clusters=n_clusters, **kwargs)
        
        logger.info("Treinando modelo K-Means...")
        
        X_train = self.data['X_train_vec']
        y_train = self.data['y_train_encoded']
        
        import time
        start_time = time.time()
        
        # Treinar K-Means
        self.model.fit(X_train)
        
        training_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {training_time:.2f}s")
        logger.info(f"Inércia final: {self.model.inertia_:.2f}")
        
        # Mapear clusters para naturezas
        cluster_labels = self.model.labels_
        
        for cluster_id in range(self.model.n_clusters):
            # Encontrar amostras neste cluster
            mask = cluster_labels == cluster_id
            if mask.sum() == 0:
                continue
            
            # Encontrar natureza mais frequente
            naturezas_no_cluster = y_train[mask]
            most_common = Counter(naturezas_no_cluster).most_common(1)[0]
            natureza_majoritaria = most_common[0]
            count_majoritaria = most_common[1]
            
            # Mapear
            self.cluster_to_natureza[cluster_id] = natureza_majoritaria
            
            # Calcular pureza (proporção da classe majoritária)
            purity = count_majoritaria / mask.sum()
            self.cluster_purities[cluster_id] = purity
        
        # Calcular índice silhueta (métrica de qualidade do clustering)
        if X_train.shape[0] < 50000:  # Silhueta é caro para datasets grandes
            logger.info("Calculando índice silhueta...")
            self.silhouette_avg = silhouette_score(X_train, cluster_labels)
            logger.info(f"Índice Silhueta: {self.silhouette_avg:.4f}")
        
        self.metrics['training_time'] = training_time
        
        # Log de pureza média
        avg_purity = np.mean(list(self.cluster_purities.values()))
        logger.info(f"Pureza média dos clusters: {avg_purity:.4f}")
    
    def predict(self, text: str) -> Tuple[str, float]:
        """
        Faz predição atribuindo ao cluster mais próximo.
        
        Args:
            text: Descrição da despesa
            
        Returns:
            Tupla (natureza prevista, confiança baseada em pureza)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar e vetorizar
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Predizer cluster
        cluster_id = self.model.predict(text_vec)[0]
        
        # Obter natureza correspondente
        pred_encoded = self.cluster_to_natureza.get(cluster_id, 0)
        natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Confiança baseada na pureza do cluster
        confidence = self.cluster_purities.get(cluster_id, 0.0)
        
        return natureza, confidence
    
    def predict_top_k(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retorna as top-k predições baseadas nos k clusters mais próximos.
        
        Args:
            text: Descrição da despesa
            k: Número de predições a retornar
            
        Returns:
            Lista de tuplas (natureza, confiança)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar e vetorizar
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        text_vec = self.pipeline.vectorizer.transform([text_processed])
        
        # Calcular distâncias para todos os centróides
        distances = self.model.transform(text_vec)[0]
        
        # Top-k clusters mais próximos
        top_k_clusters = np.argsort(distances)[:k]
        
        # Obter naturezas e confianças
        results = []
        for cluster_id in top_k_clusters:
            if cluster_id in self.cluster_to_natureza:
                pred_encoded = self.cluster_to_natureza[cluster_id]
                natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
                confidence = self.cluster_purities.get(cluster_id, 0.0)
                results.append((natureza, confidence))
        
        return results
    
    def analyze_clusters(self):
        """
        Analisa a qualidade dos clusters formados.
        
        Conforme metodologia, fornece insights sobre a estrutura dos dados.
        """
        logger.info("\n" + "="*80)
        logger.info("ANÁLISE DE CLUSTERS")
        logger.info("="*80)
        
        logger.info(f"\nNúmero de clusters: {self.model.n_clusters}")
        logger.info(f"Inércia (soma das distâncias ao quadrado): {self.model.inertia_:.2f}")
        
        if self.silhouette_avg is not None:
            logger.info(f"Índice Silhueta: {self.silhouette_avg:.4f}")
            logger.info("  (Quanto mais próximo de 1, melhor a qualidade do clustering)")
        
        logger.info(f"\nPureza média dos clusters: {np.mean(list(self.cluster_purities.values())):.4f}")
        logger.info("  (Proporção da classe majoritária em cada cluster)")
        
        # Distribuição de pureza
        purities = list(self.cluster_purities.values())
        logger.info(f"\nDistribuição de Pureza:")
        logger.info(f"  Mínima: {np.min(purities):.4f}")
        logger.info(f"  Máxima: {np.max(purities):.4f}")
        logger.info(f"  Mediana: {np.median(purities):.4f}")
        
        # Clusters com pureza muito baixa (possível redundância)
        low_purity_clusters = [(cid, purity) for cid, purity in self.cluster_purities.items() 
                               if purity < 0.5]
        
        if low_purity_clusters:
            logger.info(f"\n⚠️  Clusters com baixa pureza (< 0.5): {len(low_purity_clusters)}")
            logger.info("   (Podem indicar classes similares ou redundantes)")
        
        logger.info("="*80)
    
    def get_cluster_prototype(self, cluster_id: int, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Retorna as features mais importantes do centróide de um cluster.
        
        Conforme metodologia, os centróides podem ser interpretados como
        "protótipos" de naturezas de despesa.
        
        Args:
            cluster_id: ID do cluster
            top_n: Número de features a retornar
            
        Returns:
            Lista de tuplas (feature, peso)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Obter centróide
        centroid = self.model.cluster_centers_[cluster_id]
        
        # Obter nomes das features
        feature_names = self.pipeline.vectorizer.get_feature_names_out()
        
        # Top features
        top_indices = np.argsort(centroid)[-top_n:][::-1]
        
        results = [(feature_names[idx], centroid[idx]) for idx in top_indices]
        
        return results


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: K-Means Classifier")
    print("="*80)
    
    # 1. Criar classificador
    classifier = KMeansNaturezaClassifier()
    
    # 2. Pré-processar dados
    print("\n[1/4] Pré-processando dados...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. Treinar (o número de clusters será igual ao número de classes)
    print("\n[2/4] Treinando modelo K-Means...")
    classifier.train(n_init=10)
    
    # 4. Analisar clusters
    print("\n[3/4] Analisando qualidade dos clusters...")
    classifier.analyze_clusters()
    
    # 5. Avaliar
    print("\n[4/4] Avaliando modelo...")
    test_metrics = classifier.evaluate(dataset='test')
    
    # 6. Salvar
    print("\nSalvando modelo...")
    classifier.save_model()
    
    # Informações do modelo
    print("\n" + "="*80)
    print("INFORMAÇÕES DO MODELO")
    print("="*80)
    info = classifier.get_model_info()
    print(f"Modelo: {info['model_type']}")
    print(f"Clusters: {classifier.model.n_clusters}")
    print(f"Pureza média: {np.mean(list(classifier.cluster_purities.values())):.4f}")
    if classifier.silhouette_avg:
        print(f"Índice Silhueta: {classifier.silhouette_avg:.4f}")
    print(f"\nMétrica Principal (Macro F1-Score): {test_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Teste de predição
    print("\n" + "="*80)
    print("TESTE DE PREDIÇÃO")
    print("="*80)
    texto_teste = "Pagamento de despesas com pessoal ativo"
    print(f"Texto: {texto_teste}")
    
    natureza, conf = classifier.predict(texto_teste)
    print(f"\nPredição: {natureza}")
    print(f"Confiança (pureza do cluster): {conf:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)
    print("\n💡 OBJETIVO: Descobrir agrupamentos naturais e validar")
    print("   se todas as naturezas são necessárias ou se há redundâncias!")