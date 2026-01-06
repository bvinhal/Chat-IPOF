"""
Classificador MLP com Sentence-BERT Embeddings

Implementação conforme descrito em Materiais e Métodos (Reimers e Gurevych, 2019).
Este modelo combina embeddings semânticos pré-treinados com uma rede neural
feed-forward simples (MLP).

O Sentence-BERT gera representações vetoriais semanticamente significativas.
Para o contexto brasileiro, usa BERTimbau ou Legal-BERTimbau (Rodrigues et al., 2021).

Integrado ao projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
Referências: Reimers e Gurevych (2019), Rodrigues et al. (2021), Haykin (2001)
"""

import logging
import numpy as np
from typing import Optional
from sklearn.neural_network import MLPClassifier

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning(
        "sentence-transformers não instalado. "
        "Execute: pip install sentence-transformers"
    )

from models.base_classifier import BaseNaturezaClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLPSBERTNaturezaClassifier(BaseNaturezaClassifier):
    """
    Classificador de Natureza de Despesa usando MLP + Sentence-BERT.
    
    Conforme metodologia, combina embeddings semânticos pré-treinados
    com uma rede neural feed-forward simples.
    
    Diferentemente do BERT padrão que requer fine-tuning completo
    (computacionalmente custoso), o S-BERT permite usar embeddings
    congelados, onde apenas um classificador simples é treinado.
    
    Vantagens (Metodologia):
    1. Redução drástica do tempo de treinamento (minutos vs. horas)
    2. Viabilidade em hardware comum sem GPU
    3. Aproveitamento de conhecimento semântico aprendido em milhões de textos
    
    Arquitetura conforme metodologia:
    - Entrada: 768 dimensões (embeddings do S-BERT)
    - Camadas ocultas: 512 e 256 neurônios
    - Saída: n_classes neurônios (uma para cada natureza)
    - Ativação: ReLU (introduz não-linearidade)
    - Dropout: Previne overfitting
    
    Modelos disponíveis para português:
    - 'neuralmind/bert-base-portuguese-cased': BERTimbau
    - 'rufimelo/Legal-BERTimbau': Legal-BERTimbau (especializado em textos jurídicos)
    - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': Multilingual
    """
    
    def __init__(self, project_root: str = None, 
                 model_name_sbert: str = 'neuralmind/bert-base-portuguese-cased'):
        """
        Inicializa o classificador MLP + S-BERT.
        
        Args:
            project_root: Diretório raiz do projeto
            model_name_sbert: Nome do modelo Sentence-BERT a usar
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers não está instalado. "
                "Instale com: pip install sentence-transformers"
            )
        
        super().__init__(model_name='mlp_sbert', project_root=project_root)
        
        # Carregar modelo Sentence-BERT
        logger.info(f"Carregando modelo Sentence-BERT: {model_name_sbert}")
        try:
            self.sbert_model = SentenceTransformer(model_name_sbert)
            logger.info("Modelo Sentence-BERT carregado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo S-BERT: {e}")
            logger.info("Tentando modelo alternativo: paraphrase-multilingual-MiniLM-L12-v2")
            self.sbert_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Embeddings já computados (cache)
        self.X_train_embeddings = None
        self.X_val_embeddings = None
        self.X_test_embeddings = None
        
        logger.info("MLP + S-BERT Classifier inicializado")
    
    def _generate_embeddings(self, texts, batch_size: int = 32, show_progress: bool = True):
        """
        Gera embeddings usando Sentence-BERT.
        
        Args:
            texts: Lista de textos
            batch_size: Tamanho do batch para processamento
            show_progress: Mostrar barra de progresso
            
        Returns:
            Array numpy com embeddings
        """
        logger.info(f"Gerando embeddings para {len(texts)} textos...")
        
        embeddings = self.sbert_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        logger.info(f"Embeddings gerados: shape {embeddings.shape}")
        return embeddings
    
    def preprocess_data(self, **kwargs):
        """
        Pré-processa dados E gera embeddings.
        
        Sobrescreve o método base para incluir geração de embeddings.
        """
        # Executar pré-processamento padrão
        data = super().preprocess_data(**kwargs)
        
        # Gerar embeddings para textos pré-processados
        logger.info("\nGerando embeddings com Sentence-BERT...")
        
        self.X_train_embeddings = self._generate_embeddings(
            data['X_train_processed'],
            show_progress=True
        )
        
        self.X_val_embeddings = self._generate_embeddings(
            data['X_val_processed'],
            show_progress=False
        )
        
        self.X_test_embeddings = self._generate_embeddings(
            data['X_test_processed'],
            show_progress=False
        )
        
        return data
    
    def build_model(self, hidden_layer_sizes: tuple = (512, 256),
                   activation: str = 'relu', alpha: float = 0.0001,
                   learning_rate_init: float = 0.001, max_iter: int = 200,
                   early_stopping: bool = True, validation_fraction: float = 0.1,
                   **kwargs):
        """
        Constrói o modelo MLP.
        
        Arquitetura conforme metodologia (Haykin, 2001; Faceli et al., 2011):
        - Camadas ocultas: (512, 256) neurônios por padrão
        - Ativação ReLU: Introduz não-linearidade
        - Dropout implícito via regularização alpha
        
        Args:
            hidden_layer_sizes: Tupla com número de neurônios em cada camada oculta
            activation: Função de ativação ('relu', 'tanh', 'logistic')
            alpha: Parâmetro de regularização L2 (equivalente a Dropout)
            learning_rate_init: Taxa de aprendizado inicial
            max_iter: Número máximo de épocas
            early_stopping: Se True, para quando validação não melhora
            validation_fraction: Fração para validação (se early_stopping=True)
            **kwargs: Outros parâmetros do MLPClassifier
        """
        logger.info(
            f"Construindo modelo MLP "
            f"(hidden_layers={hidden_layer_sizes}, activation={activation})"
        )
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver='adam',  # Otimizador Adam (eficiente)
            alpha=alpha,
            learning_rate='adaptive',  # Ajusta learning rate automaticamente
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            random_state=42,
            verbose=True,  # Mostrar progresso
            **kwargs
        )
        
        logger.info("Modelo MLP construído com sucesso")
    
    def train(self, **kwargs):
        """
        Treina o modelo MLP usando embeddings.
        """
        if self.X_train_embeddings is None:
            raise ValueError(
                "Embeddings não gerados. Execute preprocess_data() primeiro."
            )
        
        if self.model is None:
            self.build_model(**kwargs)
        
        logger.info("Treinando modelo MLP com embeddings S-BERT...")
        
        import time
        start_time = time.time()
        
        # Treinar usando embeddings
        self.model.fit(self.X_train_embeddings, self.data['y_train_encoded'])
        
        training_time = time.time() - start_time
        logger.info(f"Treinamento concluído em {training_time:.2f}s")
        
        if hasattr(self.model, 'loss_'):
            logger.info(f"Loss final: {self.model.loss_:.4f}")
        
        if hasattr(self.model, 'n_iter_'):
            logger.info(f"Número de iterações: {self.model.n_iter_}")
        
        self.metrics['training_time'] = training_time
    
    def evaluate(self, dataset: str = 'test'):
        """
        Avalia o modelo usando embeddings.
        
        Sobrescreve método base para usar embeddings em vez de vetores TF-IDF.
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Selecionar conjunto de dados e embeddings correspondentes
        if dataset == 'train':
            X = self.X_train_embeddings
            y = self.data['y_train_encoded']
        elif dataset == 'val':
            X = self.X_val_embeddings
            y = self.data['y_val_encoded']
        elif dataset == 'test':
            X = self.X_test_embeddings
            y = self.data['y_test_encoded']
        else:
            raise ValueError(f"Dataset inválido: {dataset}")
        
        logger.info(f"Avaliando modelo em conjunto de {dataset}...")
        
        import time
        start_time = time.time()
        
        y_pred = self.model.predict(X)
        
        prediction_time = time.time() - start_time
        
        # Importar módulo de avaliação
        from utils.evaluation_metrics import NaturezaEvaluator
        
        evaluator = NaturezaEvaluator(model_name=self.model_name)
        
        # Obter probabilidades se disponível
        y_proba = None
        if hasattr(self.model, 'predict_proba'):
            y_proba = self.model.predict_proba(X)
        
        # Calcular métricas
        metrics = evaluator.calculate_all_metrics(
            y_true=y,
            y_pred=y_pred,
            y_proba=y_proba,
            training_time=self.metrics.get('training_time', 0.0),
            prediction_time=prediction_time,
            dataset_name=dataset
        )
        
        # Salvar métricas
        self.metrics[f'{dataset}_metrics'] = metrics
        
        # Imprimir relatório
        evaluator.print_metrics_report(metrics)
        
        return metrics
    
    def predict(self, text: str) -> tuple:
        """
        Faz predição gerando embedding do texto.
        
        Args:
            text: Descrição da despesa
            
        Returns:
            Tupla (natureza prevista, confiança)
        """
        if self.model is None:
            raise ValueError("Modelo não treinado.")
        
        # Pré-processar texto
        text_processed = self.pipeline.preprocessor.preprocess_text(text)
        
        # Gerar embedding
        embedding = self.sbert_model.encode([text_processed], convert_to_numpy=True)
        
        # Predição
        pred_encoded = self.model.predict(embedding)[0]
        natureza = self.pipeline.label_encoder.inverse_transform([pred_encoded])[0]
        
        # Confiança
        confidence = 1.0
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(embedding)[0]
            confidence = float(proba[pred_encoded])
        
        return natureza, confidence


if __name__ == "__main__":
    # Exemplo de uso
    print("="*80)
    print("EXEMPLO DE USO: MLP + Sentence-BERT Classifier")
    print("="*80)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ sentence-transformers não está instalado!")
        print("Instale com: pip install sentence-transformers")
        exit(1)
    
    # 1. Criar classificador
    print("\nCriando classificador (pode demorar ao carregar S-BERT)...")
    classifier = MLPSBERTNaturezaClassifier()
    
    # 2. Pré-processar dados e gerar embeddings
    print("\n[1/4] Pré-processando dados e gerando embeddings...")
    print("⚠️  Isso pode demorar alguns minutos...")
    data = classifier.preprocess_data(min_samples_per_class=3)
    
    # 3. Treinar modelo
    print("\n[2/4] Treinando modelo MLP...")
    classifier.train(
        hidden_layer_sizes=(512, 256),
        max_iter=200,
        early_stopping=True
    )
    
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
    print(f"Dimensão dos embeddings: {classifier.X_train_embeddings.shape[1]}")
    print(f"Amostras de treino: {info['data_info']['n_samples_train']}")
    print(f"Classes: {info['data_info']['n_classes']}")
    print(f"\nMétrica Principal (Macro F1-Score): {test_metrics['f1_macro']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    if 'top_3_accuracy' in test_metrics:
        print(f"Top-3 Accuracy: {test_metrics['top_3_accuracy']:.4f}")
    
    # Teste de predição
    print("\n" + "="*80)
    print("TESTE DE PREDIÇÃO")
    print("="*80)
    texto_teste = "Contratação de empresa para serviços de tecnologia"
    print(f"Texto: {texto_teste}")
    
    natureza, conf = classifier.predict(texto_teste)
    print(f"\nPredição: {natureza}")
    print(f"Confiança: {conf:.4f}")
    
    print("\n" + "="*80)
    print("CONCLUÍDO!")
    print("="*80)
    print("\n💡 Este modelo representa o paradigma neural com melhor custo-benefício!")
    print("   Aproveitando conhecimento semântico aprendido em milhões de textos.")