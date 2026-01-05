"""
Módulo de Pré-processamento de Dados para Classificação de Naturezas de Despesa

Este módulo implementa as etapas de pré-processamento típicas de mineração de texto
conforme descrito em Wives (2002) e Baeza-Yates e Ribeiro-Neto (2013).

Módulo integrado ao projeto Chat-IPOF para classificação de naturezas de despesa.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import os
import pandas as pd
import numpy as np
import re
import pickle
import logging
from typing import Tuple, List, Optional, Dict, Union
from pathlib import Path

# Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# NLTK para processamento de linguagem natural (opcional)
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK não disponível. Usando tokenização simples.")


class TextPreprocessor:
    """
    Classe para pré-processamento de textos de descrições de despesas.
    
    Implementa as etapas de normalização, tokenização, remoção de stopwords
    conforme metodologia descrita em Materiais e Métodos.
    """
    
    def __init__(self, language: str = 'portuguese', remove_stopwords: bool = True,
                 min_token_length: int = 2):
        """
        Inicializa o pré-processador de texto.
        
        Args:
            language: Idioma para remoção de stopwords
            remove_stopwords: Se True, remove stopwords
            min_token_length: Comprimento mínimo de tokens
        """
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        
        # Carregar stopwords
        if NLTK_AVAILABLE and remove_stopwords:
            self.stopwords_set = set(stopwords.words(language))
            # Adicionar stopwords customizadas para textos administrativos
            custom_stopwords = {
                'sendo', 'sido', 'tendo', 'ser', 'ter', 'fazer', 'feito',
                'mediante', 'conforme', 'referente', 'relativo', 'destinado'
            }
            self.stopwords_set.update(custom_stopwords)
        else:
            self.stopwords_set = set()
        
        logger.info(f"TextPreprocessor inicializado (NLTK: {NLTK_AVAILABLE})")
    
    def normalize_text(self, text: str) -> str:
        """Normaliza o texto: minúsculas e remove caracteres especiais."""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokeniza o texto em palavras."""
        if not text:
            return []
        
        if NLTK_AVAILABLE:
            try:
                tokens = word_tokenize(text, language=self.language)
            except:
                tokens = text.split()
        else:
            tokens = text.split()
        
        return [t for t in tokens if len(t) >= self.min_token_length]
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """Remove stopwords da lista de tokens."""
        if not self.remove_stopwords or not self.stopwords_set:
            return tokens
        return [token for token in tokens if token not in self.stopwords_set]
    
    def preprocess_text(self, text: str) -> str:
        """Aplica pipeline completo de pré-processamento."""
        normalized = self.normalize_text(text)
        tokens = self.tokenize(normalized)
        tokens = self.remove_stopwords_from_tokens(tokens)
        return ' '.join(tokens)
    
    def preprocess_corpus(self, texts: Union[List[str], pd.Series]) -> List[str]:
        """Processa um corpus de textos."""
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        return [self.preprocess_text(text) for text in texts]


class NaturezaDataLoader:
    """
    Classe para carregar e preparar dados de naturezas de despesa.
    
    Integrado ao projeto Chat-IPOF, carrega dados do diretório data/natureza/.
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o carregador de dados.
        
        Args:
            project_root: Diretório raiz do projeto (opcional)
        """
        if project_root is None:
            # Tenta encontrar o diretório raiz do projeto
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.project_root = project_root
        self.data_dir = os.path.join(project_root, 'data', 'natureza')
        self.df = None
        self.label_encoder = LabelEncoder()
        
        logger.info(f"NaturezaDataLoader inicializado (project_root: {project_root})")
    
    def load_data(self, filename: str = 'natureza_despesa_final.xlsx',
                  drop_na: bool = True, min_samples_per_class: int = 3) -> pd.DataFrame:
        """
        Carrega dados do arquivo Excel.
        
        Args:
            filename: Nome do arquivo
            drop_na: Remove linhas com valores nulos
            min_samples_per_class: Mínimo de amostras por classe
            
        Returns:
            DataFrame com os dados
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Arquivo não encontrado: {filepath}\n"
                f"Certifique-se de que o arquivo está em: {self.data_dir}"
            )
        
        logger.info(f"Carregando dados de: {filepath}")
        self.df = pd.read_excel(filepath)
        
        logger.info(f"Shape original: {self.df.shape}")
        
        if drop_na:
            original_len = len(self.df)
            self.df = self.df.dropna()
            logger.info(f"Removidos {original_len - len(self.df)} registros com valores nulos")
        
        # Filtrar classes com poucas amostras
        if min_samples_per_class > 1:
            class_counts = self.df['natureza'].value_counts()
            valid_classes = class_counts[class_counts >= min_samples_per_class].index
            original_len = len(self.df)
            self.df = self.df[self.df['natureza'].isin(valid_classes)]
            
            removed = original_len - len(self.df)
            if removed > 0:
                logger.info(f"Removidos {removed} registros de classes com < {min_samples_per_class} amostras")
        
        logger.info(f"Shape final: {self.df.shape}")
        logger.info(f"Naturezas únicas: {self.df['natureza'].nunique()}")
        
        return self.df
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas básicas do dataset."""
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        return {
            'total_samples': len(self.df),
            'unique_naturezas': self.df['natureza'].nunique(),
            'avg_text_length': self.df['despesa'].str.len().mean(),
            'min_text_length': self.df['despesa'].str.len().min(),
            'max_text_length': self.df['despesa'].str.len().max(),
            'class_distribution': self.df['natureza'].value_counts().to_dict()
        }
    
    def split_data(self, test_size: float = 0.15, val_size: float = 0.15,
                   random_state: int = 42) -> Tuple:
        """
        Divide dados em treino, validação e teste (70/15/15).
        
        Returns:
            Tupla (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        X = self.df['despesa'].values
        y = self.df['natureza'].values
        
        # Primeira divisão: treino+val vs teste
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=None
        )
        
        # Segunda divisão: treino vs validação
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=random_state, stratify=None
        )
        
        logger.info(f"Divisão - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test


class NaturezaPreprocessingPipeline:
    """
    Pipeline completo de pré-processamento integrado ao Chat-IPOF.
    
    Gerencia todo o fluxo de preparação dos dados para treinamento de modelos
    de classificação de naturezas de despesa.
    """
    
    def __init__(self, project_root: str = None, vectorization_method: str = 'tfidf',
                 max_features: int = 5000, remove_stopwords: bool = True):
        """
        Inicializa o pipeline de pré-processamento.
        
        Args:
            project_root: Diretório raiz do projeto
            vectorization_method: Método de vetorização ('tfidf' ou 'count')
            max_features: Número máximo de features
            remove_stopwords: Remove stopwords
        """
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.loader = NaturezaDataLoader(self.project_root)
        self.preprocessor = TextPreprocessor(remove_stopwords=remove_stopwords)
        
        # Vetorizador
        if vectorization_method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=False,
                strip_accents=None
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                lowercase=False,
                strip_accents=None
            )
        
        self.label_encoder = LabelEncoder()
        self.models_dir = os.path.join(self.project_root, 'data', 'models')
        
        logger.info(f"Pipeline inicializado (método: {vectorization_method}, max_features: {max_features})")
    
    def run(self, filename: str = 'natureza_despesa_final.xlsx',
            test_size: float = 0.15, val_size: float = 0.15,
            random_state: int = 42, min_samples_per_class: int = 3) -> Dict:
        """
        Executa pipeline completo de pré-processamento.
        
        Returns:
            Dicionário com dados processados e objetos do pipeline
        """
        logger.info("="*80)
        logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO")
        logger.info("="*80)
        
        # 1. Carregar dados
        logger.info("\n[1/5] Carregando dados...")
        self.loader.load_data(filename=filename, min_samples_per_class=min_samples_per_class)
        
        # 2. Dividir dados
        logger.info("\n[2/5] Dividindo dados...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.loader.split_data(
            test_size=test_size, val_size=val_size, random_state=random_state
        )
        
        # 3. Pré-processar textos
        logger.info("\n[3/5] Pré-processando textos...")
        X_train_processed = self.preprocessor.preprocess_corpus(X_train)
        X_val_processed = self.preprocessor.preprocess_corpus(X_val)
        X_test_processed = self.preprocessor.preprocess_corpus(X_test)
        
        # 4. Vetorizar
        logger.info("\n[4/5] Vetorizando textos...")
        X_train_vec = self.vectorizer.fit_transform(X_train_processed)
        X_val_vec = self.vectorizer.transform(X_val_processed)
        X_test_vec = self.vectorizer.transform(X_test_processed)
        logger.info(f"Shape da matriz TF-IDF: {X_train_vec.shape}")
        
        # 5. Codificar labels
        logger.info("\n[5/5] Codificando labels...")
        all_labels = np.concatenate([y_train, y_val, y_test])
        self.label_encoder.fit(all_labels)
        
        y_train_encoded = self.label_encoder.transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        logger.info(f"Número de classes: {len(self.label_encoder.classes_)}")
        logger.info("="*80)
        logger.info("PIPELINE CONCLUÍDO!")
        logger.info("="*80)
        
        return {
            'X_train_raw': X_train,
            'X_val_raw': X_val,
            'X_test_raw': X_test,
            'y_train_raw': y_train,
            'y_val_raw': y_val,
            'y_test_raw': y_test,
            'X_train_processed': X_train_processed,
            'X_val_processed': X_val_processed,
            'X_test_processed': X_test_processed,
            'X_train_vec': X_train_vec,
            'X_val_vec': X_val_vec,
            'X_test_vec': X_test_vec,
            'y_train_encoded': y_train_encoded,
            'y_val_encoded': y_val_encoded,
            'y_test_encoded': y_test_encoded,
            'preprocessor': self.preprocessor,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }
    
    def save_artifacts(self, model_name: str = 'default'):
        """
        Salva artefatos do pipeline.
        
        Args:
            model_name: Nome do modelo para organizar artefatos
        """
        os.makedirs(self.models_dir, exist_ok=True)
        model_dir = os.path.join(self.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Salvar vetorizador
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Salvar label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Salvar configuração do preprocessor
        config_path = os.path.join(model_dir, 'preprocessor_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump({
                'language': self.preprocessor.language,
                'remove_stopwords': self.preprocessor.remove_stopwords,
                'min_token_length': self.preprocessor.min_token_length,
                'stopwords_set': self.preprocessor.stopwords_set
            }, f)
        
        logger.info(f"Artefatos salvos em: {model_dir}")
    
    @classmethod
    def load_artifacts(cls, model_name: str = 'default', project_root: str = None):
        """
        Carrega artefatos de um pipeline salvo.
        
        Args:
            model_name: Nome do modelo
            project_root: Diretório raiz do projeto
            
        Returns:
            Tupla (preprocessor, vectorizer, label_encoder)
        """
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        model_dir = os.path.join(project_root, 'data', 'models', model_name)
        
        # Carregar vetorizador
        vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Carregar label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        # Carregar configuração do preprocessor
        config_path = os.path.join(model_dir, 'preprocessor_config.pkl')
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Recriar preprocessor
        preprocessor = TextPreprocessor(
            language=config['language'],
            remove_stopwords=config['remove_stopwords'],
            min_token_length=config['min_token_length']
        )
        preprocessor.stopwords_set = config['stopwords_set']
        
        logger.info(f"Artefatos carregados de: {model_dir}")
        
        return preprocessor, vectorizer, label_encoder


if __name__ == "__main__":
    # Exemplo de uso
    pipeline = NaturezaPreprocessingPipeline()
    data = pipeline.run()
    
    print(f"\nDados prontos para treinamento:")
    print(f"  X_train: {data['X_train_vec'].shape}")
    print(f"  y_train: {data['y_train_encoded'].shape}")
    print(f"  Classes: {len(np.unique(data['y_train_encoded']))}")
    
    # Salvar artefatos
    pipeline.save_artifacts(model_name='baseline')