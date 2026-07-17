"""
Data Processor - Pré-processamento Simplificado

Classe única para processar dados ANTES da divisão em treino/teste/validação.
Conforme metodologia (Wives, 2002; Baeza-Yates e Ribeiro-Neto, 2013).

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Processa dados de naturezas de despesa de forma simplificada.
    
    Pipeline:
    1. Carrega dados do Excel
    2. Limpa e normaliza textos
    3. Remove classes com poucos exemplos
    4. Vetoriza com TF-IDF
    5. Divide em treino/validação/teste (70/15/15)
    """
    
    def __init__(self, excel_path: str, min_samples_per_class: int = 5):
        """
        Inicializa o processador.
        
        Args:
            excel_path: Caminho do arquivo Excel
            min_samples_per_class: Mínimo de amostras por classe (mínimo recomendado: 5)
        """
        self.excel_path = excel_path
        # Garante mínimo de 5 para evitar problemas na divisão estratificada
        self.min_samples = max(5, min_samples_per_class)
        
        # Componentes
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.label_encoder = LabelEncoder()
        
        # Dados processados
        self.data = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        logger.info(f"DataProcessor inicializado (min_samples={min_samples_per_class})")
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza texto.
        
        Etapas:
        1. Converte para minúsculas
        2. Remove caracteres especiais
        3. Remove espaços extras
        
        Args:
            text: Texto original
            
        Returns:
            Texto limpo
        """
        if pd.isna(text):
            return ""
        
        # Minúsculas
        text = str(text).lower()
        
        # Remove caracteres especiais (mantém letras, números e espaços)
        text = re.sub(r'[^a-záàâãéêíóôõúçA-Z0-9\s]', ' ', text)
        
        # Remove espaços extras
        text = ' '.join(text.split())
        
        return text
    
    def load_and_process(self):
        """
        Carrega e processa os dados completamente.
        
        Pipeline completo:
        1. Carrega Excel
        2. Remove nulos
        3. Limpa textos
        4. Remove classes pequenas
        5. Vetoriza com TF-IDF
        6. Divide em treino/val/teste
        
        Returns:
            Dict com dados processados
        """
        logger.info("="*80)
        logger.info("PROCESSANDO DADOS")
        logger.info("="*80)
        
        # 1. Carregar dados
        logger.info(f"\n[1/6] Carregando dados de: {self.excel_path}")
        df = pd.read_excel(self.excel_path)
        logger.info(f"   Shape original: {df.shape}")
        
        # 2. Remover nulos
        logger.info("\n[2/6] Removendo valores nulos...")
        inicial = len(df)
        df = df.dropna()
        logger.info(f"   Removidos: {inicial - len(df)} registros")
        
        # 3. Limpar textos
        logger.info("\n[3/6] Limpando e normalizando textos...")
        df['texto_limpo'] = df.iloc[:, 0].apply(self.clean_text)
        df['natureza'] = df.iloc[:, 1]
        
        # 4. Remover classes pequenas
        logger.info(f"\n[4/6] Removendo classes com < {self.min_samples} amostras...")
        counts = df['natureza'].value_counts()
        classes_validas = counts[counts >= self.min_samples].index
        df = df[df['natureza'].isin(classes_validas)]
        logger.info(f"   Registros finais: {len(df)}")
        logger.info(f"   Classes únicas: {df['natureza'].nunique()}")
        
        # 5. Vetorizar com TF-IDF
        logger.info("\n[5/6] Vetorizando com TF-IDF...")
        
        # Guardar textos limpos ANTES da vetorização (para MLP+S-BERT)
        texts = df['texto_limpo'].values
        labels = df['natureza'].values
        
        X = self.vectorizer.fit_transform(texts)
        y = self.label_encoder.fit_transform(labels)
        logger.info(f"   Shape da matriz: {X.shape}")
        logger.info(f"   Número de classes: {len(self.label_encoder.classes_)}")
        
        # 6. Dividir em treino/val/teste (70/15/15)
        logger.info("\n[6/6] Dividindo em treino/validação/teste...")
        
        # Primeiro: treino (70%) vs temp (30%)
        X_train, X_temp, y_train, y_temp, texts_train, texts_temp = train_test_split(
            X, y, texts, test_size=0.3, stratify=y, random_state=42
        )
        
        # Verificar se todas as classes no temp têm pelo menos 2 amostras
        # Isso é necessário para a segunda divisão estratificada
        temp_counts = pd.Series(y_temp).value_counts()
        classes_with_one_sample = temp_counts[temp_counts < 2].index.tolist()
        
        if len(classes_with_one_sample) > 0:
            # Se houver classes com 1 amostra, usar divisão simples (sem stratify)
            logger.warning(f"   ⚠️  {len(classes_with_one_sample)} classes com <2 amostras no conjunto temp")
            logger.warning("   Usando divisão não estratificada para val/teste")
            X_val, X_test, y_val, y_test, texts_val, texts_test = train_test_split(
                X_temp, y_temp, texts_temp, test_size=0.5, random_state=42
            )
        else:
            # Divisão estratificada normal
            X_val, X_test, y_val, y_test, texts_val, texts_test = train_test_split(
                X_temp, y_temp, texts_temp, test_size=0.5, stratify=y_temp, random_state=42
            )
        
        logger.info(f"   Treino:     {X_train.shape[0]} amostras")
        logger.info(f"   Validação:  {X_val.shape[0]} amostras")
        logger.info(f"   Teste:      {X_test.shape[0]} amostras")
        
        # Salvar
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Salvar textos (necessário para MLP+S-BERT)
        self.texts_train = texts_train
        self.texts_val = texts_val
        self.texts_test = texts_test
        
        self.data = df
        
        logger.info("\n" + "="*80)
        logger.info("PROCESSAMENTO CONCLUÍDO!")
        logger.info("="*80)
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'texts_train': texts_train,
            'texts_val': texts_val,
            'texts_test': texts_test,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'n_classes': len(self.label_encoder.classes_),
            'n_features': X_train.shape[1]
        }
    
    def get_class_name(self, encoded_label: int) -> str:
        """Converte label codificado para nome da natureza."""
        return self.label_encoder.inverse_transform([encoded_label])[0]
    
    def predict_text(self, text: str):
        """
        Processa um texto novo para predição.
        
        Args:
            text: Texto a processar
            
        Returns:
            Vetor TF-IDF
        """
        text_limpo = self.clean_text(text)
        return self.vectorizer.transform([text_limpo])


if __name__ == '__main__':
    # Teste rápido
    processor = DataProcessor(
        excel_path='data/natureza/natureza_despesa_final.xlsx',
        min_samples_per_class=5
    )
    
    data = processor.load_and_process()
    
    print("\n✅ Dados processados com sucesso!")
    print(f"   Classes: {data['n_classes']}")
    print(f"   Features: {data['n_features']}")