# models/sentence_transformer_classifier.py

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import pickle
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentenceTransformerClassifier:
    """
    Classificador de natureza de despesa usando SentenceTransformer.
    Utiliza embeddings de texto para classificar descrições de despesas.
    """
    
    def __init__(self):
        """Inicializa o classificador de natureza com SentenceTransformer."""
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.model = None
        self.embeddings = None
        self.naturezas = None
        self.descricoes = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Caminho para os arquivos do modelo
        self.model_path = os.path.join(active_config.MODELS_DIR, 'st_classifier')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Flag para indicar se o modelo foi treinado
        self.is_trained = False
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Carrega os dados do arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo CSV
        
        Returns:
            pd.DataFrame: DataFrame com os dados carregados
        """
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Arquivo CSV carregado com sucesso: {len(df)} registros")
            return df
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo CSV: {str(e)}")
            raise
    
    def train(self, file_path: str) -> bool:
        """
        Treina o classificador usando o arquivo CSV.
        
        Args:
            file_path: Caminho para o arquivo CSV
        
        Returns:
            bool: True se o treinamento foi bem-sucedido
        """
        try:
            # Carrega o arquivo CSV
            df = self.load_csv(file_path)
            
            # Verifica se as colunas necessárias existem
            required_columns = ['natureza', 'Descrição']
            for col in required_columns:
                if col not in df.columns:
                    self.logger.error(f"Coluna '{col}' não encontrada no arquivo CSV")
                    return False
            
            # Inicializa o modelo SentenceTransformer
            self.logger.info(f"Inicializando modelo SentenceTransformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Limpa os dados
            df['natureza'] = df['natureza'].astype(str).str.strip()
            df['Descrição'] = df['Descrição'].astype(str).str.strip()
            
            # Remove linhas com valores vazios
            df = df[df['natureza'].notna() & df['Descrição'].notna() & 
                    (df['natureza'] != '') & (df['Descrição'] != '')]
            
            # Armazena as naturezas e descrições
            self.naturezas = df['natureza'].tolist()
            self.descricoes = df['Descrição'].tolist()
            
            # Gera os embeddings das descrições
            self.logger.info(f"Gerando embeddings para {len(self.descricoes)} descrições")
            self.embeddings = self.model.encode(self.descricoes, show_progress_bar=True)
            
            # Marca como treinado
            self.is_trained = True
            
            # Salva o modelo
            self.save_model()
            
            self.logger.info("Treinamento concluído com sucesso")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao treinar o classificador: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def predict(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Prediz as naturezas de despesa mais prováveis para um texto.
        
        Args:
            text: Texto para classificação
            top_k: Número de naturezas mais prováveis a retornar
        
        Returns:
            List[Dict[str, Any]]: Lista de previsões ordenadas por confiança
        """
        if not self.is_trained:
            # Tenta carregar o modelo se não estiver treinado
            if not self.load_model():
                self.logger.error("Modelo não treinado e não foi possível carregar")
                raise ValueError("Modelo não treinado e não foi possível carregar")
        
        try:
            # Verifica se o texto está vazio
            if not text or not text.strip():
                self.logger.warning("Texto vazio fornecido para previsão")
                return []
            
            # Gera o embedding do texto
            self.logger.info(f"Gerando embedding para o texto: {text[:100]}...")
            text_embedding = self.model.encode([text])[0]
            
            # Calcula a similaridade com todas as descrições
            similarities = cosine_similarity([text_embedding], self.embeddings)[0]
            
            # Obtém os índices dos top_k mais similares
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Cria a lista de previsões
            predictions = []
            for idx in top_indices:
                predictions.append({
                    'codigo': self.naturezas[idx],
                    'nome': self.descricoes[idx],
                    'confianca': float(similarities[idx])
                })
            
            self.logger.info(f"Geradas {len(predictions)} previsões para o texto")
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erro ao fazer previsão: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def save_model(self) -> bool:
        """
        Salva o modelo treinado.
        
        Returns:
            bool: True se o salvamento foi bem-sucedido
        """
        if not self.is_trained:
            self.logger.error("Tentativa de salvar modelo não treinado")
            return False
        
        try:
            # Cria o diretório se não existir
            os.makedirs(self.model_path, exist_ok=True)
            
            # Salva os dados
            data = {
                'naturezas': self.naturezas,
                'descricoes': self.descricoes,
                'embeddings': self.embeddings,
                'model_name': self.model_name
            }
            
            # Salva os dados em formato pickle
            data_path = os.path.join(self.model_path, 'model_data.pkl')
            with open(data_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Salva informações básicas em JSON para referência
            info = {
                'model_name': self.model_name,
                'num_naturezas': len(self.naturezas),
                'embedding_dim': self.embeddings.shape[1] if self.embeddings is not None else 0,
                'trained_at': pd.Timestamp.now().isoformat()
            }
            
            info_path = os.path.join(self.model_path, 'model_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Modelo salvo com sucesso em {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def load_model(self) -> bool:
        """
        Carrega um modelo salvo anteriormente.
        
        Returns:
            bool: True se o carregamento foi bem-sucedido
        """
        try:
            # Verifica se o arquivo de dados existe
            data_path = os.path.join(self.model_path, 'model_data.pkl')
            
            if not os.path.exists(data_path):
                self.logger.warning(f"Arquivo de dados não encontrado: {data_path}")
                return False
            
            # Carrega os dados
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
            
            # Verifica se todos os dados necessários estão presentes
            required_keys = ['naturezas', 'descricoes', 'embeddings', 'model_name']
            for key in required_keys:
                if key not in data:
                    self.logger.error(f"Dados incompletos: '{key}' não encontrado")
                    return False
            
            # Carrega os dados no objeto
            self.naturezas = data['naturezas']
            self.descricoes = data['descricoes']
            self.embeddings = data['embeddings']
            self.model_name = data['model_name']
            
            # Inicializa o modelo SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            
            self.is_trained = True
            
            self.logger.info(f"Modelo carregado com sucesso: {len(self.naturezas)} naturezas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False