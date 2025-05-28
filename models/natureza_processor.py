# models/natureza_processor.py

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import pickle
import json
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaProcessor:
    """
    Classe para processamento dos dados de natureza de despesa.
    """
    
    def __init__(self):
        """Inicializa o processador de natureza de despesa."""
        self.data_path = os.path.join(active_config.DATA_DIR, 'natureza')
        os.makedirs(self.data_path, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.df = None
        self.label_mapping = {}  # Mapeamento de código para índice
        self.inverse_mapping = {}  # Mapeamento de índice para código
    
    def load_excel(self, file_path: str) -> pd.DataFrame:
        """
        Carrega os dados do arquivo Excel.
        
        Args:
            file_path: Caminho para o arquivo Excel
            
        Returns:
            pd.DataFrame: DataFrame com os dados carregados
        """
        try:
            self.logger.info(f"Carregando arquivo {file_path}")
            df = pd.read_excel(file_path)
            
            # Verificando se as colunas esperadas existem
            expected_columns = [
                'Natureza Despesa (Codigo)', 
                'Natureza Despesa (Nome)', 
                'Descrição Despesa', 
                'Orientação Natureza'
            ]
            
            for col in expected_columns:
                if col not in df.columns:
                    self.logger.error(f"Coluna '{col}' não encontrada no arquivo Excel")
                    raise ValueError(f"Coluna '{col}' não encontrada no arquivo Excel")
            
            self.logger.info(f"Arquivo Excel carregado com sucesso. {len(df)} registros encontrados.")
            self.df = df
            return df
        
        except Exception as e:
            self.logger.error(f"Erro ao carregar arquivo Excel: {str(e)}")
            raise
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Pré-processa os dados para treinamento do modelo.
        
        Returns:
            pd.DataFrame: DataFrame com os dados pré-processados
        """
        if self.df is None:
            self.logger.error("Dados não carregados. Execute load_excel primeiro.")
            raise ValueError("Dados não carregados")
        
        try:
            # Cria cópia para não alterar os dados originais
            df = self.df.copy()
            
            # Limpeza básica
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna('').astype(str).str.strip()
                else:
                    df[col] = df[col].fillna('')
            
            # Combina os campos de texto
            df['texto_combinado'] = df['Natureza Despesa (Nome)'] + ' ' + df['Orientação Natureza']
            df['texto_combinado'] = df['texto_combinado'].str.strip()
            
            # Simplifica os nomes das colunas
            df = df.rename(columns={
                'Natureza Despesa (Codigo)': 'codigo',
                'Natureza Despesa (Nome)': 'nome',
                'Descrição Despesa': 'orientacao',
                'Orientação Natureza': 'orientacao'
            })
            
            # Garante que os códigos sejam strings
            df['codigo'] = df['codigo'].astype(str)
            
            # Cria mapeamentos para códigos (para uso em algoritmos de ML)
            unique_codes = df['codigo'].unique()
            self.label_mapping = {code: idx for idx, code in enumerate(unique_codes)}
            self.inverse_mapping = {idx: code for code, idx in self.label_mapping.items()}
            
            # Adiciona coluna com o código transformado em índice numérico
            df['label'] = df['codigo'].map(self.label_mapping)
            
            self.logger.info(f"Dados pré-processados. {len(df)} registros. {len(unique_codes)} códigos únicos.")
            self.df = df
            return df
        
        except Exception as e:
            self.logger.error(f"Erro no pré-processamento: {str(e)}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame = None) -> bool:
        """
        Salva os dados processados.
        
        Args:
            df: DataFrame a ser salvo. Se None, usa o DataFrame atual.
            
        Returns:
            bool: True se o salvamento foi bem-sucedido, False caso contrário
        """
        try:
            if df is None:
                if self.df is None:
                    self.logger.error("Sem dados para salvar")
                    return False
                df = self.df
            
            # Salva o DataFrame como CSV
            csv_path = os.path.join(self.data_path, 'natureza_despesa.csv')
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Salva o DataFrame como pickle para preservar tipos de dados
            pkl_path = os.path.join(self.data_path, 'natureza_despesa.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(df, f)
            
            # Salva os mapeamentos de códigos
            mappings = {
                'label_mapping': self.label_mapping,
                'inverse_mapping': self.inverse_mapping
            }
            
            map_path = os.path.join(self.data_path, 'code_mappings.json')
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Dados processados salvos em {self.data_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar dados processados: {str(e)}")
            return False
    
    def load_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Carrega dados processados anteriormente.
        
        Returns:
            Optional[pd.DataFrame]: DataFrame com os dados processados ou None se falhar
        """
        try:
            # Tenta carregar o arquivo pickle primeiro
            pkl_path = os.path.join(self.data_path, 'natureza_despesa.pkl')
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    df = pickle.load(f)
                
                self.df = df
                
                # Carrega também os mapeamentos
                map_path = os.path.join(self.data_path, 'code_mappings.json')
                if os.path.exists(map_path):
                    with open(map_path, 'r', encoding='utf-8') as f:
                        mappings = json.load(f)
                    
                    self.label_mapping = {k: int(v) if isinstance(v, str) else v 
                                         for k, v in mappings['label_mapping'].items()}
                    self.inverse_mapping = {int(k) if isinstance(k, str) else k: v 
                                           for k, v in mappings['inverse_mapping'].items()}
                
                self.logger.info(f"Dados processados carregados com {len(df)} registros")
                return df
            
            # Se não encontrar o pickle, tenta o CSV
            csv_path = os.path.join(self.data_path, 'natureza_despesa.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, encoding='utf-8')
                self.df = df
                
                # Carrega também os mapeamentos
                map_path = os.path.join(self.data_path, 'code_mappings.json')
                if os.path.exists(map_path):
                    with open(map_path, 'r', encoding='utf-8') as f:
                        mappings = json.load(f)
                    
                    self.label_mapping = {k: int(v) if isinstance(v, str) else v 
                                         for k, v in mappings['label_mapping'].items()}
                    self.inverse_mapping = {int(k) if isinstance(k, str) else k: v 
                                           for k, v in mappings['inverse_mapping'].items()}
                
                self.logger.info(f"Dados processados carregados com {len(df)} registros")
                return df
            
            self.logger.warning("Nenhum arquivo de dados processados encontrado")
            return None
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados processados: {str(e)}")
            return None
    
    def get_code_name_mapping(self) -> Dict[str, str]:
        """
        Retorna um mapeamento de código para nome de natureza.
        
        Returns:
            Dict[str, str]: Mapeamento de código para nome
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis para criar mapeamento")
                return {}
        
        return dict(zip(self.df['codigo'], self.df['nome']))
    
    def get_texto_by_codigo(self, codigo: str) -> Optional[str]:
        """
        Obtém o texto combinado para um determinado código de natureza.
        
        Args:
            codigo: Código da natureza de despesa
            
        Returns:
            Optional[str]: Texto combinado ou None se não encontrado
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis")
                return None
        
        filtered = self.df[self.df['codigo'] == codigo]
        if filtered.empty:
            return None
        
        return filtered['texto_combinado'].iloc[0]
    
    def get_orientacao_by_codigo(self, codigo: str) -> Optional[str]:
        """
        Obtém o texto combinado para um determinado código de natureza.
        
        Args:
            codigo: Código da natureza de despesa
            
        Returns:
            Optional[str]: Texto combinado ou None se não encontrado
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis")
                return None
        
        filtered = self.df[self.df['codigo'] == codigo]
        if filtered.empty:
            return None
        
        return filtered['orientacao'].iloc[0]
        
    def get_all_textos(self) -> List[str]:
        """
        Retorna todos os textos combinados.
        
        Returns:
            List[str]: Lista de todos os textos combinados
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis")
                return []
        
        return self.df['texto_combinado'].tolist()
    
    def get_all_codigos(self) -> List[str]:
        """
        Retorna todos os códigos de natureza.
        
        Returns:
            List[str]: Lista de todos os códigos
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis")
                return []
        
        return self.df['codigo'].unique().tolist()
    
    def get_textos_with_codigos(self) -> List[Tuple[str, str]]:
        """
        Retorna os textos combinados com seus respectivos códigos.
        
        Returns:
            List[Tuple[str, str]]: Lista de tuplas (texto, código)
        """
        if self.df is None:
            if not self.load_processed_data():
                self.logger.error("Sem dados disponíveis")
                return []
        
        return list(zip(self.df['texto_combinado'], self.df['codigo']))