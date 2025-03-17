# models/openai_embedding_model.py

import os
import numpy as np
from typing import List
import time
from tqdm import tqdm
from config import active_config
from models.embedding_model_base import EmbeddingModelBase

class OpenAIEmbeddingModel(EmbeddingModelBase):
    """
    Implementação do modelo de embeddings usando a API da OpenAI.
    """
    
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        """
        Inicializa o modelo de embeddings da OpenAI.
        
        Args:
            model_name: Nome do modelo de embeddings da OpenAI
        """
        super().__init__(model_name)
        self.api_key = active_config.OPENAI_API_KEY
        self.embedding_dim = 1536  # Dimensão dos embeddings do Ada-002
    
    def initialize(self) -> bool:
        """
        Inicializa o cliente da OpenAI.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        if not self.validate_api_key():
            self.logger.error("API key da OpenAI não configurada")
            return False
        
        try:
            from openai import OpenAI
            
            # Inicializa o cliente da OpenAI
            self.client = OpenAI(api_key=self.api_key)
            
            self.logger.info(f"Cliente OpenAI inicializado para modelo {self.model_name}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar cliente OpenAI: {str(e)}")
            return False
    
    def get_embeddings(self, texts: List[str], batch_size: int = 20) -> np.ndarray:
        """
        Gera embeddings para os textos fornecidos usando a API da OpenAI,
        resumindo automaticamente textos longos quando necessário.
        
        Args:
            texts: Lista de textos para gerar embeddings
            batch_size: Tamanho do lote para processamento em batch
            
        Returns:
            np.ndarray: Matriz de embeddings com shape (n_texts, embedding_dim)
        """
        if self.client is None and not self.initialize():
            self.logger.error("Cliente OpenAI não inicializado")
            raise ValueError("Cliente OpenAI não inicializado. Verifique a API key.")
        
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
            
            # Limite seguro para embeddings (deixamos uma margem)
            TOKEN_LIMIT = 8000
            
            # Função para contar tokens precisamente
            def count_tokens(text):
                return len(encoding.encode(text))
            
            # Função para resumir texto usando o modelo GPT
            def summarize_text(text, token_count):
                try:
                    # Calculamos a taxa de compressão necessária
                    compression_ratio = TOKEN_LIMIT / token_count
                    
                    # Criamos uma instrução apropriada para o resumo
                    prompt = f"""
                    Resumo o texto abaixo preservando ao máximo as informações essenciais sobre classificação orçamentária, 
                    natureza de despesa e quaisquer códigos ou identificadores importantes.
                    O resumo deve ter aproximadamente {int(len(text) * compression_ratio)} caracteres, 
                    mantendo os termos técnicos e específicos da área de orçamento público.
                    
                    TEXTO:
                    {text}
                    
                    RESUMO:
                    """
                    
                    # Fazemos a chamada à API para resumir
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Usando GPT 3.5 para economizar custos 
                        messages=[
                            {"role": "system", "content": "Você é um especialista em finanças públicas e orçamento governamental."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=2000,
                        temperature=0.3  # Baixa temperatura para resumos mais concisos e precisos
                    )
                    
                    summarized_text = response.choices[0].message.content.strip()
                    self.logger.info(f"Texto resumido de {token_count} tokens para {count_tokens(summarized_text)} tokens")
                    return summarized_text
                except Exception as e:
                    self.logger.error(f"Erro ao resumir texto: {str(e)}")
                    # Em caso de erro, fazemos um truncamento simples como fallback
                    return text[:int(len(text) * (TOKEN_LIMIT / token_count))]
            
            # Pré-processamento dos textos - verificar e resumir longos
            processed_texts = []
            for i, text in enumerate(texts):
                token_count = count_tokens(text)
                
                if token_count > TOKEN_LIMIT:
                    self.logger.warning(f"Texto {i+1}/{len(texts)} excede o limite de tokens ({token_count}/{TOKEN_LIMIT}). Resumindo...")
                    processed_texts.append(summarize_text(text, token_count))
                else:
                    processed_texts.append(text)
            
            all_embeddings = []
            
            # Processa em lotes para evitar limites de requisição
            for i in tqdm(range(0, len(processed_texts), batch_size), desc="Gerando embeddings OpenAI"):
                batch_texts = processed_texts[i:i + batch_size]
                
                # Faz a chamada à API para gerar embeddings
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                
                # Extrai os embeddings
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Pausa para respeitar limites de taxa
                if i + batch_size < len(processed_texts):
                    time.sleep(0.5)
            
            # Converte para array numpy
            embeddings_array = np.array(all_embeddings)
            
            self.logger.info(f"Gerados {len(all_embeddings)} embeddings com dimensão {embeddings_array.shape[1]}")
            return embeddings_array
        
        except Exception as e:
            self.logger.error(f"Erro ao gerar embeddings OpenAI: {str(e)}")
            raise
            """
            Gera embeddings para os textos fornecidos usando a API da OpenAI.
            
            Args:
                texts: Lista de textos para gerar embeddings
                batch_size: Tamanho do lote para processamento em batch
                
            Returns:
                np.ndarray: Matriz de embeddings com shape (n_texts, embedding_dim)
            """
            if self.client is None and not self.initialize():
                self.logger.error("Cliente OpenAI não inicializado")
                raise ValueError("Cliente OpenAI não inicializado. Verifique a API key.")
            
            try:
                all_embeddings = []
                
                # Processa em lotes para evitar limites de requisição
                for i in tqdm(range(0, len(texts), batch_size), desc="Gerando embeddings OpenAI"):
                    batch_texts = texts[i:i + batch_size]
                    
                    # Faz a chamada à API
                    response = self.client.embeddings.create(
                        model=self.model_name,
                        input=batch_texts
                    )
                    
                    # Extrai os embeddings
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    # Pausa para respeitar limites de taxa
                    if i + batch_size < len(texts):
                        time.sleep(0.5)
                
                # Converte para array numpy
                embeddings_array = np.array(all_embeddings)
                
                self.logger.info(f"Gerados {len(all_embeddings)} embeddings com dimensão {self.embedding_dim}")
                return embeddings_array
            
            except Exception as e:
                self.logger.error(f"Erro ao gerar embeddings OpenAI: {str(e)}")
                raise