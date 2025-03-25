# models/natureza_evaluator.py

import os
import logging
import pickle
import json
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from config import active_config
from models.embedding_model_base import EmbeddingModelBase
from models.openai_embedding_model import OpenAIEmbeddingModel
from models.claude_embedding_model import ClaudeEmbeddingModel 
from models.gemini_embedding_model import GeminiEmbeddingModel

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaEvaluator:
    """
    Avaliador de natureza de despesa que utiliza o MCASP como base de conhecimento.
    Avalia se uma natureza de despesa é adequada para uma determinada descrição.
    """
    
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa o avaliador de natureza.
        
        Args:
            embedding_provider: Provedor de embeddings ('openai', 'claude', 'gemini')
        """
        self.embedding_provider = embedding_provider
        self.model_path = os.path.join(active_config.MODELS_DIR, f'natureza_evaluator_{embedding_provider}')
        os.makedirs(self.model_path, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Componentes do modelo
        self.embedding_model = self._create_embedding_model()
        self.embeddings = None
        self.texts = None  # Textos do MCASP
        self.natureza_map = {}  # Mapeamento de código para informações da natureza
        self.is_trained = False
        
    def _create_embedding_model(self) -> EmbeddingModelBase:
        """
        Cria o modelo de embedding apropriado para o provedor especificado.
        
        Returns:
            EmbeddingModelBase: Modelo de embedding
        """
        try:
            if self.embedding_provider == 'openai':
                return OpenAIEmbeddingModel()
            elif self.embedding_provider == 'claude':
                return ClaudeEmbeddingModel()
            elif self.embedding_provider == 'gemini':
                return GeminiEmbeddingModel()
            else:
                self.logger.error(f"Provedor de embeddings desconhecido: {self.embedding_provider}. Usando claude como fallback.")
                return ClaudeEmbeddingModel()
        except Exception as e:
            self.logger.error(f"Erro ao criar modelo de embedding: {str(e)}. Usando claude como fallback.")
            return ClaudeEmbeddingModel()
    
    def extract_mcasp_content(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extrai conteúdo do PDF do MCASP relacionado às naturezas de despesa.
        
        Args:
            file_path: Caminho para o arquivo PDF do MCASP
            
        Returns:
            List[Dict[str, Any]]: Lista de dicionários com informações de natureza
        """
        from utils.document_utils import extract_text_from_file
        
        self.logger.info(f"Iniciando extração de conteúdo do MCASP: {file_path}")
        
        try:
            # Extrai o texto completo do PDF
            texto_completo = extract_text_from_file(file_path)
            
            if not texto_completo:
                self.logger.error(f"Não foi possível extrair texto do arquivo {file_path}")
                return []
                
            # Log do tamanho do texto extraído para verificação
            self.logger.info(f"Texto extraído com sucesso: {len(texto_completo)} caracteres")
            
            # Verificar se temos conteúdo suficiente para processamento
            if len(texto_completo) < 1000:
                self.logger.warning(f"Texto extraído é muito curto: apenas {len(texto_completo)} caracteres")
                # Salvar o texto em um arquivo para diagnóstico
                debug_file = os.path.join(os.path.dirname(file_path), 'mcasp_debug.txt')
                try:
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(texto_completo)
                    self.logger.info(f"Texto extraído salvo em {debug_file} para diagnóstico")
                except Exception as e:
                    self.logger.error(f"Erro ao salvar arquivo de debug: {str(e)}")
            
            # Processa o texto para extrair informações sobre naturezas de despesa
            # Primeiro, divide por seções
            naturezas_data = []
            
            # Vamos usar expressões regulares para identificar padrões de natureza de despesa
            
            # Padrão básico para códigos de natureza (pode precisar de ajustes)
            # Procura por padrões como "3.3.90.30 - Material de Consumo"
            natureza_pattern = r'(\d\.\d\.\d{1,2}\.\d{1,2})\s*[-–]\s*([^\n]+)'
            matches = re.finditer(natureza_pattern, texto_completo)
            
            match_count = 0
            for match in matches:
                match_count += 1
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                
                # Tenta extrair a descrição completa (contexto após o título)
                pos_inicio = match.end()
                next_match = re.search(natureza_pattern, texto_completo[pos_inicio:])
                
                if next_match:
                    descricao = texto_completo[pos_inicio:pos_inicio + next_match.start()].strip()
                else:
                    # Se for o último item, pega até 1000 caracteres após
                    descricao = texto_completo[pos_inicio:pos_inicio + 1000].strip()
                
                # Limita a descrição para não ficar muito longa
                descricao = descricao[:1000]
                
                naturezas_data.append({
                    'codigo': codigo,
                    'nome': nome,
                    'descricao': descricao,
                    'texto_completo': f"{codigo} - {nome}\n{descricao}"
                })
                
                if match_count <= 5 or match_count % 20 == 0:
                    self.logger.info(f"Natureza extraída: {codigo} - {nome}")
            
            # Se a extração regular falhar, tente uma abordagem mais básica
            if len(naturezas_data) == 0:
                self.logger.warning("Nenhuma natureza encontrada com padrão regular. Tentando método alternativo.")
                
                # Procurar por linhas que contenham padrões numéricos que parecem códigos de natureza
                basic_pattern = r'\d\.\d\.\d{1,2}\.\d{1,2}'
                lines = texto_completo.split('\n')
                for i, line in enumerate(lines):
                    if re.search(basic_pattern, line):
                        # Se encontrou um possível código, cria uma entrada básica
                        match = re.search(basic_pattern, line)
                        codigo = match.group(0)
                        # Tenta extrair o nome (o resto da linha após o código)
                        nome = line[match.end():].strip()
                        if not nome:
                            nome = "Natureza não especificada"
                            
                        # Se tiver próximas linhas, usa como descrição
                        descricao = ""
                        if i < len(lines) - 1:
                            descricao = lines[i+1]
                            
                        naturezas_data.append({
                            'codigo': codigo,
                            'nome': nome,
                            'descricao': descricao,
                            'texto_completo': f"{codigo} - {nome}\n{descricao}"
                        })
                
                self.logger.info(f"Método alternativo extraiu {len(naturezas_data)} naturezas")
            
            # Se ainda não encontrou nada, cria um conjunto mínimo de naturezas para não quebrar o sistema
            if len(naturezas_data) == 0:
                self.logger.warning("Nenhuma natureza encontrada. Criando conjunto básico de naturezas para fallback.")
                fallback_naturezas = [
                    {
                        'codigo': '3.3.90.30',
                        'nome': 'Material de Consumo',
                        'descricao': 'Despesas com materiais de consumo, como material de expediente, material de limpeza, etc.',
                        'texto_completo': "3.3.90.30 - Material de Consumo\nDespesas com materiais de consumo, como material de expediente, material de limpeza, etc."
                    },
                    {
                        'codigo': '3.3.90.39',
                        'nome': 'Outros Serviços de Terceiros - Pessoa Jurídica',
                        'descricao': 'Despesas com serviços prestados por pessoas jurídicas, como manutenção de equipamentos, serviços de informática, etc.',
                        'texto_completo': "3.3.90.39 - Outros Serviços de Terceiros - Pessoa Jurídica\nDespesas com serviços prestados por pessoas jurídicas, como manutenção de equipamentos, serviços de informática, etc."
                    },
                    {
                        'codigo': '4.4.90.52',
                        'nome': 'Equipamentos e Material Permanente',
                        'descricao': 'Despesas com aquisição de equipamentos e materiais permanentes, como mobiliário, veículos, etc.',
                        'texto_completo': "4.4.90.52 - Equipamentos e Material Permanente\nDespesas com aquisição de equipamentos e materiais permanentes, como mobiliário, veículos, etc."
                    }
                ]
                naturezas_data.extend(fallback_naturezas)
                self.logger.info(f"Adicionadas {len(fallback_naturezas)} naturezas de fallback")
            
            self.logger.info(f"Total de {len(naturezas_data)} naturezas extraídas do MCASP")
            return naturezas_data
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair conteúdo do MCASP: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            
            # Retorna um conjunto mínimo de naturezas para não quebrar o sistema
            fallback_naturezas = [
                {
                    'codigo': '3.3.90.30',
                    'nome': 'Material de Consumo',
                    'descricao': 'Despesas com materiais de consumo, como material de expediente, material de limpeza, etc.',
                    'texto_completo': "3.3.90.30 - Material de Consumo\nDespesas com materiais de consumo, como material de expediente, material de limpeza, etc."
                },
                {
                    'codigo': '3.3.90.39',
                    'nome': 'Outros Serviços de Terceiros - Pessoa Jurídica',
                    'descricao': 'Despesas com serviços prestados por pessoas jurídicas, como manutenção de equipamentos, serviços de informática, etc.',
                    'texto_completo': "3.3.90.39 - Outros Serviços de Terceiros - Pessoa Jurídica\nDespesas com serviços prestados por pessoas jurídicas, como manutenção de equipamentos, serviços de informática, etc."
                },
                {
                    'codigo': '4.4.90.52',
                    'nome': 'Equipamentos e Material Permanente',
                    'descricao': 'Despesas com aquisição de equipamentos e materiais permanentes, como mobiliário, veículos, etc.',
                    'texto_completo': "4.4.90.52 - Equipamentos e Material Permanente\nDespesas com aquisição de equipamentos e materiais permanentes, como mobiliário, veículos, etc."
                }
            ]
            self.logger.info(f"Usando {len(fallback_naturezas)} naturezas de fallback devido a erro na extração")
            return fallback_naturezas
    
    def train(self, mcasp_path: str = None, force_rebuild: bool = False) -> bool:
        """
        Treina o avaliador de natureza usando o MCASP.
        
        Args:
            mcasp_path: Caminho para o arquivo PDF do MCASP.
                        Se None, procura no diretório padrão.
            force_rebuild: Se True, força a recriação dos embeddings mesmo se já existirem
            
        Returns:
            bool: True se o treinamento foi bem-sucedido
        """
        try:
            # Define o caminho padrão se não for fornecido
            if mcasp_path is None:
                mcasp_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp', 'mcasp.pdf')
                
            # Verificação detalhada do arquivo
            if not os.path.exists(mcasp_path):
                self.logger.error(f"Arquivo MCASP não encontrado em {mcasp_path}")
                return False
            
            # Verificar tamanho do arquivo para garantir que não está vazio
            file_size = os.path.getsize(mcasp_path)
            if file_size == 0:
                self.logger.error(f"Arquivo MCASP em {mcasp_path} está vazio (0 bytes)")
                return False
                
            self.logger.info(f"Arquivo MCASP encontrado em {mcasp_path}, tamanho: {file_size/1024:.2f} KB")
            
            # Verifica se já existem embeddings salvos
            if not force_rebuild:
                embeddings, texts, metadata = self.embedding_model.load_embeddings()
                if embeddings is not None and texts is not None and metadata is not None:
                    # Verifica se os metadados contêm informações sobre naturezas
                    if 'natureza_map' in metadata:
                        self.embeddings = embeddings
                        self.texts = texts
                        self.natureza_map = metadata['natureza_map']
                        self.is_trained = True
                        self.logger.info(f"Modelo carregado com {len(self.texts)} textos e {len(self.natureza_map)} naturezas")
                        return True
            
            # Se chegou aqui, precisa extrair
# Se chegou aqui, precisa extrair do MCASP e gerar embeddings
            self.logger.info("Extraindo naturezas do MCASP...")
            naturezas_data = self.extract_mcasp_content(mcasp_path)
            
            if not naturezas_data:
                self.logger.error("Nenhuma natureza extraída do MCASP")
                return False
            
            # Mapeia códigos para informações de natureza
            self.natureza_map = {item['codigo']: item for item in naturezas_data}
            
            # Extrai textos para geração de embeddings
            texts = [item['texto_completo'] for item in naturezas_data]
            
            # Gera embeddings
            try:
                self.logger.info(f"Gerando embeddings com provedor {self.embedding_provider}")
                embeddings = self.embedding_model.get_embeddings(texts)
                
                # Salva os embeddings
                self.embedding_model.save_embeddings(
                    embeddings, 
                    texts, 
                    {"natureza_map": self.natureza_map}
                )
                
                self.embeddings = embeddings
                self.texts = texts
                self.is_trained = True
                
                # Salva o modelo
                self.save_model()
                
                self.logger.info(f"Avaliador treinado com {len(naturezas_data)} naturezas")
                return True
                
            except Exception as e:
                if "openai.error" in str(e) or "api key" in str(e).lower() or "authentication" in str(e).lower():
                    self.logger.error(f"Erro de API do OpenAI: {str(e)}")
                    self.logger.info("Verificando a chave API...")
                    
                    # Tentar verificar a chave de API
                    if hasattr(self.embedding_model, 'validate_api_key'):
                        if not self.embedding_model.validate_api_key():
                            self.logger.error("A chave de API não é válida ou não está configurada")
                        else:
                            self.logger.info("Chave de API validada, o erro pode ser relacionado a limites ou conexão")
                    
                    # Tentar fallback para outro modelo de embeddings
                    self.logger.info("Tentando fallback para o modelo local de embeddings...")
                    try:
                        from sentence_transformers import SentenceTransformer
                        
                        self.logger.info("Inicializando modelo local de embeddings")
                        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
                        
                        self.logger.info(f"Embeddings gerados com modelo local: {embeddings.shape}")
                        
                        # Atualiza o estado do modelo
                        self.embeddings = embeddings
                        self.texts = texts
                        self.is_trained = True
                        
                        # Salva o modelo (usando nova estrutura para evitar conflitos)
                        self.save_model()
                        
                        self.logger.info(f"Avaliador treinado com modelo local: {len(naturezas_data)} naturezas")
                        return True
                        
                    except Exception as fallback_error:
                        self.logger.error(f"Erro no fallback para modelo local: {str(fallback_error)}")
                else:
                    self.logger.error(f"Erro ao gerar embeddings: {str(e)}")
                
                # Log do traceback completo
                self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
                return False
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento do avaliador: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def evaluate(self, descricao: str, natureza_codigo: str) -> Dict[str, Any]:
        """
        Avalia se uma natureza de despesa é adequada para uma descrição.
        
        Args:
            descricao: Descrição da despesa
            natureza_codigo: Código da natureza a ser avaliada
            
        Returns:
            Dict[str, Any]: Resultado da avaliação com score e justificativa
        """
        if not self.is_trained:
            # Tenta carregar o modelo se não estiver treinado
            if not self.load_model():
                self.logger.error("Modelo não treinado e não foi possível carregar. Tentando treinar...")
                # Tenta treinar com o caminho padrão
                mcasp_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp', 'mcasp.pdf')
                if not self.train(mcasp_path):
                    raise ValueError("Modelo não treinado e não foi possível treinar automaticamente")
        
        try:
            # Verifica se a natureza existe no mapa
            if natureza_codigo not in self.natureza_map:
                return {
                    'is_valid': False,
                    'score': 0.0,
                    'justificativa': f"Natureza {natureza_codigo} não encontrada na base de conhecimento.",
                    'natureza_info': None
                }
            
            # Obtém informações da natureza
            natureza_info = self.natureza_map[natureza_codigo]
            
            # Gera embedding para a descrição
            try:
                query_embedding = self.embedding_model.get_embeddings([descricao])
            except Exception as e:
                self.logger.error(f"Erro ao gerar embedding para consulta: {str(e)}")
                # Tenta usar o modelo local como fallback
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                    query_embedding = model.encode([descricao], show_progress_bar=False, convert_to_numpy=True)
                    self.logger.info("Embedding de consulta gerado com modelo local de fallback")
                except Exception as fallback_error:
                    self.logger.error(f"Erro no fallback para modelo local: {str(fallback_error)}")
                    # Se tudo falhar, retorna uma avaliação neutra
                    return {
                        'is_valid': True,  # Assume válido para não bloquear o fluxo
                        'score': 0.5,      # Score neutro
                        'justificativa': f"Não foi possível avaliar a adequação da natureza {natureza_codigo} devido a erros no processamento.",
                        'natureza_info': natureza_info
                    }
# Busca pelas naturezas mais relevantes para a descrição
            from sklearn.preprocessing import normalize
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Normaliza os embeddings para comparação de similaridade de cosseno
            normalized_query = normalize(query_embedding)
            normalized_embeddings = normalize(self.embeddings)
            
            # Calcula a similaridade entre a descrição e todas as naturezas
            similarities = cosine_similarity(normalized_query, normalized_embeddings)[0]
            
            # Encontra o índice da natureza solicitada
            natureza_idx = None
            for i, text in enumerate(self.texts):
                if natureza_codigo in text:
                    natureza_idx = i
                    break
            
            if natureza_idx is None:
                self.logger.warning(f"Índice para natureza {natureza_codigo} não encontrado nos embeddings")
                # Busca alternativa mais flexível
                for i, text in enumerate(self.texts):
                    if re.search(re.escape(natureza_codigo.replace('.', r'\.')), text):
                        natureza_idx = i
                        self.logger.info(f"Índice encontrado com busca flexível: {i}")
                        break
                
                # Se ainda não encontrou, usa comparação de similaridade direta
                if natureza_idx is None:
                    natureza_embed = self.embedding_model.get_embeddings([natureza_info['texto_completo']])
                    natureza_similarities = cosine_similarity(normalize(natureza_embed), normalized_embeddings)[0]
                    natureza_idx = np.argmax(natureza_similarities)
                    self.logger.info(f"Índice determinado por similaridade direta: {natureza_idx}")
            
            # Similaridade da natureza solicitada
            natureza_similarity = similarities[natureza_idx] if natureza_idx is not None else 0.0
            
            # Encontra o índice da natureza mais similar
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            # Determina o código da natureza mais similar
            best_code = None
            for code, info in self.natureza_map.items():
                if info['texto_completo'] == self.texts[best_idx]:
                    best_code = code
                    break
            
            # Se não encontrou o código, busca por substring
            if best_code is None:
                for code in self.natureza_map.keys():
                    if code in self.texts[best_idx]:
                        best_code = code
                        break
            
            # Fallback se ainda não encontrou
            if best_code is None:
                self.logger.warning(f"Não foi possível determinar o código da natureza mais similar")
                # Usa o primeiro código como fallback
                best_code = list(self.natureza_map.keys())[0] if self.natureza_map else natureza_codigo
            
            # Calcula um score relativo (quanto mais próximo de 1, melhor)
            # Se a natureza solicitada for a mais similar, o score é 1.0
            # Caso contrário, é a razão entre a similaridade da natureza solicitada e a melhor similaridade
            if natureza_idx == best_idx:
                score = 1.0
                justificativa = f"A natureza {natureza_codigo} é a mais adequada para a descrição fornecida."
                is_valid = True
            else:
                score = natureza_similarity / best_similarity if best_similarity > 0 else 0.0
                
                # Classifica a adequação
                if score >= 0.9:
                    is_valid = True
                    justificativa = f"A natureza {natureza_codigo} é adequada para a descrição fornecida."
                elif score >= 0.7:
                    is_valid = True
                    justificativa = f"A natureza {natureza_codigo} é aceitável para a descrição, mas {best_code} pode ser mais adequada."
                else:
                    is_valid = False
                    justificativa = f"A natureza {natureza_codigo} não parece adequada para a descrição. A natureza {best_code} seria mais apropriada."
            
            return {
                'is_valid': is_valid,
                'score': float(score),
                'justificativa': justificativa,
                'natureza_info': natureza_info,
                'best_match': {
                    'codigo': best_code,
                    'similarity': float(best_similarity),
                    'info': self.natureza_map.get(best_code)
                } if best_code != natureza_codigo else None
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao avaliar natureza: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            # Retorna um resultado neutro para não interromper o fluxo
            return {
                'is_valid': True,  # Assume válido por padrão em caso de erro
                'score': 0.5,
                'justificativa': f"Ocorreu um erro ao avaliar a natureza. Recomendamos revisão manual.",
                'natureza_info': self.natureza_map.get(natureza_codigo),
                'error': str(e)
            }
    def save_model(self) -> bool:
        """
        Salva o modelo treinado.
        
        Returns:
            bool: True se o salvamento foi bem-sucedido, False caso contrário
        """
        if not self.is_trained:
            self.logger.error("Tentativa de salvar modelo não treinado")
            return False
        
        try:
            # Cria diretório se não existir
            os.makedirs(self.model_path, exist_ok=True)
            
            # Salva informações básicas para reconstrução
            model_info = {
                'embedding_provider': self.embedding_provider,
                'created_at': datetime.now().isoformat(),
                'natureza_count': len(self.natureza_map) if self.natureza_map else 0,
                'texts_count': len(self.texts) if self.texts else 0,
                'embeddings_shape': self.embeddings.shape if hasattr(self.embeddings, 'shape') else None
            }
            
            # Salva informações do modelo
            info_path = os.path.join(self.model_path, 'model_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            # Salva o mapeamento de naturezas
            data_path = os.path.join(self.model_path, 'natureza_map.json')
            with open(data_path, 'w', encoding='utf-8') as f:
                # Converte para JSON serializável (remove objetos numpy)
                json_map = {}
                for codigo, info in self.natureza_map.items():
                    json_map[codigo] = {k: v for k, v in info.items()}
                json.dump(json_map, f, ensure_ascii=False, indent=2)
            
            # Salva os embeddings e textos diretamente se não foram salvos pelo modelo de embedding
            try:
                embeddings_file = os.path.join(self.model_path, 'embeddings.npy')
                np.save(embeddings_file, self.embeddings)
                
                texts_file = os.path.join(self.model_path, 'texts.json')
                with open(texts_file, 'w', encoding='utf-8') as f:
                    json.dump(self.texts, f, ensure_ascii=False, indent=2)
                    
                self.logger.info(f"Embeddings e textos salvos diretamente em {self.model_path}")
            except Exception as e:
                self.logger.warning(f"Erro ao salvar embeddings e textos diretamente: {str(e)}")
                self.logger.info("Os embeddings podem ter sido salvos pelo modelo de embedding")
            
            self.logger.info(f"Modelo salvo com sucesso em {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def load_model(self) -> bool:
        """
        Carrega um modelo salvo anteriormente.
        
        Returns:
            bool: True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            # Verifica se os arquivos existem
            info_path = os.path.join(self.model_path, 'model_info.json')
            data_path = os.path.join(self.model_path, 'natureza_map.json')
            
            if not all(os.path.exists(p) for p in [info_path, data_path]):
                # Se os arquivos específicos não existem, tenta carregar embeddings diretamente
                embeddings, texts, metadata = self.embedding_model.load_embeddings()
                if embeddings is not None and texts is not None and metadata is not None:
                    if 'natureza_map' in metadata:
                        self.embeddings = embeddings
                        self.texts = texts
                        self.natureza_map = metadata['natureza_map']
                        self.is_trained = True
                        self.logger.info(f"Modelo carregado dos embeddings com {len(self.texts)} textos")
                        return True
                
                self.logger.warning(f"Arquivos do modelo não encontrados em {self.model_path}")
                
                # Tente carregar os embeddings e textos salvos diretamente
                embeddings_file = os.path.join(self.model_path, 'embeddings.npy')
                texts_file = os.path.join(self.model_path, 'texts.json')
                
                if os.path.exists(embeddings_file) and os.path.exists(texts_file):
                    self.logger.info("Tentando carregar embeddings e textos diretos...")
                    try:
                        self.embeddings = np.load(embeddings_file)
                        with open(texts_file, 'r', encoding='utf-8') as f:
                            self.texts = json.load(f)
                            
                        # Se temos embeddings e textos mas não o mapa, tenta criar um mapa básico
                        if not self.natureza_map:
                            self.logger.info("Criando mapeamento básico a partir dos textos...")
                            self.natureza_map = {}
                            for text in self.texts:
                                # Extrai o código do texto (assumindo formato padrão)
                                match = re.search(r'(\d\.\d\.\d{1,2}\.\d{1,2})', text)
                                if match:
                                    codigo = match.group(1)
                                    # Tenta extrair o nome
                                    parts = text.split(' - ', 1)
                                    nome = parts[1].split('\n')[0] if len(parts) > 1 else "Natureza de Despesa"
                                    self.natureza_map[codigo] = {
                                        'codigo': codigo,
                                        'nome': nome,
                                        'texto_completo': text
                                    }
                            
                            self.logger.info(f"Criado mapeamento básico com {len(self.natureza_map)} naturezas")
                            
                        self.is_trained = True
                        return True
                    except Exception as e:
                        self.logger.error(f"Erro ao carregar embeddings e textos diretos: {str(e)}")
                
                return False
            
            # Carrega informações do modelo
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            # Carrega o mapeamento de naturezas
            with open(data_path, 'r', encoding='utf-8') as f:
                self.natureza_map = json.load(f)
            
            # Carrega os embeddings pelo método normal
            embeddings, texts, _ = self.embedding_model.load_embeddings()
            if embeddings is not None and texts is not None:
                self.embeddings = embeddings
                self.texts = texts
                self.logger.info(f"Embeddings carregados via modelo de embedding")
            else:
                # Tenta carregar os embeddings salvos diretamente
                embeddings_file = os.path.join(self.model_path, 'embeddings.npy')
                texts_file = os.path.join(self.model_path, 'texts.json')
                
                if os.path.exists(embeddings_file) and os.path.exists(texts_file):
                    self.logger.info("Carregando embeddings e textos diretos...")
                    self.embeddings = np.load(embeddings_file)
                    with open(texts_file, 'r', encoding='utf-8') as f:
                        self.texts = json.load(f)
                else:
                    self.logger.warning("Embeddings não encontrados. O modelo pode não funcionar corretamente.")
                    
                    # Cria embeddings vazios para evitar erros
                    # Cria embeddings vazios para evitar erros
                    if self.natureza_map:
                        self.logger.info("Criando textos a partir do mapeamento de naturezas...")
                        self.texts = [info.get('texto_completo', f"{code} - Natureza de Despesa") 
                                     for code, info in self.natureza_map.items()]
                        
                        # Tenta gerar embeddings em tempo real
                        try:
                            self.logger.info("Gerando embeddings a partir dos textos...")
                            self.embeddings = self.embedding_model.get_embeddings(self.texts)
                            self.logger.info(f"Embeddings gerados com sucesso")
                        except Exception as e:
                            self.logger.error(f"Erro ao gerar embeddings: {str(e)}")
                            # Cria embeddings vazios com a dimensão correta
                            embedding_dim = 1536 if self.embedding_provider == 'openai' else 384
                            self.embeddings = np.zeros((len(self.texts), embedding_dim))
            
            self.is_trained = True
            
            self.logger.info(f"Modelo carregado com sucesso: {len(self.natureza_map)} naturezas")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            self.logger.error(f"Traceback completo:\n{traceback.format_exc()}")
            return False
    
    def get_all_naturezas(self) -> List[Dict[str, Any]]:
        """
        Retorna todas as naturezas disponíveis no modelo.
        
        Returns:
            List[Dict[str, Any]]: Lista de naturezas com seus detalhes
        """
        if not self.is_trained:
            # Tenta carregar o modelo
            if not self.load_model():
                self.logger.warning("Modelo não treinado. Retornando lista vazia.")
                return []
        
        result = []
        for codigo, info in self.natureza_map.items():
            result.append({
                'codigo': codigo,
                'nome': info['nome'],
                'descricao': info.get('descricao', '')
            })
        
        # Ordena por código
        result.sort(key=lambda x: x['codigo'])
        return result