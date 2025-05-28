# models/natureza_evaluator.py

import os
import logging
import pickle
import json
import re
import traceback
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage

from config import active_config
from models.ai_model import AIModel  # Importando a classe base abstrata
from utils.document_utils import extract_text_from_file

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaEvaluator(AIModel):
    """
    Avaliador de natureza de despesa usando RAG (Retrieval-Augmented Generation).
    Utiliza o MCASP como base de conhecimento para avaliar se uma natureza é adequada.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o avaliador de natureza.
        
        Args:
            model_name: Nome do modelo de linguagem a ser utilizado
                        Se None, usa o modelo padrão configurado
        """
        model_type = model_name or active_config.DEFAULT_MODEL
        # Se model_name for 'claude', 'openai' ou 'gemini', usamos o modelo padrão desse tipo
        if model_type in ['claude', 'openai', 'gemini']:
            if model_type == 'claude':
                model_name = active_config.CLAUDE_MODEL
            elif model_type == 'openai':
                model_name = active_config.OPENAI_MODEL
            elif model_type == 'gemini':
                model_name = active_config.GEMINI_MODEL
        
        # Chamada ao construtor da classe pai (AIModel)
        super().__init__(model_name)
        self.model_type = model_type
        
        # Caminho específico para modelo de avaliador
        self.model_path = os.path.join(active_config.MODELS_DIR, f'natureza_evaluator_{model_type}')
        os.makedirs(self.model_path, exist_ok=True)
        
        # Chain para o fluxo RAG
        self.chain = None
        
        # Inicializa o modelo de linguagem
        if not self.initialize():
            logger.warning(f"Falha ao inicializar modelo {model_type}. Algumas funcionalidades podem não estar disponíveis.")
        
        # Dicionário para mapear códigos de natureza para informações adicionais
        self.natureza_map = {}
        
        # Armazena documentos originais
        self.original_documents = []
        
        # Flag para indicar se o modelo foi treinado
        self.is_trained = False
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        try:
            # Importa o módulo correto baseado no tipo de modelo
            if self.model_type == 'claude':
                from langchain_anthropic import ChatAnthropic
                
                api_key = active_config.CLAUDE_API_KEY
                if not api_key:
                    logger.error("API key do Claude não configurada")
                    return False
                
                self.llm = ChatAnthropic(
                    model=self.model_name,
                    anthropic_api_key=api_key,
                    temperature=0.0,
                    max_tokens=4096
                )
                
            elif self.model_type == 'openai':
                from langchain_openai import ChatOpenAI
                
                api_key = active_config.OPENAI_API_KEY
                if not api_key:
                    logger.error("API key da OpenAI não configurada")
                    return False
                
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=api_key,
                    temperature=0.0,
                    max_tokens=4096
                )
                
            elif self.model_type == 'gemini':
                from langchain_google_genai import ChatGoogleGenerativeAI
                
                api_key = active_config.GEMINI_API_KEY
                if not api_key:
                    logger.error("API key do Google Gemini não configurada")
                    return False
                
                self.llm = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=api_key,
                    temperature=0.0,
                    max_output_tokens=4096
                )
            
            else:
                logger.error(f"Tipo de modelo não suportado: {self.model_type}")
                return False
            
            logger.info(f"Modelo {self.model_type} inicializado com sucesso: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inicializar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key está configurada para o provedor especificado.
        
        Returns:
            bool: True se a chave estiver configurada, False caso contrário
        """
        if self.model_type == 'claude':
            return active_config.CLAUDE_API_KEY is not None and len(active_config.CLAUDE_API_KEY) > 0
        elif self.model_type == 'openai':
            return active_config.OPENAI_API_KEY is not None and len(active_config.OPENAI_API_KEY) > 0
        elif self.model_type == 'gemini':
            return active_config.GEMINI_API_KEY is not None and len(active_config.GEMINI_API_KEY) > 0
        return False
    
    def _get_embeddings_model(self):
        """
        Obtém o modelo de embeddings apropriado para o provedor.
        
        Returns:
            O modelo de embeddings correspondente
        """
        if self.model_type == 'openai':
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(
                openai_api_key=active_config.OPENAI_API_KEY,
                model="text-embedding-ada-002"
            )
        else:
            # Para Claude e Gemini (ou qualquer outro), usamos Hugging Face
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
    
    def train(self, documents_path: str = None, force_rebuild: bool = False) -> bool:
        """
        Treina o avaliador usando o MCASP.
        
        Args:
            documents_path: Caminho para o arquivo PDF do MCASP ou diretório.
                          Se None, procura no diretório padrão.
            force_rebuild: Se True, força a recriação dos vetores mesmo se já existirem.
            
        Returns:
            bool: True se o treinamento foi bem-sucedido, False caso contrário
        """
        try:
            # Define o caminho padrão se não for fornecido
            if documents_path is None:
                mcasp_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp', 'mcasp.pdf')
                if os.path.exists(mcasp_path):
                    documents_path = mcasp_path
                else:
                    documents_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp')
            
            # Verifica se o arquivo/diretório existe
            if not os.path.exists(documents_path):
                logger.error(f"Caminho não encontrado: {documents_path}")
                return False
            
            # Tenta carregar modelo existente a menos que force_rebuild seja True
            if not force_rebuild and self.load_model():
                logger.info("Modelo existente carregado com sucesso")
                return True
            
            # Inicializa o LLM se ainda não estiver inicializado
            if self.llm is None and not self.initialize():
                logger.error("Falha ao inicializar o modelo de linguagem")
                return False
            
            # Carrega documentos
            logger.info(f"Carregando documentos de {documents_path}")
            documents = self._load_documents(documents_path)
            
            if not documents:
                logger.error("Nenhum documento encontrado para treinamento")
                return False
            
            # Armazena os documentos originais para uso futuro
            self.original_documents = documents
            
            # Divide os documentos em chunks
            logger.info("Dividindo documentos em chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Extrai informações de natureza de despesa
            self._extract_natureza_info(documents)
            
            # Inicializa os embeddings
            logger.info(f"Inicializando embeddings para {self.model_type}")
            embeddings = self._get_embeddings_model()
            
            # Cria vectorstore
            logger.info("Criando vectorstore com FAISS")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Cria cadeia de recuperação e pergunta
            self.chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                verbose=True
            )
            
            self.is_trained = True
            
            # Salva o modelo treinado
            self.save_model()
            
            logger.info(f"Avaliador de natureza treinado com sucesso, utilizando {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Erro durante treinamento do avaliador: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _load_documents(self, documents_path: str) -> List[Any]:
        """
        Carrega documentos de um arquivo ou diretório.
        
        Args:
            documents_path: Caminho para o arquivo ou diretório
            
        Returns:
            List[Any]: Lista de documentos carregados
        """
        documents = []
        
        # Verifica se é um arquivo ou diretório
        if os.path.isfile(documents_path):
            # Verifica a extensão do arquivo
            if documents_path.lower().endswith('.pdf'):
                try:
                    # Carrega o PDF
                    loader = PyPDFLoader(documents_path)
                    documents.extend(loader.load())
                    logger.info(f"Carregado arquivo PDF com {len(documents)} páginas")
                except Exception as e:
                    logger.error(f"Erro ao carregar PDF: {str(e)}")
            else:
                # Para outros tipos de arquivo, tenta extrair texto geral
                try:
                    text = extract_text_from_file(documents_path)
                    if text:
                        # Cria um documento no formato esperado pelo LangChain
                        from langchain.schema import Document
                        documents.append(Document(
                            page_content=text,
                            metadata={"source": documents_path}
                        ))
                        logger.info(f"Carregado arquivo como texto genérico")
                except Exception as e:
                    logger.error(f"Erro ao extrair texto do arquivo: {str(e)}")
        
        elif os.path.isdir(documents_path):
            # Carrega todos os arquivos PDF no diretório
            from langchain_community.document_loaders import DirectoryLoader
            
            # Tenta carregar PDFs
            try:
                pdf_loader = DirectoryLoader(
                    documents_path, 
                    glob="**/*.pdf", 
                    loader_cls=PyPDFLoader
                )
                pdf_docs = pdf_loader.load()
                documents.extend(pdf_docs)
                logger.info(f"Carregados {len(pdf_docs)} documentos PDF do diretório")
            except Exception as e:
                logger.error(f"Erro ao carregar PDFs do diretório: {str(e)}")
            
            # Tenta carregar TXT
            try:
                from langchain_community.document_loaders import TextLoader
                txt_loader = DirectoryLoader(
                    documents_path, 
                    glob="**/*.txt", 
                    loader_cls=TextLoader
                )
                txt_docs = txt_loader.load()
                documents.extend(txt_docs)
                logger.info(f"Carregados {len(txt_docs)} documentos TXT do diretório")
            except Exception as e:
                logger.error(f"Erro ao carregar TXTs do diretório: {str(e)}")
        
        return documents
    
    def _extract_natureza_info(self, documents: List[Any]) -> None:
        """
        Extrai informações detalhadas sobre naturezas de despesa, considerando a estrutura hierárquica.
        
        Args:
            documents: Lista de documentos carregados
        """
        try:
            import re
            
            # Combina o texto de todos os documentos
            all_text = ""
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    all_text += doc.page_content + "\n\n"
            
            # Procura por padrões mais específicos da estrutura de classificação
            # Busca padrões para categorias econômicas
            categoria_pattern = r'([1-4])\s*[-–]\s*([^\.]+?)(?=\n|\d\s*[-–])'
            categoria_matches = re.finditer(categoria_pattern, all_text)
            
            categorias = {}
            for match in categoria_matches:
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                categorias[codigo] = nome
                
            # Busca padrões para grupos de natureza
            grupo_pattern = r'([1-4]\.[1-9])\s*[-–]\s*([^\.]+?)(?=\n|\d\.\d\s*[-–])'
            grupo_matches = re.finditer(grupo_pattern, all_text)
            
            grupos = {}
            for match in grupo_matches:
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                grupos[codigo] = nome
                
            # Busca padrões para modalidades de aplicação
            modalidade_pattern = r'([1-4]\.[1-9]\.\d{2})\s*[-–]\s*([^\.]+?)(?=\n|\d\.\d\.\d{2}\s*[-–])'
            modalidade_matches = re.finditer(modalidade_pattern, all_text)
            
            modalidades = {}
            for match in modalidade_matches:
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                modalidades[codigo] = nome
                
            # Busca padrões para elementos de despesa (até quatro níveis)
            elemento_pattern = r'([1-4]\.[1-9]\.\d{2}\.\d{2})\s*[-–]\s*([^\.]+?)(?=\n|\d\.\d\.\d{2}\.\d{2}\s*[-–])'
            elemento_matches = re.finditer(elemento_pattern, all_text)
            
            elementos = {}
            for match in elemento_matches:
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                elementos[codigo] = nome
            
            # Procura por padrões completos (incluindo desdobramentos)
            natureza_pattern = r'(\d\.\d\.\d{2}\.\d{2}(?:\.\d{2})?)\s*[-–]\s*([^\n]+)'
            matches = re.finditer(natureza_pattern, all_text)
            
            # Mapeia códigos para nomes
            for match in matches:
                codigo = match.group(1).strip()
                nome = match.group(2).strip()
                
                # Normaliza o código para ignorar os dois últimos dígitos (dd)
                codigo_norm = codigo
                partes = codigo.split('.')
                if len(partes) > 4:  # Se tiver o desdobramento
                    codigo_norm = '.'.join(partes[:4])  # Mantém apenas c.g.mm.ee
                
                # Registra a natureza no mapa
                self.natureza_map[codigo_norm] = {
                    'codigo': codigo_norm,
                    'nome': nome,
                    'codigo_completo': codigo,
                    'categoria': partes[0] if len(partes) > 0 else "",
                    'grupo': '.'.join(partes[:2]) if len(partes) > 1 else "",
                    'modalidade': '.'.join(partes[:3]) if len(partes) > 2 else "",
                    'elemento': '.'.join(partes[:4]) if len(partes) > 3 else "",
                    'desdobramento': partes[4] if len(partes) > 4 else ""
                }
            
            # Combina todas as estruturas extraídas
            for codigo, nome in categorias.items():
                self.natureza_map[codigo] = {'codigo': codigo, 'nome': nome, 'tipo': 'categoria'}
                
            for codigo, nome in grupos.items():
                self.natureza_map[codigo] = {'codigo': codigo, 'nome': nome, 'tipo': 'grupo'}
                
            for codigo, nome in modalidades.items():
                self.natureza_map[codigo] = {'codigo': codigo, 'nome': nome, 'tipo': 'modalidade'}
                
            for codigo, nome in elementos.items():
                if codigo not in self.natureza_map:  # Não sobrescrever se já existe com info completa
                    self.natureza_map[codigo] = {'codigo': codigo, 'nome': nome, 'tipo': 'elemento'}
            
            # Se não encontrou naturezas, cria algumas básicas para referência
            if not self.natureza_map:
                logger.warning("Não foi possível extrair naturezas. Criando mapa básico de referência.")
                basic_naturezas = {
                    '3': 'Despesas Correntes',
                    '4': 'Despesas de Capital',
                    '3.3': 'Outras Despesas Correntes',
                    '4.4': 'Investimentos',
                    '3.3.90': 'Aplicações Diretas',
                    '4.4.90': 'Aplicações Diretas',
                    '3.3.90.30': 'Material de Consumo',
                    '3.3.90.39': 'Outros Serviços de Terceiros - Pessoa Jurídica',
                    '4.4.90.52': 'Equipamentos e Material Permanente'
                }
                
                for codigo, nome in basic_naturezas.items():
                    self.natureza_map[codigo] = {'codigo': codigo, 'nome': nome}
            
            logger.info(f"Extraídas {len(self.natureza_map)} naturezas de despesa para referência")
            
        except Exception as e:
            logger.error(f"Erro ao extrair informações de natureza: {str(e)}")
            logger.error(traceback.format_exc())
    
    def evaluate(self, descricao: str, natureza_codigo: str) -> Dict[str, Any]:
        """
        Avalia se uma natureza de despesa é adequada para uma descrição.
        Versão modificada para normalizar o código, desconsiderando os dois últimos dígitos.
        
        Args:
            descricao: Descrição da despesa
            natureza_codigo: Código da natureza a ser avaliada
                
        Returns:
            Dict[str, Any]: Resultado da avaliação com score e justificativa
        """
        if not self.is_trained:
            # Tenta carregar o modelo se não estiver treinado
            if not self.load_model():
                logger.error("Modelo não treinado e não foi possível carregar. Tentando treinar...")
                # Tenta treinar com o caminho padrão
                mcasp_path = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp', 'mcasp.pdf')
                if not self.train(mcasp_path):
                    raise ValueError("Modelo não treinado e não foi possível treinar automaticamente")
        
        try:
            # Normalizar o código (remover os últimos dois dígitos - dd)
            partes = natureza_codigo.split('.')
            codigo_norm = natureza_codigo
            if len(partes) > 4:  # Se tiver o desdobramento
                codigo_norm = '.'.join(partes[:4])  # Mantém apenas c.g.mm.ee
            
            # Formatação do nome da natureza (se disponível)
            natureza_nome = ""
            if codigo_norm in self.natureza_map:
                natureza_nome = self.natureza_map[codigo_norm].get('nome', '')
            
            # Construção da query para o modelo
            query = f"""
            Avalie se a natureza de despesa "{natureza_codigo}{' - ' + natureza_nome if natureza_nome else ''}" 
            é adequada para a seguinte descrição:
            
            Descrição da despesa: {descricao}
            
            Com base no Manual de Contabilidade Aplicada ao Setor Público (MCASP), 
            forneça uma avaliação detalhada sobre a adequação desta natureza para a descrição.
            IMPORTANTE: Desconsidere os dois últimos dígitos (dd) do código ao fazer a avaliação,
            considerando apenas a estrutura c.g.mm.ee (categoria econômica, grupo, modalidade, elemento).
            
            Na sua resposta, inclua:
            1. Se a natureza é adequada ou não (dê uma classificação clara: adequada, parcialmente adequada ou inadequada)
            2. Uma justificativa detalhada com base no MCASP
            3. Um score numérico de adequação de 0 a 1, onde 1 significa totalmente adequada
            4. Se a natureza não for adequada, sugira uma natureza mais apropriada se possível
            
            Responda em um formato estruturado que permita a extração fácil das informações.
            """
            
            # Faz a consulta ao modelo
            result = self.chain.invoke({"query": query})
            
            # Processa a resposta para extrair as informações estruturadas
            response = result['result'] if isinstance(result, dict) and 'result' in result else result
            
            # Parseamento da resposta
            response_lower = response.lower()
            
            # Determina se é válido
            is_valid = False
            if "adequada" in response_lower and not ("não adequada" in response_lower or "inadequada" in response_lower):
                is_valid = True
            elif "parcialmente adequada" in response_lower:
                is_valid = True  # Consideramos parcialmente adequada como válida, mas com score menor
            
            # Tenta extrair um score numérico
            score_match = re.search(r'score.*?(\d+[.,]\d+|\d+)', response_lower)
            score = 0.5  # Valor padrão se não encontrar
            
            if score_match:
                try:
                    score_str = score_match.group(1).replace(',', '.')
                    score = float(score_str)
                    # Normaliza o score para 0-1 se estiver em outra escala
                    if score > 1:
                        score = score / 10 if score <= 10 else score / 100
                except ValueError:
                    logger.warning(f"Não foi possível converter o score '{score_match.group(1)}' para float")
            
            # Para naturezas parcialmente adequadas, ajusta o score
            if "parcialmente adequada" in response_lower:
                # Ajusta para um valor intermediário se não tiver um score explícito
                if not score_match:
                    score = 0.7
            
            # Tenta encontrar uma natureza sugerida
            best_match = None
            
            # Procura por padrões como "3.3.90.30" no texto da resposta
            alternative_codes = re.findall(r'\d\.\d\.\d{1,2}\.\d{1,2}', response)
            
            # Filtra o código original e sua versão normalizada
            alternative_codes = [code for code in alternative_codes 
                                if code != natureza_codigo and code != codigo_norm]
            
            if alternative_codes:
                # Pega o primeiro código alternativo
                alt_code = alternative_codes[0]
                
                # Tenta encontrar o nome do código
                alt_name = ""
                alt_code_norm = '.'.join(alt_code.split('.')[:4])  # Normaliza
                if alt_code_norm in self.natureza_map:
                    alt_name = self.natureza_map[alt_code_norm].get('nome', '')
                else:
                    # Tenta extrair o nome do texto da resposta
                    name_pattern = f"{alt_code}\\s*[-–]\\s*([^\n.,]+)"
                    name_match = re.search(name_pattern, response)
                    if name_match:
                        alt_name = name_match.group(1).strip()
                
                best_match = {
                    'codigo': alt_code,
                    'similarity': 0.9,  # Valor simbólico, já que foi sugerido pelo modelo
                    'info': {
                        'codigo': alt_code,
                        'nome': alt_name
                    }
                }
            
            # Extrai a justificativa, considerando o texto após "justificativa" ou "justificação"
            justificativa = response
            justificativa_match = re.search(r'(?:justificativa|justificação).*?:(.*?)(?:\d\.|$)', response_lower, re.DOTALL)
            if justificativa_match:
                justificativa = justificativa_match.group(1).strip()
            
            # Se a justificativa ainda for muito longa, tenta extrair um resumo
            if len(justificativa) > 500:
                justificativa = justificativa[:500] + "..."
            
            return {
                'is_valid': is_valid,
                'score': float(score),
                'justificativa': justificativa,
                'natureza_info': self.natureza_map.get(codigo_norm, {'codigo': codigo_norm, 'nome': natureza_nome}),
                'best_match': best_match,
                'full_response': response  # Incluído para referência e depuração
            }
            
        except Exception as e:
            logger.error(f"Erro ao avaliar natureza: {str(e)}")
            logger.error(traceback.format_exc())
            # Retorna um resultado neutro para não interromper o fluxo
            return {
                'is_valid': True,  # Assume válido por padrão em caso de erro
                'score': 0.5,
                'justificativa': f"Ocorreu um erro ao avaliar a natureza. Recomendamos revisão manual.",
                'natureza_info': self.natureza_map.get(natureza_codigo, {'codigo': natureza_codigo, 'nome': ''}),
                'error': str(e)
            }
                
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Método obrigatório para compatibilidade com a classe base AIModel.
        Gera uma resposta para a consulta do usuário.
        
        Args:
            query: Consulta do usuário
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        if not self.is_trained or self.chain is None:
            return "O avaliador de natureza ainda não foi treinado. Por favor, realize o treinamento primeiro."
        
        try:
            # Converte o histórico para o formato esperado, se fornecido
            history = []
            if chat_history:
                for msg in chat_history:
                    if msg.get('role') == 'user':
                        history.append({"role": "human", "content": msg.get('content', '')})
                    elif msg.get('role') == 'assistant':
                        history.append({"role": "ai", "content": msg.get('content', '')})
            
            # Usa a cadeia para responder à consulta
            result = self.chain.invoke({"query": query})
            
            return result['result'] if isinstance(result, dict) and 'result' in result else result
        
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
    
    def save_model(self, model_name: str = None) -> bool:
        """
        Salva o modelo treinado.
        
        Args:
            model_name: Nome opcional para o modelo
            
        Returns:
            bool: True se o salvamento foi bem-sucedido, False caso contrário
        """
        if not self.is_trained or self.vectorstore is None:
            logger.error("Tentativa de salvar modelo não treinado")
            return False
        
        try:
            # Define nome do modelo se não fornecido
            if model_name is None:
                model_name = f"natureza_evaluator_{self.model_type}"
            
            # Caminho para salvar o modelo
            model_dir = os.path.join(active_config.MODELS_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Salva o vectorstore
            vectorstore_dir = os.path.join(model_dir, "vectorstore")
            os.makedirs(vectorstore_dir, exist_ok=True)
            
            # Usa o método save_local do FAISS
            self.vectorstore.save_local(vectorstore_dir)
            
            # Salva o mapeamento de naturezas
            natureza_path = os.path.join(model_dir, "natureza_map.json")
            with open(natureza_path, 'w', encoding='utf-8') as f:
                json.dump(self.natureza_map, f, ensure_ascii=False, indent=2)
            
            # Informações do modelo
            model_info = {
                'model_type': self.model_type,
                'model_name': self.model_name,
                'created_at': datetime.now().isoformat(),
                'natureza_count': len(self.natureza_map),
                'embedding_type': 'openai' if self.model_type == 'openai' else 'huggingface'
            }
            
            # Salva informações do modelo
            info_path = os.path.join(model_dir, "model_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            # Salva metadata em formato pickle para compatibilidade
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(model_info, f)
            
            # Atualiza o caminho do modelo
            self.model_path = model_dir
            
            logger.info(f"Modelo salvo com sucesso em {model_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def load_model(self, model_path: str = None) -> bool:
        """
        Carrega um modelo salvo anteriormente.
        
        Args:
            model_path: Caminho para o modelo. Se None, usa o caminho padrão.
            
        Returns:
            bool: True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            # Define o caminho padrão se não fornecido
            if model_path is None:
                model_path = self.model_path
            
            # Verifica se o diretório existe
            if not os.path.exists(model_path):
                logger.error(f"Caminho do modelo não existe: {model_path}")
                return False
            
            # Tenta carregar informações do modelo
            info_path = os.path.join(model_path, "model_info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    model_info = json.load(f)
                    
                # Atualiza as propriedades do modelo
                if 'model_type' in model_info:
                    self.model_type = model_info['model_type']
                if 'model_name' in model_info:
                    self.model_name = model_info['model_name']
            
            # Inicializa o LLM se ainda não estiver inicializado
            if self.llm is None and not self.initialize():
                logger.error("Falha ao inicializar o modelo de linguagem")
                return False
            
            # Carrega o mapeamento de naturezas
            natureza_path = os.path.join(model_path, "natureza_map.json")
            if os.path.exists(natureza_path):
                with open(natureza_path, 'r', encoding='utf-8') as f:
                    self.natureza_map = json.load(f)
            
            # Carrega os embeddings apropriados para o modelo
            embeddings = self._get_embeddings_model()
            
            # Caminho para o vectorstore
            vectorstore_dir = os.path.join(model_path, "vectorstore")
            if not os.path.exists(vectorstore_dir):
                vectorstore_dir = model_path  # Compatibilidade com versões anteriores
            
            # Verifica se existe index.faiss
            index_path = os.path.join(vectorstore_dir, "index.faiss")
            if not os.path.exists(index_path):
                logger.error(f"Arquivo index.faiss não encontrado em {vectorstore_dir}")
                return False
            
            # Carrega o vectorstore
            try:
                self.vectorstore = FAISS.load_local(
                    vectorstore_dir, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                
                # Cria cadeia de recuperação e pergunta
                self.chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": 5}
                    ),
                    return_source_documents=True,
                    verbose=True
                )
                
                self.is_trained = True
                self.model_path = model_path
                
                logger.info(f"Modelo carregado com sucesso de {model_path}")
                return True
                
            except Exception as e:
                logger.error(f"Erro ao carregar vectorstore: {str(e)}")
                logger.error(traceback.format_exc())
                return False
                
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def get_all_naturezas(self) -> List[Dict[str, Any]]:
        """
        Retorna uma lista de todas as naturezas disponíveis no mapa.
        
        Returns:
            List[Dict[str, Any]]: Lista com informações de cada natureza
        """
        result = []
        for codigo, info in self.natureza_map.items():
            result.append({
                'codigo': codigo,
                'nome': info.get('nome', ''),
                'descricao': info.get('descricao', '')
            })
        
        # Ordena por código
        result.sort(key=lambda x: x['codigo'])
        return result# models/natureza_evaluator.py

    def evaluate_candidates(self, descricao: str, candidatos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Avalia múltiplos candidatos de natureza de despesa para uma descrição.
        
        Args:
            descricao: Descrição da despesa
            candidatos: Lista de dicionários, cada um com 'codigo' e 'confianca'
                
        Returns:
            Dict[str, Any]: Resultado da avaliação com candidatos ranqueados e recomendação
        """
        if not self.is_trained:
            if not self.load_model():
                raise ValueError("Modelo não treinado e não foi possível carregar.")
        
        try:
            # Normalizar os códigos de candidatos (remover os últimos dois dígitos - dd)
            candidatos_normalizados = []
            codigos_originais = []  # Lista simples para verificação rápida
            for candidato in candidatos:
                codigo = candidato['codigo']
                codigos_originais.append(codigo)
                # Divide por pontos e remove os últimos dois dígitos se houver
                partes = codigo.split('.')
                if len(partes) >= 4:  # Certifica que tem pelo menos c.g.mm.ee
                    codigo_norm = '.'.join(partes[:4])  # Mantém apenas c.g.mm.ee
                else:
                    codigo_norm = codigo  # Mantém o código original se não tiver formato completo
                
                candidatos_normalizados.append({
                    'codigo_original': codigo,
                    'codigo_norm': codigo_norm,
                    'confianca': candidato.get('confianca', 0.0),
                    'nome': candidato.get('nome', '')
                })
            
            # Construir a consulta para o modelo
            candidatos_str = ""
            for i, cand in enumerate(candidatos_normalizados, 1):
                nome = cand['nome'] if cand['nome'] else self.natureza_map.get(cand['codigo_norm'], {}).get('nome', '')
                candidatos_str += f"{i}. {cand['codigo_original']} - {nome} (Confiança: {cand['confianca']:.2%})\n"
            
            query = f"""
            Avalie qual das seguintes naturezas de despesa é mais adequada para a descrição:
            
            Descrição da despesa: {descricao}
            
            Candidatos de natureza de despesa:
            {candidatos_str}
            
            Com base no Manual de Contabilidade Aplicada ao Setor Público (MCASP), 
            forneça uma avaliação detalhada sobre qual natureza é mais adequada. 

            ATENÇÃO: Sua resposta deve se basear EXCLUSIVAMENTE nas informações do Manual de Contabilidade 
            Aplicada ao Setor Público (MCASP).
            NÃO invente, complete ou sugira códigos que não estejam explicitamente no MCASP.
            Limite-se a códigos até o nível de elemento (c.g.mm.ee) sem subelemento.
            Se você não tiver certeza, indique claramente que a informação não está disponível no MCASP.

            IMPORTANTE: Desconsidere os dois últimos dígitos (dd) do código ao fazer a avaliação,
            considerando apenas a estrutura c.g.mm.ee (categoria econômica, grupo, modalidade, elemento).
                        
            Na sua resposta, inclua os seguintes itens de forma esturuturada, obedecendo o seguinte padrão:
            
            Item 1. Um ranking numerado das naturezas candidatas, da mais adequada para a menos adequada, com os 
            subitens iniciando com 1.1 para a mais relevante, 1.2 para a segunda mais relevante e 1.3 para a menos relevante
            
            Item 2. Justificativa: uma justificativa baseada APENAS em texto explícito do MCASP indicando quando possível
            a página ou seção do MCASP onde a informação foi encontrada
            
            Item 3. Se nenhuma das naturezas candidatas for adequada, sugira uma natureza mais apropriada
            especificando o código c.g.mm.ee, porém ela deve aparecer textualmente no MCASP. Esta sugestão deverá ser
            feita somente se nenhuma das naturezas candidatas estiver entre as mais adequadas. Se uma das naturezas
            candidatas estiver entre as mais adequadas, deverá ser retornado "Item 3. Sem sugestão"

            
            """
            
            # Faz a consulta ao modelo
            result = self.chain.invoke({"query": query})
            
            # Processa a resposta
            response = result['result'] if isinstance(result, dict) and 'result' in result else result
            
            # ABORDAGEM COMPLETAMENTE NOVA: Extração rigorosa da ordenação do modelo
            
            # 1. Primeiro tenta localizar um ranking numerado explícito
            # Padrão: "1. 3.3.90.30", "2. 4.4.90.52", etc.
            ranking_explicito = []
            ranking_pattern = r'(?:^|\n)\s*(\d+)\s*\.\s*(\d\.\d\.\d{1,2}\.\d{1,2})'
            for match in re.finditer(ranking_pattern, response):
                posicao = int(match.group(1))
                codigo = match.group(2)
                if codigo in codigos_originais:
                    ranking_explicito.append((posicao, codigo))
            
            # 2. Se encontrou ranking explícito, vamos usá-lo
            if ranking_explicito:
                # Ordena pelo número de posição
                ranking_explicito.sort(key=lambda x: x[0])
                ordered_codigos = [codigo for _, codigo in ranking_explicito]
                
                # Adiciona qualquer código que não estava no ranking explícito ao final
                for codigo in codigos_originais:
                    if codigo not in ordered_codigos:
                        ordered_codigos.append(codigo)
            
            # 3. Se não encontrou ranking explícito, tenta pela primeira ocorrência de cada código
            else:
                # Lista para armazenar a posição da primeira ocorrência de cada código
                primeira_ocorrencia = []
                
                for codigo in codigos_originais:
                    match = re.search(re.escape(codigo), response)
                    if match:
                        primeira_ocorrencia.append((match.start(), codigo))
                    else:
                        # Se o código não for encontrado, coloca no final com uma posição alta
                        primeira_ocorrencia.append((len(response) + 1, codigo))
                
                # Ordena pela posição da primeira ocorrência
                primeira_ocorrencia.sort(key=lambda x: x[0])
                ordered_codigos = [codigo for _, codigo in primeira_ocorrencia]
            
            # Cria o ranking final usando a ordem determinada
            ranking = []
            codigos_to_info = {c['codigo_original']: c for c in candidatos_normalizados}
            
            # Agora vamos construir o ranking respeitando a ordem determinada
            for i, codigo in enumerate(ordered_codigos, 1):
                info = codigos_to_info.get(codigo, {})
                # Score baseado exclusivamente na posição
                score = max(0.1, 1.0 - (i-1) * 0.15)  # Decai mais rapidamente com a posição
                
                ranking.append({
                    'codigo': codigo,
                    'codigo_norm': info.get('codigo_norm', codigo),
                    'score': score,
                    'position': i,
                    'nome': info.get('nome', ''),
                    'confianca_original': info.get('confianca', 0.0)
                })
            
            # Modificação para o método evaluate_candidates da classe NaturezaEvaluator
            # Este código deve substituir o trecho que trata dos códigos alternativos (aproximadamente linhas 971-999)

            # Procura por uma natureza alternativa sugerida
            alternative_codes = re.findall(r'\d\.\d\.\d{1,2}\.\d{1,2}', response)

            # Filtra códigos de candidatos
            alternative_codes = [code for code in alternative_codes 
                                if code not in codigos_originais]

            # Verifica se o modelo indica explicitamente que nenhuma natureza é adequada
            nenhuma_adequada = False
            padroes_inadequacao = [
                r'nenhuma\s+das\s+naturezas\s+(?:é|parece|seria|se\s+mostra)\s+adequada',
                r'nenhuma\s+das\s+opções\s+(?:é|parece|seria|se\s+mostra)\s+adequada',
                r'nenhum\s+dos\s+candidatos\s+(?:é|parece|seria|se\s+mostra)\s+adequad[oa]',
                r'não\s+há\s+natureza\s+adequada\s+entre\s+as\s+(?:opções|candidatas)',
                r'sugiro\s+uma\s+natureza\s+(?:alternativa|diferente)',
                r'recomendo\s+utilizar\s+(?:outra|uma\s+natureza\s+diferente)',
                r'(?:todas|ambas)\s+as\s+naturezas\s+(?:são|estão|parecem)\s+inadequadas',
                r'nenhuma\s+(?:se adequa|está adequada|corresponde)'
            ]

            for padrao in padroes_inadequacao:
                if re.search(padrao, response.lower()):
                    nenhuma_adequada = True
                    logger.info(f"Detectada indicação de que nenhuma natureza é adequada usando padrão: {padrao}")
                    break

            best_alternative = None
            if alternative_codes and nenhuma_adequada:
                alt_code = alternative_codes[0]
                
                # Tenta encontrar o nome do código
                alt_name = ""
                alt_code_norm = '.'.join(alt_code.split('.')[:4])  # Normaliza
                if alt_code_norm in self.natureza_map:
                    alt_name = self.natureza_map[alt_code_norm].get('nome', '')
                else:
                    # Tenta extrair o nome do texto da resposta
                    name_pattern = f"{alt_code}\\s*[-–]\\s*([^\n.,]+)"
                    name_match = re.search(name_pattern, response)
                    if name_match:
                        alt_name = name_match.group(1).strip()
                
                best_alternative = {
                    'codigo': alt_code,
                    'codigo_norm': alt_code_norm,
                    'score': 0.9 if nenhuma_adequada else 0.6,  # Score mais alto se nenhuma é adequada
                    'nome': alt_name,
                    'sugerido_pelo_modelo': True,
                    'e_alternativa': True
                }
                
                logger.info(f"Código alternativo sugerido: {alt_code} - {alt_name}")

            # Modificar também a lógica de recomendação:
            # Determina a recomendação final baseada na ordem
            recommended = ranking[0] if ranking else None

            # Só recomenda a alternativa se nenhuma das originais for adequada
            if best_alternative and nenhuma_adequada:
                # Se o modelo indicou explicitamente que nenhuma é adequada, usa a alternativa
                recommended = best_alternative
                logger.info(f"Recomendando alternativa {best_alternative['codigo']} por indicação explícita de inadequação")
            
            # Extrai a justificativa
            justificativa_pattern = justificativa_pattern = r'(?:^|\n)\s*(?:Item\s*)?2\.\s*(?:justificativa|justificação).*?:?(.*?)(?=(?:^|\n)\s*(?:Item\s*)?3\.|\Z)' #r'(?:^|\n)\s*2\.(?:Item|*)\s*(?:justificativa|justificação).*?:?(.*?)(?:(?:^|\n)\s*3\.|\Z)'
            justificativa_match = re.search(justificativa_pattern, response, re.IGNORECASE | re.DOTALL)

            if justificativa_match:
                justificativa = justificativa_match.group(1).strip()
            else:
                # Tenta outra abordagem se o padrão específico não for encontrado
                alt_pattern = r'(?:justificativa|justificação).*?:(.*?)(?:(?:^|\n)\s*\d\.|\Z)'
                alt_match = re.search(alt_pattern, response.lower(), re.DOTALL)
                if alt_match:
                    justificativa = alt_match.group(1).strip()
                else:
                    justificativa = response

            # Limpa linhas em branco e espaços extras
            justificativa = re.sub(r'\n\s*\n', '\n', justificativa)

            # Se a justificativa ainda for muito longa, tenta extrair um resumo
            if len(justificativa) > 800:
                last_sentence_end = justificativa[:800].rfind('.')
                if last_sentence_end > 0:
                    justificativa = justificativa[:last_sentence_end + 1]
                else:
                    justificativa = justificativa[:800] + "..."
                                
            # Se não encontrou uma justificativa clara, usa um trecho do texto
            if not justificativa or len(justificativa) < 50:
                justificativa = response[:500] + "..." if len(response) > 500 else response
            
            # Determina a recomendação final baseada APENAS na ordem
            recommended = ranking[0] if ranking else None
            
            # Se temos uma alternativa que parece melhor, priorize-a
            if best_alternative and (not recommended or best_alternative['score'] > recommended['score']):
                recommended = best_alternative
            
            # Monta o resultado final
            resultado = {
                'ranking': ranking,
                'justificativa': justificativa,
                'best_alternative': best_alternative if nenhuma_adequada else None,  # Só inclui a alternativa se nenhuma é adequada
                'recommended': recommended,
                'nenhuma_adequada': nenhuma_adequada,  # Nova flag para indicar a inadequação das naturezas originais
                'full_response': response
            }
            
            return resultado
            
        except Exception as e:
            logger.error(f"Erro ao avaliar candidatos: {str(e)}")
            logger.error(traceback.format_exc())
            # Retorna um resultado com os candidatos originais ordenados por confiança
            candidatos_ordenados = sorted(candidatos, key=lambda x: -x.get('confianca', 0))
            return {
                'ranking': [{'codigo': c['codigo'], 'score': c.get('confianca', 0)} for c in candidatos_ordenados],
                'justificativa': f"Ocorreu um erro ao avaliar as naturezas. Recomendamos revisão manual.",
                'best_alternative': None,
                'recommended': candidatos_ordenados[0] if candidatos_ordenados else None,
                'error': str(e)
            }

    def analyze_natureza_options(self, description: str, natureza_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa uma lista de opções de natureza de despesa e seleciona as mais adequadas
        para a descrição fornecida.
        
        Args:
            description: Descrição da despesa
            natureza_options: Lista de opções de natureza de despesa, 
                            cada uma com 'codigo', 'nome' e 'confianca'
            
        Returns:
            List[Dict[str, Any]]: Lista das naturezas mais adequadas, ordenadas por relevância
        """
        pass

    def analyze_natureza_complete(self, description: str, natureza_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa uma lista de opções de natureza de despesa completa (incluindo subelementos)
        e seleciona as mais adequadas para a descrição fornecida.
        
        Args:
            description: Descrição da despesa
            natureza_options: Lista de opções de natureza de despesa completa, 
                            cada uma com 'codigo', 'nome', 'descricao' e outros metadados
            
        Returns:
            List[Dict[str, Any]]: Lista das naturezas mais adequadas, ordenadas por relevância
        """
        pass

    def resume_texto(self, texto: str) -> str:
        """
        Resumir texto usando o modelo de avaliação e extrair informações relevantes.
        
        Args:
            texto: Texto a ser resumido
                
        Returns:
            Dict[str, Any]: Resumo e informações extraídas (valor, meses, datas)
        """
        pass