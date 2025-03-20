import os
import numpy as np
from typing import List, Dict, Any, Optional
import glob
import pickle
import json
import shutil
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

from models.ai_model import AIModel
from config import active_config


class GeminiModel(AIModel):
    """
    Implementação do modelo de IA baseado no Google Gemini.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo Gemini.
        
        Args:
            model_name: Nome do modelo Gemini a ser utilizado
        """
        model_name = model_name or active_config.GEMINI_MODEL
        super().__init__(model_name)
        self.api_key = active_config.GEMINI_API_KEY
        self.llm = None
        self.chain = None
        self.original_documents = None  # Armazena os documentos originais para facilitar reconstrução
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo Gemini e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        if not self.validate_api_key():
            self.logger.error("API key do Google Gemini não configurada")
            return False
        
        try:
            # Inicializa o modelo de linguagem Gemini
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.2,
                max_output_tokens=4096
            )
            
            self.logger.info(f"Modelo Gemini '{self.model_name}' inicializado com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelo Gemini: {str(e)}")
            return False
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key do Google Gemini está configurada.
        
        Returns:
            bool: True se a API key está configurada, False caso contrário
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def train(self, documents_path: str) -> bool:
        """
        Treina o modelo Gemini com os documentos fornecidos.
        
        Args:
            documents_path: Caminho para os documentos de treinamento
            
        Returns:
            bool: True se o treinamento foi bem-sucedido, False caso contrário
        """
        if not os.path.exists(documents_path):
            self.logger.error(f"Caminho de documentos não existe: {documents_path}")
            return False
        
        try:
            # Inicializa o modelo se ainda não estiver inicializado
            if self.llm is None and not self.initialize():
                return False
            
            # Carrega documentos
            self.logger.info("Carregando documentos para treinamento...")
            documents = self._load_documents(documents_path)
            
            if not documents:
                self.logger.error("Nenhum documento encontrado para treinamento")
                return False
            
            # Armazena os documentos originais para uso futuro
            self.original_documents = documents
            
            # Divide documentos em chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Inicializa embeddings
            self.logger.info("Inicializando embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Cria vectorstore
            self.logger.info("Criando vectorstore com FAISS...")
            self.vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Cria cadeia de retrieval conversacional
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                verbose=True
            )
            
            self.is_trained = True
            self.logger.info("Treinamento do modelo Gemini concluído com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro no treinamento do modelo Gemini: {str(e)}")
            return False
    
    def _load_documents(self, documents_path: str) -> List[Any]:
        """
        Carrega documentos de um diretório.
        
        Args:
            documents_path: Caminho para o diretório com documentos
            
        Returns:
            List[Any]: Lista de documentos carregados
        """
        # Define os loaders para diferentes tipos de arquivo
        pdf_loader = DirectoryLoader(
            documents_path, 
            glob="**/*.pdf", 
            loader_cls=PyPDFLoader
        )
        
        docx_loader = DirectoryLoader(
            documents_path, 
            glob="**/*.docx", 
            loader_cls=Docx2txtLoader
        )
        
        txt_loader = DirectoryLoader(
            documents_path, 
            glob="**/*.txt", 
            loader_cls=TextLoader
        )
        
        # Carrega os documentos
        documents = []
        for loader in [pdf_loader, docx_loader, txt_loader]:
            try:
                docs = loader.load()
                self.logger.info(f"Carregados {len(docs)} documentos de {loader.__class__.__name__}")
                documents.extend(docs)
            except Exception as e:
                self.logger.warning(f"Erro ao carregar documentos com {loader.__class__.__name__}: {str(e)}")
        
        return documents
    
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Gera uma resposta para a consulta do usuário.
        
        Args:
            query: Consulta do usuário
            chat_history: Histórico da conversa (opcional)
            
        Returns:
            str: Resposta gerada pelo modelo
        """
        if not self.is_trained or self.chain is None:
            self.logger.error("Tentativa de gerar resposta com modelo não treinado")
            return "O modelo ainda não foi treinado. Por favor, realize o treinamento primeiro."
        
        try:
            # Converte o formato do histórico do chat se fornecido
            langchain_history = []
            if chat_history:
                for msg in chat_history:
                    if msg.get('role') == 'user':
                        langchain_history.append(HumanMessage(content=msg.get('content', '')))
                    elif msg.get('role') == 'assistant':
                        langchain_history.append(AIMessage(content=msg.get('content', '')))
            
            # Gera resposta usando o modelo treinado
            result = self.chain({
                'question': query,
                'chat_history': langchain_history
            })
            
            return result.get('answer', "Não foi possível gerar uma resposta.")
        except Exception as e:
            self.logger.error(f"Erro ao gerar resposta: {str(e)}")
            return f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"

    def save_model(self, model_name: str = None) -> bool:
        """
        Salva o modelo treinado para uso posterior.
        
        Args:
            model_name: Nome para salvar o modelo (opcional)
            
        Returns:
            bool: True se o salvamento foi bem-sucedido, False caso contrário
        """
        if not self.is_trained or self.vectorstore is None:
            self.logger.error("Tentativa de salvar modelo não treinado")
            return False
        
        try:
            if model_name is None:
                model_name = f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Cria o diretório do modelo se não existir
            model_dir = os.path.join(active_config.MODELS_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # 1. Salvar o vectorstore usando o método save_local do FAISS
            self.logger.info(f"Salvando vectorstore em {model_dir}")
            index_path = os.path.join(model_dir, "index.faiss")
            self.vectorstore.save_local(model_dir)
            
            # 2. Salvar os documentos originais
            if self.original_documents:
                self.logger.info("Salvando documentos originais")
                docs_path = os.path.join(model_dir, "original_documents.pkl")
                with open(docs_path, 'wb') as f:
                    pickle.dump(self.original_documents, f)
            
            # 3. Salvar metadados
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            metadata = {
                'model_name': self.model_name,
                'model_type': self.__class__.__name__,
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat(),
                'embedding_model': "sentence-transformers/all-MiniLM-L6-v2",
                'version': 2  # Versão do formato de salvamento
            }
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # 4. Salvar também uma versão em JSON para fácil leitura humana
            metadata_json_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_json_path, 'w', encoding='utf-8') as f:
                # Convertemos datetime para string para poder serializar para JSON
                json_metadata = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in metadata.items()}
                json.dump(json_metadata, f, ensure_ascii=False, indent=2)
            
            self.model_path = model_dir
            self.logger.info(f"Modelo salvo com sucesso em {model_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
            return False
    
    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo treinado anteriormente.
        
        Args:
            model_path: Caminho para o modelo salvo
            
        Returns:
            bool: True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            # Verifica se o diretório existe
            if not os.path.exists(model_path):
                self.logger.error(f"Caminho do modelo não existe: {model_path}")
                return False
            
            # Carrega metadados
            metadata_path = os.path.join(model_path, "metadata.pkl")
            if not os.path.exists(metadata_path):
                self.logger.error(f"Arquivo de metadados não encontrado: {metadata_path}")
                return False
                
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verifica se o tipo de modelo é compatível
            if metadata['model_type'] != self.__class__.__name__:
                self.logger.error(f"Tipo de modelo incompatível: {metadata['model_type']}")
                return False
            
            # Inicializa o modelo de linguagem Gemini
            if self.llm is None and not self.initialize():
                self.logger.error("Falha ao inicializar o modelo Gemini")
                return False
            
            # Inicializa embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Verifica se existe arquivos index.faiss e index.pkl (formato padrão do FAISS.save_local)
            index_faiss_path = os.path.join(model_path, "index.faiss")
            index_pkl_path = os.path.join(model_path, "index.pkl")
            
            if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
                # Carrega o vectorstore usando FAISS.load_local
                self.logger.info("Carregando vectorstore usando FAISS.load_local")
                self.vectorstore = FAISS.load_local(
                    model_path, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Vectorstore carregado com sucesso")
            else:
                # Verifica se temos os documentos originais para reconstruir o vectorstore
                docs_path = os.path.join(model_path, "original_documents.pkl")
                if os.path.exists(docs_path):
                    self.logger.info("Reconstruindo vectorstore a partir dos documentos originais")
                    try:
                        # Carrega os documentos originais
                        with open(docs_path, 'rb') as f:
                            documents = pickle.load(f)
                        
                        # Divide documentos em chunks
                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=1000,
                            chunk_overlap=200,
                            length_function=len,
                        )
                        chunks = text_splitter.split_documents(documents)
                        
                        # Recria o vectorstore
                        self.vectorstore = FAISS.from_documents(chunks, embeddings)
                        self.logger.info("Vectorstore reconstruído com sucesso")
                        
                        # Armazena os documentos originais
                        self.original_documents = documents
                    except Exception as e:
                        self.logger.error(f"Erro ao reconstruir vectorstore: {str(e)}")
                        return False
                else:
                    self.logger.error("Não foi possível encontrar os arquivos necessários para carregar o modelo")
                    return False
            
            # Recria a cadeia de retrieval
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                verbose=True
            )
            
            # Atualiza propriedades do modelo
            self.model_name = metadata['model_name']
            self.is_trained = metadata.get('is_trained', True)
            self.model_path = model_path
            
            self.logger.info(f"Modelo carregado com sucesso de {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return False