import os
from typing import List, Dict, Any, Optional
import glob
import pickle
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    TextLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

from models.ai_model import AIModel
from config import active_config


class OpenAIModel(AIModel):
    """
    Implementação do modelo de IA baseado no OpenAI GPT.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo OpenAI.
        
        Args:
            model_name: Nome do modelo OpenAI a ser utilizado
        """
        model_name = model_name or active_config.OPENAI_MODEL
        super().__init__(model_name)
        self.api_key = active_config.OPENAI_API_KEY
        self.llm = None
        self.chain = None
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo OpenAI e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        if not self.validate_api_key():
            self.logger.error("API key da OpenAI não configurada")
            return False
        
        try:
            # Inicializa o modelo de linguagem OpenAI
            self.llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                temperature=0.2,
                max_tokens=4096
            )
            
            self.logger.info(f"Modelo OpenAI '{self.model_name}' inicializado com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelo OpenAI: {str(e)}")
            return False
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key da OpenAI está configurada.
        
        Returns:
            bool: True se a API key está configurada, False caso contrário
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def train(self, documents_path: str) -> bool:
        """
        Treina o modelo OpenAI com os documentos fornecidos.
        
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
            
            # Divide documentos em chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Inicializa embeddings da OpenAI
            self.logger.info("Inicializando embeddings...")
            embeddings = OpenAIEmbeddings(
                openai_api_key=self.api_key
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
            self.logger.info("Treinamento do modelo OpenAI concluído com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro no treinamento do modelo OpenAI: {str(e)}")
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
        Salva o modelo OpenAI treinado para uso posterior utilizando métodos específicos do FAISS.
        
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
                model_name = f"{self.__class__.__name__}_{self.model_name}"
            
            # Cria o diretório do modelo
            model_dir = os.path.join(active_config.MODELS_DIR, model_name)
            os.makedirs(model_dir, exist_ok=True)
            
            # Verifica se o vectorstore é um objeto FAISS
            if not isinstance(self.vectorstore, FAISS):
                self.logger.error("O vectorstore não é um objeto FAISS")
                return False
            
            # Usar o método save_local do FAISS, que salva os arquivos index.faiss e index.pkl
            self.logger.info("Salvando vectorstore usando FAISS.save_local")
            self.vectorstore.save_local(model_dir)
            
            # Salva metadados com informações específicas da OpenAI
            metadata_path = os.path.join(model_dir, "metadata.pkl")
            
            # Captura informações do embedding para reconstrução posterior
            embedding_info = {
                'model': 'text-embedding-ada-002',  # Valor padrão
                'api_key': None  # Não salvamos a API key por segurança
            }
            
            # Tenta extrair informações específicas do modelo de embedding
            if hasattr(self.vectorstore, '_embeddings'):
                embeddings = self.vectorstore._embeddings
                if hasattr(embeddings, 'model'):
                    embedding_info['model'] = embeddings.model
            
            metadata = {
                'model_name': self.model_name,
                'model_type': self.__class__.__name__,
                'is_trained': self.is_trained,
                'format_version': 3,  # Nova versão do formato
                'embedding_info': embedding_info,
                'saved_with': 'FAISS.save_local'  # Indica método de salvamento
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.model_path = model_dir
            self.logger.info(f"Modelo OpenAI salvo com sucesso em {model_dir} usando FAISS.save_local")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo OpenAI: {str(e)}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo OpenAI treinado anteriormente usando métodos específicos do FAISS.
        
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
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verifica se o tipo de modelo é compatível
            if metadata['model_type'] != self.__class__.__name__:
                self.logger.error(f"Tipo de modelo incompatível: {metadata['model_type']}")
                return False
            
            # Inicializa o modelo de linguagem OpenAI
            if self.llm is None and not self.initialize():
                self.logger.error("Falha ao inicializar o modelo de linguagem OpenAI")
                return False
            
            # Determina o método de carregamento com base nos metadados ou na disponibilidade de arquivos
            saved_with = metadata.get('saved_with', None)
            
            # Tenta criar embeddings da OpenAI para carregar o FAISS
            try:
                # Extrai informações de embedding dos metadados
                embedding_info = metadata.get('embedding_info', {})
                embedding_model = embedding_info.get('model', 'text-embedding-ada-002')
                
                # Inicializa embeddings
                embeddings = OpenAIEmbeddings(
                    openai_api_key=self.api_key,
                    model=embedding_model
                )
                
                self.logger.info(f"Embeddings OpenAI inicializados com modelo: {embedding_model}")
            except Exception as e:
                self.logger.error(f"Erro ao inicializar embeddings OpenAI: {str(e)}")
                return False
            
            # Carrega o vectorstore usando FAISS.load_local
            if saved_with == 'FAISS.save_local' or os.path.exists(os.path.join(model_path, 'index.faiss')):
                try:
                    self.logger.info("Carregando vectorstore usando FAISS.load_local")
                    # Adicionamos o parâmetro allow_dangerous_deserialization=True
                    self.vectorstore = FAISS.load_local(
                        model_path, 
                        embeddings, 
                        allow_dangerous_deserialization=True
                    )
                    self.logger.info("Vectorstore carregado com sucesso usando FAISS.load_local")
                except Exception as e:
                    self.logger.error(f"Erro ao carregar vectorstore usando FAISS.load_local: {str(e)}")
                    return False
            # Código para lidar com formatos legados permanece o mesmo...
            else:
                self.logger.warning("Formato antigo detectado, tentando métodos alternativos de carregamento")
                # ... resto do código de carregamento de formatos legados ...
            
            # Se chegamos aqui, temos um vectorstore carregado
            # Agora criamos a chain
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=retriever,
                    return_source_documents=True,
                    verbose=True
                )
                self.logger.info("Chain criada com sucesso para o modelo OpenAI")
            except Exception as e:
                self.logger.error(f"Erro ao criar chain para o modelo OpenAI: {str(e)}")
                return False
            
            # Atualiza propriedades do modelo
            self.model_name = metadata['model_name']
            self.is_trained = True
            self.model_path = model_path
            
            self.logger.info(f"Modelo OpenAI carregado com sucesso de {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo OpenAI: {str(e)}")
            return False