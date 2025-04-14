import os
import logging
import pickle
import json
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import hashlib

from langchain_anthropic import ChatAnthropic
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


class ClaudeModel(AIModel):
    """
    Implementação otimizada do modelo de IA baseado no Claude da Anthropic.
    Com melhorias no salvamento e carregamento de modelos.
    """
    
    def __init__(self, model_name: str = None):
        """
        Inicializa o modelo Claude.
        
        Args:
            model_name: Nome do modelo Claude a ser utilizado
        """
        model_name = model_name or active_config.CLAUDE_MODEL
        super().__init__(model_name)
        self.api_key = active_config.CLAUDE_API_KEY
        self.llm = None
        self.chain = None
        self.original_documents = None  # Armazena os documentos originais para facilitar reconstrução
        self.embeddings_model = None  # Referência ao modelo de embeddings usado
    
    def initialize(self) -> bool:
        """
        Inicializa o modelo Claude e seus componentes.
        
        Returns:
            bool: True se a inicialização foi bem-sucedida, False caso contrário
        """
        if not self.validate_api_key():
            self.logger.error("API key do Claude não configurada")
            return False
        
        try:
            # Inicializa o modelo de linguagem Claude
            self.llm = ChatAnthropic(
                model=self.model_name,
                anthropic_api_key=self.api_key,
                temperature=0.2,
                max_tokens=4096
            )
            
            self.logger.info(f"Modelo Claude '{self.model_name}' inicializado com sucesso")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao inicializar modelo Claude: {str(e)}")
            return False
    
    def validate_api_key(self) -> bool:
        """
        Verifica se a API key do Claude está configurada.
        
        Returns:
            bool: True se a API key está configurada, False caso contrário
        """
        return self.api_key is not None and len(self.api_key) > 0
    
    def train(self, documents_path: str) -> bool:
        """
        Treina o modelo Claude com os documentos fornecidos.
        
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
            
            # Gera um hash dos documentos para verificação futura
            docs_hash = self._generate_documents_hash(documents)
            
            # Divide documentos em chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(documents)
            
            # Inicializa embeddings
            self.logger.info("Inicializando embeddings...")
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Cria vectorstore
            self.logger.info("Criando vectorstore com FAISS...")
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings_model)
            
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
            self.logger.info(f"Treinamento do modelo Claude concluído com sucesso. Hash dos documentos: {docs_hash}")
            return True
        except Exception as e:
            self.logger.error(f"Erro no treinamento do modelo Claude: {str(e)}")
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
    
    def _generate_documents_hash(self, documents: List[Any]) -> str:
        """
        Gera um hash único para os documentos carregados.
        
        Args:
            documents: Lista de documentos
            
        Returns:
            str: Hash SHA-256 dos documentos
        """
        # Cria um resumo dos documentos para gerar um hash consistente
        doc_summary = ""
        for doc in documents:
            if hasattr(doc, 'page_content'):
                # Adiciona os primeiros 100 caracteres de cada documento
                doc_summary += doc.page_content[:100] + "\n"
            
            if hasattr(doc, 'metadata') and doc.metadata:
                # Adiciona o nome do arquivo se disponível
                if 'source' in doc.metadata:
                    doc_summary += doc.metadata['source'] + "\n"
        
        # Gera um hash SHA-256 do resumo
        hash_obj = hashlib.sha256(doc_summary.encode('utf-8'))
        return hash_obj.hexdigest()
    
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
                # Limita o histórico apenas às últimas 5 mensagens para evitar tokens desnecessários
                recent_history = chat_history[-5:] if len(chat_history) > 5 else chat_history
                
                for msg in recent_history:
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
            
            # Tenta gerar resposta sem usar o histórico em caso de erro
            try:
                if langchain_history:
                    self.logger.info("Tentando gerar resposta sem histórico após erro...")
                    result = self.chain({
                        'question': query,
                        'chat_history': []
                    })
                    return result.get('answer', "Não foi possível gerar uma resposta com o histórico completo.")
            except:
                pass
            
            return f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"

    def save_model(self, model_name: str = None) -> bool:
        """
        Salva o modelo treinado para uso posterior usando método otimizado.
        
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
                # Usa timestamp para garantir unicidade
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_name = f"{self.__class__.__name__}_{timestamp}"
            
            # Cria o diretório do modelo se não existir
            model_dir = os.path.join(active_config.MODELS_DIR, model_name)
            
            # Se já existir, faz backup e recria
            if os.path.exists(model_dir):
                backup_dir = f"{model_dir}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.logger.info(f"Diretório do modelo já existe. Criando backup em {backup_dir}")
                shutil.move(model_dir, backup_dir)
            
            os.makedirs(model_dir, exist_ok=True)
            
            # 1. Salva o FAISS vectorstore diretamente
            vectorstore_dir = os.path.join(model_dir, "vectorstore")
            os.makedirs(vectorstore_dir, exist_ok=True)
            
            self.logger.info(f"Salvando FAISS vectorstore em {vectorstore_dir}")
            self.vectorstore.save_local(vectorstore_dir)
            
            # 2. Salva os documentos originais se disponíveis
            if self.original_documents:
                self.logger.info("Salvando documentos originais...")
                docs_dir = os.path.join(model_dir, "documents")
                os.makedirs(docs_dir, exist_ok=True)
                
                # Salva os documentos
                docs_path = os.path.join(docs_dir, "original_documents.pkl")
                with open(docs_path, 'wb') as f:
                    pickle.dump(self.original_documents, f)
                
                # Gera um arquivo de resumo para referência rápida
                docs_summary = []
                for i, doc in enumerate(self.original_documents[:20]):  # Limita a 20 docs para o resumo
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        summary = {
                            'id': i,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'preview': doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                        }
                        docs_summary.append(summary)
                
                # Salva o resumo em JSON para fácil visualização
                summary_path = os.path.join(docs_dir, "documents_summary.json")
                with open(summary_path, 'w', encoding='utf-8') as f:
                    json.dump(docs_summary, f, ensure_ascii=False, indent=2)
            
            # 3. Salva os metadados do modelo
            self.logger.info("Salvando metadados do modelo...")
            metadata = {
                'model_name': self.model_name,
                'model_type': self.__class__.__name__,
                'is_trained': self.is_trained,
                'saved_at': datetime.now().isoformat(),
                'documents_count': len(self.original_documents) if self.original_documents else 0,
                'embeddings_model': "sentence-transformers/all-MiniLM-L6-v2",
                'save_format_version': 2,
                'vectorstore_type': 'FAISS',
                'vectorstore_path': os.path.relpath(vectorstore_dir, model_dir)
            }
            
            # Salva em formato binário (pickle) para o sistema
            metadata_pkl_path = os.path.join(model_dir, "metadata.pkl")
            with open(metadata_pkl_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Salva em formato JSON para leitura humana
            metadata_json_path = os.path.join(model_dir, "metadata.json")
            with open(metadata_json_path, 'w', encoding='utf-8') as f:
                # Converter datetime para string
                json_metadata = metadata.copy()
                if isinstance(json_metadata.get('saved_at'), datetime):
                    json_metadata['saved_at'] = json_metadata['saved_at'].isoformat()
                
                json.dump(json_metadata, f, ensure_ascii=False, indent=2)
            
            # 4. Cria um arquivo de verificação para validação rápida
            version_file = os.path.join(model_dir, "claude_model_v2.check")
            with open(version_file, 'w') as f:
                f.write(f"Claude Model Verification File\nCreated: {datetime.now().isoformat()}\nModel: {self.model_name}")
            
            self.model_path = model_dir
            self.logger.info(f"Modelo Claude salvo com sucesso em {model_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo Claude: {str(e)}")
            return False

    def load_model(self, model_path: str) -> bool:
        """
        Carrega um modelo Claude treinado anteriormente com método otimizado.
        
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
            
            # Verifica se é um modelo Claude v2
            version_check = os.path.join(model_path, "claude_model_v2.check")
            is_v2_model = os.path.exists(version_check)
            
            # Carrega metadados
            metadata_path = os.path.join(model_path, "metadata.pkl")
            if not os.path.exists(metadata_path):
                self.logger.error(f"Arquivo de metadados não encontrado: {metadata_path}")
                return False
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Verifica compatibilidade do modelo
            if metadata.get('model_type') != self.__class__.__name__:
                self.logger.error(f"Tipo de modelo incompatível: {metadata.get('model_type')}")
                return False
            
            # Inicializa o modelo Claude
            if self.llm is None and not self.initialize():
                self.logger.error("Falha ao inicializar o modelo Claude")
                return False
            
            # Inicializa embeddings
            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            
            # Determina o caminho do vectorstore com base na versão do modelo
            if is_v2_model and metadata.get('save_format_version', 0) >= 2:
                # Para modelos v2, usa o caminho relativo especificado nos metadados
                vectorstore_path = os.path.join(model_path, metadata.get('vectorstore_path', 'vectorstore'))
            else:
                # Para modelos antigos, assume o padrão
                vectorstore_path = model_path
            
            # Tenta carregar o vectorstore
            self.logger.info(f"Carregando vectorstore de {vectorstore_path}")
            
            if os.path.exists(os.path.join(vectorstore_path, 'index.faiss')) and os.path.exists(os.path.join(vectorstore_path, 'index.pkl')):
                # Carrega o vectorstore usando FAISS.load_local
                self.vectorstore = FAISS.load_local(
                    vectorstore_path, 
                    self.embeddings_model,
                    allow_dangerous_deserialization=True
                )
                self.logger.info("Vectorstore FAISS carregado com sucesso")
            else:
                # Se o vectorstore não estiver disponível, verifica se temos os documentos originais
                docs_path = os.path.join(model_path, "documents", "original_documents.pkl")
                
                if os.path.exists(docs_path):
                    self.logger.info("Vectorstore não encontrado. Reconstruindo a partir dos documentos originais...")
                    
                    # Carrega os documentos originais
                    with open(docs_path, 'rb') as f:
                        self.original_documents = pickle.load(f)
                    
                    # Divide documentos em chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                    )
                    chunks = text_splitter.split_documents(self.original_documents)
                    
                    # Cria vectorstore
                    self.vectorstore = FAISS.from_documents(chunks, self.embeddings_model)
                    self.logger.info("Vectorstore reconstruído com sucesso a partir dos documentos originais")
                else:
                    self.logger.error("Não foi possível encontrar nem o vectorstore nem os documentos originais")
                    return False
            
            # Cria a cadeia de retrieval conversacional
            self.chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                verbose=True
            )
            
            # Atualiza propriedades do modelo
            self.model_name = metadata.get('model_name', self.model_name)
            self.is_trained = True
            self.model_path = model_path
            
            self.logger.info(f"Modelo Claude carregado com sucesso de {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo Claude: {str(e)}")
            # Informações adicionais para depuração
            import traceback
            self.logger.error(f"Detalhes do erro:\n{traceback.format_exc()}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Obtém informações sobre o modelo carregado.
        
        Returns:
            Dict[str, Any]: Dicionário com informações do modelo
        """
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'model_path': self.model_path if self.is_trained else None,
            'documents_count': len(self.original_documents) if self.original_documents else 0,
            'embedding_model': "sentence-transformers/all-MiniLM-L6-v2",
            'vectorstore_type': 'FAISS'
        }
        
        # Adiciona informações do vectorstore se disponível
        if self.vectorstore is not None:
            try:
                info['vectorstore_size'] = self.vectorstore.index.ntotal
            except:
                info['vectorstore_size'] = 'Unknown'
        
        return info

    def analyze_natureza_options(self, description: str, natureza_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa uma lista de opções de natureza de despesa usando o modelo Claude
        e seleciona as mais adequadas para a descrição fornecida.
        
        Args:
            description: Descrição da despesa
            natureza_options: Lista de opções de natureza de despesa, 
                            cada uma com 'codigo', 'nome' e 'confianca'
            
        Returns:
            List[Dict[str, Any]]: Lista das naturezas mais adequadas, ordenadas por relevância
        """
        if self.llm is None and not self.initialize():
            self.logger.error("Cliente Claude não inicializado")
            raise ValueError("Cliente Claude não inicializado. Verifique a API key.")
        
        try:
            # Importa o Anthropic client
            import anthropic
            
            # Inicializa o cliente Anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            
            # Monta a lista de opções formatada
            options_text = ""
            for i, option in enumerate(natureza_options, 1):
                codigo = option.get('codigo', '')
                nome = option.get('nome', '')
                confianca = option.get('confianca', 0)
                options_text += f"{i}. {codigo} - {nome} (Confiança inicial: {confianca:.2%})\n"
            
            # Cria o prompt para análise
            prompt = f"""
            Com base na descrição de despesa abaixo, analise as opções de natureza de despesa listadas 
            e escolha as 3 mais adequadas em ordem de relevância. Sua análise deve considerar a 
            classificação orçamentária do setor público brasileiro.
            
            Descrição da despesa:
            {description}
            
            Opções de natureza de despesa:
            {options_text}
            
            Para cada opção escolhida, forneça:
            1. O código da natureza
            2. Uma justificativa de por que esta é uma boa classificação
            3. Um score de 0 a 100 indicando o grau de adequação
            
            Você deve responder com um JSON no seguinte formato (sem incluir comentários, limitando-se exatamente a esta estrutura):
            {{
                "analise": [
                    {{
                        "codigo": "código da natureza",
                        "justificativa": "justificativa clara e objetiva",
                        "score": número entre 0 e 100
                    }},
                    ...
                ]
            }}
            
            Certifique-se de que seu JSON seja válido. Retorne APENAS o JSON, sem texto adicional antes ou depois.
            """
            
            # Chama a API para análise
            response = client.messages.create(
                model=self.model_name,
                max_tokens=2000,
                temperature=0.2,  # Baixa temperatura para resultados mais determinísticos
                system="Você é um especialista em classificação orçamentária do setor público brasileiro, particularmente em naturezas de despesa. Respondas sempre em JSON válido quando solicitado.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extrai a resposta
            response_text = response.content[0].text.strip()
            
            # Parseia a resposta JSON
            import json
            import re
            
            # Tenta extrair JSON da resposta, mesmo se tiver texto ao redor
            json_match = re.search(r'({[\s\S]*})', response_text)
            
            if json_match:
                try:
                    result = json.loads(json_match.group(1))
                    analysis = result.get('analise', [])
                    
                    # Reorganiza a lista de naturezas de acordo com a análise
                    ranked_options = []
                    
                    for analysis_item in analysis:
                        codigo = analysis_item.get('codigo')
                        
                        # Busca a natureza original correspondente
                        found = False
                        for option in natureza_options:
                            if option.get('codigo') == codigo:
                                # Cria uma cópia com os novos valores
                                ranked_option = option.copy()
                                ranked_option['justificativa'] = analysis_item.get('justificativa', '')
                                ranked_option['score'] = analysis_item.get('score', 0) / 100  # Normaliza para 0-1
                                ranked_options.append(ranked_option)
                                found = True
                                break
                        
                        # Se não encontrou, adiciona como nova entrada
                        if not found and codigo:
                            ranked_options.append({
                                'codigo': codigo,
                                'nome': f"(Nome não disponível para {codigo})",
                                'confianca': analysis_item.get('score', 0) / 100,
                                'justificativa': analysis_item.get('justificativa', '')
                            })
                    
                    # Limitando aos 3 primeiros resultados
                    return ranked_options[:3]
                
                except json.JSONDecodeError:
                    self.logger.error(f"Falha ao decodificar resposta JSON do Claude: {response_text}")
            
            # Tenta extrair os códigos diretamente usando regex se o JSON falhar
            self.logger.warning("Usando extração alternativa de códigos para Claude")
            
            # Procura por códigos no formato c.g.mm.ee
            natureza_pattern = r'(\d\.\d\.\d{2}\.\d{2})'
            codigo_matches = re.findall(natureza_pattern, response_text)
            
            # Se encontrou códigos, reorganiza com base na ordem de aparição
            if codigo_matches:
                codigo_ordered = []
                for codigo in codigo_matches:
                    # Evita duplicação de códigos
                    if codigo not in codigo_ordered:
                        codigo_ordered.append(codigo)
                
                # Filtra os resultados para códigos encontrados
                ranked_options = []
                for codigo in codigo_ordered[:3]:  # Limita a 3
                    # Busca nas opções originais
                    found = False
                    for option in natureza_options:
                        if option.get('codigo') == codigo:
                            ranked_options.append(option)
                            found = True
                            break
                    
                    # Se o código não estava nas opções originais, cria uma entrada básica
                    if not found:
                        ranked_options.append({
                            'codigo': codigo,
                            'nome': f"(Nome não disponível para {codigo})",
                            'confianca': 0.7  # Valor padrão
                        })
                
                return ranked_options
            
            # Em último caso, retorna as 3 melhores opções originais
            self.logger.warning("Fallback para opções originais ordenadas por confiança")
            return sorted(natureza_options, key=lambda x: x.get('confianca', 0), reverse=True)[:3]
        
        except Exception as e:
            self.logger.error(f"Erro ao analisar opções de natureza com Claude: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Em caso de erro, retorna as 3 melhores opções da lista original
            return sorted(natureza_options, key=lambda x: x.get('confianca', 0), reverse=True)[:3]