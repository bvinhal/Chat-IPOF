import os
from typing import List, Dict, Any, Optional
import glob
import pickle
import tiktoken
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
                for doc in docs:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        source_path = doc.metadata['source']
                        doc.metadata['filename'] = os.path.basename(source_path)
                        # Registra o nome no log para depuração
                        self.logger.debug(f"Arquivo carregado: {doc.metadata['filename']}")
    
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
            # Importa o OpenAI client para resumir a consulta se necessário
            from openai import OpenAI
            
            # Inicializa o contador de tokens
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
                
                # Calcula o tamanho da consulta em tokens
                query_tokens = len(encoding.encode(query))
                self.logger.info(f"Tokens na consulta original: {query_tokens}")
                
                # Define limites mais conservadores
                TOKEN_LIMIT = 16000  # Limite total do contexto
                RESERVE_TOKENS = 10000  # Reservado para documentos recuperados e respostas
                QUERY_TOKEN_LIMIT = TOKEN_LIMIT - RESERVE_TOKENS  # Limite para a consulta
                
                # Converte o formato do histórico do chat para estimar os tokens
                history_tokens = 0
                if chat_history:
                    for msg in chat_history:
                        if msg.get('content'):
                            history_tokens += len(encoding.encode(msg.get('content', '')))
                            # Adicione alguns tokens para metadados de cada mensagem
                            history_tokens += 25  # Estimativa para role, formatting, etc.
                
                # Calcula tokens disponíveis considerando o histórico
                available_tokens = QUERY_TOKEN_LIMIT - history_tokens
                
                # Se a consulta excede os tokens disponíveis, resumimos
                if query_tokens > available_tokens and available_tokens > 0:
                    self.logger.warning(f"Consulta excede o limite de tokens disponíveis ({query_tokens} > {available_tokens}). Resumindo...")
                    
                    # Inicializa o cliente OpenAI para resumir
                    client = OpenAI(api_key=self.api_key)
                    
                    # Define um comprimento alvo em termos de proporção
                    target_proportion = 0.8 * (available_tokens / query_tokens)
                    target_length = int(len(query) * target_proportion)
                    
                    # Ajusta para um mínimo razoável
                    target_length = max(target_length, 200)
                    
                    # Cria um prompt de resumo mais assertivo
                    summarize_prompt = f"""
                    IMPORTANTE: Resuma a seguinte consulta em NO MÁXIMO {target_length} caracteres.
                    Priorize os pontos principais, palavras-chave e parâmetros essenciais.
                    Mantenha apenas as informações absolutamente cruciais.
                    
                    CONSULTA ORIGINAL:
                    {query}
                    
                    RESUMO CONCISO (máximo {target_length} caracteres):
                    """
                    
                    # Chama a API para resumir com parâmetros mais restritos
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",  # Modelo mais rápido e econômico para resumos
                        messages=[
                            {"role": "system", "content": "Você é um especialista em resumir textos de forma extremamente concisa e precisa."},
                            {"role": "user", "content": summarize_prompt}
                        ],
                        max_tokens=int(available_tokens * 0.8),  # Usa apenas 80% dos tokens disponíveis
                        temperature=0.2  # Temperatura baixa para resumos mais determinísticos e concisos
                    )
                    
                    # Extrai o resumo da resposta
                    summarized_query = response.choices[0].message.content.strip()
                    
                    # Verifica o tamanho do resumo em tokens
                    summary_tokens = len(encoding.encode(summarized_query))
                    self.logger.info(f"Consulta resumida de {query_tokens} para {summary_tokens} tokens (redução de {int((1-(summary_tokens/query_tokens))*100)}%)")
                    
                    # Verifica se o resumo foi realmente eficiente
                    if summary_tokens > 0.9 * query_tokens:
                        # Se o resumo não foi eficiente, faz um corte mais drástico
                        self.logger.warning("Resumo não foi eficiente. Realizando corte direto...")
                        words = query.split()
                        # Preserva apenas 40% das palavras originais
                        max_words = int(len(words) * 0.4)
                        summarized_query = " ".join(words[:max_words]) + "..."
                        summary_tokens = len(encoding.encode(summarized_query))
                        self.logger.info(f"Consulta truncada para {summary_tokens} tokens após corte direto")
                    
                    # Usa o resumo como consulta
                    query = summarized_query
                    
                    # Adiciona uma nota sobre o resumo no início da consulta
                    query = f"[Esta é uma consulta resumida automaticamente para caber no limite de tokens] {query}"
                
            except ImportError:
                self.logger.warning("Tiktoken não disponível, não foi possível verificar o tamanho da consulta")
            except Exception as e:
                self.logger.error(f"Erro ao resumir consulta: {str(e)}. Tentando abordagem alternativa...")
                
                # Abordagem de contingência se o resumo falhar
                try:
                    words = query.split()
                    # Se a consulta for muito longa, faz um corte direto preservando apenas metade
                    if len(words) > 200:
                        max_words = min(200, int(len(words) * 0.5))
                        query = " ".join(words[:max_words]) + "... [consulta truncada devido ao tamanho]"
                        self.logger.info(f"Consulta truncada para {len(query)} caracteres")
                except Exception as e2:
                    self.logger.error(f"Erro ao truncar consulta: {str(e2)}. Usando consulta original.")
            
            # Converte o formato do histórico do chat se fornecido
            langchain_history = []
            if chat_history:
                # Limita o histórico para as últimas 5 mensagens se for muito grande
                if len(chat_history) > 5:
                    self.logger.info(f"Histórico grande ({len(chat_history)} mensagens). Limitando para as últimas 5.")
                    chat_history = chat_history[-5:]
                
                for msg in chat_history:
                    if msg.get('role') == 'user':
                        langchain_history.append(HumanMessage(content=msg.get('content', '')))
                    elif msg.get('role') == 'assistant':
                        langchain_history.append(AIMessage(content=msg.get('content', '')))
            
            # Tenta gerar resposta com limite explícito de tokens
            try:
                # Gera resposta usando o modelo treinado
                result = self.chain({
                    'question': query,
                    'chat_history': langchain_history
                })
                
                # Se a consulta foi resumida, adiciona uma nota à resposta
                answer = result.get('answer', "Não foi possível gerar uma resposta.")
                if query_tokens > available_tokens:
                    answer = f"{answer}"
                
                return answer
                
            except Exception as chain_error:
                # Se falhar devido a limite de contexto, tenta novamente sem histórico
                if "context_length_exceeded" in str(chain_error) or "maximum context length" in str(chain_error):
                    self.logger.warning("Erro de contexto muito longo. Tentando novamente sem histórico...")
                    
                    result = self.chain({
                        'question': query,
                        'chat_history': []  # Sem histórico
                    })
                    
                    answer = result.get('answer', "Não foi possível gerar uma resposta.")
                    return f"Nota: O histórico da conversa foi ignorado devido ao limite de contexto do modelo. A resposta a seguir é baseada apenas na sua consulta atual.\n\n{answer}"
                else:
                    raise  # Re-levanta outros erros
        
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

    def analyze_natureza_options(self, description: str, natureza_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa uma lista de opções de natureza de despesa usando o modelo OpenAI
        e seleciona as mais adequadas para a descrição fornecida.
        
        Args:
            description: Descrição da despesa
            natureza_options: Lista de opções de natureza de despesa, 
                            cada uma com 'codigo', 'nome' e 'confianca'
            
        Returns:
            List[Dict[str, Any]]: Lista das naturezas mais adequadas, ordenadas por relevância
        """
        if self.llm is None and not self.initialize():
            self.logger.error("Cliente OpenAI não inicializado")
            raise ValueError("Cliente OpenAI não inicializado. Verifique a API key.")
        
        try:
            # Importa o OpenAI client diretamente
            from openai import OpenAI
            
            # Inicializa o cliente OpenAI
            client = OpenAI(api_key=self.api_key)
            
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
            
            Responda no seguinte formato JSON:
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
            
            Retorne APENAS o JSON, sem texto adicional antes ou depois.
            """
            
            # Chama a API para análise
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Você é um especialista em classificação orçamentária do setor público brasileiro, particularmente em naturezas de despesa."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Baixa temperatura para resultados mais determinísticos
                response_format={"type": "json_object"}  # Garante resposta em formato JSON
            )
            
            # Extrai a resposta
            response_text = response.choices[0].message.content.strip()
            
            # Parseia a resposta JSON
            import json
            try:
                result = json.loads(response_text)
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
                self.logger.error(f"Falha ao decodificar resposta JSON: {response_text}")
                
                # Em caso de falha, retorne as 3 primeiras opções originais
                return sorted(natureza_options, key=lambda x: x.get('confianca', 0), reverse=True)[:3]
        
        except Exception as e:
            self.logger.error(f"Erro ao analisar opções de natureza: {str(e)}")
            # Em caso de erro, retorna as 3 melhores opções da lista original
            return sorted(natureza_options, key=lambda x: x.get('confianca', 0), reverse=True)[:3]

    def analyze_natureza_complete(self, description: str, natureza_options: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analisa uma lista de opções de natureza de despesa completa (incluindo subelementos)
        usando o modelo OpenAI e seleciona as mais adequadas para a descrição fornecida.
        
        Args:
            description: Descrição da despesa
            natureza_options: Lista de opções de natureza de despesa completa, 
                            cada uma com 'codigo', 'nome', 'descricao' e outros metadados
            
        Returns:
            List[Dict[str, Any]]: Lista das naturezas mais adequadas, ordenadas por relevância
        """
        if self.llm is None and not self.initialize():
            self.logger.error("Cliente OpenAI não inicializado")
            raise ValueError("Cliente OpenAI não inicializado. Verifique a API key.")
        
        try:
            # Importa o OpenAI client diretamente
            from openai import OpenAI
            
            # Inicializa o cliente OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Monta a lista de opções formatada com descrições completas
            options_text = ""
            for i, option in enumerate(natureza_options, 1):
                codigo = option.get('codigo', '')
                nome = option.get('nome', '')
                descricao = option.get('descricao', '')
                options_text += f"{i}. {codigo} - {nome}\n   Descrição: {descricao}\n\n"
            
            # Cria o prompt para análise
            prompt = f"""
            Com base na descrição de despesa abaixo, analise as opções de natureza de despesa COMPLETAS 
            (incluindo subelementos) listadas e ordene-as da mais adequada para a menos adequada.
            
            ATENÇÃO: Você deve considerar APENAS as naturezas listadas abaixo e suas descrições. 
            Não inclua naturezas adicionais que não estejam explicitamente listadas.
            
            Descrição da despesa:
            {description}
            
            Opções de natureza de despesa (com códigos completos):
            {options_text}
            
            Para cada opção, forneça:
            1. O código completo da natureza no formato c.g.mm.ee.ss
            2. Uma justificativa detalhada explicando por que esta natureza é adequada ou não para a descrição fornecida
            3. Um score de 0 a 100 indicando o grau de adequação
            
            Responda no seguinte formato JSON:
            {{
                "analise": [
                    {{
                        "codigo": "código completo da natureza no formato c.g.mm.ee.ss",
                        "justificativa": "justificativa detalhada",
                        "score": número entre 0 e 100
                    }},
                    ...
                ]
            }}
            
            Retorne APENAS o JSON, sem texto adicional antes ou depois.
            """
            
            # Chama a API para análise
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Você é um especialista em classificação orçamentária do setor público brasileiro, particularmente em naturezas de despesa, incluindo os subelementos (formato c.g.mm.ee.ss)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Baixa temperatura para resultados mais determinísticos
                response_format={"type": "json_object"}  # Garante resposta em formato JSON
            )
            
            # Extrai a resposta
            response_text = response.choices[0].message.content.strip()
            
            # Parseia a resposta JSON
            import json
            try:
                result = json.loads(response_text)
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
                        # Procura com comparação parcial
                        for option in natureza_options:
                            if codigo.startswith(option.get('codigo', '')):
                                ranked_option = option.copy()
                                ranked_option['justificativa'] = analysis_item.get('justificativa', '')
                                ranked_option['score'] = analysis_item.get('score', 0) / 100
                                ranked_options.append(ranked_option)
                                found = True
                                break
                        
                        if not found:
                            # Adiciona mesmo sem correspondência
                            ranked_options.append({
                                'codigo': codigo,
                                'nome': f"(Nome não disponível para {codigo})",
                                'score': analysis_item.get('score', 0) / 100,
                                'justificativa': analysis_item.get('justificativa', '')
                            })
                
                # Se não obteve resultados do modelo, retorna as opções originais na ordem fornecida
                if not ranked_options:
                    self.logger.warning("Não foi possível obter ranking do modelo, retornando opções originais")
                    return natureza_options
                
                # Retorna as opções ranqueadas
                return ranked_options
            
            except json.JSONDecodeError:
                self.logger.error(f"Falha ao decodificar resposta JSON: {response_text}")
                
                # Em caso de falha, retorne as opções originais na ordem fornecida
                return natureza_options
        
        except Exception as e:
            self.logger.error(f"Erro ao analisar opções de natureza completa: {str(e)}")
            # Em caso de erro, retorna as opções originais
            return natureza_options