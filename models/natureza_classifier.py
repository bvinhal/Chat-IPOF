# models/natureza_classifier.py

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import logging
import pickle
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from config import active_config

# Importações dos modelos de embeddings
from models.openai_embedding_model import OpenAIEmbeddingModel
from models.claude_embedding_model import ClaudeEmbeddingModel
from models.gemini_embedding_model import GeminiEmbeddingModel
from models.natureza_processor import NaturezaProcessor
from models.sentence_transformer_classifier import SentenceTransformerClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaClassifier:
    """
    Classificador de natureza de despesa baseado em embeddings.
    """
    
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa o classificador de natureza.
        
        Args:
            embedding_provider: Provedor de embeddings ('openai', 'claude', 'gemini')
        """
        self.embedding_provider = embedding_provider
        self.model_path = os.path.join(active_config.MODELS_DIR, f'natureza_classifier_{embedding_provider}')
        os.makedirs(self.model_path, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Componentes do modelo
        self.embedding_model = self._create_embedding_model()
        self.processor = NaturezaProcessor()
        self.embeddings = None
        self.texts = None
        self.codigos = None
        self.knn_model = None
        self.code_name_mapping = {}
        self.is_trained = False
            
    def _create_embedding_model(self):
        """
        Cria o modelo de embedding apropriado.
        
        Returns:
            Um modelo de embeddings para o provedor especificado
        """
        if self.embedding_provider == 'openai':
            return OpenAIEmbeddingModel()
        elif self.embedding_provider == 'claude':
            return ClaudeEmbeddingModel()
        elif self.embedding_provider == 'gemini':
            return GeminiEmbeddingModel()
        else:
            raise ValueError(f"Provedor de embeddings desconhecido: {self.embedding_provider}")
    
    def train(self, excel_path: str = None, force_rebuild: bool = False) -> bool:
        """
        Treina o classificador usando embeddings.
        
        Args:
            excel_path: Caminho para o arquivo Excel com dados de natureza.
                        Se None, tenta carregar dados processados.
            force_rebuild: Se True, força a recriação dos embeddings mesmo se já existirem
            
        Returns:
            bool: True se o treinamento foi bem-sucedido
        """
        try:
            # Carrega os dados
            if excel_path:
                self.processor.load_excel(excel_path)
                data = self.processor.preprocess_data()
                self.processor.save_processed_data(data)
            else:
                data = self.processor.load_processed_data()
                if data is None:
                    self.logger.error("Sem dados para treinar. Forneça um caminho para o arquivo Excel.")
                    return False
            
            # Mapeamento de código para nome
            self.code_name_mapping = self.processor.get_code_name_mapping()
            
            # Verifica se já existem embeddings salvos
            if not force_rebuild:
                embeddings, texts, metadata = self.embedding_model.load_embeddings()
                if embeddings is not None and texts is not None:
                    self.embeddings = embeddings
                    self.texts = texts
                    
                    # Extrai os códigos dos textos originais
                    text_codigo_pairs = self.processor.get_textos_with_codigos()
                    
                    # Cria um dicionário para mapear texto para código
                    text_to_code = {text: code for text, code in text_codigo_pairs}
                    
                    # Mapeia os textos carregados para códigos
                    self.codigos = [text_to_code.get(text, None) for text in texts]
                    
                    # Verifica se todos os textos foram mapeados a códigos
                    if None in self.codigos:
                        self.logger.warning("Alguns textos não puderam ser mapeados para códigos. " +
                                          "Recomendado recriar embeddings.")
                        
                        if force_rebuild:
                            # Forçar a recriação
                            self.embeddings = None
                            self.texts = None
                            self.codigos = None
                        else:
                            # Tenta corrigir mapeando apenas textos válidos
                            valid_indices = [i for i, code in enumerate(self.codigos) if code is not None]
                            if valid_indices:
                                self.embeddings = self.embeddings[valid_indices]
                                self.texts = [self.texts[i] for i in valid_indices]
                                self.codigos = [self.codigos[i] for i in valid_indices]
                            else:
                                # Se nenhum texto for válido, força a recriação
                                self.embeddings = None
                                self.texts = None
                                self.codigos = None
            
            # Se não temos embeddings válidos, precisamos gerá-los
            if self.embeddings is None or self.texts is None or self.codigos is None:
                # Obtém textos e códigos
                text_codigo_pairs = self.processor.get_textos_with_codigos()
                texts = [text for text, _ in text_codigo_pairs]
                codigos = [code for _, code in text_codigo_pairs]
                
                # Gera embeddings
                self.logger.info(f"Gerando embeddings com provedor {self.embedding_provider}")
                embeddings = self.embedding_model.get_embeddings(texts)
                
                # Salva os embeddings
                self.embedding_model.save_embeddings(
                    embeddings, 
                    texts, 
                    {"codigos": codigos}
                )
                
                self.embeddings = embeddings
                self.texts = texts
                self.codigos = codigos
            
            # Normaliza os embeddings para cálculo de similaridade por cosseno
            self.embeddings = normalize(self.embeddings)
            
            # Treina um modelo KNN para encontrar vizinhos mais próximos
            self.knn_model = NearestNeighbors(
                n_neighbors=min(5, len(self.embeddings)),
                metric='cosine'
            )
            self.knn_model.fit(self.embeddings)
            
            # Salva o modelo
            self.is_trained = True
            self.save_model()
            
            self.logger.info(f"Classificador treinado com {len(self.embeddings)} exemplos e {len(set(self.codigos))} códigos únicos")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro durante treinamento do classificador: {str(e)}")
            return False
    
    def predict_simplify(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Prediz as naturezas de despesa mais prováveis para um texto e as avalia.
        
        Args:
            text: Texto para classificação
            top_k: Número de naturezas mais prováveis a retornar
                
        Returns:
            List[Dict[str, Any]]: Lista de previsões ordenadas por confiança, incluindo avaliação
        """
        if not self.is_trained or self.knn_model is None:
            self.logger.error("Modelo não treinado. Execute train() primeiro.")
            raise ValueError("Modelo não treinado")
        
        try:
            # Parte original: Gera embedding para o texto de consulta
            query_embedding = self.embedding_model.get_embeddings([text])
            
            # Normaliza o embedding
            query_embedding = normalize(query_embedding)
            
            # Encontra os k vizinhos mais próximos
            distances, indices = self.knn_model.kneighbors(
                query_embedding, 
                n_neighbors=min(top_k, len(self.embeddings))
            )
            
            # Converte distâncias de cosseno para similaridades
            # (distância de cosseno = 1 - similaridade de cosseno)
            similarities = 1 - distances[0]
            
            # Obtém os códigos e textos correspondentes
            result_indices = indices[0]
            classificacao_results = []
            
            for i, idx in enumerate(result_indices):
                codigo = self.codigos[idx]
                similarity = similarities[i]
                
                # Busca o nome correspondente ao código
                nome = self.code_name_mapping.get(codigo, "Nome não encontrado")
                
                classificacao_results.append({
                    'codigo': codigo,
                    'nome': nome,
                    'confianca': float(similarity),
                    'texto_referencia': self.texts[idx]
                })
            
            # NOVO: Avaliação das classificações por meio do NaturezaEvaluator
            try:
                from models.natureza_evaluator import NaturezaEvaluator
                
                # Inicializa o avaliador com o mesmo embedding provider
                avaliador = NaturezaEvaluator(self.embedding_provider)
                if not avaliador.load_model():
                    self.logger.warning("Não foi possível carregar o avaliador. Usando apenas classificação.")
                    return classificacao_results
                
                # Avalia os candidatos
                avaliacao_result = avaliador.evaluate_candidates(text, classificacao_results)
                
                # Atualiza as classificações com os resultados da avaliação
                if avaliacao_result and 'ranking' in avaliacao_result:
                    # Cria um mapeamento de código para classificação original
                    codigo_to_classificacao = {item['codigo']: item for item in classificacao_results}
                    
                    # Prepara o resultado final combinando classificação e avaliação
                    final_results = []
                    
                    # Adiciona os itens do ranking de avaliação na nova ordem
                    for rank_item in avaliacao_result['ranking']:
                        codigo = rank_item.get('codigo')
                        if codigo in codigo_to_classificacao:
                            # Combina o item original com dados da avaliação
                            item_combinado = codigo_to_classificacao[codigo].copy()
                            item_combinado['avaliacao_score'] = rank_item.get('score', 0)
                            item_combinado['ranking_position'] = rank_item.get('position', -1)
                            final_results.append(item_combinado)
                    
                    # Adiciona a melhor alternativa se existir e não estiver nas classificações originais
                    if avaliacao_result.get('best_alternative'):
                        alt = avaliacao_result['best_alternative']
                        alt_codigo = alt.get('codigo')
                        # Verifica se essa alternativa já não está nos resultados
                        if alt_codigo and not any(r['codigo'] == alt_codigo for r in final_results):
                            final_results.append({
                                'codigo': alt_codigo,
                                'nome': alt.get('nome', ''),
                                'confianca': alt.get('score', 0.8),  # Confiança padrão para alternativas
                                'texto_referencia': '',  # Não há texto de referência
                                'avaliacao_score': alt.get('score', 0.8),
                                'ranking_position': 0,  # Posição 0 indica que é uma alternativa sugerida
                                'is_alternative': True
                            })
                    
                    # Adiciona a justificativa ao primeiro item (mais recomendado)
                    if final_results and 'justificativa' in avaliacao_result:
                        final_results[0]['justificativa'] = avaliacao_result.get('justificativa', '')
                    
                    # Se o ranking estiver vazio, usa os resultados originais
                    if not final_results:
                        final_results = classificacao_results
                    
                    return final_results
            
            except Exception as e:
                self.logger.error(f"Erro durante a avaliação: {str(e)}")
                # Se houver erro na avaliação, retorna apenas as classificações originais
                return classificacao_results
            
            # Se não conseguiu avaliar, retorna apenas as classificações originais
            return classificacao_results
            
        except Exception as e:
            self.logger.error(f"Erro ao fazer previsão: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def predict(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Prediz as naturezas de despesa mais prováveis para um texto.
        Fluxo aprimorado com RAG:
        1. Usa o SentenceTransformerClassifier para obter candidatos iniciais
        2. Usa o modelo LLM para selecionar os melhores dentre esses candidatos
        3. Busca as naturezas completas (com subelementos) para cada candidato selecionado
        4. Gera embeddings para as naturezas completas e seleciona as mais adequadas
        5. Usa o modelo RAG para definir a ordem final das naturezas mais apropriadas
        
        Args:
            text: Texto para classificação
            top_k: Número de naturezas mais prováveis a retornar
                
        Returns:
            List[Dict[str, Any]]: Lista de previsões ordenadas por relevância
        """
        if not self.is_trained or self.knn_model is None:
            # Tenta carregar o modelo se não estiver treinado
            if not self.load_model():
                self.logger.error("Modelo não treinado e não foi possível carregar")
                raise ValueError("Modelo não treinado e não foi possível carregar")
        
        try:
            # ETAPA 1: Usa o SentenceTransformerClassifier para obter candidatos iniciais
            st_classifier = SentenceTransformerClassifier()
            
            # Número de candidatos a serem selecionados na primeira etapa
            # Vamos pegar mais do que o resultado final para dar mais opções ao LLM
            intermediate_k = 10
            
            try:
                # Obtém previsões do SentenceTransformer (10 candidatos)
                st_predictions = st_classifier.predict(text, top_k=intermediate_k)
                self.logger.info(f"SentenceTransformer retornou {len(st_predictions)} candidatos iniciais")
                
                # Se não tiver candidatos suficientes, tenta o método original como fallback
                if len(st_predictions) < 3:
                    self.logger.warning("Poucos candidatos do SentenceTransformer, usando método original")
                    raise Exception("Poucos candidatos")
                
            except Exception as st_error:
                # Se falhar com o SentenceTransformer, usa o método original como fallback
                self.logger.warning(f"Erro ao usar SentenceTransformerClassifier: {str(st_error)}. Usando método original como fallback.")
                
                # Parte original: Gera embedding para o texto de consulta
                query_embedding = self.embedding_model.get_embeddings([text])
                
                # Normaliza o embedding
                query_embedding = normalize(query_embedding)
                
                # Encontra os k vizinhos mais próximos
                distances, indices = self.knn_model.kneighbors(
                    query_embedding, 
                    n_neighbors=min(intermediate_k, len(self.embeddings))
                )
                
                # Converte distâncias de cosseno para similaridades
                similarities = 1 - distances[0]
                
                # Obtém os códigos e textos correspondentes
                result_indices = indices[0]
                st_predictions = []
                
                for i, idx in enumerate(result_indices):
                    codigo = self.codigos[idx]
                    similarity = similarities[i]
                    
                    # Busca o nome correspondente ao código
                    nome = self.code_name_mapping.get(codigo, "Nome não encontrado")
                    
                    st_predictions.append({
                        'codigo': codigo,
                        'nome': nome,
                        'confianca': float(similarity),
                        'texto_referencia': self.texts[idx]
                    })
            
            # ETAPA 2: Usa o modelo LLM para refinar a seleção dos candidatos iniciais
            chat_controller_instance = None
            try:
                # Importa o módulo necessário
                import sys
                sys.path.append('.')
                
                # Obtém o controlador de chat atual
                
                # Tenta obter uma instância existente ou criar uma nova
                try:
                    # Tenta importar a instância do app.py
                    from app import chat_controller
                    chat_controller_instance = chat_controller
                    self.logger.info("Usando instância do chat_controller de app.py")
                except ImportError:
                    self.logger.warning("Não foi possível importar chat_controller de app.py")
                    # Tenta criar uma nova instância
                    try:
                        from controllers.enhanced_chat_controller import EnhancedChatController
                        chat_controller_instance = EnhancedChatController()
                        self.logger.info("Criada nova instância de EnhancedChatController")
                    except Exception as controller_error:
                        self.logger.error(f"Erro ao criar EnhancedChatController: {str(controller_error)}")
                
                # Se temos uma instância válida do controlador, usa o modelo atual para análise
                if chat_controller_instance and chat_controller_instance.current_model:
                    self.logger.info(f"Usando modelo {chat_controller_instance.current_model_type} para análise LLM")
                    
                    # Usa o modelo para analisar as opções iniciais
                    initial_results = chat_controller_instance.current_model.analyze_natureza_options(
                        text, st_predictions
                    )
                    
                    # Verifica se temos resultados válidos
                    if initial_results and len(initial_results) > 0:
                        self.logger.info(f"Análise LLM concluída com sucesso, retornando {len(initial_results)} candidatos intermediários")
                        st_predictions = initial_results
            
            except Exception as llm_error:
                self.logger.error(f"Erro na etapa de análise LLM inicial: {str(llm_error)}")
                self.logger.info("Continuando com candidatos originais do SentenceTransformer")
            
            # ETAPA 3: Busca naturezas completas (com subelemento) para cada candidato
            try:
                # Importar o processador de natureza
                from models.natureza_processor import NaturezaProcessor
                
                # Obtém os top_k * 2 candidatos iniciais (para ter margem)
                top_candidates = st_predictions[:top_k * 2]
                
                # Inicializa o processador
                processor = NaturezaProcessor()
                processor.load_processed_data()
                
                # Busca as naturezas completas (com subelementos) para cada candidato
                complete_candidates = []
                
                # Simula uma base de naturezas completas (subelementos)
                # Em um ambiente real, isso seria buscado de uma base de dados
                complete_naturezas = {}
                
                # Obtém todos os códigos de natureza disponíveis
                all_codigos = processor.get_all_codigos()
                
                # Para cada candidato, busca as naturezas completas correspondentes
                for candidate in top_candidates:
                    codigo_base = candidate.get('codigo', '')
                    
                    # Procura naturezas que começam com o código base (são subelementos)
                    matching_completes = []
                    for codigo in all_codigos:
                        if codigo.startswith(codigo_base): #and len(codigo.split('.')) > len(codigo_base.split('.')):
                            natureza_info = {
                                'codigo': codigo,
                                'nome': processor.get_code_name_mapping().get(codigo, f"Subelemento de {codigo_base}"),
                                'confianca': candidate.get('confianca', 0.0) * 0.9,  # Reduz um pouco a confiança
                                'base_code': codigo_base
                            }
                            matching_completes.append(natureza_info)
                    
                    # Se não encontrar subelementos, usa o próprio código base
                    if not matching_completes:
                        complete_candidates.append({
                            'codigo': codigo_base,
                            'nome': candidate.get('nome', ''),
                            'confianca': candidate.get('confianca', 0.0),
                            'descricao': processor.get_orientacao_by_codigo(codigo_base) or "Não há descrição disponível.",
                            'base_code': codigo_base
                        })
                    else:
                        # Adiciona os subelementos encontrados
                        for complete in matching_completes:
                            descricao = processor.get_orientacao_by_codigo(complete['codigo'])
                            if not descricao:  # Se não tem descrição específica, usa a do código base
                                descricao = processor.get_orientacao_by_codigo(codigo_base) or "Não há descrição disponível."
                            
                            complete['descricao'] = descricao
                            complete_candidates.append(complete)
                
                self.logger.info(f"Encontradas {len(complete_candidates)} naturezas completas (com subelementos)")
                
            except Exception as complete_error:
                self.logger.error(f"Erro ao buscar naturezas completas: {str(complete_error)}")
                # Em caso de erro, continua com os candidatos originais
                complete_candidates = []
                for candidate in top_candidates:
                    candidate['descricao'] = "Descrição não disponível devido a erro no processamento."
                    complete_candidates.append(candidate)
            
            # ETAPA 4: Gera embeddings para as naturezas completas e seleciona as mais adequadas
            try:
                # Se temos poucos candidatos, não precisamos fazer esta etapa
                if len(complete_candidates) <= top_k:
                    self.logger.info(f"Apenas {len(complete_candidates)} candidatos completos disponíveis, pulando etapa de embeddings")
                    selected_candidates = complete_candidates
                else:
                    # Gera embeddings para o texto de consulta
                    query_embedding = self.embedding_model.get_embeddings([text])
                    query_embedding = normalize(query_embedding)
                    
                    # Prepara textos das naturezas completas para embeddings
                    candidate_texts = []
                    for candidate in complete_candidates:
                        # Combina código, nome e descrição
                        combined_text = f"{candidate['codigo']} - {candidate['nome']}\n{candidate['descricao']}"
                        candidate_texts.append(combined_text)
                    
                    # Gera embeddings para as naturezas completas
                    candidate_embeddings = self.embedding_model.get_embeddings(candidate_texts)
                    candidate_embeddings = normalize(candidate_embeddings)
                    
                    # Calcula similaridade de cosseno
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
                    
                    # Ordena os candidatos por similaridade
                    candidate_with_similarities = []
                    for i, candidate in enumerate(complete_candidates):
                        candidate_with_sim = candidate.copy()
                        candidate_with_sim['similarity'] = similarities[i]
                        candidate_with_similarities.append(candidate_with_sim)
                    
                    # Ordena por similaridade, depois por confiança original
                    sorted_candidates = sorted(
                        candidate_with_similarities, 
                        key=lambda x: (-x['similarity'], -x['confianca'])
                    )
                    
                    # Seleciona os top_k candidatos mais similares
                    selected_candidates = sorted_candidates[:top_k]
                    self.logger.info(f"Selecionados {len(selected_candidates)} candidatos após análise de embeddings")
            
            except Exception as embedding_error:
                self.logger.error(f"Erro ao gerar embeddings para naturezas completas: {str(embedding_error)}")
                # Em caso de erro, seleciona os candidatos por confiança original
                selected_candidates = sorted(complete_candidates, key=lambda x: -x['confianca'])[:top_k]
            
            # ETAPA 5: Usa o modelo RAG para definir a ordem final das naturezas mais apropriadas
            try:
                # Verifica se temos um modelo e controlador válido
                if chat_controller_instance and chat_controller_instance.current_model:
                    self.logger.info(f"Usando modelo RAG {chat_controller_instance.current_model_type} para ordenação final")
                    
                    # Chama o método específico de análise para naturezas completas
                    if hasattr(chat_controller_instance.current_model, 'analyze_natureza_complete'):
                        final_results = chat_controller_instance.current_model.analyze_natureza_complete(
                            text, selected_candidates
                        )
                        
                        # Verifica se temos resultados válidos
                        if final_results and len(final_results) > 0:
                            self.logger.info(f"Análise RAG final concluída com sucesso, retornando {len(final_results)} resultados")
                            # Limita ao número solicitado
                            return final_results[:top_k]
                    else:
                        self.logger.warning(f"Modelo {chat_controller_instance.current_model_type} não implementa analyze_natureza_complete")
            
            except Exception as rag_error:
                self.logger.error(f"Erro na etapa final de análise RAG: {str(rag_error)}")
            
            # Se a etapa 5 falhar ou não for possível, retorna os candidatos selecionados na etapa 4
            return selected_candidates[:top_k]
            
        except Exception as e:
            self.logger.error(f"Erro global ao fazer previsão: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
                                                    
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
            
            # Não salvamos os embeddings aqui pois já foram salvos pelo embedding_model
            # Salvamos apenas as referências e o modelo KNN
            
            # Informações básicas para reconstrução
            model_info = {
                'embedding_provider': self.embedding_provider,
                'num_examples': len(self.embeddings) if self.embeddings is not None else 0,
                'is_trained': self.is_trained,
                'code_name_mapping': self.code_name_mapping
            }
            
            # Salva informações do modelo
            info_path = os.path.join(self.model_path, 'model_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, ensure_ascii=False, indent=2)
            
            # Salva textos e códigos
            data_path = os.path.join(self.model_path, 'model_data.pkl')
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'texts': self.texts,
                    'codigos': self.codigos
                }, f)
            
            # Salva o modelo KNN
            knn_path = os.path.join(self.model_path, 'knn_model.pkl')
            with open(knn_path, 'wb') as f:
                pickle.dump(self.knn_model, f)
            
            self.logger.info(f"Modelo salvo com sucesso em {self.model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar modelo: {str(e)}")
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
            data_path = os.path.join(self.model_path, 'model_data.pkl')
            knn_path = os.path.join(self.model_path, 'knn_model.pkl')
            
            if not all(os.path.exists(p) for p in [info_path, data_path, knn_path]):
                self.logger.warning("Arquivos do modelo não encontrados")
                return False
            
            # Carrega informações do modelo
            with open(info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
            
            # Verifica se o provedor de embeddings é compatível
            if model_info['embedding_provider'] != self.embedding_provider:
                self.logger.warning(f"Provedor de embeddings incompatível: " +
                                   f"modelo salvo com {model_info['embedding_provider']}, " +
                                   f"atual é {self.embedding_provider}")
            
            # Carrega textos e códigos
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data['texts']
                self.codigos = data['codigos']
            
            # Carrega o modelo KNN
            with open(knn_path, 'rb') as f:
                self.knn_model = pickle.load(f)
            
            # Carrega o mapeamento de código para nome
            self.code_name_mapping = model_info.get('code_name_mapping', {})
            if not self.code_name_mapping:
                # Se o mapeamento não foi salvo, tenta carregá-lo do processador
                self.code_name_mapping = self.processor.get_code_name_mapping()
            
            # Carrega embeddings existentes (apenas referência, os vetores reais ficam no arquivo do embedding_model)
            embeddings, _, _ = self.embedding_model.load_embeddings()
            if embeddings is not None:
                self.embeddings = embeddings
            
            self.is_trained = True
            
            self.logger.info(f"Modelo carregado com sucesso: {len(self.texts)} exemplos e {len(set(self.codigos))} códigos únicos")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar modelo: {str(e)}")
            return False