# Importação das bibliotecas necessárias
import os
import re
import json
import tempfile
import numpy as np
from typing import Dict, List, Optional, Union
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from config import active_config

# Processamento de PDFs
import pypdf

# LLM APIs
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage

# Para embeddings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Definição da classe para o classificador com JSON
class ClassificadorDespesasMCASP:
    def __init__(self, embedding_provider: str = 'openai'):
        """
        Inicializa o classificador de despesas baseado em JSON

        Args:
            mcasp_json_path: Caminho para o arquivo JSON do MCASP principal
            subelementos_json_path: Caminho para o arquivo JSON de subelementos (opcional)
            api_key_openai: Chave da API OpenAI
            api_key_anthropic: Chave da API Anthropic (Claude)
            provider: Provedor a ser usado ("openai" ou "anthropic")
        """
        self.embedding_provider = embedding_provider
        self.mcasp_json_path = os.path.join(active_config.DATA_DIR, 'natureza_v2', 'mcasp_oficial_enriquecido_v2.json')
        self.subelementos_json_path = os.path.join(active_config.DATA_DIR, 'natureza_v2', 'subelementos_goias_processados.json')

        self.api_key_openai = active_config.OPENAI_API_KEY

        self.provider = embedding_provider
        self.mcasp_data = None
        self.subelementos_data = None
        self.embedding_model = None
        self.mcasp_embeddings = None
        self.mcasp_texts = None

        # Carregar os dados do MCASP
        self._load_mcasp_data()

        # Carregar os dados de subelementos, se fornecidos
        if self.subelementos_json_path:
            self._load_subelementos_data()

        # Inicializar o modelo de embeddings
        self._init_embedding_model()

        # Inicializar o modelo LLM adequado
        self._init_llm()

        # Preparar os embeddings do MCASP
        self._prepare_mcasp_embeddings()

    def _load_mcasp_data(self):
        """Carrega os dados do MCASP do arquivo JSON principal"""
        try:
            with open(self.mcasp_json_path, 'r', encoding='utf-8') as f:
                self.mcasp_data = json.load(f)
            print(f"Dados do MCASP carregados com sucesso do arquivo: {self.mcasp_json_path}")
        except Exception as e:
            print(f"Erro ao carregar o arquivo JSON do MCASP: {e}")
            raise

    def _load_subelementos_data(self):
        """Carrega e indexa os dados de subelementos para busca rápida"""
        if not self.subelementos_json_path:
            print("Caminho para o arquivo de subelementos não fornecido.")
            self.subelementos_data = []
            self.subelementos_index = {}
            return

        try:
            print(f"Carregando e indexando subelementos de: {self.subelementos_json_path}")
            with open(self.subelementos_json_path, 'r', encoding='utf-8') as f:
                subelementos_json = json.load(f)

            # Determinar o formato e extrair a lista de subelementos
            if isinstance(subelementos_json, list):
                self.subelementos_data = subelementos_json
            elif isinstance(subelementos_json, dict) and "subelementos" in subelementos_json:
                if isinstance(subelementos_json["subelementos"], list):
                    self.subelementos_data = subelementos_json["subelementos"]
                else:
                    print("Erro: A chave 'subelementos' não contém uma lista")
                    self.subelementos_data = []
            else:
                print("Formato de arquivo de subelementos não reconhecido.")
                self.subelementos_data = []

            # Criar índice para busca rápida
            self.subelementos_index = {}
            for sub in self.subelementos_data:
                if all(campo in sub for campo in ["categoria", "grupo", "modalidade", "elemento"]):
                    # Normalizar para strings para garantir consistência
                    cat = str(sub["categoria"])
                    grp = str(sub["grupo"])
                    mod = str(sub["modalidade"])
                    ele = str(sub["elemento"])

                    # Criar chave composta
                    chave = f"{cat}.{grp}.{mod}.{ele}"

                    # Adicionar ao índice
                    if chave not in self.subelementos_index:
                        self.subelementos_index[chave] = []
                    self.subelementos_index[chave].append(sub)

            print(f"Subelementos indexados com sucesso: {len(self.subelementos_data)} subelementos, {len(self.subelementos_index)} combinações únicas")

        except Exception as e:
            print(f"Erro ao carregar arquivo de subelementos: {e}")
            self.subelementos_data = []
            self.subelementos_index = {}

    def _init_embedding_model(self):
        """Inicializa o modelo de embeddings, apenas se necessário"""
        # Verificar se existe um arquivo de cache de embeddings
        cache_path = f"{os.path.splitext(self.mcasp_json_path)[0]}_embeddings.npz"
        cache_texts_path = f"{os.path.splitext(self.mcasp_json_path)[0]}_texts.json"

        # Se ambos existirem, pode pular a inicialização do modelo por enquanto
        if os.path.exists(cache_path) and os.path.exists(cache_texts_path):
            print("Cache de embeddings encontrado, inicialização do modelo pode ser adiada.")
            self.embedding_model = None  # Será inicializado posteriormente se necessário
            return

        # Se não houver cache, inicializar o modelo normalmente
        try:
            print("Carregando modelo de embeddings...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Modelo de embeddings carregado com sucesso.")

            # Adicionar cache para embeddings de subelementos
            self.subelementos_embeddings_cache = {}

        except Exception as e:
            print(f"Erro ao carregar modelo de embeddings: {e}")
            raise

    def _prepare_mcasp_embeddings(self):
        """Prepara embeddings para todos os itens do MCASP com opção de cachear"""
        if not self.mcasp_data or not self.embedding_model:
            print("Dados do MCASP ou modelo de embeddings não disponíveis")
            return

        # Verificar se existe um arquivo de cache de embeddings
        cache_path = f"{os.path.splitext(self.mcasp_json_path)[0]}_embeddings.npz"
        cache_texts_path = f"{os.path.splitext(self.mcasp_json_path)[0]}_texts.json"

        # Tentar carregar do cache
        if os.path.exists(cache_path) and os.path.exists(cache_texts_path):
            try:
                print(f"Tentando carregar embeddings do cache: {cache_path}")
                # Carregar embeddings
                cached_data = np.load(cache_path)
                self.mcasp_embeddings = cached_data['embeddings']

                # Carregar textos
                with open(cache_texts_path, 'r', encoding='utf-8') as f:
                    self.mcasp_texts = json.load(f)

                print(f"Embeddings carregados com sucesso do cache: {len(self.mcasp_texts)} itens")
                return
            except Exception as e:
                print(f"Erro ao carregar embeddings do cache: {e}. Gerando novamente.")

        # Se não conseguir carregar do cache, gerar embeddings
        print("Preparando embeddings para itens do MCASP...")

        # Preparar todos os textos do MCASP
        self.mcasp_texts = []
        for tipo in ["categorias", "grupos", "modalidades", "elementos"]:
            if tipo in self.mcasp_data:  # Verificar se o tipo existe no MCASP
                for item in self.mcasp_data.get(tipo, []):
                    texto = f"Tipo: {tipo}. Código: {item['codigo']}. Descrição: {item['descricao']}. Definição: {item.get('definicao', '')}"
                    self.mcasp_texts.append({
                        "tipo": tipo,
                        "codigo": item['codigo'],
                        "texto": texto,
                        "item": item
                    })

        # Gerar embeddings para todos os textos
        if self.mcasp_texts:
            texts_for_embedding = [item["texto"] for item in self.mcasp_texts]
            print(f"Gerando embeddings para {len(texts_for_embedding)} itens...")
            self.mcasp_embeddings = self.embedding_model.encode(texts_for_embedding)
            print("Embeddings gerados com sucesso.")

            # Salvar embeddings em cache para uso futuro
            try:
                # Salvar embeddings como numpy array
                np.savez_compressed(cache_path, embeddings=self.mcasp_embeddings)

                # Salvar textos (sem o modelo de item completo para economizar espaço)
                texts_to_save = []
                for item in self.mcasp_texts:
                    # Criar versão simplificada sem o item completo
                    simplified = {
                        "tipo": item["tipo"],
                        "codigo": item["codigo"],
                        "texto": item["texto"],
                        # Salvar apenas informações essenciais do item
                        "item": {
                            "codigo": item["item"]["codigo"],
                            "descricao": item["item"]["descricao"],
                            "definicao": item["item"].get("definicao", "")
                        }
                    }
                    texts_to_save.append(simplified)

                with open(cache_texts_path, 'w', encoding='utf-8') as f:
                    json.dump(texts_to_save, f, ensure_ascii=False, indent=2)

                print(f"Embeddings e textos salvos em cache: {cache_path} e {cache_texts_path}")
            except Exception as e:
                print(f"Erro ao salvar embeddings em cache: {e}")
        else:
            print("Nenhum item encontrado no MCASP para gerar embeddings.")
    '''
    def _restore_mcasp_items_from_cache(self):
        """Restaura os itens completos do MCASP a partir dos textos carregados do cache"""
        if not self.mcasp_texts or not self.mcasp_data:
            return

        # Mapear códigos para itens completos para cada tipo
        item_maps = {}
        for tipo in ["categorias", "grupos", "modalidades", "elementos"]:
            if tipo in self.mcasp_data:
                item_maps[tipo] = {item["codigo"]: item for item in self.mcasp_data[tipo]}

        # Atualizar os itens simplificados com os completos
        for item in self.mcasp_texts:
            tipo = item["tipo"]
            codigo = item["codigo"]
            if tipo in item_maps and codigo in item_maps[tipo]:
                item["item"] = item_maps[tipo][codigo]
    '''
    def _init_llm(self):
        """Inicializa o modelo de linguagem baseado no provedor escolhido"""
        try:
            if self.provider == "openai":
                if not self.api_key_openai:
                    raise ValueError("É necessário fornecer uma chave da API OpenAI")

                self.llm = ChatOpenAI(
                    api_key=self.api_key_openai,
                    model_name="gpt-4o",#"gpt-4-turbo",
                    temperature=0.1
                )
                print("Modelo OpenAI (GPT-4) inicializado com sucesso.")

            elif self.provider == "anthropic":
                if not self.api_key_anthropic:
                    raise ValueError("É necessário fornecer uma chave da API Anthropic")

                self.llm = ChatAnthropic(
                    api_key=self.api_key_anthropic,
                    model="claude-3-opus-20240229",
                    temperature=0.1
                )
                print("Modelo Anthropic (Claude) inicializado com sucesso.")

            else:
                raise ValueError("Provedor não suportado. Use 'openai' ou 'anthropic'")
        except Exception as e:
            print(f"Erro ao inicializar o modelo LLM: {e}")
            raise

    def buscar_subelementos_por_natureza(self, categoria, grupo, modalidade, elemento):
        """
        Busca subelementos compatíveis com uma natureza de despesa específica
        usando o índice de subelementos para acesso rápido

        Args:
            categoria: Código da categoria
            grupo: Código do grupo
            modalidade: Código da modalidade
            elemento: Código do elemento

        Returns:
            Lista de subelementos compatíveis ou lista vazia se não houver
        """
        # Verificar se o índice está disponível
        if not hasattr(self, 'subelementos_index') or not self.subelementos_index:
            print("Índice de subelementos não disponível")
            return []

        # Normalizar para strings
        cat = str(categoria)
        grp = str(grupo)
        mod = str(modalidade)
        ele = str(elemento)

        # Criar chave composta
        chave = f"{cat}.{grp}.{mod}.{ele}"

        # Buscar no índice
        subelementos_compativeis = self.subelementos_index.get(chave, [])

        # Fazer cópias dos subelementos para evitar modificar os originais
        resultado = []
        for sub in subelementos_compativeis:
            # Fazer cópia do subelemento
            sub_copia = sub.copy()

            # Garantir que existe um código completo
            if "codigo_completo" not in sub_copia or not sub_copia["codigo_completo"]:
                sub_copia["codigo_completo"] = f"{cat}.{grp}.{mod}.{ele}.{sub_copia['codigo']}"

            resultado.append(sub_copia)

        print(f"Encontrados {len(resultado)} subelementos para {chave}")
        return resultado

    def encontrar_melhores_subelementos_por_embedding(self, subelementos, descricao, top_n=3, limiar_compatibilidade=0.3):
        """
        Encontra os melhores subelementos baseados em similaridade de embeddings

        Args:
            subelementos: Lista de subelementos para comparar
            descricao: Descrição da despesa
            top_n: Número de subelementos a retornar
            limiar_compatibilidade: Pontuação mínima para considerar um subelemento compatível

        Returns:
            Lista dos top_n subelementos mais relevantes com suas pontuações
        """
        if not subelementos:
            return []

        try:
            # Verificar se o modelo de embeddings está inicializado
            if self.embedding_model is None:
                print("Inicializando modelo de embeddings para busca de subelementos...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Criar embedding da descrição
            descricao_emb = self.embedding_model.encode(descricao)

            # Preparar subelementos e seus textos para embeddings
            textos_subelementos = []
            for sub in subelementos:
                texto = f"Subelemento {sub.get('codigo', '')}: {sub.get('descricao', '')}"
                textos_subelementos.append(texto)

            # Gerar embeddings para todos os subelementos
            subelementos_emb = self.embedding_model.encode(textos_subelementos)

            # Calcular similaridade de cosseno
            similaridades = cosine_similarity([descricao_emb], subelementos_emb)[0]

            # Atribuir pontuações aos subelementos
            for i, sub in enumerate(subelementos):
                sub["similaridade"] = float(similaridades[i])

            # Filtrar subelementos com pontuação acima do limiar
            subelementos_compativeis = [s for s in subelementos if s.get("similaridade", 0) >= limiar_compatibilidade]

            # Ordenar por similaridade e retornar os top_n
            subelementos_pontuados = sorted(
                subelementos_compativeis,
                key=lambda x: x.get("similaridade", 0),
                reverse=True
            )

            return subelementos_pontuados[:top_n]

        except Exception as e:
            print(f"Erro ao avaliar subelementos com embeddings: {e}")
            return []

    def selecionar_melhores_subelementos_finais(self, subelementos_compativeis, descricao, top_n=3):
        """
        Usa o LLM para selecionar os melhores subelementos entre todos os compatíveis,
        independentemente da classificação principal
        """
        if not subelementos_compativeis:
            return []

        try:
            # Preparar os dados dos subelementos para enviar ao LLM
            subelementos_info = []
            for i, sub in enumerate(subelementos_compativeis):
                codigo = sub.get("codigo", "")
                codigo_completo = sub.get("codigo_completo", "")
                descricao_sub = sub.get("descricao", "")
                similaridade = sub.get("similaridade", 0)
                classificacao_ref = sub.get("classificacao_ref", {})

                subelementos_info.append({
                    "indice": i,
                    "codigo": codigo,
                    "codigo_completo": codigo_completo,
                    "descricao": descricao_sub,
                    "similaridade_embedding": similaridade,
                    "classificacao_associada": {
                        "codigo": classificacao_ref.get("codigo_completo", ""),
                        "posicao": classificacao_ref.get("posicao", 0)
                    }
                })

            # Criar o prompt para o LLM com instruções para selecionar os melhores subelementos
            prompt = f"""
            Você é um especialista em contabilidade pública e precisa selecionar os 3 subelementos
            que melhor classificam a despesa descrita.

            DESCRIÇÃO DA DESPESA:
            "{descricao}"

            SUBELEMENTOS PRÉ-SELECIONADOS (por análise de similaridade semântica):
            {json.dumps(subelementos_info, indent=2, ensure_ascii=False)}

            TAREFA:
            Analise todos os subelementos apresentados e selecione os 3 que melhor classificam
            a despesa descrita, considerando:

            1. A adequação conceitual do subelemento à descrição da despesa
            2. A especificidade do subelemento (preferir subelementos mais específicos)
            3. A compatibilidade com as normas contábeis do MCASP
            4. A similaridade semântica já calculada (campo "similaridade_embedding")

            Para cada subelemento selecionado, forneça uma justificativa detalhada
            explicando por que ele é adequado para classificar esta despesa específica.

            Responda no seguinte formato JSON:
            {{
              "subelementos_selecionados": [
                {{
                  "indice": X,
                  "justificativa": "Justificativa detalhada sobre por que este subelemento é apropriado"
                }},
                ...
              ]
            }}
            """

            # Enviar ao LLM
            mensagens = [HumanMessage(content=prompt)]
            response = self.llm.invoke(mensagens)

            # Extrair o JSON da resposta
            resultado_llm = self._extrair_json(response.content)

            # Verificar se o resultado tem o formato esperado
            if not resultado_llm or "subelementos_selecionados" not in resultado_llm:
                print("Aviso: O LLM não retornou um formato válido. Retornando os top subelementos por similaridade.")
                # Fallback: retornar os top_n por similaridade
                return sorted(
                    subelementos_compativeis,
                    key=lambda x: x.get("similaridade", 0),
                    reverse=True
                )[:top_n]

            # Mapear as justificativas para os subelementos originais e retornar os selecionados
            subelementos_selecionados = []
            for selecao in resultado_llm["subelementos_selecionados"]:
                indice = selecao.get("indice")
                justificativa = selecao.get("justificativa", "")

                if isinstance(indice, int) and 0 <= indice < len(subelementos_compativeis):
                    subelemento = subelementos_compativeis[indice].copy()
                    subelemento["justificativa"] = justificativa
                    subelementos_selecionados.append(subelemento)

            return subelementos_selecionados

        except Exception as e:
            print(f"Erro ao selecionar subelementos finais com o LLM: {e}")
            # Em caso de erro, retornar os top_n por similaridade como fallback
            return sorted(
                subelementos_compativeis,
                key=lambda x: x.get("similaridade", 0),
                reverse=True
            )[:top_n]

    def classificar_despesa(self, descricao):
        """
        Classificação híbrida:
        1. LLM para classificação principal (c.g.mm.ee)
        2. Embeddings para pré-selecionar subelementos compatíveis
        3. LLM para escolha final dos 3 melhores subelementos
        """
        # Gerar contexto relevante do MCASP
        contexto = self.gerar_contexto_mcasp(descricao)

        # Prompt para o LLM para obter as naturezas de despesa
        prompt = f"""
        Você é um especialista em contabilidade pública e deve classificar a seguinte despesa de acordo com o Manual de Contabilidade Aplicada ao Setor Público (MCASP):

        Descrição da despesa: {descricao}

        Aqui estão as informações relevantes do MCASP que podem ajudar na classificação:

        {contexto}

        **Tarefa: Forneça as 3 melhores possibilidades de classificação para esta despesa, em ordem decrescente de probabilidade.**

        Para cada classificação, defina:
        1. Categoria Econômica (código e descrição)
        2. Grupo de Natureza da Despesa (código e descrição)
        3. Modalidade de Aplicação (código e descrição)
        4. Elemento de Despesa (código e descrição)
        5. Código completo no formato c.g.mm.ee (categoria, grupo, modalidade, elemento)
        6. Justificativa detalhada de por que esta classificação é apropriada

        Classifique as opções como:
        - Opção 1: A mais provável classificação
        - Opção 2: A segunda mais provável classificação
        - Opção 3: A terceira mais provável classificação

        Para cada opção, forneça uma justificativa completa baseada no MCASP, explicando por que esta classificação específica é adequada para a despesa descrita.

        Responda no seguinte formato JSON:
        ```json
        {{
          "classificacoes": [
            {{
              "posicao": 1,
              "classificacao": {{
                "categoria_economica": {{
                  "codigo": "X",
                  "descricao": "Nome da categoria"
                }},
                "grupo_natureza_despesa": {{
                  "codigo": "X",
                  "descricao": "Nome do grupo"
                }},
                "modalidade_aplicacao": {{
                  "codigo": "XX",
                  "descricao": "Nome da modalidade"
                }},
                "elemento_despesa": {{
                  "codigo": "XX",
                  "descricao": "Nome do elemento"
                }}
              }},
              "codigo_completo": "X.X.XX.XX",
              "justificativa": "Justificativa detalhada para esta classificação..."
            }},
            {{
              "posicao": 2,
              "classificacao": {{
                "categoria_economica": {{
                  "codigo": "X",
                  "descricao": "Nome da categoria"
                }},
                "grupo_natureza_despesa": {{
                  "codigo": "X",
                  "descricao": "Nome do grupo"
                }},
                "modalidade_aplicacao": {{
                  "codigo": "XX",
                  "descricao": "Nome da modalidade"
                }},
                "elemento_despesa": {{
                  "codigo": "XX",
                  "descricao": "Nome do elemento"
                }}
              }},
              "codigo_completo": "X.X.XX.XX",
              "justificativa": "Justificativa detalhada para esta classificação..."
            }},
            {{
              "posicao": 3,
              "classificacao": {{
                "categoria_economica": {{
                  "codigo": "X",
                  "descricao": "Nome da categoria"
                }},
                "grupo_natureza_despesa": {{
                  "codigo": "X",
                  "descricao": "Nome do grupo"
                }},
                "modalidade_aplicacao": {{
                  "codigo": "XX",
                  "descricao": "Nome da modalidade"
                }},
                "elemento_despesa": {{
                  "codigo": "XX",
                  "descricao": "Nome do elemento"
                }}
              }},
              "codigo_completo": "X.X.XX.XX",
              "justificativa": "Justificativa detalhada para esta classificação..."
            }}
          ]
        }}
        ```
        """

        # Enviar a solicitação para o LLM
        try:
            mensagens = [HumanMessage(content=prompt)]
            response = self.llm.invoke(mensagens)

            # Extrair o JSON da resposta
            resultado = self._extrair_json(response.content)

            # Verificar estrutura esperada
            if "classificacoes" not in resultado:
                print("Erro: A resposta do modelo não contém o campo 'classificacoes'")
                resultado = {"classificacoes": []}

            # Lista para armazenar todos os subelementos compatíveis
            todos_subelementos_compativeis = []

            # Para cada classificação, buscar os melhores subelementos, com tratamento de erros
            for classificacao in resultado["classificacoes"]:
                try:
                    # Garantir que os campos necessários existem
                    if "classificacao" not in classificacao:
                        classificacao["subelementos"] = []
                        continue

                    # Extrair códigos da classificação
                    classificacao_dict = classificacao["classificacao"]
                    if (not all(campo in classificacao_dict for campo in ["categoria_economica", "grupo_natureza_despesa",
                                                                    "modalidade_aplicacao", "elemento_despesa"])):
                        classificacao["subelementos"] = []
                        continue

                    cat = classificacao_dict["categoria_economica"].get("codigo", "")
                    grp = classificacao_dict["grupo_natureza_despesa"].get("codigo", "")
                    mod = classificacao_dict["modalidade_aplicacao"].get("codigo", "")
                    ele = classificacao_dict["elemento_despesa"].get("codigo", "")

                    # Verificar se todos os códigos estão presentes
                    if not (cat and grp and mod and ele):
                        classificacao["subelementos"] = []
                        continue

                    # Buscar subelementos compatíveis
                    subelementos_compativeis = self.buscar_subelementos_por_natureza(cat, grp, mod, ele)

                    # Se existirem subelementos, encontrar os compatíveis por EMBEDDINGS
                    if subelementos_compativeis:
                        compativeis = self.encontrar_melhores_subelementos_por_embedding(
                            subelementos_compativeis,
                            descricao,
                            top_n=5,  # Pegar mais subelementos por natureza para análise LLM posterior
                            limiar_compatibilidade=0.2
                        )

                        if compativeis:
                            # Adicionar informação da classificação aos subelementos
                            for sub in compativeis:
                                sub["classificacao_ref"] = {
                                    "posicao": classificacao.get("posicao", 0),
                                    "codigo_completo": classificacao.get("codigo_completo", ""),
                                    "categoria": cat,
                                    "grupo": grp,
                                    "modalidade": mod,
                                    "elemento": ele,
                                    "justificativa": classificacao.get("justificativa", "")
                                }

                            # Adicionar à lista global
                            todos_subelementos_compativeis.extend(compativeis)

                    # Manter estrutura original para compatibilidade
                    classificacao["subelementos"] = []

                except Exception as e:
                    print(f"Erro ao processar subelementos para classificação: {e}")
                    classificacao["subelementos"] = []

            # Se temos subelementos compatíveis, enviar ao LLM para escolha final
            if todos_subelementos_compativeis:
                melhores_subelementos = self.selecionar_melhores_subelementos_finais(
                    todos_subelementos_compativeis,
                    descricao
                )

                # Criar resultado final apenas com as classificações dos subelementos escolhidos
                resultado_final = {"classificacoes": []}

                # Mapear códigos de classificação para evitar duplicações
                codigos_processados = set()

                # Adicionar classificações correspondentes aos subelementos escolhidos
                for sub in melhores_subelementos:
                    class_ref = sub.get("classificacao_ref", {})
                    codigo = class_ref.get("codigo_completo", "")

                    # Evitar duplicação de classificações
                    if codigo and codigo not in codigos_processados:
                        # Encontrar a classificação completa original
                        for class_orig in resultado["classificacoes"]:
                            if class_orig.get("codigo_completo") == codigo:
                                # Copiar classificação
                                nova_class = class_orig.copy()

                                # Limpar subelementos existentes (se houver)
                                nova_class["subelementos"] = []

                                # Adicionar à lista de classificações finais
                                resultado_final["classificacoes"].append(nova_class)
                                codigos_processados.add(codigo)
                                break

                # Adicionar os subelementos escolhidos como um campo separado no resultado
                resultado_final["melhores_subelementos"] = [
                    {
                        "codigo": sub.get("codigo", ""),
                        "descricao": sub.get("descricao", ""),
                        "codigo_completo": sub.get("codigo_completo", ""),
                        "similaridade": sub.get("similaridade", 0),
                        "classificacao_associada": sub.get("classificacao_ref", {}).get("codigo_completo", ""),
                        "justificativa": sub.get("justificativa", "")
                    } for sub in melhores_subelementos
                ]

                return resultado_final
            else:
                # Se não encontrou subelementos compatíveis, retornar classificações originais
                return resultado

        except Exception as e:
            print(f"Erro ao chamar o LLM: {e}")
            import traceback
            traceback.print_exc()
            return {"classificacoes": []}

    def extrair_texto_pdf(self, pdf_file_path):
        """
        Extrai texto de um arquivo PDF

        Args:
            pdf_file_path: Caminho para o arquivo PDF

        Returns:
            Texto extraído do PDF
        """
        texto_completo = ""

        try:
            with open(pdf_file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                num_paginas = len(pdf_reader.pages)

                for i in range(num_paginas):
                    page = pdf_reader.pages[i]
                    texto_completo += page.extract_text() + "\n\n"

            return texto_completo
        except Exception as e:
            print(f"Erro ao extrair texto do PDF: {e}")
            return ""

    def extrair_descricoes_do_texto(self, texto):
        """
        Extrai descrições de despesas de um texto longo (como um contrato)

        Args:
            texto: Texto do documento

        Returns:
            Texto resumido focando nas descrições de despesas
        """
        # Limitar o tamanho do texto para evitar exceder o contexto máximo do modelo
        texto_limitado = texto[:10000] if len(texto) > 10000 else texto

        # Usar o próprio LLM para extrair as descrições relevantes
        prompt = f"""
        Analise o documento abaixo (que pode ser um Contrato ou Termo de Referência) e extraia objetivamente as informações essenciais necessárias para a correta classificação contábil da despesa pública conforme o Manual de Contabilidade Aplicada ao Setor Público (MCASP). Apresente especificamente:

        1. Tipo do documento analisado (Contrato ou Termo de Referência);
        2. Objeto detalhado do documento (o que exatamente será executado ou adquirido);
        3. Partes envolvidas ou unidades responsáveis (identificando claramente se a entidade que receberá recursos é pública, privada com ou sem fins lucrativos);
        4. Forma prevista de transferência financeira ou pagamento (ex.: aplicação direta, transferência fundo a fundo, transferência para instituições privadas sem fins lucrativos, contrato de gestão, etc.);
        5. Especificação do tipo de despesa (corrente ou capital) com detalhamento dos gastos previstos (contrato de gestão, materiais, serviços contínuos, obras, equipamentos, desenvolvimento de software, etc.).

        Utilize exclusivamente as informações contidas no documento analisado, destacando claramente os trechos relevantes que embasam a classificação recomendada.

        Documento:
        {texto_limitado}

        Forneça apenas as descrições das despesas, cada uma em uma linha.
        """

        # Obter as descrições usando o LLM
        try:
            mensagens = [HumanMessage(content=prompt)]
            response = self.llm.invoke(mensagens)
            return response.content
        except Exception as e:
            print(f"Erro ao extrair descrições do texto: {e}")
            return "Não foi possível extrair descrições do texto."

    def gerar_contexto_mcasp(self, descricao):
        """
        Gera contexto do MCASP usando busca semântica por embeddings

        Args:
            descricao: Descrição da despesa

        Returns:
            Texto com informações relevantes do MCASP
        """
        if self.mcasp_embeddings is None or not self.mcasp_texts:
            print("Embeddings do MCASP não preparados. Gerando contexto básico.")
            return self._gerar_contexto_basico()

        try:

            # Verificar se o modelo de embeddings está inicializado
            if self.embedding_model is None:
                print("Inicializando modelo de embeddings para busca...")
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

            # Criar embedding da descrição
            descricao_emb = self.embedding_model.encode(descricao)

            # Calcular similaridade com todos os itens do MCASP
            similarities = cosine_similarity([descricao_emb], self.mcasp_embeddings)[0]

            # Ordenar por similaridade e pegar os top 10
            top_indices = similarities.argsort()[-10:][::-1]

            # Formatar os resultados para o contexto
            contexto_items = []
            for idx in top_indices:
                item = self.mcasp_texts[idx]
                contexto_items.append(f"Tipo: {item['tipo']}\nCódigo: {item['codigo']}\nDescrição: {item['item']['descricao']}\nDefinição: {item['item'].get('definicao', 'N/A')}")

            # Adicionar informações de estrutura do MCASP
            estrutura = self._gerar_estrutura_mcasp()

            # Combinar tudo em um único contexto
            contexto = estrutura + "\n\nINFORMAÇÕES RELEVANTES DO MCASP BASEADAS EM SIMILARIDADE SEMÂNTICA:\n" + "\n\n".join(contexto_items)

            return contexto
        except Exception as e:
            print(f"Erro ao gerar contexto MCASP: {e}")
            return self._gerar_contexto_basico()

    def _gerar_estrutura_mcasp(self):
        """Gera a estrutura básica do MCASP"""
        return """
            ESTRUTURA MCASP:
            1. Categoria Econômica:
            3 - Despesas Correntes: não contribuem para a formação de bem de capital
            4 - Despesas de Capital: contribuem para a formação de bem de capital

            2. Grupo de Natureza da Despesa:
            1 - Pessoal e Encargos Sociais: despesas com pessoal ativo, inativo e pensionistas
            2 - Juros e Encargos da Dívida: pagamento de juros e encargos de dívidas
            3 - Outras Despesas Correntes: aquisição de material de consumo, pagamento de serviços, etc.
            4 - Investimentos: despesas com softwares, planejamento de obras, imóveis, equipamentos e material permanente
            5 - Inversões Financeiras: aquisição de imóveis ou bens de capital em utilização, aquisição de títulos ou aumento de capital
            6 - Amortização da Dívida: pagamento ou refinanciamento do principal da dívida

            3. Modalidade de Aplicação:
            20 - Transferências à União
            30 - Transferências a Estados e ao Distrito Federal
            40 - Transferências a Municípios
            50 - Transferências a Instituições Privadas sem Fins Lucrativos
            60 - Transferências a Instituições Privadas com Fins Lucrativos
            70 - Transferências a Instituições Multigovernamentais
            80 - Transferências ao Exterior
            90 - Aplicações Diretas
            91 - Aplicações Diretas (operações entre órgãos)
            """

    def _gerar_contexto_basico(self):
        """Gera um contexto básico com a estrutura do MCASP"""
        return self._gerar_estrutura_mcasp() + """
            4. Elementos de Despesa:
            30 - Material de Consumo
            35 - Serviços de Consultoria
            36 - Outros Serviços de Terceiros - Pessoa Física
            39 - Outros Serviços de Terceiros - Pessoa Jurídica
            40 - Serviços de Tecnologia da Informação e Comunicação - Pessoa Jurídica
            51 - Obras e Instalações
            52 - Equipamentos e Material Permanente
            """

    def _extrair_json(self, texto):
        """
        Extrai JSON da resposta do LLM

        Args:
            texto: Texto da resposta

        Returns:
            Dicionário com os dados extraídos
        """
        try:
            # Padrão para encontrar conteúdo JSON entre crases
            match = re.search(r'```json\s*(.*?)\s*```', texto, re.DOTALL)

            if match:
                json_str = match.group(1)
            else:
                # Tentar encontrar chaves diretamente se não houver marcadores de código
                match = re.search(r'(\{.*\})', texto, re.DOTALL)
                if match:
                    json_str = match.group(1)
                else:
                    raise ValueError("Não foi possível extrair JSON da resposta")

            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Se falhar, tentar limpar o texto antes de decodificar
                json_str = re.sub(r'[\n\t]', '', json_str)
                return json.loads(json_str)
        except Exception as e:
            print(f"Erro ao extrair JSON: {e}")
            return {"classificacoes": []}
