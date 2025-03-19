# controllers/ipof_controller.py

import os
import re
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, Union
import json

import openai
from openai import OpenAI
from config import active_config
from models.ipof_model import IPOF, ParcelaIPOF
from models.natureza_classifier import NaturezaClassifier
from controllers.chat_controller import ChatController
from utils.document_utils import extract_text_from_file, extract_valor_monetario, extract_meses
from utils.ipof_html_generator import IPOFHtmlGenerator

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IPOFController:
    """
    Controlador para gerenciar o fluxo de criação de IPOF.
    """
    
    # Estados do fluxo de criação do IPOF
    ESTADOS = {
        'INATIVO': 0,
        'DESCRICAO': 1,
        'NATUREZA': 2,
        'VALOR': 3,
        'MESES': 4,
        'PROCESSO': 5,
        'DOTACAO': 6,
        'DATA_INICIO': 7,
        'CONFIRMACAO': 8
    }
    
    def __init__(self, chat_controller: ChatController = None):
        """
        Inicializa o controlador de IPOF.
        
        Args:
            chat_controller: Controlador de chat para referenciar o modelo de IA atual
        """
        self.chat_controller = chat_controller
        self.estado_atual = self.ESTADOS['INATIVO']
        self.classificador = NaturezaClassifier(active_config.DEFAULT_MODEL)
        self.classificador.load_model()
        
        # Dados temporários do IPOF em criação
        self.dados_temp = {
            'descricao': '',
            'natureza_despesa': '',
            'valor_total': 0.0,
            'quantidade_meses': 0,
            'numero_processo': '',
            'dotacao_orcamentaria': '',
            'data_inicio': None
        }
        
        # IPOF atual
        self.ipof_atual = None
        
        # Gerador de HTML para IPOF
        self.html_generator = IPOFHtmlGenerator()
        
        # Nome do arquivo HTML gerado (se houver)
        self.html_file_name = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def iniciar_criacao_ipof(self) -> str:
        """
        Inicia o fluxo de criação de um novo IPOF.
        
        Returns:
            str: Mensagem de resposta para o usuário
        """
        # Reinicia o estado e os dados temporários
        self.estado_atual = self.ESTADOS['DESCRICAO']
        self.dados_temp = {
            'descricao': '',
            'natureza_despesa': '',
            'valor_total': 0.0,
            'quantidade_meses': 0,
            'numero_processo': '',
            'dotacao_orcamentaria': '',
            'data_inicio': None
        }
        self.ipof_atual = None
        self.html_file_name = None
        
        # Retorna mensagem solicitando a descrição
        return (
            "Vamos iniciar a criação de um novo IPOF.\n\n"
            "Por favor, informe a descrição da despesa ou anexe um documento "
            "(Termo de Referência, ETP, Contrato ou outro) que contenha a descrição.\n\n"
            "O documento pode ser nos formatos DOC, DOCX ou PDF.\n\n"
            "A qualquer momento, você pode digitar 'cancelar' para interromper a criação do IPOF "
            "ou 'reiniciar' para começar novamente."
        )
    
    def processar_mensagem(self, mensagem: str, ai_model=None) -> str:
        """
        Processa uma mensagem do usuário no fluxo de criação do IPOF.
        
        Args:
            mensagem: Mensagem do usuário
            ai_model: Modelo de IA para processamento de texto
            
        Returns:
            str: Resposta ao usuário
        """
        # Verifica comandos especiais
        if mensagem.lower() == 'cancelar':
            return self._cancelar_criacao()
        
        if mensagem.lower() == 'reiniciar':
            return self.iniciar_criacao_ipof()
        
        # Processa a mensagem de acordo com o estado atual
        if self.estado_atual == self.ESTADOS['DESCRICAO']:
            return self._processar_descricao(mensagem, ai_model)
        
        elif self.estado_atual == self.ESTADOS['NATUREZA']:
            return self._processar_natureza(mensagem)
        
        elif self.estado_atual == self.ESTADOS['VALOR']:
            return self._processar_valor(mensagem)
        
        elif self.estado_atual == self.ESTADOS['MESES']:
            return self._processar_meses(mensagem)
        
        elif self.estado_atual == self.ESTADOS['PROCESSO']:
            return self._processar_processo(mensagem)
        
        elif self.estado_atual == self.ESTADOS['DOTACAO']:
            return self._processar_dotacao(mensagem)
        
        elif self.estado_atual == self.ESTADOS['DATA_INICIO']:
            return self._processar_data_inicio(mensagem)
        
        elif self.estado_atual == self.ESTADOS['CONFIRMACAO']:
            return self._processar_confirmacao(mensagem)
        
        else:
            return "Erro: Estado inválido na criação do IPOF. Por favor, tente novamente."
    
    def processar_arquivo(self, arquivo_path: str, tipo_arquivo: str, ai_model=None) -> str:
        """
        Processa um arquivo anexado pelo usuário.
        
        Args:
            arquivo_path: Caminho para o arquivo
            tipo_arquivo: Tipo do arquivo (doc, pdf, etc.)
            ai_model: Modelo de IA para processamento de texto
            
        Returns:
            str: Resposta ao usuário
        """
        # Verifica se estamos no estado correto
        if self.estado_atual != self.ESTADOS['DESCRICAO']:
            return "Por favor, anexe o arquivo quando solicitada a descrição da despesa."
        
        try:
            # Extrai o texto do arquivo
            texto_extraido = extract_text_from_file(arquivo_path)
            
            if not texto_extraido:
                return "Não foi possível extrair texto do arquivo. Por favor, informe a descrição manualmente."
            
            # Processa o texto extraído
            return self._processar_descricao(texto_extraido, ai_model, fonte="arquivo")
            
        except Exception as e:
            self.logger.error(f"Erro ao processar arquivo: {str(e)}")
            return f"Erro ao processar o arquivo: {str(e)}. Por favor, informe a descrição manualmente."
    
    def _resumir_texto(self, texto: str, ai_model) -> Dict[str, Any]:
        """
        Resumir texto usando o modelo de IA.
        
        Args:
            texto: Texto a ser resumido
            ai_model: Modelo de IA para resumir o texto
                
        Returns:
            Dict[str, Any]: Resumo, valor identificado e quantidade de meses identificada
        """
        try:
            # Inicializa o cliente OpenAI
            client = OpenAI(api_key=active_config.OPENAI_API_KEY)
            
            # Cria um prompt para resumir o texto
            prompt = (
                    f"Resuma este arquivo destacando os itens mais importantes. Procure informações como objeto da contratação "
                    f"para integrae o resumo, sendo esta informação a primeira que deve aparecer na resposta. Dentre estes itens, "
                    f"tente encontrar valores monetário globais que possamos entender e periodos de tempo que nos permita encontrar "
                    f"um padrão de quantidade de meses. Seja criado mais duas linhas sendo uma contendo o valor total identficado e "
                    f"a outra o tempo de validade do contrato ou coisa parecida. Este padrão deverá ser Valor total da despesa: R$ e "
                    f"Quantidade de meses previstos: . Caso não encontrem tais informações, deverá ser informado não encontrado no "
                    f"lugar dos valores. No resumo não coloque caracteres especiais como * para serparar os principais itens pedidos "
                    f"que foram encontrados :\n\n{texto}"
            )
            
            # Chama a API da OpenAI para resumir o texto usando GPT-3.5 Turbo
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", #gpt-4-turbo
                messages=[
                    {"role": "system", "content": "Você é um especialista em finanças públicas e orçamento governamental."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3  # Temperatura baixa para manter o resumo mais factual
            )
            
            # Extrai o resumo da resposta
            resumo = response.choices[0].message.content.strip()
            
            # Limita o resumo a 500 caracteres
            if len(resumo) > 500:
                resumo = resumo[:497] + "..."
            
            # Extrai valor e meses do texto original
            valor_identificado = extract_valor_monetario(texto)
            meses_identificados = extract_meses(texto)
            
            return {
                'resumo': resumo,
                'valor_identificado': valor_identificado,
                'meses_identificados': meses_identificados
            }
                
        except Exception as e:
            self.logger.error(f"Erro ao resumir texto com GPT-3.5 Turbo: {str(e)}")
            # Truncamento simples em caso de erro
            resumo = texto[:500] + "..." if len(texto) > 500 else texto
            return {
                'resumo': resumo,
                'valor_identificado': extract_valor_monetario(texto),
                'meses_identificados': extract_meses(texto)
            }
                
    def _processar_descricao(self, conteudo: str, ai_model=None, fonte: str = "mensagem") -> str:
        """
        Processa a descrição da despesa.
        
        Args:
            conteudo: Descrição informada pelo usuário ou extraída de arquivo
            ai_model: Modelo de IA para processamento
            fonte: Fonte do conteúdo ('mensagem' ou 'arquivo')
            
        Returns:
            str: Resposta ao usuário
        """
        # Resumir o texto
        resultado = self._resumir_texto(conteudo, ai_model)
        resumo = resultado['resumo']
        valor_identificado = resultado['valor_identificado']
        meses_identificados = resultado['meses_identificados']
        
        # Armazena o resumo
        self.dados_temp['descricao'] = resumo
        
        # Classifica a natureza de despesa
        sugestoes_natureza = []
        try:
            if self.classificador.is_trained:
                predicoes = self.classificador.predict(resumo, top_k=1)
                if predicoes:
                    sugestao = predicoes[0]
                    sugestoes_natureza.append(sugestao)
        except Exception as e:
            self.logger.error(f"Erro ao classificar natureza: {str(e)}")
        
        # Monta resposta para o usuário
        fonte_msg = "o arquivo fornecido" if fonte == "arquivo" else "sua descrição"
        resposta = f"Resumo da descrição extraída de {fonte_msg}:\n\n{resumo}\n\n"
        
        # Informações sobre o valor
        if valor_identificado:
            resposta += f"Valor total da despesa: R$ {valor_identificado:.2f}\n"
            self.dados_temp['valor_total'] = valor_identificado
        else:
            resposta += "Valor total da despesa: Não encontrado\n"
        
        # Informações sobre o período
        if meses_identificados:
            resposta += f"Quantidade de meses previstos: {meses_identificados}\n"
            self.dados_temp['quantidade_meses'] = meses_identificados
        else:
            resposta += "Quantidade de meses previstos: Não encontrado\n"
        
        resposta += "\n"
        
        # Avança para o próximo estado (natureza)
        self.estado_atual = self.ESTADOS['NATUREZA']
        
        # Sugestão de natureza de despesa
        if sugestoes_natureza:
            sugestao = sugestoes_natureza[0]
            self.dados_temp['natureza_despesa'] = sugestao['codigo']
            resposta += (
                f"Para essa descrição, sugiro a seguinte natureza de despesa:\n"
                f"{sugestao['codigo']} - {sugestao['nome']}\n"
                f"(Confiança: {sugestao['confianca']:.2%})\n\n"
                f"Você aceita esta natureza sugerida? Responda 'sim' ou informe uma natureza mais adequada."
            )
        else:
            resposta += "Por favor, informe a natureza de despesa adequada para esta descrição."
        
        return resposta
    
    def _processar_natureza(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre a natureza de despesa.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        # Se o usuário respondeu "sim", mantém a natureza sugerida que já está armazenada
        if mensagem.lower() in ['sim', 's', 'yes', 'y']:
            # Verifica se temos natureza sugerida
            if not self.dados_temp['natureza_despesa']:
                # Se não tivermos, pede para o usuário informar
                return "Por favor, informe o código da natureza de despesa."
        else:
            # Se o usuário informou outra natureza, armazena
            self.dados_temp['natureza_despesa'] = mensagem
        
        # Avança para o próximo estado
        self.estado_atual = self.ESTADOS['VALOR']
        
        # Se já temos o valor, pula para o próximo estado
        if self.dados_temp['valor_total'] > 0:
            return self._avancar_com_valor_existente()
        
        # Se não temos o valor, pergunta ao usuário
        return "Qual é o valor total da despesa? (Informe apenas números, ex: 1000,00)"
    
    def _avancar_com_valor_existente(self) -> str:
        """
        Avança para o próximo estado após o valor quando já temos um valor identificado.
        
        Returns:
            str: Próxima pergunta ao usuário
        """
        # Avança para o próximo estado
        self.estado_atual = self.ESTADOS['MESES']
        
        # Se já temos a quantidade de meses, pula para o próximo estado
        if self.dados_temp['quantidade_meses'] > 0:
            return self._avancar_com_meses_existentes()
        
        # Se não temos a quantidade de meses, pergunta ao usuário
        return "Qual é a quantidade de meses previstos para esta despesa?"
    
    def _avancar_com_meses_existentes(self) -> str:
        """
        Avança para o próximo estado após os meses quando já temos meses identificados.
        
        Returns:
            str: Próxima pergunta ao usuário
        """
        # Avança para o próximo estado
        self.estado_atual = self.ESTADOS['PROCESSO']
        
        # Pergunta pelo número do processo
        return "Qual é o número do processo?"
    
    def _processar_valor(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre o valor total da despesa.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        try:
            # Limpa a mensagem e substitui vírgula por ponto
            valor_str = mensagem.strip().replace('R$', '').replace('.', '').replace(',', '.')
            
            # Converte para float
            valor = float(valor_str)
            
            # Armazena o valor
            self.dados_temp['valor_total'] = valor
            
            # Avança para o próximo estado
            self.estado_atual = self.ESTADOS['MESES']
            
            # Se já temos a quantidade de meses, pula para o próximo estado
            if self.dados_temp['quantidade_meses'] > 0:
                return self._avancar_com_meses_existentes()
            
            # Se não temos a quantidade de meses, pergunta ao usuário
            return "Qual é a quantidade de meses previstos para esta despesa?"
            
        except ValueError:
            return "Por favor, informe um valor numérico válido (ex: 1000,00)."
        except Exception as e:
            self.logger.error(f"Erro ao processar valor: {str(e)}")
            return "Ocorreu um erro ao processar o valor. Por favor, tente novamente."
    
    def _processar_meses(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre a quantidade de meses.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        try:
            # Converte para inteiro
            meses = int(mensagem.strip())
            
            # Armazena a quantidade de meses
            self.dados_temp['quantidade_meses'] = meses
            
            # Avança para o próximo estado
            self.estado_atual = self.ESTADOS['PROCESSO']
            
            # Pergunta pelo número do processo
            return "Qual é o número do processo?"
            
        except ValueError:
            return "Por favor, informe um número inteiro válido para a quantidade de meses."
        except Exception as e:
            self.logger.error(f"Erro ao processar meses: {str(e)}")
            return "Ocorreu um erro ao processar a quantidade de meses. Por favor, tente novamente."
    
    def _processar_processo(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre o número do processo.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        # Armazena o número do processo
        self.dados_temp['numero_processo'] = mensagem.strip()
        
        # Avança para o próximo estado
        self.estado_atual = self.ESTADOS['DOTACAO']
        
        # Pergunta pela dotação orçamentária
        return "Qual é a dotação orçamentária?"
    
    def _processar_dotacao(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre a dotação orçamentária.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        # Armazena a dotação orçamentária
        self.dados_temp['dotacao_orcamentaria'] = mensagem.strip()
        
        # Avança para o próximo estado
        self.estado_atual = self.ESTADOS['DATA_INICIO']
        
        # Pergunta pela data de início do desembolso
        return "Qual é a data de início do desembolso? (formato: dd/mm/aaaa)"
    
    def _processar_data_inicio(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre a data de início.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Próxima pergunta ao usuário
        """
        try:
            # Tenta converter a string para data
            data_str = mensagem.strip()
            
            # Verifica formato da data (dd/mm/aaaa)
            if not re.match(r'\d{2}/\d{2}/\d{4}', data_str):
                return "Por favor, informe a data no formato dd/mm/aaaa."
            
            # Converte para datetime
            dia, mes, ano = map(int, data_str.split('/'))
            data_inicio = datetime(ano, mes, dia)
            
            # Armazena a data de início
            self.dados_temp['data_inicio'] = data_inicio
            
            # Gera o IPOF
            self._gerar_ipof()
            
            # Avança para o estado de confirmação
            self.estado_atual = self.ESTADOS['CONFIRMACAO']
            
            # Pergunta se o usuário deseja visualizar o IPOF
            return (
                "Todas as informações foram coletadas. O IPOF foi gerado.\n\n"
                "Você deseja visualizar o IPOF? (sim/não)"
            )
            
        except ValueError:
            return "Data inválida. Por favor, informe a data no formato dd/mm/aaaa."
        except Exception as e:
            self.logger.error(f"Erro ao processar data: {str(e)}")
            return "Ocorreu um erro ao processar a data. Por favor, tente novamente."
    
    def _processar_confirmacao(self, mensagem: str) -> str:
        """
        Processa a resposta do usuário sobre a visualização do IPOF.
        
        Args:
            mensagem: Resposta do usuário
            
        Returns:
            str: Resposta ao usuário
        """
        mensagem_lower = mensagem.lower()
        
        if mensagem_lower in ['sim', 's', 'yes', 'y', 'visualizar']:
            # Mostra o IPOF formatado
            return self._visualizar_ipof()
        
        elif mensagem_lower in ['json']:
            # Retorna o IPOF em formato JSON
            return self._gerar_json()
        
        elif mensagem_lower in ['novo', 'reiniciar', 'novo ipof']:
            # Inicia um novo IPOF
            return self.iniciar_criacao_ipof()
        
        else:
            # Finaliza e volta ao chat normal
            self.estado_atual = self.ESTADOS['INATIVO']
            return (
                "A criação do IPOF foi concluída. Voltando ao chat normal.\n\n"
                "Você pode criar um novo IPOF a qualquer momento digitando algo como "
                "'criar IPOF', 'novo IPOF' ou similar."
            )
    
    def _gerar_ipof(self) -> None:
        """
        Gera o IPOF com base nos dados coletados.
        """
        try:
            # Cria o IPOF
            self.ipof_atual = IPOF(
                numero_processo=self.dados_temp['numero_processo'],
                descricao=self.dados_temp['descricao'],
                valor_total=self.dados_temp['valor_total']
            )
            
            # Extrai a unidade orçamentária
            self.ipof_atual.extrair_unidade_orcamentaria(self.dados_temp['dotacao_orcamentaria'])
            
            # Cria as parcelas
            self.ipof_atual.criar_parcelas_automaticas(
                quantidade_meses=self.dados_temp['quantidade_meses'],
                data_inicio=self.dados_temp['data_inicio'],
                dotacao_orcamentaria=self.dados_temp['dotacao_orcamentaria'],
                natureza_despesa=self.dados_temp['natureza_despesa']
            )
            
            # Gera o arquivo HTML do IPOF
            self.html_file_name = self._gerar_html_ipof()
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar IPOF: {str(e)}")
    
    def _gerar_html_ipof(self) -> Optional[str]:
        """
        Gera um arquivo HTML para o IPOF atual.
        
        Returns:
            Optional[str]: Nome do arquivo HTML gerado ou None em caso de erro
        """
        if not self.ipof_atual:
            self.logger.error("Tentativa de gerar HTML com IPOF não inicializado")
            return None
        
        try:
            # Converte o IPOF para dicionário
            ipof_dict = self.ipof_atual.to_dict()
            
            # Gera o HTML usando o gerador
            return self.html_generator.generate_html(ipof_dict)
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar HTML do IPOF: {str(e)}")
            return None
    
    def _visualizar_ipof(self) -> str:
        """
        Retorna um resumo do IPOF com link para versão HTML completa.
        
        Returns:
            str: Resumo do IPOF com link para visualização HTML
        """
        if not self.ipof_atual:
            return "Erro: IPOF não encontrado. Por favor, tente criar novamente."
        
        # Criar um resumo do IPOF em vez de mostrar a formatação completa
        ipof = self.ipof_atual
        
        # Resumo com informações principais
        resposta = "### IPOF - Resumo ###\n\n"
        resposta += f"Processo: {ipof.numero_processo}\n"
        resposta += f"Valor Total: R$ {ipof.valor_total:.2f}\n"
        resposta += f"Quantidade de Parcelas: {len(ipof.parcelas)}\n"
        
        # Adiciona o link para o HTML
        if self.html_file_name:
            link_html = f"/ipof/{self.html_file_name}"
            resposta += f"\n\n### Visualização Completa do IPOF ###\n"
            resposta += f"Para visualizar todos os detalhes do IPOF, incluindo todas as parcelas, clique no link abaixo:\n"
            resposta += f"[Abrir IPOF Completo em Nova Página]({link_html})\n\n"
        else:
            resposta += "\n\nOcorreu um erro ao gerar a visualização em HTML. Por favor, tente novamente ou contate o suporte.\n\n"
        
        resposta += (
            "O que deseja fazer agora?\n"
            "- Digite 'novo' para criar um novo IPOF\n"
            "- Digite qualquer outra coisa para voltar ao chat normal"
        )
        
        return resposta
                
    def _gerar_json(self) -> str:
        """
        Retorna o IPOF em formato JSON com link para visualização HTML.
        
        Returns:
            str: IPOF em formato JSON com link para visualização
        """
        if not self.ipof_atual:
            return "Erro: IPOF não encontrado. Por favor, tente criar novamente."
        
        html_link = ""
        if self.html_file_name:
            link_html = f"/ipof/{self.html_file_name}"
            html_link = f"\n\n### Visualização HTML Disponível ###\n"
            html_link += f"Você pode visualizar e imprimir o IPOF em formato HTML clicando no link abaixo:\n"
            html_link += f"[Abrir IPOF em nova página]({link_html})"
        
        return (
            "Aqui está o IPOF em formato JSON:\n\n"
            f"```json\n{self.ipof_atual.to_json()}\n```"
            f"{html_link}\n\n"
            "O que deseja fazer agora?\n"
            "- Digite 'visualizar' para ver o IPOF formatado\n"
            "- Digite 'novo' para criar um novo IPOF\n"
            "- Digite qualquer outra coisa para voltar ao chat normal"
        )
    
    def _cancelar_criacao(self) -> str:
        """
        Cancela a criação do IPOF.
        
        Returns:
            str: Mensagem de cancelamento
        """
        self.estado_atual = self.ESTADOS['INATIVO']
        return "A criação do IPOF foi cancelada. Voltando ao chat normal."
    
    def esta_ativo(self) -> bool:
        """
        Verifica se o fluxo de criação de IPOF está ativo.
        
        Returns:
            bool: True se o fluxo está ativo, False caso contrário
        """
        return self.estado_atual != self.ESTADOS['INATIVO']
    
    def verificar_inicio_ipof(self, mensagem: str) -> bool:
        """
        Verifica se a mensagem indica o desejo de iniciar a criação de um IPOF.
        
        Args:
            mensagem: Mensagem do usuário
            
        Returns:
            bool: True se a mensagem indica intenção de criar IPOF, False caso contrário
        """
        # Lista de padrões que indicam criação de IPOF
        padroes = [
            r'criar\s+(?:um\s+)?ipof',
            r'iniciar\s+(?:um\s+)?ipof',
            r'novo\s+(?:um\s+)?ipof',
            r'fazer\s+(?:um\s+)?ipof',
            r'elaborar\s+(?:um\s+)?ipof',
            r'incluir\s+(?:um\s+)?ipof',
            r'planejar\s+(?:um\s+)?ipof',
            r'programar\s+(?:um\s+)?ipof',
            r'cadastrar\s+(?:um\s+)?ipof',  # Adicionado para sua solicitação
            r'registrar\s+(?:um\s+)?ipof',  # Outra variação comum
            r'quero\s+(?:criar|fazer|elaborar|cadastrar)\s+(?:um\s+)?ipof',
            r'desejo\s+(?:criar|fazer|elaborar|cadastrar)\s+(?:um\s+)?ipof',
            r'gostaria\s+de\s+(?:criar|fazer|elaborar|cadastrar)\s+(?:um\s+)?ipof',
            r'criar\s+(?:um\s+)?planejamento\s+orçamentário',
            r'instrumento\s+de\s+planejamento',
            r'ipof'  # Verificação simples apenas da palavra IPOF isolada
        ]
        
        mensagem_lower = mensagem.lower()
        
        for padrao in padroes:
            if re.search(padrao, mensagem_lower):
                return True
        
        return False