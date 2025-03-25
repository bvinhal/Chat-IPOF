# utils/document_utils.py

import os
import logging
from datetime import datetime
from typing import Optional
import re

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> Optional[str]:
    """
    Extrai texto de um arquivo PDF.
    
    Args:
        file_path: Caminho para o arquivo PDF
        
    Returns:
        Optional[str]: Texto extraído ou None em caso de erro
    """
    try:
        from pdfminer.high_level import extract_text
        
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return None
        
        # Extrai o texto do PDF
        text = extract_text(file_path)
        
        # Verifica se conseguiu extrair texto
        if not text or text.strip() == "":
            logger.warning(f"Não foi possível extrair texto do PDF: {file_path}")
            return None
        
        return text
        
    except ImportError:
        logger.error("Biblioteca 'pdfminer' não instalada. Execute 'pip install pdfminer.six'.")
        return None
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return None

def extract_text_from_docx(file_path: str) -> Optional[str]:
    """
    Extrai texto de um arquivo DOCX.
    
    Args:
        file_path: Caminho para o arquivo DOCX
        
    Returns:
        Optional[str]: Texto extraído ou None em caso de erro
    """
    try:
        import docx
        
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return None
        
        # Carrega o documento
        doc = docx.Document(file_path)
        
        # Extrai o texto de cada parágrafo
        texts = [paragraph.text for paragraph in doc.paragraphs]
        
        # Combina os parágrafos em um único texto
        text = "\n".join(texts)
        
        # Verifica se conseguiu extrair texto
        if not text or text.strip() == "":
            logger.warning(f"Não foi possível extrair texto do DOCX: {file_path}")
            return None
        
        return text
        
    except ImportError:
        logger.error("Biblioteca 'python-docx' não instalada. Execute 'pip install python-docx'.")
        return None
    except Exception as e:
        logger.error(f"Erro ao extrair texto do DOCX: {str(e)}")
        return None

def extract_text_from_doc(file_path: str) -> Optional[str]:
    """
    Extrai texto de um arquivo DOC (formato antigo do Word).
    
    Args:
        file_path: Caminho para o arquivo DOC
        
    Returns:
        Optional[str]: Texto extraído ou None em caso de erro
    """
    try:
        # Tenta primeiro com textract (mais completo, mas requer mais dependências)
        try:
            import textract
            text = textract.process(file_path).decode('utf-8')
            return text
        except ImportError:
            logger.warning("Biblioteca 'textract' não instalada. Tentando alternativa...")
        
        # Alternativa: usar antiword (Unix/Linux/Mac)
        try:
            import subprocess
            result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
            else:
                logger.warning(f"antiword falhou: {result.stderr}")
        except Exception as e:
            logger.warning(f"Falha ao usar antiword: {str(e)}")
        
        # Se todas as tentativas falharem, informa o erro
        logger.error(f"Não foi possível extrair texto do DOC: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair texto do DOC: {str(e)}")
        return None

def extract_text_from_txt(file_path: str) -> Optional[str]:
    """
    Extrai texto de um arquivo TXT.
    
    Args:
        file_path: Caminho para o arquivo TXT
        
    Returns:
        Optional[str]: Texto extraído ou None se não encontrado
    """
    try:
        # Verifica se o arquivo existe
        if not os.path.exists(file_path):
            logger.error(f"Arquivo não encontrado: {file_path}")
            return None
        
        # Tenta diferentes codificações para lidar com caracteres especiais
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    text = file.read()
                return text
            except UnicodeDecodeError:
                continue
        
        logger.warning(f"Não foi possível decodificar o arquivo TXT com as codificações conhecidas: {file_path}")
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair texto do TXT: {str(e)}")
        return None

def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    Extrai texto de um arquivo, detectando automaticamente o tipo de arquivo.
    
    Args:
        file_path: Caminho para o arquivo
        
    Returns:
        Optional[str]: Texto extraído ou None em caso de erro
    """
    try:
        # Obtém a extensão do arquivo
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Extrai texto de acordo com a extensão
        if file_ext == '.pdf':
            return extract_text_from_pdf(file_path)
        elif file_ext == '.docx':
            return extract_text_from_docx(file_path)
        elif file_ext == '.doc':
            return extract_text_from_doc(file_path)
        elif file_ext == '.txt':
            return extract_text_from_txt(file_path)
        else:
            logger.error(f"Tipo de arquivo não suportado: {file_ext}")
            return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair texto do arquivo: {str(e)}")
        return None

def extract_valor_monetario(texto: str) -> Optional[float]:
    """
    Extrai valor monetário de um texto, seguindo o padrão brasileiro (vírgula para separar casas decimais).
    
    Args:
        texto: Texto para extrair o valor
        
    Returns:
        Optional[float]: Valor extraído ou None se não encontrado
    """
    try:
        # Primeiro procura por padrões mais específicos com palavra "valor"
        padrao_valor = r'valor\s+(?:total|global|estimado|contratual)?\s*(?:de|:)?\s*(?:R\$\s*)?([\d\.]+,\d{1,2}|[\d\.]+|\d+)'
        padrao_no_valor = r'no\s+valor\s+(?:total|global|estimado)?\s*(?:de|:)?\s*(?:R\$\s*)?([\d\.]+,\d{1,2}|[\d\.]+|\d+)'
        
        # Tenta os padrões mais específicos primeiro
        for padrao in [padrao_no_valor, padrao_valor]:
            match = re.search(padrao, texto, re.IGNORECASE)
            if match:
                valor_str = match.group(1)
                logger.info(f"Valor monetário encontrado: {valor_str}")
                
                # Converte para formato que o Python entende
                # Remove espaços e caracteres extras
                valor_str = valor_str.strip()
                
                # Trata separadores decimais
                if ',' in valor_str:
                    partes = valor_str.split(',')
                    parte_inteira = partes[0].replace('.', '')  # Remove pontos de milhar
                    parte_decimal = partes[1] if len(partes) > 1 else '0'
                    valor_str = f"{parte_inteira}.{parte_decimal}"
                else:
                    # Se não tem vírgula, assume valor inteiro
                    valor_str = valor_str.replace('.', '')
                
                logger.info(f"Valor monetário convertido para processamento: {valor_str}")
                
                try:
                    valor = float(valor_str)
                    return valor
                except ValueError:
                    logger.warning(f"Não foi possível converter '{valor_str}' para float")
                    continue
        
        # Procura por padrões genéricos com R$
        padrao_reais = r'R\$\s*([\d\.]+,\d{1,2}|[\d\.]+|\d+)'
        match = re.search(padrao_reais, texto)
        if match:
            valor_str = match.group(1)
            logger.info(f"Valor monetário com R$ encontrado: {valor_str}")
            
            # Mesmo processo de conversão
            if ',' in valor_str:
                partes = valor_str.split(',')
                parte_inteira = partes[0].replace('.', '')
                parte_decimal = partes[1] if len(partes) > 1 else '0'
                valor_str = f"{parte_inteira}.{parte_decimal}"
            else:
                valor_str = valor_str.replace('.', '')
            
            try:
                return float(valor_str)
            except ValueError:
                pass
        
        # Procura por números grandes que poderiam ser valores monetários
        # Este é um último recurso para textos sem contexto explícito
        numeros = re.findall(r'(\d{4,}(?:,\d{1,2})?)', texto)
        for num in numeros:
            logger.info(f"Número possível de ser valor monetário: {num}")
            
            # Mesmo processo de conversão
            if ',' in num:
                partes = num.split(',')
                parte_inteira = partes[0].replace('.', '')
                parte_decimal = partes[1] if len(partes) > 1 else '0'
                valor_str = f"{parte_inteira}.{parte_decimal}"
            else:
                valor_str = num.replace('.', '')
            
            try:
                valor = float(valor_str)
                # Aplica filtro para evitar falsos positivos
                if 1000 <= valor <= 1_000_000_000:
                    return valor
            except ValueError:
                continue
        
        logger.info("Nenhum valor monetário encontrado no texto")
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair valor monetário: {str(e)}")
        return None
                        
def extract_meses(texto: str) -> Optional[int]:
    """
    Extrai quantidade de meses do texto.
    
    Args:
        texto: Texto para extrair os meses
        
    Returns:
        Optional[int]: Quantidade de meses extraída ou None se não encontrada
    """
    try:
        # Padrões para identificar períodos de tempo em meses
        padroes = [
            r'(?:período|prazo|vigência|duração|tempo)\s+(?:de|:)?\s+(\d+)\s+(?:meses|mês)',
            r'(?:contrato|prestação)\s+(?:de|por|:)?\s+(\d+)\s+(?:meses|mês)',
            r'(?:validade|vigência)\s+(?:de|:)?\s+(\d+)\s+(?:meses|mês)',
            r'(?:durante|por)\s+(\d+)\s+(?:meses|mês)',
            r'(\d+)\s+(?:meses|mês)\s+(?:de|para)\s+(?:execução|prestação|contratação|vigência|duração)'
        ]
        
        for padrao in padroes:
            match = re.search(padrao, texto, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        # Padrão para períodos em anos, convertendo para meses
        padrao_anos = r'(?:período|prazo|vigência|duração|tempo)\s+(?:de|:)?\s+(\d+)\s+(?:anos|ano)'
        match = re.search(padrao_anos, texto, re.IGNORECASE)
        if match:
            return int(match.group(1)) * 12
        
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair meses: {str(e)}")
        return None

def extract_data_inicio(texto: str) -> Optional[datetime]:
    """
    Extrai a data de início de desembolso/vigência de um texto.
    
    Args:
        texto: Texto para extrair a data
        
    Returns:
        Optional[datetime]: Data de início extraída ou None se não encontrada
    """
    try:
        # Padrões para identificar datas de início
        padrao_data = r'(\d{2}[/.-]\d{2}[/.-]\d{4}|\d{2}[/.-]\d{2}[/.-]\d{2})'
        
        # Padrões de contexto para data de início
        padroes_contexto = [
            r'(?:data\s+(?:de|do)\s+(?:início|inicio|desembolso|pagamento|vigência|vigencia))\s*(?:do contrato|da despesa|do pagamento|da vigência|da execução|:)?\s*' + padrao_data,
            r'(?:início|inicio|começo|desembolso inicial|primeira parcela)\s+(?:em|na data|no dia|a partir de|previsto para)\s*(?::)?\s*' + padrao_data,
            r'(?:início|inicio)\s+(?:dos|das|de)\s+(?:pagamentos|desembolsos|serviços|atividades)\s*(?::)?\s*' + padrao_data,
            r'(?:primeiro|1º|1o)\s+(?:pagamento|desembolso)\s*(?::)?\s*' + padrao_data,
            r'(?:vigência|vigencia|prazo)\s+(?:contratual|do contrato)\s*(?:a partir de|iniciando em)\s*(?::)?\s*' + padrao_data,
            r'(?:contrato|despesa)\s+(?:inicia(?:-se)?|começa|tem início)\s*(?:em|no dia|na data|a partir de)\s*(?::)?\s*' + padrao_data,
            r'(?:a partir de|desde)\s*' + padrao_data
        ]
        
        # Tenta encontrar com contexto específico primeiro
        for padrao in padroes_contexto:
            match = re.search(padrao, texto.lower())
            if match:
                data_str = match.group(1)
                # Tenta converter a string para data
                try:
                    # Normaliza o formato da data
                    data_str = re.sub(r'[.-]', '/', data_str)
                    
                    # Verifica o formato e converte
                    if re.match(r'\d{2}/\d{2}/\d{4}', data_str):
                        dia, mes, ano = map(int, data_str.split('/'))
                        return datetime(ano, mes, dia)
                    elif re.match(r'\d{2}/\d{2}/\d{2}', data_str):
                        dia, mes, ano = map(int, data_str.split('/'))
                        # Ajusta para 2000 se o ano for pequeno
                        if ano < 50:
                            ano += 2000
                        else:
                            ano += 1900
                        return datetime(ano, mes, dia)
                except (ValueError, IndexError):
                    continue  # Tenta o próximo padrão se a conversão falhar
        
        # Tenta outras abordagens mais específicas se não tiver encontrado
        meses_abrev = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
        meses_completos = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
        
        # Combinando os meses para um padrão mais completo
        meses_str = '|'.join(meses_abrev + meses_completos)
        
        # Padrão para datas por extenso (ex: 15 de janeiro de 2023)
        padrao_extenso = r'(\d{1,2})\s+(?:de\s+)?(' + meses_str + r')(?:\s+de)?\s+(\d{4}|\d{2})'
        
        # Busca por datas por extenso no contexto de início
        for contexto in ['início', 'inicio', 'começo', 'data inicial', 'vigência', 'vigencia', 'a partir de']:
            padrao = r'(?:' + contexto + r')\s+(?:em|no dia|na data|a partir de|previsto para)?\s*(?::)?\s*' + padrao_extenso
            match = re.search(padrao, texto.lower())
            if match:
                try:
                    dia = int(match.group(1))
                    mes_str = match.group(2).lower()
                    ano = int(match.group(3))
                    
                    # Ajusta o ano se necessário
                    if ano < 50:
                        ano += 2000
                    elif ano < 100:
                        ano += 1900
                    
                    # Determina o número do mês
                    if mes_str in meses_abrev:
                        mes = meses_abrev.index(mes_str) + 1
                    else:
                        mes = meses_completos.index(mes_str) + 1
                    
                    return datetime(ano, mes, dia)
                except (ValueError, IndexError):
                    pass
        
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair data de início: {str(e)}")
        return None

def extract_data_termino(texto: str) -> Optional[datetime]:
    """
    Extrai a data de término/conclusão de um texto.
    
    Args:
        texto: Texto para extrair a data
        
    Returns:
        Optional[datetime]: Data de término extraída ou None se não encontrada
    """
    try:
        # Padrões para identificar datas
        padrao_data = r'(\d{2}[/.-]\d{2}[/.-]\d{4}|\d{2}[/.-]\d{2}[/.-]\d{2})'
        
        # Padrões de contexto para data de término
        padroes_contexto = [
            r'(?:data\s+(?:de|do)\s+(?:término|termino|fim|conclusão|conclusao|encerramento))\s*(?:do contrato|da despesa|do pagamento|da vigência|da execução|:)?\s*' + padrao_data,
            r'(?:término|termino|fim|encerramento|conclusão|conclusao)\s+(?:em|na data|no dia|previsto para)\s*(?::)?\s*' + padrao_data,
            r'(?:até|ate)\s+(?:o dia|a data)\s*(?::)?\s*' + padrao_data,
            r'(?:última|ultima)\s+(?:parcela|pagamento|desembolso)\s*(?:em|na data|no dia|previsto para)?\s*(?::)?\s*' + padrao_data,
            r'(?:vigência|vigencia|prazo)\s+(?:contratual|do contrato)\s*(?:até|ate)\s*(?::)?\s*' + padrao_data,
            r'(?:contrato|despesa)\s+(?:termina|finaliza|encerra(?:-se)?)\s*(?:em|no dia|na data)\s*(?::)?\s*' + padrao_data,
            r'(?:válido|valido)\s+(?:até|ate)\s*' + padrao_data
        ]
        
        # Tenta encontrar com contexto específico primeiro
        for padrao in padroes_contexto:
            match = re.search(padrao, texto.lower())
            if match:
                data_str = match.group(1)
                # Tenta converter a string para data
                try:
                    # Normaliza o formato da data
                    data_str = re.sub(r'[.-]', '/', data_str)
                    
                    # Verifica o formato e converte
                    if re.match(r'\d{2}/\d{2}/\d{4}', data_str):
                        dia, mes, ano = map(int, data_str.split('/'))
                        return datetime(ano, mes, dia)
                    elif re.match(r'\d{2}/\d{2}/\d{2}', data_str):
                        dia, mes, ano = map(int, data_str.split('/'))
                        # Ajusta para 2000 se o ano for pequeno
                        if ano < 50:
                            ano += 2000
                        else:
                            ano += 1900
                        return datetime(ano, mes, dia)
                except (ValueError, IndexError):
                    continue  # Tenta o próximo padrão se a conversão falhar
        
        # Tenta outras abordagens mais específicas se não tiver encontrado
        meses_abrev = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
        meses_completos = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
        
        # Combinando os meses para um padrão mais completo
        meses_str = '|'.join(meses_abrev + meses_completos)
        
        # Padrão para datas por extenso (ex: 15 de janeiro de 2023)
        padrao_extenso = r'(\d{1,2})\s+(?:de\s+)?(' + meses_str + r')(?:\s+de)?\s+(\d{4}|\d{2})'
        
        # Busca por datas por extenso no contexto de término
        for contexto in ['término', 'termino', 'fim', 'conclusão', 'conclusao', 'encerramento', 'até', 'ate']:
            padrao = r'(?:' + contexto + r')\s+(?:em|no dia|na data|previsto para)?\s*(?::)?\s*' + padrao_extenso
            match = re.search(padrao, texto.lower())
            if match:
                try:
                    dia = int(match.group(1))
                    mes_str = match.group(2).lower()
                    ano = int(match.group(3))
                    
                    # Ajusta o ano se necessário
                    if ano < 50:
                        ano += 2000
                    elif ano < 100:
                        ano += 1900
                    
                    # Determina o número do mês
                    if mes_str in meses_abrev:
                        mes = meses_abrev.index(mes_str) + 1
                    else:
                        mes = meses_completos.index(mes_str) + 1
                    
                    return datetime(ano, mes, dia)
                except (ValueError, IndexError):
                    pass
        
        return None
        
    except Exception as e:
        logger.error(f"Erro ao extrair data de término: {str(e)}")
        return None

def calcular_quantidade_meses(data_inicio: datetime, data_termino: datetime) -> int:
    """
    Calcula a quantidade de meses entre duas datas.
    
    Args:
        data_inicio: Data de início
        data_termino: Data de término
        
    Returns:
        int: Quantidade de meses entre as datas
    """
    try:
        # Calcula a diferença em meses
        meses = (data_termino.year - data_inicio.year) * 12 + (data_termino.month - data_inicio.month)
        
        # Ajusta se o dia do mês de término for menor que o dia do mês de início
        if data_termino.day < data_inicio.day:
            meses -= 1
        
        # Garante pelo menos 1 mês
        return max(1, meses + 1)  # +1 porque estamos contando meses completos
    except Exception as e:
        logger.error(f"Erro ao calcular quantidade de meses: {str(e)}")
        return 1  # Retorna 1 em caso de erro