# utils/document_utils.py

import os
import logging
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
        Optional[str]: Texto extraído ou None em caso de erro
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
    Extrai valor monetário de um texto.
    
    Args:
        texto: Texto para extrair o valor
        
    Returns:
        Optional[float]: Valor extraído ou None se não encontrado
    """
    try:
        # Padrões para identificar valores monetários
        # Busca por "R$ X.XXX,XX" ou "X.XXX,XX reais" ou variações
        padrao_valor = r'R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)'
        padrao_valor_alt = r'(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)\s*(?:reais|REAIS)'
        
        # Busca por palavras-chave antes de valores
        padrao_contexto = [
            r'valor\s+(?:total|global|estimado|contratual)?\s*(?:de|:)?\s*R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)',
            r'(?:total|global|estimado|contratual)\s+(?:de|:)?\s*R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)',
            r'orçamento\s+(?:de|:)?\s*R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)',
            r'custo\s+(?:total|estimado)?\s*(?:de|:)?\s*R\$\s*(\d{1,3}(?:\.\d{3})*(?:,\d{1,2})?)'
        ]
        
        # Primeiro, tenta encontrar com contexto
        for padrao in padrao_contexto:
            match = re.search(padrao, texto, re.IGNORECASE)
            if match:
                valor_str = match.group(1).replace('.', '').replace(',', '.')
                return float(valor_str)
        
        # Se não encontrou com contexto, tenta padrões simples
        match = re.search(padrao_valor, texto)
        if match:
            valor_str = match.group(1).replace('.', '').replace(',', '.')
            return float(valor_str)
        
        match = re.search(padrao_valor_alt, texto)
        if match:
            valor_str = match.group(1).replace('.', '').replace(',', '.')
            return float(valor_str)
        
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