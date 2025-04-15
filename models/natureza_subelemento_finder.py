import os
import json
import logging
from typing import List, Dict, Any
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NaturezaSubelementoFinder:
    """
    Classe para buscar naturezas completas (com subelementos) a partir
    de naturezas no nível de elemento.
    """
    
    def __init__(self):
        """Inicializa o buscador de subelementos."""
        self.data_path = os.path.join(active_config.DATA_DIR, 'natureza')
        self.subelementos_file = os.path.join(self.data_path, 'subelementos_goias_processados.json')
        self.subelementos_data = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Tenta carregar o arquivo de subelementos
        self._load_subelementos()
    
    def _load_subelementos(self) -> bool:
        """
        Carrega os dados de subelementos do arquivo JSON.
        
        Returns:
            bool: True se o carregamento foi bem-sucedido, False caso contrário
        """
        try:
            if not os.path.exists(self.subelementos_file):
                self.logger.error(f"Arquivo de subelementos não encontrado: {self.subelementos_file}")
                return False
            
            with open(self.subelementos_file, 'r', encoding='utf-8') as f:
                self.subelementos_data = json.load(f)
            
            self.logger.info(f"Dados de subelementos carregados com sucesso: {len(self.subelementos_data)} registros")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados de subelementos: {str(e)}")
            return False
    
    def find_complete_naturezas(self, natureza_elementos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Busca as naturezas completas (com subelementos) a partir das naturezas
        no nível de elemento.
        
        Args:
            natureza_elementos: Lista de dicionários contendo as naturezas no nível de elemento,
                              cada um com pelo menos a chave 'codigo'
            
        Returns:
            List[Dict[str, Any]]: Lista de dicionários contendo as naturezas completas,
                                cada um com 'codigo_completo' e 'descricao'
        """
        if not self.subelementos_data:
            if not self._load_subelementos():
                self.logger.error("Não foi possível carregar os dados de subelementos")
                return []
        
        result = []
        
        # Para cada natureza no nível de elemento
        for natureza_elem in natureza_elementos:
            # Obtém o código no formato c.g.mm.ee
            codigo_elem = natureza_elem.get('codigo', '')
            
            # Verifica se o código está no formato correto
            if not codigo_elem or not self._validate_codigo_elem(codigo_elem):
                self.logger.warning(f"Código de natureza inválido ou não encontrado: {codigo_elem}")
                continue
            
            # Busca todos os subelementos correspondentes
            subelementos_encontrados = self._filter_subelementos(codigo_elem)
            
            if not subelementos_encontrados:
                self.logger.warning(f"Nenhum subelemento encontrado para o código: {codigo_elem}")
                # Adiciona apenas o elemento original como fallback
                result.append({
                    'codigo_completo': codigo_elem,
                    'descricao': natureza_elem.get('nome', 'Descrição não disponível'),
                    'natureza_elemento': natureza_elem
                })
            else:
                # Adiciona todos os subelementos encontrados
                for sub in subelementos_encontrados:
                    result.append({
                        'codigo_completo': sub.get('codigo_completo', ''),
                        'descricao': sub.get('descricao', 'Descrição não disponível'),
                        'natureza_elemento': natureza_elem
                    })
                    
                self.logger.info(f"Encontrados {len(subelementos_encontrados)} subelementos para {codigo_elem}")
        
        # Ordena por código completo para melhor apresentação
        result.sort(key=lambda x: x.get('codigo_completo', ''))
        
        return result
    
    def _validate_codigo_elem(self, codigo: str) -> bool:
        """
        Valida se o código está no formato c.g.mm.ee
        
        Args:
            codigo: Código da natureza no nível de elemento
            
        Returns:
            bool: True se o código é válido, False caso contrário
        """
        import re
        # Padrão para o formato c.g.mm.ee
        pattern = r'^\d\.\d\.\d{2}\.\d{2}$'
        return bool(re.match(pattern, codigo))
    
    def _filter_subelementos(self, codigo_elem: str) -> List[Dict[str, Any]]:
        """
        Filtra os subelementos com base no código do elemento.
        
        Args:
            codigo_elem: Código da natureza no nível de elemento (c.g.mm.ee)
            
        Returns:
            List[Dict[str, Any]]: Lista de subelementos encontrados
        """
        try:
            # Lista para armazenar os resultados
            result = []
            
            # Percorre todos os subelementos
            for subelemento in self.subelementos_data.get('subelementos'):
                # Verifica se o código_completo começa com o código do elemento
                codigo_completo = subelemento.get('codigo_completo', '')
                
                # Usamos o prefixo para garantir que estamos combinando exatamente com c.g.mm.ee
                if codigo_completo.startswith(codigo_elem + '.'):
                    result.append(subelemento)
                elif codigo_completo == codigo_elem:
                    # Caso especial: o código completo é igual ao código do elemento
                    # (não possui subelemento específico)
                    result.append(subelemento)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro ao filtrar subelementos: {str(e)}")
            return []