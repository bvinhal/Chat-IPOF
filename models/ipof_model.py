# models/ipof_model.py

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import json
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ParcelaIPOF:
    """
    Classe que representa uma parcela do IPOF.
    """
    
    def __init__(
        self, 
        data_referencia: datetime, 
        data_desembolso: datetime, 
        dotacao_orcamentaria: str, 
        natureza_despesa: str, 
        valor_parcela: float
    ):
        """
        Inicializa uma parcela do IPOF.
        
        Args:
            data_referencia: Data de referência da parcela
            data_desembolso: Data de desembolso da parcela
            dotacao_orcamentaria: Dotação orçamentária
            natureza_despesa: Código da natureza de despesa
            valor_parcela: Valor da parcela
        """
        self.data_referencia = data_referencia
        self.data_desembolso = data_desembolso
        self.dotacao_orcamentaria = dotacao_orcamentaria
        self.natureza_despesa = natureza_despesa
        self.valor_parcela = valor_parcela
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a parcela para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação da parcela como dicionário
        """
        return {
            'data_referencia': self.data_referencia.strftime('%d/%m/%Y'),
            'data_desembolso': self.data_desembolso.strftime('%d/%m/%Y'),
            'dotacao_orcamentaria': self.dotacao_orcamentaria,
            'natureza_despesa': self.natureza_despesa,
            'valor_parcela': self.valor_parcela
        }
    
    def __str__(self) -> str:
        """
        Retorna a representação em string da parcela.
        
        Returns:
            str: Representação em string da parcela
        """
        return (
            f"Data de Referência: {self.data_referencia.strftime('%d/%m/%Y')}\n"
            f"Data de Desembolso: {self.data_desembolso.strftime('%d/%m/%Y')}\n"
            f"Dotação Orçamentária: {self.dotacao_orcamentaria}\n"
            f"Natureza de Despesa: {self.natureza_despesa}\n"
            f"Valor da Parcela: R$ {self.valor_parcela:.2f}"
        )


class IPOF:
    """
    Classe que representa um IPOF (Instrumento de Planejamento, Orçamento e Finanças).
    """
    
    def __init__(
        self, 
        numero_processo: str, 
        descricao: str, 
        valor_total: float, 
        parcelas: List[ParcelaIPOF] = None,
        unidade_orcamentaria: str = ""
    ):
        """
        Inicializa um IPOF.
        
        Args:
            numero_processo: Número do processo
            descricao: Descrição do IPOF
            valor_total: Valor total do IPOF
            parcelas: Lista de parcelas do IPOF
            unidade_orcamentaria: Unidade orçamentária (extraída da dotação)
        """
        self.numero_processo = numero_processo
        self.descricao = descricao
        self.valor_total = valor_total
        self.parcelas = parcelas or []
        self.unidade_orcamentaria = unidade_orcamentaria
    
    def adicionar_parcela(self, parcela: ParcelaIPOF) -> None:
        """
        Adiciona uma parcela ao IPOF.
        
        Args:
            parcela: Parcela a ser adicionada
        """
        self.parcelas.append(parcela)
    
    def calcular_valor_total(self) -> float:
        """
        Calcula o valor total do IPOF com base nas parcelas.
        
        Returns:
            float: Valor total calculado
        """
        return sum(parcela.valor_parcela for parcela in self.parcelas)
    
    def criar_parcelas_automaticas(
        self, 
        quantidade_meses: int, 
        data_inicio: datetime, 
        dotacao_orcamentaria: str, 
        natureza_despesa: str
    ) -> None:
        """
        Cria parcelas automaticamente com base na quantidade de meses.
        
        Args:
            quantidade_meses: Quantidade de meses para criar parcelas
            data_inicio: Data de início do desembolso
            dotacao_orcamentaria: Dotação orçamentária para todas as parcelas
            natureza_despesa: Natureza de despesa para todas as parcelas
        """
        # Limpa parcelas existentes
        self.parcelas = []
        
        # Calcula valor de cada parcela
        valor_parcela = self.valor_total / quantidade_meses
        
        # Cria parcelas
        for i in range(quantidade_meses):
            # Calcula a data da parcela (incrementando o mês)
            data_parcela = data_inicio + timedelta(days=30 * i)
            
            # Cria a parcela
            parcela = ParcelaIPOF(
                data_referencia=data_parcela,
                data_desembolso=data_parcela,
                dotacao_orcamentaria=dotacao_orcamentaria,
                natureza_despesa=natureza_despesa,
                valor_parcela=valor_parcela
            )
            
            # Adiciona a parcela
            self.adicionar_parcela(parcela)
    
    def extrair_unidade_orcamentaria(self, dotacao_orcamentaria: str) -> None:
        """
        Extrai a unidade orçamentária da dotação.
        
        Args:
            dotacao_orcamentaria: Dotação orçamentária
        """
        try:
            # Verifica se a dotação tem pelo menos 8 caracteres
            if len(dotacao_orcamentaria) >= 8:
                # Extrai 4 dígitos a partir da quinta posição (índice 5)
                self.unidade_orcamentaria = dotacao_orcamentaria[5:4]
            else:
                self.unidade_orcamentaria = "N/A"
        except Exception as e:
            logger.error(f"Erro ao extrair unidade orçamentária: {str(e)}")
            self.unidade_orcamentaria = "N/A"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte o IPOF para um dicionário.
        
        Returns:
            Dict[str, Any]: Representação do IPOF como dicionário
        """
        return {
            'numero_processo': self.numero_processo,
            'unidade_orcamentaria': self.unidade_orcamentaria,
            'descricao': self.descricao,
            'valor_total': self.valor_total,
            'parcelas': [parcela.to_dict() for parcela in self.parcelas]
        }
    
    def to_json(self) -> str:
        """
        Converte o IPOF para uma string JSON.
        
        Returns:
            str: Representação JSON do IPOF
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    def formatado(self) -> str:
        """
        Retorna uma representação formatada do IPOF.
        
        Returns:
            str: Representação formatada do IPOF
        """
        resultado = (
            f"### IPOF - Instrumento de Planejamento, Orçamento e Finanças ###\n\n"
            f"Número do Processo: {self.numero_processo}\n"
            f"Unidade Orçamentária: {self.unidade_orcamentaria}\n\n"
            f"Descrição:\n{self.descricao}\n\n"
            f"Valor Total: R$ {self.valor_total:.2f}\n\n"
            f"Parcelas:\n"
        )
        
        for i, parcela in enumerate(self.parcelas, 1):
            resultado += f"\nParcela {i}:\n{parcela}\n"
        
        return resultado