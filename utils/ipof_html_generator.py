# utils/ipof_html_generator.py

import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
import logging
import jinja2
from config import active_config

# Configuração de logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IPOFHtmlGenerator:
    """
    Gerador de arquivos HTML para IPOF.
    """
    
    def __init__(self):
        """Inicializa o gerador de HTML para IPOF."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Diretório para armazenar os arquivos HTML
        self.html_dir = os.path.join(active_config.DATA_DIR, 'ipof_html')
        os.makedirs(self.html_dir, exist_ok=True)
        
        # Configurar o ambiente Jinja2
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Verifica se o template existe, caso contrário cria um template padrão
        self._ensure_template_exists()
    
    def _ensure_template_exists(self) -> None:
        """
        Verifica se o template do IPOF existe, caso contrário cria o template padrão.
        """
        template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'ipof_template.html')
        
        if not os.path.exists(template_path):
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            
            # Conteúdo do template HTML
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write("""<!DOCTYPE html>
<html lang="pt-br">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IPOF - {{ ipof_data.numero_processo }}</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #60a5fa;
            --dark: #1e293b;
            --medium-dark: #334155;
            --medium: #64748b;
            --light: #e2e8f0;
            --ultra-light: #f8fafc;
            --success: #22c55e;
            --info: #0ea5e9;
            --warning: #f59e0b;
            --error: #ef4444;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.5;
            color: var(--dark);
            background-color: var(--ultra-light);
            margin: 0;
            padding: 0;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .ipof-header {
            background-color: var(--primary);
            color: white;
            padding: 2rem;
            border-radius: 8px 8px 0 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .ipof-header h1 {
            margin: 0 0 1rem 0;
            font-size: 1.75rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .ipof-header h1 svg {
            width: 24px;
            height: 24px;
        }
        
        .ipof-header .processo {
            font-size: 1.25rem;
            font-weight: 500;
            opacity: 0.9;
        }
        
        .ipof-content {
            background-color: white;
            padding: 2rem;
            border-radius: 0 0 8px 8px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        .ipof-card {
            border-radius: 8px;
            padding: 1.5rem;
            background-color: var(--ultra-light);
            margin-bottom: 1.5rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        }
        
        .ipof-card h2 {
            margin-top: 0;
            color: var(--primary-dark);
            font-size: 1.25rem;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .ipof-info {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.5rem;
        }
        
        .info-item {
            margin-bottom: 1rem;
        }
        
        .info-item .label {
            font-size: 0.875rem;
            color: var(--medium);
            margin-bottom: 0.25rem;
        }
        
        .info-item .value {
            font-size: 1rem;
            font-weight: 500;
            color: var(--dark);
        }
        
        .info-item .value.highlight {
            font-size: 1.125rem;
            color: var(--primary-dark);
            font-weight: 600;
        }
        
        .descricao {
            padding: 1rem;
            background-color: white;
            border-radius: 6px;
            border-left: 3px solid var(--primary);
            margin-bottom: 1.5rem;
        }
        
        .parcelas-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
            overflow: hidden;
            border-radius: 8px;
        }
        
        .parcelas-table thead {
            background-color: var(--primary);
            color: white;
        }
        
        .parcelas-table th {
            text-align: left;
            padding: 1rem;
            font-weight: 500;
        }
        
        .parcelas-table td {
            padding: 1rem;
            border-bottom: 1px solid var(--light);
        }
        
        .parcelas-table tr:last-child td {
            border-bottom: none;
        }
        
        .parcelas-table tr:nth-child(even) {
            background-color: var(--ultra-light);
        }
        
        .parcelas-table tr:hover {
            background-color: var(--light);
        }
        
        .footer {
            text-align: center;
            margin-top: 2rem;
            color: var(--medium);
            font-size: 0.875rem;
        }
        
        .print-button {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            cursor: pointer;
            margin-top: 1.5rem;
            transition: background-color 0.2s;
        }
        
        .print-button:hover {
            background-color: var(--primary-dark);
        }
        
        @media print {
            body {
                background-color: white;
            }
            
            .container {
                max-width: 100%;
                padding: 0;
            }
            
            .ipof-header, .ipof-content {
                box-shadow: none;
                border-radius: 0;
            }
            
            .print-button {
                display: none;
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 1rem;
            }
            
            .ipof-header {
                padding: 1.5rem;
            }
            
            .ipof-content {
                padding: 1.5rem;
            }
            
            .ipof-info {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="ipof-header">
            <h1>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path>
                    <rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect>
                    <path d="M12 11h4"></path>
                    <path d="M12 16h4"></path>
                    <path d="M8 11h.01"></path>
                    <path d="M8 16h.01"></path>
                </svg>
                Instrumento de Planejamento, Orçamento e Finanças
            </h1>
            <div class="processo">Processo: {{ ipof_data.numero_processo }}</div>
        </div>
        
        <div class="ipof-content">
            <div class="ipof-card">
                <h2>Informações Gerais</h2>
                <div class="ipof-info">
                    <div class="info-item">
                        <div class="label">Número do Processo</div>
                        <div class="value">{{ ipof_data.numero_processo }}</div>
                    </div>
                    <div class="info-item">
                        <div class="label">Unidade Orçamentária</div>
                        <div class="value">{{ ipof_data.unidade_orcamentaria }}</div>
                    </div>
                    <div class="info-item">
                        <div class="label">Valor Total</div>
                        <div class="value highlight">R$ {{ "{:,.2f}".format(ipof_data.valor_total).replace(',', '.').replace('.', ',', 1) }}</div>
                    </div>
                    <div class="info-item">
                        <div class="label">Quantidade de Parcelas</div>
                        <div class="value">{{ len(ipof_data.parcelas) }}</div>
                    </div>
                </div>
            </div>
            
            <div class="ipof-card">
                <h2>Descrição</h2>
                <div class="descricao">
                    {{ ipof_data.descricao }}
                </div>
            </div>
            
            <div class="ipof-card">
                <h2>Parcelas</h2>
                <table class="parcelas-table">
                    <thead>
                        <tr>
                            <th>Nº</th>
                            <th>Data de Referência</th>
                            <th>Data de Desembolso</th>
                            <th>Natureza de Despesa</th>
                            <th>Valor da Parcela</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for index, parcela in enumerate(ipof_data.parcelas) %}
                        <tr>
                            <td>{{ index + 1 }}</td>
                            <td>{{ parcela.data_referencia }}</td>
                            <td>{{ parcela.data_desembolso }}</td>
                            <td>{{ parcela.natureza_despesa }}</td>
                            <td>R$ {{ "{:,.2f}".format(parcela.valor_parcela).replace(',', '.').replace('.', ',', 1) }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            
            <button class="print-button" onclick="window.print()">
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="6 9 6 2 18 2 18 9"></polyline>
                    <path d="M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2"></path>
                    <rect x="6" y="14" width="12" height="8"></rect>
                </svg>
                Imprimir IPOF
            </button>
        </div>
        
        <div class="footer">
            <p>IPOF gerado em {{ datetime.now().strftime('%d/%m/%Y às %H:%M') }}</p>
        </div>
    </div>
</body>
</html>""")
    
    def generate_html(self, ipof_data: Dict[str, Any]) -> Optional[str]:
        """
        Gera um arquivo HTML com os dados do IPOF.
        
        Args:
            ipof_data: Dicionário com os dados do IPOF
            
        Returns:
            Optional[str]: Nome do arquivo HTML gerado ou None em caso de erro
        """
        try:
            # Gera um ID único para o arquivo
            file_id = str(uuid.uuid4())
            
            # Nome do arquivo HTML
            file_name = f"ipof_{file_id}.html"
            file_path = os.path.join(self.html_dir, file_name)
            
            # Obtém o template
            template = self.jinja_env.get_template('ipof_template.html')
            
            # Adiciona quantidade de parcelas diretamente ao dicionário de dados
            ipof_data['quantidade_parcelas'] = len(ipof_data.get('parcelas', []))
            
            # Converte as datas das parcelas para formato legível
            for parcela in ipof_data.get('parcelas', []):
                if isinstance(parcela.get('data_referencia'), datetime):
                    parcela['data_referencia'] = parcela['data_referencia'].strftime('%d/%m/%Y')
                if isinstance(parcela.get('data_desembolso'), datetime):
                    parcela['data_desembolso'] = parcela['data_desembolso'].strftime('%d/%m/%Y')
            
            # Renderiza o template com os dados do IPOF
            html_content = template.render(
                ipof_data=ipof_data,
                enumerate=enumerate,
                datetime=datetime
            )
            
            # Salva o conteúdo no arquivo HTML
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Arquivo HTML do IPOF gerado com sucesso: {file_path}")
            
            # Retorna o nome do arquivo (sem o caminho completo)
            return file_name
            
        except Exception as e:
            self.logger.error(f"Erro ao gerar arquivo HTML do IPOF: {str(e)}")
            return None