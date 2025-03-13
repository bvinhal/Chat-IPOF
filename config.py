import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

class Config:
    """Configurações base da aplicação"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'chave-secreta-temporaria'
    DEBUG = False
    TESTING = False
    
    # Configuração do banco de dados
    DATABASE_URI = os.environ.get('DATABASE_URI') or 'sqlite:///app.db'
    
    # Diretórios da aplicação
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    TRAINING_DATA_DIR = os.path.join(DATA_DIR, 'training')
    MODELS_DIR = os.path.join(DATA_DIR, 'models')
    
    # Configurações das APIs de IA
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
    
    # Configurações dos modelos
    DEFAULT_MODEL = os.environ.get('DEFAULT_MODEL') or 'claude'  # claude, openai, gemini
    OPENAI_MODEL = os.environ.get('OPENAI_MODEL') or 'gpt-3.5-turbo' #'gpt-4'
    CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL') or 'claude-3-haiku-20240307'
    GEMINI_MODEL = os.environ.get('GEMINI_MODEL') or 'gemini-pro'
    
    # Configurações da interface de chat
    MAX_HISTORY_LENGTH = 10  # Número de mensagens para manter no histórico
    MAX_TOKEN_LIMIT = 4096   # Limite de tokens para respostas


class DevelopmentConfig(Config):
    """Configurações para ambiente de desenvolvimento"""
    DEBUG = True
    

class TestingConfig(Config):
    """Configurações para ambiente de testes"""
    TESTING = True
    DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """Configurações para ambiente de produção"""
    pass


# Configuração ativa baseada na variável de ambiente
config_by_name = {
    'dev': DevelopmentConfig,
    'test': TestingConfig,
    'prod': ProductionConfig
}

active_config = config_by_name[os.environ.get('FLASK_ENV', 'dev')]
