# Assistente de Orçamento Público

Uma aplicação de chat baseada em inteligência artificial para consultas sobre planejamento e execução orçamentária do setor público.

## Características

- **Chat interativo**: Interface amigável para tirar dúvidas sobre orçamento público
- **Múltiplos modelos de IA**: Suporte para Claude (Anthropic), GPT (OpenAI) e Gemini (Google)
- **Treinamento personalizado**: Crie e gerencie modelos treinados com documentação específica
- **Interface responsiva**: Funciona em desktops, tablets e smartphones
- **Armazenamento de conversas**: Salva histórico de conversas localmente

## Requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes Python)
- Chaves de API para os serviços desejados (Claude, OpenAI, Gemini)

## Instalação

1. Clone o repositório ou extraia os arquivos para uma pasta local
2. Navegue até a pasta do projeto no terminal
3. Crie um ambiente virtual (recomendado):

```bash
python -m venv venv
```

4. Ative o ambiente virtual:

- Windows:
```bash
venv\Scripts\activate
```

- Linux/Mac:
```bash
source venv/bin/activate
```

5. Instale as dependências:

```bash
pip install -r requirements.txt
```

6. Crie um arquivo `.env` na raiz do projeto com suas chaves de API:

```
FLASK_ENV=dev
SECRET_KEY=sua-chave-secreta

# Escolha pelo menos uma das seguintes APIs
OPENAI_API_KEY=sua-chave-da-openai
CLAUDE_API_KEY=sua-chave-do-claude
GEMINI_API_KEY=sua-chave-do-gemini

# Modelos específicos (opcional)
OPENAI_MODEL=gpt-4
CLAUDE_MODEL=claude-3-haiku-20240307
GEMINI_MODEL=gemini-pro

# Modelo padrão: claude, openai ou gemini
DEFAULT_MODEL=claude
```

## Preparando os dados para treinamento

1. Crie uma pasta `data/training` na raiz do projeto (se não existir):

```bash
mkdir -p data/training
```

2. Coloque seus documentos de treinamento nesta pasta. Formatos suportados:
   - PDF (.pdf)
   - Word (.docx)
   - Texto (.txt)

## Executando a aplicação

1. Execute o servidor Flask:

```bash
python app.py
```

2. Acesse a aplicação no navegador:

```
http://localhost:5000
```

## Uso básico

1. **Treinamento inicial**:
   - Na página inicial, clique em "Treinar Modelo"
   - Selecione o tipo de modelo que deseja treinar
   - Aguarde o processo de treinamento (pode levar alguns minutos)

2. **Chat**:
   - Acesse a página de chat
   - Digite suas perguntas sobre orçamento público
   - O assistente responderá com base nos documentos de treinamento

3. **Gerenciamento de modelos**:
   - Na página inicial, clique em "Gerenciar Modelos"
   - Selecione modelos para usar ou excluir conforme necessário

## Estrutura do projeto

```
orcamento_chat/
│
├── app.py                      # Arquivo principal da aplicação
├── config.py                   # Configurações da aplicação
├── requirements.txt            # Dependências do projeto
│
├── static/                     # Arquivos estáticos
│   ├── css/
│   │   ├── main.css            # Estilos principais
│   │   └── chat.css            # Estilos específicos do chat
│   ├── js/
│   │   ├── main.js             # JavaScript principal
│   │   └── chat.js             # Lógica do chat
│   └── img/                    # Imagens da aplicação
│
├── templates/                  # Templates HTML
│   ├── base.html               # Template base
│   ├── index.html              # Página inicial
│   └── chat.html               # Interface do chat
│
├── models/                     # Modelos da aplicação
│   ├── ai_model.py             # Classe base para modelos de IA
│   ├── claude_model.py         # Implementação do modelo Claude
│   ├── openai_model.py         # Implementação do modelo OpenAI
│   └── gemini_model.py         # Implementação do modelo Gemini
│
├── controllers/                # Controladores da aplicação
│   ├── chat_controller.py      # Controle da lógica do chat
│   └── training_controller.py  # Controle do treinamento de modelos
│
├── data/                       # Dados para treinamento e configuração
    ├── training/               # Documentos para treinamento
    └── models/                 # Modelos treinados serão salvos aqui
```

## Customização

### Modelos de IA
Você pode modificar os parâmetros dos modelos de IA editando os arquivos:
- `models/claude_model.py`
- `models/openai_model.py`
- `models/gemini_model.py`

### Aparência
A interface pode ser customizada através dos arquivos CSS:
- `static/css/main.css` (estilos globais)
- `static/css/chat.css` (estilos da interface de chat)

## Resolução de problemas

### Erros de API
- Verifique se suas chaves de API estão corretas no arquivo `.env`
- Confirme se você tem saldo ou créditos nas plataformas respectivas

### Falhas no treinamento
- Verifique se os documentos estão no formato correto
- Verifique o console para mensagens de erro detalhadas
- Assegure-se de que os documentos não excedem os limites das APIs

### Erros de conexão
- Verifique sua conexão com a internet
- Confirme se os serviços das APIs estão online

## Segurança

- Não compartilhe suas chaves de API
- Evite enviar dados sensíveis ou confidenciais para os modelos de IA
- Considere a execução em uma rede privada para maior segurança

## Licença

Este projeto é fornecido para o setor público sem restrições de uso.
