from config import active_config
from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory
import os
import uuid
import mimetypes
import shutil
import glob
import time
import json
from datetime import datetime
from werkzeug.utils import secure_filename

from config import active_config

# Importar controllers
from controllers.enhanced_ipof_chat_controller import EnhancedIPOFChatController
from controllers.training_controller import TrainingController

# Importar models
from models.natureza_classifier import NaturezaClassifier
from models.ipof_model import IPOF, ParcelaIPOF
from models.chat_natureza_handler import ChatNaturezaHandler
from controllers.natureza_evaluator_controller import NaturezaEvaluatorController


# Cria os diretórios necessários se não existirem
os.makedirs(active_config.DATA_DIR, exist_ok=True)
os.makedirs(active_config.TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(active_config.MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(active_config.DATA_DIR, 'temp'), exist_ok=True)
os.makedirs(os.path.join(active_config.DATA_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(active_config.DATA_DIR, 'ipof_html'), exist_ok=True)
ipof_html_dir = os.path.abspath(os.path.join(active_config.DATA_DIR, 'ipof_html'))
os.makedirs(ipof_html_dir, exist_ok=True)

app = Flask(__name__)
app.config.from_object(active_config)
app.config['UPLOAD_FOLDER'] = os.path.join(active_config.DATA_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limita uploads a 16MB

# Inicializa controladores
chat_controller = EnhancedIPOFChatController()
training_controller = TrainingController()
natureza_evaluator_controller = NaturezaEvaluatorController()


# Tipos de arquivos permitidos
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}

def allowed_file(filename):
    """Verifica se o arquivo tem uma extensão permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Rota para a página inicial"""
    return render_template('index.html')


@app.route('/chat')
def chat():
    """Rota para a página de chat"""
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Endpoint da API para processar mensagens do chat"""
    data = request.json
    user_message = data.get('message', '')
    chat_history = data.get('history', [])
    
    # Processa a mensagem usando o controlador de chat
    response = chat_controller.process_message(user_message, chat_history)
    
    return jsonify({
        'response': response
    })


@app.route('/api/upload-chat-file', methods=['POST'])
def upload_chat_file():
    """Endpoint para processar upload de arquivo durante o chat"""
    # Verifica se há um arquivo na requisição
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': 'Nenhum arquivo encontrado'
        }), 400
    
    file = request.files['file']
    
    # Verifica se um arquivo foi selecionado
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'Nenhum arquivo selecionado'
        }), 400
    
    # Verifica se o arquivo tem uma extensão permitida
    if file and allowed_file(file.filename):
        # Gera um nome seguro para o arquivo
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        
        # Caminho para salvar o arquivo
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Salva o arquivo
        file.save(file_path)
        
        # Determina o tipo de arquivo para processamento
        file_type = original_filename.rsplit('.', 1)[1].lower()
        
        # Mensagem adicional (opcional)
        message = request.form.get('message', '')
        
        # Histórico do chat (opcional)
        chat_history = request.form.get('history', '[]')
        try:
            chat_history = json.loads(chat_history)
        except:
            chat_history = []
        
        # Processa o arquivo
        response = chat_controller.process_file(file_path, file_type, message, chat_history)
        
        return jsonify({
            'success': True,
            'message': 'Arquivo processado com sucesso',
            'response': response,
            'file_path': file_path
        })
    
    return jsonify({
        'success': False,
        'message': 'Tipo de arquivo não permitido. Use PDF, DOC, DOCX ou TXT.'
    }), 400


@app.route('/api/train', methods=['POST'])
def train_model():
    """Endpoint da API para iniciar treinamento do modelo"""
    data = request.json
    model_type = data.get('model_type', active_config.DEFAULT_MODEL)
    
    # Inicia o treinamento usando o controlador de treinamento
    result = training_controller.train_model(model_type)
    
    return jsonify({
        'success': result.get('success', False),
        'message': result.get('message', 'Falha no treinamento')
    })


@app.route('/api/models', methods=['GET'])
def get_models():
    """Endpoint da API para listar modelos disponíveis"""
    models = training_controller.list_available_models()
    
    return jsonify({
        'models': models
    })


@app.route('/api/current-model', methods=['GET'])
def get_current_model():
    """Endpoint da API para obter o modelo atual"""
    model_info = {
        'type': chat_controller.current_model_type,
        'name': chat_controller.current_model.__class__.__name__ if chat_controller.current_model else 'Nenhum'
    }
    
    return jsonify({
        'model': model_info
    })


@app.route('/api/change-model', methods=['POST'])
def change_model():
    """Endpoint da API para alterar o modelo atual"""
    data = request.json
    model_type = data.get('model_type')
    
    if not model_type:
        return jsonify({
            'success': False,
            'message': 'Tipo de modelo não especificado'
        }), 400
    
    result = chat_controller.change_model(model_type)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/delete-model', methods=['POST'])
def delete_model():
    """Endpoint da API para excluir um modelo"""
    data = request.json
    model_path = data.get('model_path')
    
    if not model_path:
        return jsonify({
            'success': False,
            'message': 'Caminho do modelo não especificado'
        }), 400
    
    result = training_controller.delete_model(model_path)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route('/api/train-natureza', methods=['POST'])
def train_natureza_classifier():
    """Endpoint da API para treinar o classificador de natureza de despesa"""
    data = request.json
    embedding_provider = data.get('embedding_provider', active_config.DEFAULT_MODEL)
    excel_path = data.get('excel_path')
    force_rebuild = data.get('force_rebuild', False)
    
    # Verifica se o caminho do arquivo foi fornecido
    if not excel_path:
        # Se não foi fornecido um caminho específico, usa o arquivo padrão no diretório de dados
        excel_path = os.path.join(active_config.DATA_DIR, 'natureza', 'natureza_despesa.xlsx')
    
    try:
        # Inicializa o classificador com o provedor especificado
        classifier = NaturezaClassifier(embedding_provider)
        
        # Inicia o treinamento
        result = classifier.train(excel_path, force_rebuild)
        
        if result:
            return jsonify({
                'success': True,
                'message': f'Classificador de natureza treinado com sucesso usando provedor {embedding_provider}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Falha ao treinar classificador de natureza'
            }), 400
    
    except Exception as e:
        app.logger.error(f"Erro ao treinar classificador de natureza: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro ao treinar classificador: {str(e)}'
        }), 500


@app.route('/admin')
def admin_panel():
    """Rota para o painel de administração"""
    return render_template('admin.html')


@app.route('/api/upload-excel', methods=['POST'])
def upload_excel():
    """Endpoint para upload de arquivo Excel"""
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'message': 'Nenhum arquivo encontrado'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'Nenhum arquivo selecionado'
        }), 400
    
    if file and file.filename.endswith(('.xlsx', '.xls')):
        # Cria o diretório para natureza se não existir
        natureza_dir = os.path.join(active_config.DATA_DIR, 'natureza')
        os.makedirs(natureza_dir, exist_ok=True)
        
        # Salva o arquivo
        filename = 'natureza_despesa.xlsx'
        file_path = os.path.join(natureza_dir, filename)
        file.save(file_path)
        
        return jsonify({
            'success': True,
            'message': 'Arquivo Excel carregado com sucesso',
            'file_path': file_path
        })
    
    return jsonify({
        'success': False,
        'message': 'Formato de arquivo não suportado. Use .xlsx ou .xls'
    }), 400


@app.route('/api/natureza-models', methods=['GET'])
def get_natureza_models():
    """Endpoint para listar classificadores de natureza disponíveis"""
    try:
        # Diretório onde os classificadores são salvos
        base_dir = os.path.join(active_config.MODELS_DIR)
        
        # Padrão para identificar diretórios de classificadores
        classifier_pattern = 'natureza_classifier_*'
        
        # Lista todos os diretórios que correspondem ao padrão
        classifier_dirs = glob.glob(os.path.join(base_dir, classifier_pattern))
        
        models = []
        for dir_path in classifier_dirs:
            # Extrai o nome do provedor de embeddings do nome do diretório
            provider = os.path.basename(dir_path).replace('natureza_classifier_', '')
            
            # Verifica se existe o arquivo de informações
            info_path = os.path.join(dir_path, 'model_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                
                # Adiciona informações básicas
                model_info = {
                    'id': os.path.basename(dir_path),
                    'embedding_provider': provider,
                    'num_examples': info.get('num_examples', 0),
                    'created': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getctime(dir_path)))
                }
                
                models.append(model_info)
        
        # Ordena por data de criação (mais recente primeiro)
        models.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'success': True,
            'models': models
        })
    
    except Exception as e:
        app.logger.error(f"Erro ao listar classificadores de natureza: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro ao listar classificadores: {str(e)}',
            'models': []
        })


@app.route('/api/set-natureza-provider', methods=['POST'])
def set_natureza_provider():
    """Endpoint para definir o provedor de classificação de natureza"""
    data = request.json
    provider = data.get('provider')
    
    if not provider:
        return jsonify({
            'success': False,
            'message': 'Provedor não especificado'
        }), 400
    
    try:
        # Salva a preferência em um arquivo de configuração
        config_dir = os.path.join(active_config.DATA_DIR, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, 'natureza_config.json')
        
        config = {'provider': provider}
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        # Reinicializa o manipulador de chat com o novo provedor
        chat_controller.natureza_handler = ChatNaturezaHandler(provider)
        
        return jsonify({
            'success': True,
            'message': f'Provedor de classificação definido para {provider}'
        })
    
    except Exception as e:
        app.logger.error(f"Erro ao definir provedor de classificação: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro ao definir provedor: {str(e)}'
        }), 500


@app.route('/api/delete-natureza-model', methods=['POST'])
def delete_natureza_model():
    """Endpoint para excluir um classificador de natureza"""
    data = request.json
    model_id = data.get('model_id')
    
    if not model_id:
        return jsonify({
            'success': False,
            'message': 'ID do modelo não especificado'
        }), 400
    
    try:
        # Verifica se o diretório existe
        model_path = os.path.join(active_config.MODELS_DIR, model_id)
        if not os.path.exists(model_path):
            return jsonify({
                'success': False,
                'message': f'Modelo não encontrado: {model_id}'
            }), 404
        
        # Exclui o diretório e todo seu conteúdo
        shutil.rmtree(model_path)
        
        return jsonify({
            'success': True,
            'message': f'Classificador de natureza excluído com sucesso'    
        })
    
    except Exception as e:
        app.logger.error(f"Erro ao excluir classificador de natureza: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro ao excluir classificador: {str(e)}'
        }), 500


# Limpa arquivos temporários periodicamente (pode ser implementado como uma tarefa agendada)
def clean_temp_files():
    """Limpa arquivos temporários antigos"""
    try:
        # Diretório de uploads
        upload_dir = app.config['UPLOAD_FOLDER']
        
        # Tempo atual em segundos
        now = time.time()
        
        # Tempo limite (24 horas)
        time_limit = 24 * 60 * 60
        
        # Lista todos os arquivos no diretório de uploads
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            
            # Verifica se é um arquivo (não um diretório)
            if os.path.isfile(file_path):
                # Obtém a hora da última modificação do arquivo
                file_time = os.path.getmtime(file_path)
                
                # Se o arquivo for mais antigo que o limite, exclui
                if now - file_time > time_limit:
                    os.remove(file_path)
                    app.logger.info(f"Arquivo temporário excluído: {file_path}")
    
    except Exception as e:
        app.logger.error(f"Erro ao limpar arquivos temporários: {str(e)}")

    @app.route('/ipof/<filename>')
    def serve_ipof_html(filename):
        """
        Serve um arquivo HTML de IPOF.
        
        Args:
            filename: Nome do arquivo HTML
            
        Returns:
            O arquivo HTML solicitado
        """
        # Adicione log para depuração
        app.logger.info(f"Solicitação para servir arquivo IPOF: {filename}")
        app.logger.info(f"Procurando em: {ipof_html_dir}")
        
        # Verificar se o arquivo existe antes de tentar servi-lo
        file_path = os.path.join(ipof_html_dir, filename)
        if os.path.isfile(file_path):
            app.logger.info(f"Arquivo encontrado: {file_path}")
        else:
            app.logger.error(f"Arquivo não encontrado: {file_path}")
        
        return send_from_directory(ipof_html_dir, filename)

    @app.route('/api/ipof/export', methods=['POST'])
    def export_ipof_json():
        """Endpoint para salvar o IPOF em formato JSON para SIAFIC."""
        try:
            # Obtém os dados do IPOF do corpo da requisição
            data = request.json
            
            if not data:
                return jsonify({
                    'success': False,
                    'message': 'Dados do IPOF não fornecidos'
                }), 400
            
            # Gera um nome de arquivo único baseado no número do processo
            processo = data.get('numero_processo', 'sem_processo')
            processo_safe = processo.replace('/', '_').replace('\\', '_')
            
            # Cria diretório para exportações SIAFIC se não existir
            siafic_dir = os.path.join(active_config.DATA_DIR, 'siafic_exports')
            os.makedirs(siafic_dir, exist_ok=True)
            
            # Adiciona timestamp ao nome do arquivo para garantir unicidade
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ipof_siafic_{processo_safe}_{timestamp}.json"
            
            # Caminho completo do arquivo
            file_path = os.path.join(siafic_dir, filename)
            
            # Salva os dados em formato JSON
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return jsonify({
                'success': True,
                'message': 'IPOF exportado com sucesso para SIAFIC',
                'file_path': filename
            })
            
        except Exception as e:
            app.logger.error(f"Erro ao exportar IPOF para JSON: {str(e)}")
            return jsonify({
                'success': False,
                'message': f'Erro ao exportar IPOF: {str(e)}'
            }), 500
            
    @app.route('/api/train-natureza-evaluator', methods=['POST'])
    def train_natureza_evaluator():
        """Endpoint para treinar o avaliador de natureza de despesa."""
        data = request.json
        provider = data.get('provider', active_config.DEFAULT_MODEL)
        mcasp_path = data.get('mcasp_path')
        force_rebuild = data.get('force_rebuild', False)
        
        # Treina o avaliador
        result = natureza_evaluator_controller.train_evaluator(provider, mcasp_path, force_rebuild)
        
        # Retorna o resultado
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/natureza-evaluators', methods=['GET'])
    def get_natureza_evaluators():
        """Endpoint para listar avaliadores de natureza disponíveis."""
        evaluators = natureza_evaluator_controller.list_evaluators()
        
        return jsonify({
            'success': True,
            'evaluators': evaluators
        })

    @app.route('/api/delete-natureza-evaluator', methods=['POST'])
    def delete_natureza_evaluator():
        """Endpoint para excluir um avaliador de natureza."""
        data = request.json
        provider = data.get('provider')
        
        if not provider:
            return jsonify({
                'success': False,
                'message': 'Provedor não especificado'
            }), 400
        
        result = natureza_evaluator_controller.delete_evaluator(provider)
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 400

    @app.route('/api/evaluate-natureza', methods=['POST'])
    def evaluate_natureza():
        """Endpoint para avaliar uma ou mais naturezas para uma descrição."""
        data = request.json
        descricao = data.get('descricao', '')
        
        # Verifica se os dados necessários foram fornecidos
        if not descricao:
            return jsonify({
                'success': False,
                'message': 'Descrição é obrigatória'
            }), 400
        
        # Pode receber uma lista de candidatos ou um único código
        candidatos = data.get('candidatos', [])
        natureza_codigo = data.get('natureza_codigo', '')
        
        if not candidatos and not natureza_codigo:
            return jsonify({
                'success': False,
                'message': 'É necessário fornecer candidatos ou natureza_codigo'
            }), 400
        
        # Se forneceu apenas um código, converte para o formato de candidatos
        if natureza_codigo and not candidatos:
            candidatos = [{'codigo': natureza_codigo, 'confianca': 1.0}]
        
        # Usa o controlador de chat se disponível, caso contrário usa o controlador de avaliador
        if hasattr(chat_controller, 'evaluate_natureza_candidates'):
            result = chat_controller.evaluate_natureza_candidates(descricao, candidatos)
        elif hasattr(natureza_evaluator_controller, 'evaluate_candidates'):
            result = natureza_evaluator_controller.evaluate_candidates(descricao, candidatos)
        else:
            # Fallback para a avaliação de um único código
            if len(candidatos) == 1:
                result = natureza_evaluator_controller.evaluate_natureza(descricao, candidatos[0]['codigo'])
            else:
                return jsonify({
                    'success': False,
                    'message': 'Avaliação de múltiplos candidatos não implementada'
                }), 501
        
        if result.get('success', False):
            return jsonify(result)
        else:
            return jsonify(result), 400
                        
    @app.route('/api/upload-mcasp', methods=['POST'])
    def upload_mcasp():
        """Endpoint para upload de arquivo PDF do MCASP."""
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Nenhum arquivo encontrado'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Nenhum arquivo selecionado'
            }), 400
        
        if file and file.filename.endswith('.pdf'):
            # Cria o diretório MCASP se não existir
            mcasp_dir = os.path.join(active_config.TRAINING_DATA_DIR, 'mcasp')
            os.makedirs(mcasp_dir, exist_ok=True)
            
            # Salva o arquivo
            filename = 'mcasp.pdf'
            file_path = os.path.join(mcasp_dir, filename)
            file.save(file_path)
            
            return jsonify({
                'success': True,
                'message': 'Arquivo MCASP carregado com sucesso',
                'file_path': file_path
            })
        
        return jsonify({
            'success': False,
            'message': 'Formato de arquivo não suportado. Use .pdf'
        }), 400
        
    @app.route('/natureza-evaluator-demo')
    def natureza_evaluator_demo():
        """Rota para a página de demonstração do avaliador de natureza."""
        return render_template('natureza_evaluator_demo.html')
        
# Executa a limpeza a cada inicio da aplicação
clean_temp_files()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=active_config.DEBUG)