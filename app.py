from flask import Flask, render_template, request, jsonify
import os
from config import active_config

# Importar controllers
from controllers.enhanced_chat_controller import EnhancedChatController
from controllers.training_controller import TrainingController

# Importar models
from models.natureza_classifier import NaturezaClassifier

# Cria os diretórios necessários se não existirem
os.makedirs(active_config.DATA_DIR, exist_ok=True)
os.makedirs(active_config.TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(active_config.MODELS_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_object(active_config)

# Inicializa controladores
chat_controller = EnhancedChatController()
training_controller = TrainingController()


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

# Adicione uma rota para treinamento do classificador de natureza
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


# Adicione ao arquivo app.py

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
        
        # Exclui os arquivos dentro do diretório
        for file_name in os.listdir(model_path):
            file_path = os.path.join(model_path, file_name)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        
        # Exclui o diretório
        os.rmdir(model_path)
        
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

        
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=active_config.DEBUG)
