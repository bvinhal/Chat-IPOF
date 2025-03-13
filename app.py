from flask import Flask, render_template, request, jsonify
import os
from config import active_config

# Importar controllers
from controllers.chat_controller import ChatController
from controllers.training_controller import TrainingController

# Cria os diretórios necessários se não existirem
os.makedirs(active_config.DATA_DIR, exist_ok=True)
os.makedirs(active_config.TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(active_config.MODELS_DIR, exist_ok=True)

app = Flask(__name__)
app.config.from_object(active_config)

# Inicializa controladores
chat_controller = ChatController()
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=active_config.DEBUG)
