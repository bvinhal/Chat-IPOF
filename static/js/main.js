/**
 * Script principal da aplicação
 * Assistente de Orçamento Público
 */

// Aguarda carregamento do DOM
document.addEventListener('DOMContentLoaded', function() {
    // Botões para abrir modais
    const trainModelBtn = document.getElementById('trainModelBtn');
    const manageModelsBtn = document.getElementById('manageModelsBtn');
    
    // Modais
    const trainModelModal = document.getElementById('trainModelModal');
    const manageModelsModal = document.getElementById('manageModelsModal');
    
    // Elementos dentro dos modais
    const closeModalBtns = document.querySelectorAll('.close-modal');
    const trainModelForm = document.getElementById('trainModelForm');
    const cancelBtns = document.querySelectorAll('.cancel-button');
    const modelsList = document.getElementById('modelsList');
    
    // Estado do treinamento
    const trainingStatus = document.getElementById('trainingStatus');
    
    // Função para abrir modal
    function openModal(modal) {
        if (!modal) return;
        modal.style.display = 'block';
        
        // Se for o modal de gerenciamento de modelos, carrega a lista
        if (modal === manageModelsModal && modelsList) {
            loadModelsList();
        }
    }
    
    // Função para fechar modal
    function closeModal(modal) {
        if (!modal) return;
        modal.style.display = 'none';
        
        // Reinicia o estado do modal de treinamento
        if (modal === trainModelModal && trainingStatus) {
            trainingStatus.classList.add('hidden');
        }
    }
    
    // Função para carregar lista de modelos
    async function loadModelsList() {
        if (!modelsList) return;
        
        // Mostra mensagem de carregamento
        modelsList.innerHTML = `
            <div class="spinner"></div>
            <p>Carregando modelos disponíveis...</p>
        `;
        
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error('Falha ao carregar modelos');
            }
            
            if (!data.models || data.models.length === 0) {
                modelsList.innerHTML = `
                    <div class="status-container info">
                        <p>Nenhum modelo treinado disponível. Por favor, treine um modelo primeiro.</p>
                    </div>
                `;
                return;
            }
            
            // Renderiza lista de modelos
            let modelsHtml = '';
            data.models.forEach(model => {
                modelsHtml += `
                    <div class="model-item" data-model-path="${model.path}">
                        <div class="model-info">
                            <div class="model-name">${model.name}</div>
                            <div class="model-meta">
                                Tipo: ${model.type} | Criado em: ${model.created}
                            </div>
                        </div>
                        <div class="model-actions">
                            <button class="small-button use-model-btn" data-model-type="${model.type}">Usar</button>
                            <button class="small-button delete-model-btn">Excluir</button>
                        </div>
                    </div>
                `;
            });
            
            modelsList.innerHTML = modelsHtml;
            
            // Adiciona event listeners para os botões
            document.querySelectorAll('.use-model-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const modelType = this.getAttribute('data-model-type');
                    changeModel(modelType);
                });
            });
            
            document.querySelectorAll('.delete-model-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const modelItem = this.closest('.model-item');
                    const modelPath = modelItem.getAttribute('data-model-path');
                    deleteModel(modelPath, modelItem);
                });
            });
            
        } catch (error) {
            console.error('Erro ao carregar modelos:', error);
            modelsList.innerHTML = `
                <div class="status-container error">
                    <p>Erro ao carregar modelos: ${error.message}</p>
                </div>
            `;
        }
    }
    
    // Função para mudar o modelo atual
    async function changeModel(modelType) {
        try {
            const response = await fetch('/api/change-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType }),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Falha ao alterar modelo');
            }
            
            showNotification(data.message, 'success');
            
            // Fecha o modal após a alteração
            closeModal(manageModelsModal);
            
        } catch (error) {
            console.error('Erro ao alterar modelo:', error);
            showNotification(`Erro ao alterar modelo: ${error.message}`, 'error');
        }
    }
    
    // Função para excluir um modelo
    async function deleteModel(modelPath, modelItem) {
        if (!confirm('Tem certeza que deseja excluir este modelo?')) return;
        
        try {
            const response = await fetch('/api/delete-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_path: modelPath }),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Falha ao excluir modelo');
            }
            
            // Remove o item da lista
            if (modelItem) {
                modelItem.remove();
            }
            
            showNotification(data.message, 'success');
            
        } catch (error) {
            console.error('Erro ao excluir modelo:', error);
            showNotification(`Erro ao excluir modelo: ${error.message}`, 'error');
        }
    }
    
    // Função para treinar modelo
    async function trainModel(formData) {
        if (!trainModelForm || !trainingStatus) return;
        
        // Esconde o formulário e mostra o status de treinamento
        trainModelForm.style.display = 'none';
        trainingStatus.classList.remove('hidden');
        
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: formData.modelType }),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Falha no treinamento');
            }
            
            // Atualiza o status de treinamento
            trainingStatus.innerHTML = `
                <div class="status-container success">
                    <i class="fas fa-check-circle"></i>
                    <p>${data.message}</p>
                </div>
                <button type="button" class="primary-button close-training-btn">Fechar</button>
            `;
            
            // Adiciona listener para fechar o modal
            document.querySelector('.close-training-btn').addEventListener('click', function() {
                closeModal(trainModelModal);
            });
            
        } catch (error) {
            console.error('Erro no treinamento:', error);
            
            // Atualiza o status de treinamento com a mensagem de erro
            trainingStatus.innerHTML = `
                <div class="status-container error">
                    <i class="fas fa-exclamation-circle"></i>
                    <p>Erro no treinamento: ${error.message}</p>
                </div>
                <button type="button" class="primary-button close-training-btn">Fechar</button>
                <button type="button" class="secondary-button retry-training-btn">Tentar Novamente</button>
            `;
            
            // Adiciona listeners para os botões
            document.querySelector('.close-training-btn').addEventListener('click', function() {
                closeModal(trainModelModal);
            });
            
            document.querySelector('.retry-training-btn').addEventListener('click', function() {
                // Reinicia o modal para o estado original
                trainingStatus.classList.add('hidden');
                trainModelForm.style.display = 'block';
            });
        }
    }
    
    // Função para mostrar notificações
    function showNotification(message, type = 'info') {
        // Verifica se já existe uma notificação
        let notification = document.querySelector('.notification');
        
        if (!notification) {
            // Cria um elemento de notificação
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }
        
        // Define o conteúdo e classe da notificação
        notification.innerHTML = `
            <div class="notification-content ${type}">
                <p>${message}</p>
                <button class="close-notification">&times;</button>
            </div>
        `;
        
        // Mostra a notificação
        notification.style.display = 'block';
        
        // Adiciona event listener para fechar
        notification.querySelector('.close-notification').addEventListener('click', function() {
            notification.style.display = 'none';
        });
        
        // Fecha automaticamente após 5 segundos
        setTimeout(() => {
            if (notification.style.display === 'block') {
                notification.style.display = 'none';
            }
        }, 5000);
    }
    
    // Event listeners para botões de abrir modal
    if (trainModelBtn) {
        trainModelBtn.addEventListener('click', function() {
            openModal(trainModelModal);
        });
    }
    
    if (manageModelsBtn) {
        manageModelsBtn.addEventListener('click', function() {
            openModal(manageModelsModal);
        });
    }
    
    // Event listeners para fechar modais
    closeModalBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const modal = this.closest('.modal');
            closeModal(modal);
        });
    });
    
    cancelBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const modal = this.closest('.modal');
            closeModal(modal);
        });
    });
    
    // Fechar modais ao clicar fora deles
    window.addEventListener('click', function(event) {
        if (event.target.classList.contains('modal')) {
            closeModal(event.target);
        }
    });
    
    // Event listener para formulário de treinamento
    if (trainModelForm) {
        trainModelForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const formData = {
                modelType: document.getElementById('modelType').value
            };
            
            trainModel(formData);
        });
    }
    
    // Estilo para notificações
    const style = document.createElement('style');
    style.textContent = `
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 350px;
            display: none;
        }
        
        .notification-content {
            padding: 1rem;
            border-radius: var(--radius);
            box-shadow: var(--shadow-md);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .notification-content.success {
            background-color: var(--success);
            color: white;
        }
        
        .notification-content.error {
            background-color: var(--error);
            color: white;
        }
        
        .notification-content.info {
            background-color: var(--info);
            color: white;
        }
        
        .notification-content.warning {
            background-color: var(--warning);
            color: white;
        }
        
        .notification-content p {
            margin: 0;
            flex: 1;
        }
        
        .close-notification {
            background: none;
            border: none;
            color: white;
            font-size: 1.25rem;
            cursor: pointer;
            padding: 0 0 0 10px;
        }
    `;
    document.head.appendChild(style);
});
