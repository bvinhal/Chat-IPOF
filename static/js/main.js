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

/**
 * JavaScript para o seletor de modelo no header
 * Sistema de combobox compacto que mostra apenas o último modelo de cada LLM
 */

class HeaderModelSelector {
    constructor() {
        this.currentModel = null;
        this.availableModels = {};
        this.isLoading = false;
        
        // Elementos DOM
        this.dropdown = document.getElementById('headerModelDropdown');
        this.dropdownButton = document.getElementById('headerModelDropdownButton');
        this.dropdownMenu = document.getElementById('headerModelDropdownMenu');
        this.currentModelText = document.getElementById('headerCurrentModelText');
        this.currentModelIcon = document.getElementById('headerCurrentModelIcon');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadCurrentModel();
    }
    
    setupEventListeners() {
        // Toggle dropdown
        if (this.dropdownButton) {
            this.dropdownButton.addEventListener('click', (e) => {
                e.stopPropagation();
                this.toggleDropdown();
            });
        }
        
        // Fechar dropdown ao clicar fora
        document.addEventListener('click', (e) => {
            if (!this.dropdown.contains(e.target)) {
                this.closeDropdown();
            }
        });
        
        // Fechar dropdown com ESC
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeDropdown();
            }
        });
    }
    
    getModelIcon(type) {
        const icons = {
            claude: 'brain',
            openai: 'bolt',
            gemini: 'gem'
        };
        return icons[type] || 'robot';
    }
    
    getModelDisplayName(type) {
        const names = {
            claude: 'Claude',
            openai: 'GPT-4',
            gemini: 'Gemini'
        };
        return names[type] || type.toUpperCase();
    }
    
    async loadCurrentModel() {
        try {
            const response = await fetch('/api/current-model');
            if (response.ok) {
                const data = await response.json();
                if (data && data.model) {
                    this.currentModel = data.model.type;
                    this.updateCurrentModelDisplay(data.model.type);
                } else {
                    this.updateCurrentModelDisplay(null);
                }
            }
        } catch (error) {
            console.error('Erro ao carregar modelo atual:', error);
            this.updateCurrentModelDisplay(null);
        }
    }
    
    updateCurrentModelDisplay(modelType) {
        if (modelType) {
            const displayName = this.getModelDisplayName(modelType);
            const iconName = this.getModelIcon(modelType);
            
            this.currentModelText.textContent = displayName;
            this.currentModelIcon.className = `model-icon ${modelType}`;
            this.currentModelIcon.innerHTML = `<i class="fas fa-${iconName}"></i>`;
        } else {
            this.currentModelText.textContent = 'Nenhum modelo';
            this.currentModelIcon.className = 'model-icon';
            this.currentModelIcon.innerHTML = '<i class="fas fa-robot"></i>';
        }
    }
    
    async loadAvailableModels() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.dropdownMenu.innerHTML = '<div class="spinner" style="width: 16px; height: 16px; margin: var(--spacing-sm) auto;"></div>';
        
        try {
            const response = await fetch('/api/models');
            if (response.ok) {
                const data = await response.json();
                
                if (!data.models || data.models.length === 0) {
                    this.dropdownMenu.innerHTML = `
                        <div class="no-models-message">
                            <i class="fas fa-info-circle"></i> Nenhum modelo treinado disponível
                        </div>
                    `;
                    return;
                }
                
                // Organizar modelos por tipo e pegar apenas o mais recente de cada
                const latestModels = {};
                data.models.forEach(model => {
                    if (!latestModels[model.type] || new Date(model.created) > new Date(latestModels[model.type].created)) {
                        latestModels[model.type] = model;
                    }
                });
                
                this.availableModels = latestModels;
                this.renderModelDropdown(latestModels);
                
            } else {
                throw new Error('Falha ao carregar modelos');
            }
        } catch (error) {
            console.error('Erro ao carregar modelos:', error);
            this.dropdownMenu.innerHTML = `
                <div class="no-models-message">
                    <i class="fas fa-exclamation-triangle"></i> Erro ao carregar modelos
                </div>
            `;
        } finally {
            this.isLoading = false;
        }
    }
    
    renderModelDropdown(latestModels) {
        let html = '';
        const modelTypes = ['claude', 'openai', 'gemini'];
        
        modelTypes.forEach(type => {
            if (latestModels[type]) {
                const model = latestModels[type];
                const isActive = this.currentModel === type;
                const displayName = this.getModelDisplayName(type);
                const iconName = this.getModelIcon(type);
                
                html += `
                    <div class="dropdown-section">
                        <div class="dropdown-section-title">${displayName}</div>
                        <div class="dropdown-item ${isActive ? 'active' : ''}" data-model-type="${type}">
                            <div class="model-icon ${type}">
                                <i class="fas fa-${iconName}"></i>
                            </div>
                            <div class="dropdown-item-text">
                                <div>${model.name}</div>
                                <div class="dropdown-item-meta">Criado: ${model.created}</div>
                            </div>
                        </div>
                    </div>
                `;
            }
        });
        
        if (html === '') {
            html = '<div class="no-models-message">Nenhum modelo disponível</div>';
        }
        
        this.dropdownMenu.innerHTML = html;
        
        // Adicionar event listeners aos itens
        this.dropdownMenu.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', () => {
                const modelType = item.dataset.modelType;
                if (modelType && modelType !== this.currentModel) {
                    this.changeModel(modelType, item);
                }
            });
        });
    }
    
    async changeModel(modelType, itemElement) {
        if (this.isLoading) return;
        
        try {
            this.isLoading = true;
            
            // Mostrar loader no item
            if (itemElement) {
                const originalContent = itemElement.innerHTML;
                itemElement.innerHTML = '<div class="spinner" style="width: 12px; height: 12px; margin: 4px auto;"></div>';
            }
            
            const response = await fetch('/api/change-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType }),
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this.showNotification(data.message, 'success');
                this.currentModel = modelType;
                this.updateCurrentModelDisplay(modelType);
                this.closeDropdown();
                
                // Recarregar a página se estivermos no chat para atualizar o contexto
                if (window.location.pathname.includes('/chat')) {
                    setTimeout(() => {
                        window.location.reload();
                    }, 1000);
                }
            } else {
                this.showNotification(data.message, 'error');
                // Recarregar a lista de modelos para restaurar o estado
                await this.loadAvailableModels();
            }
        } catch (error) {
            console.error('Erro ao alterar modelo:', error);
            this.showNotification(`Erro ao alterar modelo: ${error.message}`, 'error');
            await this.loadAvailableModels();
        } finally {
            this.isLoading = false;
        }
    }
    
    toggleDropdown() {
        if (this.dropdown.classList.contains('open')) {
            this.closeDropdown();
        } else {
            this.openDropdown();
        }
    }
    
    openDropdown() {
        this.dropdown.classList.add('open');
        this.loadAvailableModels();
    }
    
    closeDropdown() {
        this.dropdown.classList.remove('open');
    }
    
    showNotification(message, type = 'info') {
        const notification = document.getElementById('globalNotification');
        if (!notification) return;
        
        const content = notification.querySelector('.notification-content');
        
        content.className = `notification-content ${type}`;
        content.querySelector('p').textContent = message;
        
        notification.style.display = 'block';
        
        // Auto close
        setTimeout(() => {
            notification.style.display = 'none';
        }, 5000);
        
        // Close button
        const closeBtn = content.querySelector('.close-notification');
        if (closeBtn) {
            closeBtn.onclick = () => {
                notification.style.display = 'none';
            };
        }
    }
}

// Inicializar quando o DOM estiver carregado
document.addEventListener('DOMContentLoaded', function() {
    // Só inicializar se os elementos existirem
    if (document.getElementById('headerModelDropdown')) {
        new HeaderModelSelector();
    }
});

// Função global para compatibilidade com outros scripts
window.HeaderModelSelector = HeaderModelSelector;
});

