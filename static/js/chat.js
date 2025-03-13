/**
 * Script específico para a interface de chat
 * Assistente de Orçamento Público
 */

// Aguarda carregamento do DOM
document.addEventListener('DOMContentLoaded', function() {
    // Elementos do chat
    const chatMessages = document.getElementById('chatMessages');
    const messageForm = document.getElementById('messageForm');
    const userInput = document.getElementById('userInput');
    const currentModelName = document.getElementById('currentModelName');
    const changeModelBtn = document.getElementById('changeModelBtn');
    const newConversationBtn = document.getElementById('newConversationBtn');
    
    // Elementos do modal de seleção de modelo
    const selectModelModal = document.getElementById('selectModelModal');
    const closeModalBtn = selectModelModal ? selectModelModal.querySelector('.close-modal') : null;
    const availableModelsList = document.getElementById('availableModelsList');
    
    // Histórico de mensagens
    let chatHistory = [];
    
    // ID da conversa atual
    let currentConversationId = generateId();
    
    // Função para gerar ID único
    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }
    
    // Função para carregar o nome do modelo atual
    async function loadCurrentModel() {
        if (!currentModelName) return;
        
        try {
            const response = await fetch('/api/current-model');
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error('Falha ao obter modelo atual');
            }
            
            if (data.model) {
                currentModelName.textContent = `${data.model.type.toUpperCase()} (${data.model.name})`;
            } else {
                currentModelName.textContent = 'Nenhum modelo selecionado';
            }
            
        } catch (error) {
            console.error('Erro ao carregar modelo atual:', error);
            currentModelName.textContent = 'Erro ao carregar modelo';
        }
    }
    
    // Carrega o modelo atual ao iniciar
    loadCurrentModel();
    
    // Função para carregar lista de modelos disponíveis
    async function loadAvailableModels() {
        if (!availableModelsList) return;
        
        // Mostra mensagem de carregamento
        availableModelsList.innerHTML = `
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
                availableModelsList.innerHTML = `
                    <div class="status-container info">
                        <p>Nenhum modelo treinado disponível. Por favor, treine um modelo primeiro.</p>
                    </div>
                `;
                return;
            }
            
            // Organiza modelos por tipo
            const modelsByType = {};
            data.models.forEach(model => {
                if (!modelsByType[model.type]) {
                    modelsByType[model.type] = [];
                }
                modelsByType[model.type].push(model);
            });
            
            // Renderiza lista de modelos agrupados por tipo
            let modelsHtml = '';
            for (const [type, models] of Object.entries(modelsByType)) {
                modelsHtml += `<h4>${type.toUpperCase()}</h4>`;
                
                models.forEach(model => {
                    modelsHtml += `
                        <div class="model-item" data-model-path="${model.path}">
                            <div class="model-info">
                                <div class="model-name">${model.name}</div>
                                <div class="model-meta">
                                    Criado em: ${model.created}
                                </div>
                            </div>
                            <div class="model-actions">
                                <button class="small-button select-model-btn" data-model-type="${model.type}">Selecionar</button>
                            </div>
                        </div>
                    `;
                });
            }
            
            availableModelsList.innerHTML = modelsHtml;
            
            // Adiciona event listeners para os botões
            document.querySelectorAll('.select-model-btn').forEach(btn => {
                btn.addEventListener('click', function() {
                    const modelType = this.getAttribute('data-model-type');
                    changeModel(modelType);
                });
            });
            
        } catch (error) {
            console.error('Erro ao carregar modelos:', error);
            availableModelsList.innerHTML = `
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
            
            // Atualiza o modelo exibido
            loadCurrentModel();
            
            // Fecha o modal após a alteração
            if (selectModelModal) {
                selectModelModal.style.display = 'none';
            }
            
        } catch (error) {
            console.error('Erro ao alterar modelo:', error);
            showNotification(`Erro ao alterar modelo: ${error.message}`, 'error');
        }
    }
    
    // Função para adicionar mensagem ao chat
    function addMessage(content, role, timestamp = new Date()) {
        if (!chatMessages) return;
        
        // Cria elemento da mensagem
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        // Formata a hora
        const formattedTime = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Conteúdo da mensagem com formatação Markdown
        messageElement.innerHTML = `
            <div class="message-content">
                <p>${formatMessage(content)}</p>
                <div class="message-time">${formattedTime}</div>
            </div>
        `;
        
        // Adiciona ao chat
        chatMessages.appendChild(messageElement);
        
        // Rola para a parte inferior do chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Adiciona ao histórico
        if (role !== 'system') {
            chatHistory.push({
                role: role,
                content: content,
                timestamp: timestamp.toISOString()
            });
            
            // Limita o histórico a 100 mensagens para evitar problemas de desempenho
            if (chatHistory.length > 100) {
                chatHistory.shift();
            }
            
            // Salva o histórico no localStorage
            try {
                localStorage.setItem(`chat_history_${currentConversationId}`, JSON.stringify(chatHistory));
            } catch (e) {
                console.warn('Falha ao salvar histórico no localStorage:', e);
            }
        }
    }
    
    // Função para adicionar indicador de digitação
    function addTypingIndicator() {
        if (!chatMessages) return;
        
        // Remove qualquer indicador existente
        removeTypingIndicator();
        
        // Cria elemento do indicador
        const indicatorElement = document.createElement('div');
        indicatorElement.className = 'typing-indicator';
        indicatorElement.id = 'typingIndicator';
        indicatorElement.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        
        // Adiciona ao chat
        chatMessages.appendChild(indicatorElement);
        
        // Rola para a parte inferior do chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Função para remover indicador de digitação
    function removeTypingIndicator() {
        const existingIndicator = document.getElementById('typingIndicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
    }
    
    // Função para formatar a mensagem (substituto simples para Markdown)
    function formatMessage(text) {
        // Escape HTML para evitar injeção de código
        text = escapeHtml(text);
        
        // Formatação de texto
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');  // Negrito
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');  // Itálico
        
        // Código inline
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Blocos de código
        text = text.replace(/```([^`]*?)```/g, function(match, code) {
            return `<pre>${code.trim()}</pre>`;
        });
        
        // Links
        text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
        
        // Quebras de linha
        text = text.replace(/\n/g, '<br>');
        
        return text;
    }
    
    // Função para escapar HTML
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    // Função para enviar mensagem para o servidor
    async function sendMessage(message) {
        try {
            // Adiciona indicador de digitação
            addTypingIndicator();
            
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    history: chatHistory
                }),
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Falha ao processar mensagem');
            }
            
            // Remove o indicador de digitação
            removeTypingIndicator();
            
            // Adiciona a resposta ao chat
            addMessage(data.response, 'assistant');
            
        } catch (error) {
            console.error('Erro ao enviar mensagem:', error);
            
            // Remove o indicador de digitação
            removeTypingIndicator();
            
            // Adiciona mensagem de erro
            addMessage(
                `Desculpe, ocorreu um erro ao processar sua mensagem: ${error.message}`,
                'system'
            );
        }
    }
    
    // Função para iniciar nova conversa
    function startNewConversation() {
        // Gera novo ID para a conversa
        currentConversationId = generateId();
        
        // Limpa histórico
        chatHistory = [];
        
        // Limpa o chat
        if (chatMessages) {
            chatMessages.innerHTML = '';
            
            // Adiciona mensagem de boas-vindas
            addMessage(
                'Olá! Sou o Assistente de Orçamento Público. Como posso ajudá-lo(a) hoje?',
                'system'
            );
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
    
    // Event listener para formulário de mensagem
    if (messageForm) {
        messageForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            if (!userInput || !userInput.value.trim()) return;
            
            const message = userInput.value.trim();
            
            // Adiciona a mensagem ao chat
            addMessage(message, 'user');
            
            // Limpa o campo de entrada
            userInput.value = '';
            
            // Envia a mensagem para o servidor
            sendMessage(message);
        });
    }
    
    // Event listener para botão de alterar modelo
    if (changeModelBtn) {
        changeModelBtn.addEventListener('click', function() {
            if (selectModelModal) {
                // Carrega modelos disponíveis
                loadAvailableModels();
                
                // Abre o modal
                selectModelModal.style.display = 'block';
            }
        });
    }
    
    // Event listener para botão de nova conversa
    if (newConversationBtn) {
        newConversationBtn.addEventListener('click', function() {
            if (confirm('Iniciar uma nova conversa? O histórico atual será perdido.')) {
                startNewConversation();
            }
        });
    }
    
    // Event listener para fechar o modal
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', function() {
            if (selectModelModal) {
                selectModelModal.style.display = 'none';
            }
        });
    }
    
    // Fechar o modal ao clicar fora dele
    window.addEventListener('click', function(event) {
        if (event.target === selectModelModal) {
            selectModelModal.style.display = 'none';
        }
    });
    
    // Permitir redimensionamento da área de texto
    if (userInput) {
        userInput.addEventListener('input', function() {
            // Ajusta a altura com base no conteúdo
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
            
            // Limita a altura máxima
            if (this.scrollHeight > 150) {
                this.style.height = '150px';
                this.style.overflowY = 'auto';
            } else {
                this.style.overflowY = 'hidden';
            }
        });
    }
    
    // Tecla Enter para enviar (com Shift+Enter para nova linha)
    if (userInput) {
        userInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                
                // Simula envio do formulário se houver conteúdo
                if (this.value.trim() && messageForm) {
                    messageForm.dispatchEvent(new Event('submit'));
                }
            }
        });
    }
    
    // Carrega histórico salvo
    function loadSavedHistory() {
        try {
            const savedHistory = localStorage.getItem(`chat_history_${currentConversationId}`);
            if (savedHistory) {
                chatHistory = JSON.parse(savedHistory);
                
                // Reconstrói o chat
                if (chatMessages) {
                    chatMessages.innerHTML = '';
                    
                    // Adiciona mensagem de boas-vindas
                    addMessage(
                        'Olá! Sou o Assistente de Orçamento Público. Como posso ajudá-lo(a) hoje?',
                        'system'
                    );
                    
                    // Adiciona mensagens do histórico
                    chatHistory.forEach(msg => {
                        addMessage(
                            msg.content,
                            msg.role,
                            new Date(msg.timestamp)
                        );
                    });
                }
            } else {
                // Não há histórico salvo, adiciona mensagem de boas-vindas
                addMessage(
                    'Olá! Sou o Assistente de Orçamento Público. Como posso ajudá-lo(a) hoje?',
                    'system'
                );
            }
        } catch (e) {
            console.warn('Falha ao carregar histórico do localStorage:', e);
            
            // Adiciona mensagem de boas-vindas
            addMessage(
                'Olá! Sou o Assistente de Orçamento Público. Como posso ajudá-lo(a) hoje?',
                'system'
            );
        }
    }
    
    // Carrega histórico ao iniciar
    loadSavedHistory();
    
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