/**
 * Script para a interface de chat com suporte a IPOF e upload de arquivos
 * Assistente de Orçamento Público
 */

// Aguarda carregamento do DOM
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM carregado - iniciando script de chat");
    
    // Elementos do chat
    const chatMessages = document.getElementById('chatMessages');
    const messageForm = document.getElementById('messageForm');
    const userInput = document.getElementById('userInput');
    const currentModelName = document.getElementById('currentModelName');
    const changeModelBtn = document.getElementById('changeModelBtn');
    const newConversationBtn = document.getElementById('newConversationBtn');
    
    // Log para depuração
    console.log("Elementos DOM encontrados:", {
        chatMessages: !!chatMessages,
        messageForm: !!messageForm,
        userInput: !!userInput,
        currentModelName: !!currentModelName,
        changeModelBtn: !!changeModelBtn,
        newConversationBtn: !!newConversationBtn
    });
    
    // Elementos de upload
    const fileUploadBtn = document.createElement('button');
    fileUploadBtn.type = 'button';
    fileUploadBtn.className = 'upload-button';
    fileUploadBtn.innerHTML = '<i class="fas fa-paperclip"></i>';
    fileUploadBtn.title = 'Anexar arquivo';
    
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.id = 'fileInput';
    fileInput.accept = '.pdf,.doc,.docx,.txt';
    fileInput.style.display = 'none';
    
    // Adiciona os elementos de upload ao formulário
    if (messageForm) {
        messageForm.appendChild(fileInput);
        const sendButton = messageForm.querySelector('.send-button');
        if (sendButton) {
            messageForm.insertBefore(fileUploadBtn, sendButton);
        } else {
            messageForm.appendChild(fileUploadBtn);
        }
    }
    
    // Elementos do modal de seleção de modelo
    const selectModelModal = document.getElementById('selectModelModal');
    const closeModalBtn = selectModelModal ? selectModelModal.querySelector('.close-modal') : null;
    const availableModelsList = document.getElementById('availableModelsList');
    
    // Histórico de mensagens
    let chatHistory = [];
    
    // ID da conversa atual
    let currentConversationId = generateId();
    
    // Flag para controlar se o assistente está "digitando"
    let isAssistantTyping = false;
    
    // Função para gerar ID único
    function generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substring(2);
    }
    
    // Função para formatar a mensagem (substituto simples para Markdown)
    function formatMessage(text) {
        if (!text) return '';
        
        // Escape HTML para evitar injeção de código
        text = escapeHtml(text);
        
        // Formatação de texto
        text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');  // Negrito
        text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');  // Itálico
        
        // Código inline
        text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
        
        // Blocos de código
        text = text.replace(/```([^`]*)```/g, function(match, code) {
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
    
    // Função para adicionar mensagem ao chat
    function addMessage(content, role, timestamp = new Date(), withTypingEffect = true) {
        if (!chatMessages) {
            console.error("Elemento chatMessages não encontrado");
            return;
        }
        
        console.log(`Adicionando mensagem como ${role}:`, content.substring(0, 50) + "...");
        
        // Cria elemento da mensagem
        const messageElement = document.createElement('div');
        messageElement.className = `message ${role}`;
        
        // Formata a hora
        const formattedTime = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Estrutura básica da mensagem
        if (role !== 'assistant' || !withTypingEffect) {
            // Para mensagens do usuário ou do sistema, ou quando não queremos efeito de digitação
            messageElement.innerHTML = `
                <div class="message-content">
                    <p>${formatMessage(content)}</p>
                    <div class="message-time">${formattedTime}</div>
                </div>
            `;
        } else {
            // Para mensagens do assistente com efeito de digitação
            messageElement.innerHTML = `
                <div class="message-content">
                    <p></p>
                    <div class="message-time">${formattedTime}</div>
                </div>
            `;
        }
        
        // Adiciona ao chat
        chatMessages.appendChild(messageElement);
        
        // Rola para a parte inferior do chat
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Adiciona efeito de digitação para mensagens do assistente
        if (role === 'assistant' && withTypingEffect) {
            const textElement = messageElement.querySelector('p');
            startTypewriter(textElement, content);
        }
        
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
    
    // Função para iniciar o efeito de digitação
    function startTypewriter(element, text) {
        isAssistantTyping = true;
        
        // Adiciona classe para indicar que está digitando
        element.classList.add('typing');
        
        // Inicializa variáveis
        let i = 0;
        let fullText = text;
        let currentText = '';
        
        // Define um temporizador mais lento para textos longos, mais rápido para textos curtos
        const baseSpeed = text.length > 1000 ? 2 : (text.length > 500 ? 5 : 10);
        
        // Função para digitar o próximo caractere
        function typeCharacter() {
            if (i < fullText.length) {
                // Adiciona o próximo caractere
                currentText += fullText.charAt(i);
                i++;
                
                // Atualiza o conteúdo do elemento
                element.innerHTML = formatMessage(currentText);
                
                // Rola para manter o texto visível
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Determina o atraso para o próximo caractere
                let delay = baseSpeed;
                
                // Adiciona variação e pausas para pontuação
                const currentChar = fullText.charAt(i-1);
                if (['.', '!', '?'].includes(currentChar)) {
                    delay = 300; // Pausa mais longa para finais de frase
                } else if ([',', ';', ':'].includes(currentChar)) {
                    delay = 150; // Pausa média para outras pontuações
                } else if (currentChar === '\n') {
                    delay = 200; // Pausa para quebras de linha
                } else {
                    // Pequena variação aleatória para parecer mais natural
                    delay = baseSpeed + Math.floor(Math.random() * 15);
                }
                
                // Agenda a digitação do próximo caractere
                setTimeout(typeCharacter, delay);
            } else {
                // Terminou de digitar
                isAssistantTyping = false;
                element.classList.remove('typing');
                element.classList.add('typing-done');
            }
        }
        
        // Inicia o processo de digitação
        typeCharacter();
    }
    
    // Função para adicionar indicador de digitação
    function addTypingIndicator() {
        if (!chatMessages) {
            console.error("Elemento chatMessages não encontrado");
            return;
        }
        
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
    
    // Função para carregar o nome do modelo atual
    async function loadCurrentModel() {
        if (!currentModelName) {
            console.error("Elemento currentModelName não encontrado");
            return;
        }
        
        try {
            console.log("Iniciando carregamento do modelo atual");
            currentModelName.textContent = "Carregando...";
            
            const response = await fetch('/api/current-model');
            console.log("Resposta da API current-model:", response);
            
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Dados do modelo atual:", data);
            
            if (data && data.model) {
                currentModelName.textContent = `${data.model.type.toUpperCase()} (${data.model.name})`;
            } else {
                currentModelName.textContent = 'Nenhum modelo selecionado';
            }
            
        } catch (error) {
            console.error('Erro ao carregar modelo atual:', error);
            currentModelName.textContent = 'Erro ao carregar modelo';
        }
    }
    
    // Função para carregar lista de modelos disponíveis
    async function loadAvailableModels() {
        if (!availableModelsList) {
            console.error("Elemento availableModelsList não encontrado");
            return;
        }
        
        // Mostra mensagem de carregamento
        availableModelsList.innerHTML = `
            <div class="spinner"></div>
            <p>Carregando modelos disponíveis...</p>
        `;
        
        try {
            console.log("Iniciando carregamento dos modelos disponíveis");
            
            const response = await fetch('/api/models');
            console.log("Resposta da API models:", response);
            
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Dados dos modelos disponíveis:", data);
            
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
            console.log(`Iniciando mudança de modelo para: ${modelType}`);
            
            const response = await fetch('/api/change-model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType }),
            });
            
            console.log("Resposta da API change-model:", response);
            
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Dados da resposta change-model:", data);
            
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
    
    // Função para enviar mensagem para o servidor
    async function sendMessage(message) {
        // Verifica se o assistente já está "digitando"
        if (isAssistantTyping) {
            showNotification("Por favor, aguarde o assistente terminar de digitar.", "warning");
            return;
        }
        
        try {
            console.log("Enviando mensagem para o servidor:", message);
            
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
            
            console.log("Resposta da API chat:", response);
            
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Dados da resposta chat:", data);
            
            // Remove o indicador de digitação
            removeTypingIndicator();
            
            // Adiciona a resposta ao chat com efeito de digitação
            if (data && data.response) {
                addMessage(data.response, 'assistant');
            } else {
                addMessage("Desculpe, recebi uma resposta vazia do servidor.", 'system');
            }
            
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
    
    // Função para enviar um arquivo para o servidor
    async function uploadFile(file, message = '') {
        // Verifica se o assistente já está "digitando"
        if (isAssistantTyping) {
            showNotification("Por favor, aguarde o assistente terminar de digitar.", "warning");
            return;
        }
        
        try {
            // Verifica se o arquivo é válido
            if (!file) {
                throw new Error('Nenhum arquivo selecionado');
            }
            
            // Verifica extensão permitida
            const fileExt = file.name.split('.').pop().toLowerCase();
            const allowedExts = ['pdf', 'doc', 'docx', 'txt'];
            
            if (!allowedExts.includes(fileExt)) {
                throw new Error('Tipo de arquivo não permitido. Use PDF, DOC, DOCX ou TXT.');
            }
            
            console.log(`Enviando arquivo: ${file.name} (${file.size} bytes)`);
            
            // Adiciona mensagem indicando o upload
            addMessage(`📎 Enviando arquivo: ${file.name}`, 'user');
            
            // Adiciona indicador de digitação
            addTypingIndicator();
            
            // Cria um objeto FormData para enviar o arquivo
            const formData = new FormData();
            formData.append('file', file);
            formData.append('message', message);
            formData.append('history', JSON.stringify(chatHistory));
            
            // Envia o arquivo para o servidor
            const response = await fetch('/api/upload-chat-file', {
                method: 'POST',
                body: formData
            });
            
            console.log("Resposta da API upload-chat-file:", response);
            
            if (!response.ok) {
                throw new Error(`Erro HTTP: ${response.status}`);
            }
            
            const data = await response.json();
            console.log("Dados da resposta upload-chat-file:", data);
            
            // Remove o indicador de digitação
            removeTypingIndicator();
            
            // Adiciona a resposta ao chat
            if (data && data.response) {
                addMessage(data.response, 'assistant');
            } else {
                addMessage("Desculpe, recebi uma resposta vazia do servidor.", 'system');
            }
            
        } catch (error) {
            console.error('Erro ao enviar arquivo:', error);
            
            // Remove o indicador de digitação
            removeTypingIndicator();
            
            // Adiciona mensagem de erro
            addMessage(
                `Desculpe, ocorreu um erro ao processar seu arquivo: ${error.message}`,
                'system'
            );
        }
    }
    
    // Função para iniciar nova conversa
    // Modificação na função startNewConversation
    function startNewConversation() {
        // Verifica se o assistente está digitando
        if (isAssistantTyping) {
            showNotification("Por favor, aguarde o assistente terminar de digitar antes de iniciar uma nova conversa.", "warning");
            return;
        }
        
        // Gera novo ID para a conversa
        currentConversationId = generateId();
        
        // Limpa histórico
        chatHistory = [];
        
        // Limpa o chat
        if (chatMessages) {
            chatMessages.innerHTML = '';
            
            // Adiciona apenas uma mensagem de boas-vindas
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
    
    // Função para carregar histórico salvo
    // Modificação na função loadSavedHistory do arquivo static/js/chat.js
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
                    
                    // Adiciona mensagens do histórico sem efeito de digitação
                    chatHistory.forEach(msg => {
                        // Adiciona mensagens diretas, sem animação
                        addMessage(msg.content, msg.role, new Date(msg.timestamp), false);
                    });
                    
                    // Rola para o final após carregar todas as mensagens
                    chatMessages.scrollTop = chatMessages.scrollHeight;
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
            
            // Se não conseguiu carregar histórico, adiciona mensagem de boas-vindas
            addMessage(
                'Olá! Sou o Assistente de Orçamento Público. Como posso ajudá-lo(a) hoje?',
                'system'
            );
        }
    }
    
    // Event listener para formulário de mensagem
    if (messageForm) {
        messageForm.addEventListener('submit', function(event) {
            event.preventDefault();
            console.log("Formulário de mensagem enviado");
            
            if (!userInput || !userInput.value.trim()) {
                console.log("Campo de entrada vazio, ignorando envio");
                return;
            }
            
            // Verifica se o assistente está digitando
            if (isAssistantTyping) {
                showNotification("Por favor, aguarde o assistente terminar de digitar.", "warning");
                return;
            }
            
            const message = userInput.value.trim();
            console.log("Mensagem capturada:", message);
            
            // Adiciona a mensagem ao chat
            addMessage(message, 'user');
            
            // Limpa o campo de entrada
            userInput.value = '';
            
            // Envia a mensagem para o servidor
            sendMessage(message);
        });
    } else {
        console.error("Formulário de mensagem não encontrado");
    }
    
    // Event listener para botão de upload
    if (fileUploadBtn) {
        fileUploadBtn.addEventListener('click', function() {
            console.log("Botão de upload clicado");
            fileInput.click();
        });
    }
    
    // Event listener para seleção de arquivo
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            console.log("Arquivo selecionado:", this.files);
            if (this.files && this.files[0]) {
                // Verifica se o assistente está digitando
                if (isAssistantTyping) {
                    showNotification("Por favor, aguarde o assistente terminar de digitar.", "warning");
                    this.value = '';  // Limpa a seleção
                    return;
                }
                
                // Obtém o arquivo selecionado
                const file = this.files[0];
                
                // Enviar o arquivo para o servidor
                uploadFile(file, userInput.value);
                
                // Limpa o campo de entrada
                userInput.value = '';
                
                // Limpa o input de arquivo
                this.value = '';
            }
        });
    }
    
    // Event listener para botão de alterar modelo
    if (changeModelBtn) {
        changeModelBtn.addEventListener('click', function() {
            console.log("Botão de alterar modelo clicado");
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
            console.log("Botão de nova conversa clicado");
            if (confirm('Iniciar uma nova conversa? O histórico atual será perdido.')) {
                startNewConversation();
            }
        });
    }
    
    // Event listener para fechar o modal
    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', function() {
            console.log("Botão de fechar modal clicado");
            if (selectModelModal) {
                selectModelModal.style.display = 'none';
            }
        });
    }
    
    // Fechar o modal ao clicar fora dele
    window.addEventListener('click', function(event) {
        if (event.target === selectModelModal) {
            console.log("Clique fora do modal detectado");
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
                    console.log("Enter pressionado, enviando formulário");
                    messageForm.dispatchEvent(new Event('submit'));
                }
            }
        });
    }
    
    // Inicialização
    console.log("Iniciando carregamento do modelo atual");
    loadCurrentModel();
    
    console.log("Iniciando carregamento do histórico");
    // Removendo qualquer mensagem que possa ter sido adicionada antes disso
    if (chatMessages) {
        chatMessages.innerHTML = '';
    }
    loadSavedHistory();
    
    // Adicionar estilo CSS para o botão de upload
    const uploadStyle = document.createElement('style');
    uploadStyle.textContent = `
        .upload-button {
            width: 36px;
            height: 36px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: var(--light);
            color: var(--medium);
            border: none;
            border-radius: 50%;
            margin-right: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .upload-button:hover {
            background-color: var(--medium-light);
            color: var(--dark);
        }
        
        #messageForm {
            display: flex;
            align-items: flex-end;
        }
    `;
    document.head.appendChild(uploadStyle);
    
    console.log("Inicialização do script de chat concluída");
});