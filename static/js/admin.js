// static/js/admin.js

document.addEventListener('DOMContentLoaded', function() {
    // Elementos DOM
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Formulários
    const trainModelForm = document.getElementById('trainModelForm');
    const trainNaturezaForm = document.getElementById('trainNaturezaForm');
    
    // Status de treinamento
    const trainingStatus = document.getElementById('trainingStatus');
    const naturezaTrainingStatus = document.getElementById('naturezaTrainingStatus');
    
    // Listas de modelos
    const modelsList = document.getElementById('modelsList');
    const naturezaModelsList = document.getElementById('naturezaModelsList');
    
    // Inicialização
    loadModels();
    loadNaturezaModels();
    
    // Gerenciamento de abas
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remove a classe active de todos os botões e conteúdos
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            // Adiciona a classe active ao botão clicado
            this.classList.add('active');
            
            // Mostra o conteúdo correspondente
            const tabId = this.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Treinamento de modelo RAG
    if (trainModelForm) {
        trainModelForm.addEventListener('submit', function(event) {
            event.preventDefault();
            
            const modelType = document.getElementById('modelType').value;
            
            // Mostra status de treinamento
            if (trainingStatus) {
                trainModelForm.style.display = 'none';
                trainingStatus.classList.remove('hidden');
            }
            
            // Envia requisição para API
            fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model_type: modelType }),
            })
            .then(response => response.json())
            .then(data => {
                if (trainingStatus) {
                    if (data.success) {
                        trainingStatus.innerHTML = `
                            <div class="status-container success">
                                <i class="fas fa-check-circle"></i>
                                <p>${data.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-form-btn">Voltar</button>
                        `;
                    } else {
                        trainingStatus.innerHTML = `
                            <div class="status-container error">
                                <i class="fas fa-exclamation-circle"></i>
                                <p>Erro no treinamento: ${data.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-form-btn">Voltar</button>
                        `;
                    }
                    
                    // Adiciona evento para resetar o formulário
                    document.querySelector('.reset-form-btn').addEventListener('click', function() {
                        trainingStatus.classList.add('hidden');
                        trainModelForm.style.display = 'block';
                        // Recarrega a lista de modelos
                        loadModels();
                    });
                }
            })
            .catch(error => {
                console.error('Erro:', error);
                if (trainingStatus) {
                    trainingStatus.innerHTML = `
                        <div class="status-container error">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Erro ao se comunicar com o servidor: ${error.message}</p>
                        </div>
                        <button type="button" class="primary-button reset-form-btn">Voltar</button>
                    `;
                    
                    document.querySelector('.reset-form-btn').addEventListener('click', function() {
                        trainingStatus.classList.add('hidden');
                        trainModelForm.style.display = 'block';
                    });
                }
            });
        });
    }
    
    // Treinamento de classificador de natureza
    if (trainNaturezaForm) {
        trainNaturezaForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const embeddingProvider = document.getElementById('embeddingProvider').value;
            const forceRebuild = document.getElementById('forceRebuild').checked;
            const excelFile = document.getElementById('excelFile').files[0];
            
            // Mostra status de treinamento
            if (naturezaTrainingStatus) {
                trainNaturezaForm.style.display = 'none';
                naturezaTrainingStatus.classList.remove('hidden');
            }
            
            try {
                // Se um arquivo foi fornecido, precisamos fazer upload primeiro
                let excelPath = null;
                
                if (excelFile) {
                    // Cria um FormData para upload do arquivo
                    const formData = new FormData();
                    formData.append('file', excelFile);
                    
                    // Faz upload do arquivo
                    const uploadResponse = await fetch('/api/upload-excel', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!uploadResponse.ok) {
                        throw new Error('Falha ao fazer upload do arquivo Excel');
                    }
                    
                    const uploadResult = await uploadResponse.json();
                    excelPath = uploadResult.file_path;
                }
                
                // Agora treina o classificador
                const trainResponse = await fetch('/api/train-natureza', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        embedding_provider: embeddingProvider,
                        excel_path: excelPath,
                        force_rebuild: forceRebuild
                    }),
                });
                
                const trainResult = await trainResponse.json();
                
                if (naturezaTrainingStatus) {
                    if (trainResult.success) {
                        naturezaTrainingStatus.innerHTML = `
                            <div class="status-container success">
                                <i class="fas fa-check-circle"></i>
                                <p>${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-natureza-form-btn">Voltar</button>
                        `;
                    } else {
                        naturezaTrainingStatus.innerHTML = `
                            <div class="status-container error">
                                <i class="fas fa-exclamation-circle"></i>
                                <p>Erro no treinamento: ${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-natureza-form-btn">Voltar</button>
                        `;
                    }
                    
                    // Adiciona evento para resetar o formulário
                    document.querySelector('.reset-natureza-form-btn').addEventListener('click', function() {
                        naturezaTrainingStatus.classList.add('hidden');
                        trainNaturezaForm.style.display = 'block';
                        // Recarrega a lista de classificadores
                        loadNaturezaModels();
                    });
                }
                
            } catch (error) {
                console.error('Erro:', error);
                if (naturezaTrainingStatus) {
                    naturezaTrainingStatus.innerHTML = `
                        <div class="status-container error">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Erro ao se comunicar com o servidor: ${error.message}</p>
                        </div>
                        <button type="button" class="primary-button reset-natureza-form-btn">Voltar</button>
                    `;
                    
                    document.querySelector('.reset-natureza-form-btn').addEventListener('click', function() {
                        naturezaTrainingStatus.classList.add('hidden');
                        trainNaturezaForm.style.display = 'block';
                    });
                }
            }
        });
    }
    
    // Função para carregar a lista de modelos RAG
    function loadModels() {
        if (!modelsList) return;
        
        modelsList.innerHTML = `
            <div class="spinner"></div>
            <p>Carregando modelos disponíveis...</p>
        `;
        
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                if (!data.models || data.models.length === 0) {
                    modelsList.innerHTML = `
                        <div class="status-container info">
                            <p>Nenhum modelo treinado disponível. Por favor, treine um modelo primeiro.</p>
                        </div>
                    `;
                    return;
                }
                
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
                        useModel(modelType);
                    });
                });
                
                document.querySelectorAll('.delete-model-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const modelItem = this.closest('.model-item');
                        const modelPath = modelItem.getAttribute('data-model-path');
                        deleteModel(modelPath, modelItem);
                    });
                });
            })
            .catch(error => {
                console.error('Erro ao carregar modelos:', error);
                modelsList.innerHTML = `
                    <div class="status-container error">
                        <p>Erro ao carregar modelos: ${error.message}</p>
                    </div>
                `;
            });
    }
    
    // Função para carregar a lista de classificadores de natureza
    function loadNaturezaModels() {
        if (!naturezaModelsList) return;
        
        naturezaModelsList.innerHTML = `
            <div class="spinner"></div>
            <p>Carregando classificadores disponíveis...</p>
        `;
        
        fetch('/api/natureza-models')
            .then(response => response.json())
            .then(data => {
                if (!data.models || data.models.length === 0) {
                    naturezaModelsList.innerHTML = `
                        <div class="status-container info">
                            <p>Nenhum classificador de natureza disponível. Por favor, treine um classificador primeiro.</p>
                        </div>
                    `;
                    return;
                }
                
                let modelsHtml = '';
                data.models.forEach(model => {
                    modelsHtml += `
                        <div class="model-item" data-model-id="${model.id}">
                            <div class="model-info">
                                <div class="model-name">Classificador ${model.embedding_provider}</div>
                                <div class="model-meta">
                                    Exemplos: ${model.num_examples} | Criado em: ${model.created}
                                </div>
                            </div>
                            <div class="model-actions">
                                <button class="small-button use-natureza-btn" data-provider="${model.embedding_provider}">Usar</button>
                                <button class="small-button delete-natureza-btn">Excluir</button>
                            </div>
                        </div>
                    `;
                });
                
                naturezaModelsList.innerHTML = modelsHtml;
                
                // Adiciona event listeners para os botões
                document.querySelectorAll('.use-natureza-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const provider = this.getAttribute('data-provider');
                        useNaturezaModel(provider);
                    });
                });
                
                document.querySelectorAll('.delete-natureza-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const modelItem = this.closest('.model-item');
                        const modelId = modelItem.getAttribute('data-model-id');
                        deleteNaturezaModel(modelId, modelItem);
                    });
                });
            })
            .catch(error => {
                console.error('Erro ao carregar classificadores:', error);
                naturezaModelsList.innerHTML = `
                    <div class="status-container error">
                        <p>Erro ao carregar classificadores: ${error.message}</p>
                    </div>
                `;
            });
    }
    
    // Funções para gerenciar modelos
    function useModel(modelType) {
        fetch('/api/change-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_type: modelType }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, 'success');
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao alterar modelo: ${error.message}`, 'error');
        });
    }
    
    function deleteModel(modelPath, modelItem) {
        if (!confirm('Tem certeza que deseja excluir este modelo?')) return;
        
        fetch('/api/delete-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_path: modelPath }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, 'success');
                if (modelItem) {
                    modelItem.remove();
                }
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao excluir modelo: ${error.message}`, 'error');
        });
    }
    
    // Funções para gerenciar classificadores de natureza
    function useNaturezaModel(provider) {
        fetch('/api/set-natureza-provider', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ provider: provider }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, 'success');
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao definir classificador: ${error.message}`, 'error');
        });
    }
    
    function deleteNaturezaModel(modelId, modelItem) {
        if (!confirm('Tem certeza que deseja excluir este classificador?')) return;
        
        fetch('/api/delete-natureza-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_id: modelId }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, 'success');
                if (modelItem) {
                    modelItem.remove();
                }
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao excluir classificador: ${error.message}`, 'error');
        });
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
});