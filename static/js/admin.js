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
    
    const trainEvaluatorForm = document.getElementById('trainEvaluatorForm');
    const evaluatorTrainingStatus = document.getElementById('evaluatorTrainingStatus');
    const evaluatorsList = document.getElementById('evaluatorsList');

    // Elementos para o classificador SentenceTransformer
    const trainSTForm = document.getElementById('trainSTForm');
    const stTrainingStatus = document.getElementById('stTrainingStatus');
    const testSTForm = document.getElementById('testSTForm');
    const stTestResults = document.getElementById('stTestResults');
    const stResultsContent = document.getElementById('stResultsContent');

    // Inicialização
    loadModels();
    loadNaturezaModels();
    loadEvaluators(); 
    
    function loadEvaluators() {
        if (!evaluatorsList) return;
        
        evaluatorsList.innerHTML = `
            <div class="spinner"></div>
            <p>Carregando avaliadores disponíveis...</p>
        `;
        
        fetch('/api/natureza-evaluators')
            .then(response => response.json())
            .then(data => {
                if (!data.evaluators || data.evaluators.length === 0) {
                    evaluatorsList.innerHTML = `
                        <div class="status-container info">
                            <p>Nenhum avaliador de natureza disponível. Por favor, treine um avaliador primeiro.</p>
                        </div>
                    `;
                    return;
                }
                
                let evaluatorsHtml = '';
                data.evaluators.forEach(evaluator => {
                    evaluatorsHtml += `
                        <div class="model-item" data-provider="${evaluator.provider}">
                            <div class="model-info">
                                <div class="model-name">Avaliador ${evaluator.provider}</div>
                                <div class="model-meta">
                                    Naturezas: ${evaluator.natureza_count} | Criado em: ${formatDate(evaluator.created_at)}
                                    ${evaluator.is_active ? '<span class="badge-active">Ativo</span>' : ''}
                                </div>
                            </div>
                            <div class="model-actions">
                                <button class="small-button use-evaluator-btn" data-provider="${evaluator.provider}">Usar</button>
                                <button class="small-button delete-evaluator-btn">Excluir</button>
                            </div>
                        </div>
                    `;
                });
                
                evaluatorsList.innerHTML = evaluatorsHtml;
                
                // Adiciona event listeners para os botões
                document.querySelectorAll('.use-evaluator-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const provider = this.getAttribute('data-provider');
                        useEvaluator(provider);
                    });
                });
                
                document.querySelectorAll('.delete-evaluator-btn').forEach(btn => {
                    btn.addEventListener('click', function() {
                        const modelItem = this.closest('.model-item');
                        const provider = modelItem.getAttribute('data-provider');
                        deleteEvaluator(provider, modelItem);
                    });
                });
            })
            .catch(error => {
                console.error('Erro ao carregar avaliadores:', error);
                evaluatorsList.innerHTML = `
                    <div class="status-container error">
                        <p>Erro ao carregar avaliadores: ${error.message}</p>
                    </div>
                `;
            });
    }
    
    // Função auxiliar para formatar datas
    function formatDate(dateString) {
        if (!dateString) return 'Desconhecido';
        
        // Verifica se é uma data ISO
        if (dateString.includes('T')) {
            const date = new Date(dateString);
            if (isNaN(date.getTime())) return dateString;
            
            return date.toLocaleString('pt-BR');
        }
        
        return dateString;
    }
    
    // Funções para gerenciar avaliadores
    function useEvaluator(provider) {
        fetch('/api/change-model', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ model_type: provider }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(`Modelo e avaliador alterados para ${provider}`, 'success');
                loadEvaluators(); // Recarrega para atualizar o status "Ativo"
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao ativar avaliador: ${error.message}`, 'error');
        });
    }
    
    function deleteEvaluator(provider, modelItem) {
        if (!confirm(`Tem certeza que deseja excluir o avaliador ${provider}?`)) return;
        
        fetch('/api/delete-natureza-evaluator', {
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
                if (modelItem) {
                    modelItem.remove();
                }
            } else {
                showNotification(data.message, 'error');
            }
        })
        .catch(error => {
            console.error('Erro:', error);
            showNotification(`Erro ao excluir avaliador: ${error.message}`, 'error');
        });
    }
    
    // Adicionar este event listener para o formulário de treinamento do avaliador
    if (trainEvaluatorForm) {
        trainEvaluatorForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const provider = document.getElementById('evaluatorProvider').value;
            const forceRebuild = document.getElementById('forceRebuildEvaluator').checked;
            const mcaspFile = document.getElementById('mcaspFile').files[0];
            
            // Mostra status de treinamento
            if (evaluatorTrainingStatus) {
                trainEvaluatorForm.style.display = 'none';
                evaluatorTrainingStatus.classList.remove('hidden');
            }
            
            try {
                // Se um arquivo foi fornecido, precisamos fazer upload primeiro
                let mcaspPath = null;
                
                if (mcaspFile) {
                    // Cria um FormData para upload do arquivo
                    const formData = new FormData();
                    formData.append('file', mcaspFile);
                    
                    // Faz upload do arquivo
                    const uploadResponse = await fetch('/api/upload-mcasp', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!uploadResponse.ok) {
                        throw new Error('Falha ao fazer upload do arquivo MCASP');
                    }
                    
                    const uploadResult = await uploadResponse.json();
                    mcaspPath = uploadResult.file_path;
                }
                
                // Agora treina o avaliador
                const trainResponse = await fetch('/api/train-natureza-evaluator', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        provider: provider,
                        mcasp_path: mcaspPath,
                        force_rebuild: forceRebuild
                    }),
                });
                
                const trainResult = await trainResponse.json();
                
                if (evaluatorTrainingStatus) {
                    if (trainResult.success) {
                        evaluatorTrainingStatus.innerHTML = `
                            <div class="status-container success">
                                <i class="fas fa-check-circle"></i>
                                <p>${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-evaluator-form-btn">Voltar</button>
                        `;
                    } else {
                        evaluatorTrainingStatus.innerHTML = `
                            <div class="status-container error">
                                <i class="fas fa-exclamation-circle"></i>
                                <p>Erro no treinamento: ${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-evaluator-form-btn">Voltar</button>
                        `;
                    }
                    
                    // Adiciona evento para resetar o formulário
                    document.querySelector('.reset-evaluator-form-btn').addEventListener('click', function() {
                        evaluatorTrainingStatus.classList.add('hidden');
                        trainEvaluatorForm.style.display = 'block';
                        // Recarrega a lista de avaliadores
                        loadEvaluators();
                    });
                }
                
            } catch (error) {
                console.error('Erro:', error);
                if (evaluatorTrainingStatus) {
                    evaluatorTrainingStatus.innerHTML = `
                        <div class="status-container error">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Erro ao se comunicar com o servidor: ${error.message}</p>
                        </div>
                        <button type="button" class="primary-button reset-evaluator-form-btn">Voltar</button>
                    `;
                    
                    document.querySelector('.reset-evaluator-form-btn').addEventListener('click', function() {
                        evaluatorTrainingStatus.classList.add('hidden');
                        trainEvaluatorForm.style.display = 'block';
                    });
                }
            }
        });
    }
    
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


    if (trainSTForm) {
        trainSTForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const csvFile = document.getElementById('stCsvFile').files[0];
            
            // Verifica se um arquivo foi selecionado
            if (!csvFile) {
                showNotification('Por favor, selecione um arquivo CSV', 'warning');
                return;
            }
            
            // Mostra status de treinamento
            if (stTrainingStatus) {
                trainSTForm.style.display = 'none';
                stTrainingStatus.classList.remove('hidden');
            }
            
            try {
                // Cria um FormData para upload do arquivo
                const formData = new FormData();
                formData.append('file', csvFile);
                
                // Faz upload do arquivo
                const uploadResponse = await fetch('/api/upload-st-csv', {
                    method: 'POST',
                    body: formData
                });
                
                if (!uploadResponse.ok) {
                    throw new Error('Falha ao fazer upload do arquivo CSV');
                }
                
                const uploadResult = await uploadResponse.json();
                
                // Inicia o treinamento
                const trainResponse = await fetch('/api/train-st-classifier', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        csv_path: uploadResult.file_path
                    }),
                });
                
                const trainResult = await trainResponse.json();
                
                if (stTrainingStatus) {
                    if (trainResult.success) {
                        stTrainingStatus.innerHTML = `
                            <div class="status-container success">
                                <i class="fas fa-check-circle"></i>
                                <p>${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-st-form-btn">Voltar</button>
                        `;
                    } else {
                        stTrainingStatus.innerHTML = `
                            <div class="status-container error">
                                <i class="fas fa-exclamation-circle"></i>
                                <p>Erro no treinamento: ${trainResult.message}</p>
                            </div>
                            <button type="button" class="primary-button reset-st-form-btn">Voltar</button>
                        `;
                    }
                    
                    // Adiciona evento para resetar o formulário
                    document.querySelector('.reset-st-form-btn').addEventListener('click', function() {
                        stTrainingStatus.classList.add('hidden');
                        trainSTForm.style.display = 'block';
                    });
                }
                
            } catch (error) {
                console.error('Erro:', error);
                if (stTrainingStatus) {
                    stTrainingStatus.innerHTML = `
                        <div class="status-container error">
                            <i class="fas fa-exclamation-circle"></i>
                            <p>Erro ao se comunicar com o servidor: ${error.message}</p>
                        </div>
                        <button type="button" class="primary-button reset-st-form-btn">Voltar</button>
                    `;
                    
                    document.querySelector('.reset-st-form-btn').addEventListener('click', function() {
                        stTrainingStatus.classList.add('hidden');
                        trainSTForm.style.display = 'block';
                    });
                }
            }
        });
    }
    
    // Event listener para o formulário de teste do SentenceTransformer
    if (testSTForm) {
        testSTForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            
            const description = document.getElementById('stTestDescription').value;
            
            if (!description) {
                showNotification('Por favor, insira uma descrição para teste', 'warning');
                return;
            }
            
            // Mostra a área de resultados com spinner
            if (stResultsContent) {
                stResultsContent.innerHTML = `
                    <div class="spinner"></div>
                    <p>Classificando a descrição...</p>
                `;
                stTestResults.classList.remove('hidden');
            }
            
            try {
                const response = await fetch('/api/test-st-classifier', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        description: description
                    }),
                });
                
                const result = await response.json();
                
                if (stResultsContent) {
                    if (result.success) {
                        // Formata os resultados
                        let html = '<div class="st-predictions">';
                        
                        // Agrupar naturezas por elemento base
                        const naturezasPorElemento = {};
                        result.naturezas_completas.forEach(nat => {
                            if (nat.tipo === 'elemento') {
                                // Se for elemento, cria uma entrada própria
                                const key = nat.codigo;
                                if (!naturezasPorElemento[key]) {
                                    naturezasPorElemento[key] = {
                                        elemento: nat,
                                        subelementos: []
                                    };
                                }
                            } else if (nat.tipo === 'subelemento') {
                                // Se for subelemento, adiciona ao elemento base
                                const key = nat.elemento_base;
                                if (!naturezasPorElemento[key]) {
                                    naturezasPorElemento[key] = {
                                        elemento: {
                                            codigo: nat.elemento_base,
                                            descricao: nat.descricao_elemento,
                                            justificativa: nat.justificativa,
                                            score: nat.score
                                        },
                                        subelementos: []
                                    };
                                }
                                naturezasPorElemento[key].subelementos.push(nat);
                            }
                        });
                        
                        // Ordena os elementos pelo score (decrescente)
                        const elementosOrdenados = Object.values(naturezasPorElemento)
                            .sort((a, b) => b.elemento.score - a.elemento.score);
                        
                        // Renderiza as naturezas agrupadas
                        elementosOrdenados.forEach((grupo, index) => {
                            const elemento = grupo.elemento;
                            const score = (elemento.score * 100).toFixed(2);
                            const confidenceClass = score > 70 ? 'high' : 
                                                score > 40 ? 'medium' : 'low';
                            
                            html += `
                                <div class="prediction-item">
                                    <div class="prediction-rank">${index + 1}</div>
                                    <div class="prediction-content">
                                        <div class="prediction-code"><strong>Código:</strong> ${elemento.codigo}</div>
                                        <div class="prediction-name"><strong>Descrição:</strong> ${elemento.descricao}</div>
                                        <div class="prediction-confidence ${confidenceClass}"><strong>Confiança:</strong> ${score}%</div>
                                        ${elemento.justificativa ? `<div class="prediction-justification"><strong>Justificativa:</strong> ${elemento.justificativa}</div>` : ''}
                                    `;
                            
                            // Adiciona subelementos
                            if (grupo.subelementos && grupo.subelementos.length > 0) {
                                html += `
                                    <div class="prediction-subelements">
                                        <strong>Naturezas Completas (c.g.mm.ee.ss):</strong>
                                        <ul class="subelements-list">
                                `;
                                
                                grupo.subelementos.forEach(sub => {
                                    html += `
                                        <li class="subelement-item">
                                            <div class="subelement-code">${sub.codigo}</div>
                                            <div class="subelement-desc">${sub.descricao}</div>
                                        </li>
                                    `;
                                });
                                
                                html += `
                                        </ul>
                                    </div>
                                `;
                            }
                            
                            html += `
                                    </div>
                                </div>
                            `;
                        });
                        
                        html += '</div>';
                        
                        // Adiciona informações sobre o processamento
                        html += `<div class="info-box">
                            <p>Os resultados acima foram processados em múltiplas etapas:</p>
                            <ol>
                                <li>Classificação inicial com SentenceTransformer para identificar candidatos</li>
                                ${result.used_llm ? `<li>Refinamento usando o modelo ${result.llm_model || 'LLM'} para selecionar as naturezas mais adequadas</li>` : ''}
                                ${result.has_subelements ? `<li>Busca de naturezas completas (c.g.mm.ee.ss) para cada natureza selecionada</li>` : ''}
                            </ol>
                        </div>`;
                        
                        stResultsContent.innerHTML = html;
                    } else {
                        stResultsContent.innerHTML = `
                            <div class="status-container error">
                                <p>Erro na classificação: ${result.message}</p>
                            </div>
                        `;
                    }
                }
                
            } catch (error) {
                console.error('Erro:', error);
                if (stResultsContent) {
                    stResultsContent.innerHTML = `
                        <div class="status-container error">
                            <p>Erro ao se comunicar com o servidor: ${error.message}</p>
                        </div>
                    `;
                }
            }
        });
    }    
    // Adicionar estilos para os resultados
    const style = document.createElement('style');
    style.textContent = `
        .st-predictions {
            margin-top: 1rem;
        }
        .prediction-item {
            display: flex;
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: var(--radius);
            background-color: var(--ultra-light);
            border-left: 4px solid var(--primary-color);
        }
        .prediction-rank {
            font-size: 1.5rem;
            font-weight: bold;
            margin-right: 1rem;
            color: var(--primary-color);
            min-width: 2rem;
            text-align: center;
        }
        .prediction-content {
            flex: 1;
        }
        .prediction-code, .prediction-name, .prediction-confidence {
            margin-bottom: 0.5rem;
        }
        .prediction-confidence.high {
            color: var(--success);
        }
        .prediction-confidence.medium {
            color: var(--warning);
        }
        .prediction-confidence.low {
            color: var(--error);
        }
        .mt-4 {
            margin-top: 2rem;
        }
    `;
    document.head.appendChild(style);
    const additionalStyle = `
        .prediction-justification {
            margin-top: 0.75rem;
            padding: 0.75rem;
            background-color: #f8f9fa;
            border-left: 3px solid var(--primary-color);
            border-radius: 0 var(--radius) var(--radius) 0;
            font-style: italic;
        }
        
        .info-box {
            margin-top: 1.5rem;
            padding: 0.75rem;
            background-color: var(--light);
            border-radius: var(--radius);
            font-size: 0.9rem;
            color: var(--medium-dark);
        }
    `;
    document.head.appendChild(document.createElement('style')).textContent += additionalStyle;

});