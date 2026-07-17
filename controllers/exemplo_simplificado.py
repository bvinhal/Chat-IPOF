"""
Exemplo de Uso Simplificado

Demonstra como usar a arquitetura simplificada para:
1. Processar dados
2. Treinar modelos
3. Avaliar em teste e validação

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import logging
import pandas as pd
from data_processor import DataProcessor
from supervised_models import SVMClassifier, LogisticClassifier
from unsupervised_models import SimilarityClassifier, KMeansClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)

# Tentar importar LightGBM
try:
    from supervised_models import LightGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️  LightGBM não disponível. Instale com: pip install lightgbm")

# Tentar importar MLP+S-BERT
try:
    from neural_models import MLPSBERTClassifier
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("⚠️  MLP+S-BERT não disponível. Instale com: pip install sentence-transformers torch")


def main():
    print("="*80)
    print("EXEMPLO DE USO SIMPLIFICADO")
    print("="*80)
    
    # ========================================================================
    # ETAPA 1: PROCESSAR DADOS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 1: PROCESSAR DADOS")
    print("="*80)
    
    processor = DataProcessor(
        excel_path='data/natureza/natureza_despesa_final.xlsx',
        min_samples_per_class=5
    )
    
    data = processor.load_and_process()
    
    # Dados prontos
    X_train = data['X_train']
    X_val = data['X_val']
    X_test = data['X_test']
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    print(f"\n✅ Dados processados!")
    print(f"   Treino: {X_train.shape[0]} amostras")
    print(f"   Validação: {X_val.shape[0]} amostras")
    print(f"   Teste: {X_test.shape[0]} amostras")
    print(f"   Classes: {data['n_classes']}")
    print(f"   Features: {data['n_features']}")
    
    # ========================================================================
    # ETAPA 2: TREINAR E AVALIAR MODELOS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 2: TREINAR E AVALIAR MODELOS")
    print("="*80)
    
    results = []
    
    # ------------------------------------------------------------------------
    # Modelo 1: SVM Linear
    # ------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODELO 1: SVM LINEAR")
    print("-"*80)
    
    svm = SVMClassifier(C=1.0, max_iter=1000)
    svm.fit(X_train, y_train)
    
    svm_val = svm.evaluate(X_val, y_val, 'validação')
    svm_test = svm.evaluate(X_test, y_test, 'teste')
    
    results.append({
        'Modelo': 'SVM Linear',
        'Val Acc': svm_val['accuracy'],
        'Val F1': svm_val['f1_macro'],
        'Test Acc': svm_test['accuracy'],
        'Test F1': svm_test['f1_macro'],
        'Treino (s)': svm_test['training_time']
    })
    

    # ------------------------------------------------------------------------
    # Modelo 2: Regressão Logística
    # ------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODELO 2: REGRESSÃO LOGISTICA")
    print("-"*80)
    
    logistic = LogisticClassifier(C=1.0, max_iter=1000)
    logistic.fit(X_train, y_train)
    
    log_val = logistic.evaluate(X_val, y_val, 'validação')
    log_test = logistic.evaluate(X_test, y_test, 'teste')
    
    results.append({
        'Modelo': 'Regressão Logística',
        'Val Acc': log_val['accuracy'],
        'Val F1': log_val['f1_macro'],
        'Test Acc': log_test['accuracy'],
        'Test F1': log_test['f1_macro'],
        'Treino (s)': log_test['training_time']
    })
    
    # ------------------------------------------------------------------------
    # Modelo 3: LightGBM (se disponível)
    # ------------------------------------------------------------------------
    if LIGHTGBM_AVAILABLE:
        print("\n" + "-"*80)
        print("MODELO 3: LIGHTGBM")
        print("-"*80)
        
        lgbm = LightGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=100)
        lgbm.fit(X_train, y_train)
        
        lgbm_val = lgbm.evaluate(X_val, y_val, 'validação')
        lgbm_test = lgbm.evaluate(X_test, y_test, 'teste')
        
        results.append({
            'Modelo': 'LightGBM',
            'Val Acc': lgbm_val['accuracy'],
            'Val F1': lgbm_val['f1_macro'],
            'Test Acc': lgbm_test['accuracy'],
            'Test F1': lgbm_test['f1_macro'],
            'Treino (s)': lgbm_test['training_time']
        })
    
    # ------------------------------------------------------------------------
    # Modelo 4: Similaridade
    # ------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODELO 4: SIMILARIDADE (LAZY LEARNING)")
    print("-"*80)
    
    similarity = SimilarityClassifier()
    similarity.fit(X_train, y_train)
    
    sim_val = similarity.evaluate(X_val, y_val, 'validação')
    sim_test = similarity.evaluate(X_test, y_test, 'teste')
    
    results.append({
        'Modelo': 'Similaridade',
        'Val Acc': sim_val['accuracy'],
        'Val F1': sim_val['f1_macro'],
        'Test Acc': sim_test['accuracy'],
        'Test F1': sim_test['f1_macro'],
        'Treino (s)': sim_test['training_time']
    })
    
    # ------------------------------------------------------------------------
    # Modelo 5: K-Means
    # ------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("MODELO 5: K-MEANS CLUSTERING")
    print("-"*80)
    
    kmeans = KMeansClassifier(n_clusters=None, max_iter=300)
    kmeans.fit(X_train, y_train)
    
    km_val = kmeans.evaluate(X_val, y_val, 'validação')
    km_test = kmeans.evaluate(X_test, y_test, 'teste')
    
    results.append({
        'Modelo': 'K-Means',
        'Val Acc': km_val['accuracy'],
        'Val F1': km_val['f1_macro'],
        'Test Acc': km_test['accuracy'],
        'Test F1': km_test['f1_macro'],
        'Treino (s)': km_test['training_time']
    })
    
    # ------------------------------------------------------------------------
    # Modelo 6: MLP + S-BERT (se disponível)
    # ------------------------------------------------------------------------
    if SBERT_AVAILABLE:
        print("\n" + "-"*80)
        print("MODELO 6: MLP + SENTENCE-BERT")
        print("-"*80)
        print("⚠️  Este modelo pode levar alguns minutos...")
        
        mlp_sbert = MLPSBERTClassifier(
            model_name='neuralmind/bert-base-portuguese-cased',  # BERTimbau
            hidden_layer_sizes=(512, 256),
            learning_rate_init=0.001,
            max_iter=200
        )
        
        # Treinar com TEXTOS (não vetores TF-IDF)
        mlp_sbert.fit(data['texts_train'], y_train)
        
        # Avaliar
        mlp_val = mlp_sbert.evaluate(data['texts_val'], y_val, 'validação')
        mlp_test = mlp_sbert.evaluate(data['texts_test'], y_test, 'teste')
        
        results.append({
            'Modelo': 'MLP + S-BERT',
            'Val Acc': mlp_val['accuracy'],
            'Val F1': mlp_val['f1_macro'],
            'Test Acc': mlp_test['accuracy'],
            'Test F1': mlp_test['f1_macro'],
            'Treino (s)': mlp_test['training_time']
        })
    
    # ========================================================================
    # ETAPA 3: COMPARAR RESULTADOS
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 3: COMPARAÇÃO DE RESULTADOS")
    print("="*80)
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Test F1', ascending=False)
    
    print("\n" + df_results.to_string(index=False))
    
    # Melhor modelo
    best = df_results.iloc[0]
    print(f"\n🏆 MELHOR MODELO: {best['Modelo']}")
    print(f"   F1-Macro (teste): {best['Test F1']:.4f}")
    print(f"   Accuracy (teste): {best['Test Acc']:.4f}")
    
    # ========================================================================
    # ETAPA 4: TESTAR PREDIÇÃO
    # ========================================================================
    print("\n" + "="*80)
    print("ETAPA 4: EXEMPLO DE PREDIÇÃO")
    print("="*80)
    
    textos_exemplo = [
        "Aquisição de materiais de escritório",
        "Contratação de serviços de consultoria",
        "Pagamento de despesas com pessoal"
    ]
    
    print(f"\nUsando modelo: {best['Modelo']}")
    
    # Selecionar melhor modelo
    if best['Modelo'] == 'SVM Linear':
        best_model = svm
    elif best['Modelo'] == 'Regressão Logística':
        best_model = logistic
    elif best['Modelo'] == 'LightGBM':
        best_model = lgbm
    elif best['Modelo'] == 'Similaridade':
        best_model = similarity
    elif best['Modelo'] == 'K-Means':
        best_model = kmeans
    elif best['Modelo'] == 'MLP + S-BERT':
        best_model = mlp_sbert
    else:
        best_model = svm  # Fallback
    
    # Verificar se é modelo neural (usa textos) ou tradicional (usa TF-IDF)
    is_neural = best['Modelo'] == 'MLP + S-BERT'
    
    for texto in textos_exemplo:
        if is_neural:
            # Modelo neural: prediz diretamente do texto
            y_pred = best_model.predict([texto])[0]
        else:
            # Modelo tradicional: precisa vetorizar com TF-IDF
            X_novo = processor.predict_text(texto)
            y_pred = best_model.predict(X_novo)[0]
        
        # Converter para nome da natureza
        natureza = processor.get_class_name(y_pred)
        
        print(f"\n  Texto: {texto}")
        print(f"  Natureza: {natureza}")
    
    print("\n" + "="*80)
    print("✅ CONCLUÍDO!")
    print("="*80)


if __name__ == '__main__':
    main()