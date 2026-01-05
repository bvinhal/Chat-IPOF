"""
Script de Testes para o Módulo de Pré-processamento

Valida a integração do módulo de pré-processamento com o projeto Chat-IPOF.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import os
import sys
import logging

# Adicionar o diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_data_loader():
    """Testa o carregador de dados"""
    logger.info("\n" + "="*80)
    logger.info("TESTE 1: Data Loader")
    logger.info("="*80)
    
    from utils.natureza_preprocessing import NaturezaDataLoader
    
    try:
        loader = NaturezaDataLoader(project_root=project_root)
        df = loader.load_data(min_samples_per_class=3)
        
        assert df is not None, "DataFrame é None"
        assert len(df) > 0, "DataFrame vazio"
        assert 'despesa' in df.columns, "Coluna 'despesa' não encontrada"
        assert 'natureza' in df.columns, "Coluna 'natureza' não encontrada"
        
        stats = loader.get_statistics()
        assert stats['total_samples'] > 0, "Nenhuma amostra carregada"
        
        logger.info(f"✅ PASSOU - {stats['total_samples']} amostras, {stats['unique_naturezas']} naturezas")
        return True
        
    except Exception as e:
        logger.error(f"❌ FALHOU - {str(e)}")
        return False


def test_text_preprocessor():
    """Testa o pré-processador de texto"""
    logger.info("\n" + "="*80)
    logger.info("TESTE 2: Text Preprocessor")
    logger.info("="*80)
    
    from utils.natureza_preprocessing import TextPreprocessor
    
    try:
        preprocessor = TextPreprocessor(remove_stopwords=True)
        
        # Teste de normalização
        texto = "A AQUISIÇÃO de Equipamentos!!! @#$"
        normalizado = preprocessor.normalize_text(texto)
        assert normalizado == "a aquisição de equipamentos", f"Normalização falhou: {normalizado}"
        
        # Teste de pipeline completo
        texto = "A compra de materiais de escritório para a secretaria"
        processado = preprocessor.preprocess_text(texto)
        assert len(processado) > 0, "Texto processado vazio"
        assert "compra" in processado or "materiais" in processado, "Palavras importantes removidas"
        
        logger.info(f"✅ PASSOU - Texto processado corretamente")
        logger.info(f"   Original: {texto}")
        logger.info(f"   Processado: {processado}")
        return True
        
    except Exception as e:
        logger.error(f"❌ FALHOU - {str(e)}")
        return False


def test_pipeline():
    """Testa o pipeline completo"""
    logger.info("\n" + "="*80)
    logger.info("TESTE 3: Pipeline Completo")
    logger.info("="*80)
    
    from utils.natureza_preprocessing import NaturezaPreprocessingPipeline
    
    try:
        pipeline = NaturezaPreprocessingPipeline(project_root=project_root)
        
        data = pipeline.run(min_samples_per_class=3)
        
        # Verificar dados retornados
        assert 'X_train_vec' in data, "X_train_vec não encontrado"
        assert 'y_train_encoded' in data, "y_train_encoded não encontrado"
        
        # Verificar shapes
        assert data['X_train_vec'].shape[0] > 0, "Conjunto de treino vazio"
        assert data['X_train_vec'].shape[1] > 0, "Nenhuma feature"
        
        # Verificar consistência
        assert data['X_train_vec'].shape[0] == len(data['y_train_encoded']), \
            "Inconsistência entre X e y"
        
        logger.info(f"✅ PASSOU - Pipeline executado com sucesso")
        logger.info(f"   Treino: {data['X_train_vec'].shape[0]} amostras")
        logger.info(f"   Features: {data['X_train_vec'].shape[1]}")
        logger.info(f"   Classes: {len(pipeline.label_encoder.classes_)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ FALHOU - {str(e)}")
        return False


def test_svm_classifier():
    """Testa o classificador SVM"""
    logger.info("\n" + "="*80)
    logger.info("TESTE 4: SVM Classifier")
    logger.info("="*80)
    
    from models.svm_classifier import SVMNaturezaClassifier
    
    try:
        # Criar classificador
        classifier = SVMNaturezaClassifier(project_root=project_root)
        
        # Pré-processar dados
        classifier.preprocess_data(min_samples_per_class=3)
        
        # Treinar
        classifier.train(C=1.0, max_iter=500)  # Menos iterações para teste rápido
        
        # Avaliar
        metrics = classifier.evaluate('test')
        
        assert metrics['accuracy'] > 0, "Accuracy zero"
        assert metrics['f1_macro'] > 0, "F1 macro zero"
        
        # Testar predição
        texto = "Aquisição de materiais de escritório"
        natureza, conf = classifier.predict(texto)
        
        assert natureza is not None, "Predição retornou None"
        assert 0 <= conf <= 1, f"Confiança fora do intervalo: {conf}"
        
        logger.info(f"✅ PASSOU - Modelo treinado e testado")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Macro F1: {metrics['f1_macro']:.4f}")
        logger.info(f"   Teste de predição: '{texto}' → {natureza}")
        return True
        
    except Exception as e:
        logger.error(f"❌ FALHOU - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load():
    """Testa salvamento e carregamento de modelo"""
    logger.info("\n" + "="*80)
    logger.info("TESTE 5: Save/Load Model")
    logger.info("="*80)
    
    from models.svm_classifier import SVMNaturezaClassifier
    import shutil
    
    try:
        # Criar e treinar modelo
        classifier = SVMNaturezaClassifier(project_root=project_root)
        classifier.preprocess_data(min_samples_per_class=3)
        classifier.train(C=1.0, max_iter=500)
        
        # Fazer uma predição antes de salvar
        texto_teste = "Contratação de serviços de consultoria"
        natureza_antes, conf_antes = classifier.predict(texto_teste)
        
        # Salvar
        classifier.save_model()
        
        # Criar novo classificador e carregar
        classifier_loaded = SVMNaturezaClassifier(project_root=project_root)
        classifier_loaded.load_model()
        
        # Fazer a mesma predição depois de carregar
        natureza_depois, conf_depois = classifier_loaded.predict(texto_teste)
        
        # Verificar consistência
        assert natureza_antes == natureza_depois, \
            f"Predições diferentes: {natureza_antes} vs {natureza_depois}"
        assert abs(conf_antes - conf_depois) < 0.0001, \
            f"Confiançasdiferentes: {conf_antes} vs {conf_depois}"
        
        logger.info(f"✅ PASSOU - Modelo salvo e carregado corretamente")
        logger.info(f"   Predição consistente: {natureza_antes}")
        
        # Limpar modelo de teste
        model_dir = os.path.join(project_root, 'data', 'models', 'svm_linear')
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
            logger.info(f"   Modelo de teste removido")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FALHOU - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Executa todos os testes"""
    logger.info("\n" + "#"*80)
    logger.info("# EXECUÇÃO DE TESTES - MÓDULO DE PRÉ-PROCESSAMENTO")
    logger.info("#"*80)
    
    tests = [
        ("Data Loader", test_data_loader),
        ("Text Preprocessor", test_text_preprocessor),
        ("Pipeline Completo", test_pipeline),
        ("SVM Classifier", test_svm_classifier),
        ("Save/Load Model", test_save_load)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            logger.error(f"Erro inesperado em {name}: {str(e)}")
            results.append((name, False))
    
    # Resumo
    logger.info("\n" + "="*80)
    logger.info("RESUMO DOS TESTES")
    logger.info("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASSOU" if success else "❌ FALHOU"
        logger.info(f"{status}: {name}")
    
    logger.info("\n" + "="*80)
    logger.info(f"RESULTADO FINAL: {passed}/{total} testes passaram")
    logger.info("="*80)
    
    if passed == total:
        logger.info("\n🎉 TODOS OS TESTES PASSARAM!")
        logger.info("O módulo está pronto para uso no Chat-IPOF.")
    else:
        logger.warning(f"\n⚠️  {total - passed} teste(s) falharam.")
        logger.warning("Verifique os logs acima para detalhes.")
    
    return passed == total


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = run_all_tests()
    sys.exit(0 if success else 1)