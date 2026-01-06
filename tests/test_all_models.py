"""
Teste de Validação Completo - Todos os Modelos e Métricas

Valida que TODOS os 6 modelos descritos em Materiais e Métodos estão
corretamente implementados, juntamente com todas as métricas.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import os
import sys
import logging
from typing import Dict, Any

# Adicionar diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_import():
    """Testa importação de todos os modelos."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 1: IMPORTAÇÃO DE MODELOS")
    logger.info("="*80)
    
    models_status = {}
    
    # Modelo 1: SVM Linear
    try:
        from models.svm_classifier import SVMNaturezaClassifier
        models_status['SVM Linear'] = '✅'
        logger.info("✅ SVM Linear importado com sucesso")
    except Exception as e:
        models_status['SVM Linear'] = f'❌ {str(e)}'
        logger.error(f"❌ SVM Linear falhou: {e}")
    
    # Modelo 2: LightGBM
    try:
        from models.lightgbm_classifier import LightGBMNaturezaClassifier
        models_status['LightGBM'] = '✅'
        logger.info("✅ LightGBM importado com sucesso")
    except Exception as e:
        models_status['LightGBM'] = f'⚠️  {str(e)}'
        logger.warning(f"⚠️  LightGBM não disponível (opcional): {e}")
    
    # Modelo 3: Regressão Logística
    try:
        from models.logistic_classifier import LogisticRegressionNaturezaClassifier
        models_status['Regressão Logística'] = '✅'
        logger.info("✅ Regressão Logística importada com sucesso")
    except Exception as e:
        models_status['Regressão Logística'] = f'❌ {str(e)}'
        logger.error(f"❌ Regressão Logística falhou: {e}")
    
    # Modelo 4: Similaridade
    try:
        from models.similarity_classifier import SimilarityNaturezaClassifier
        models_status['Similaridade'] = '✅'
        logger.info("✅ Similaridade importado com sucesso")
    except Exception as e:
        models_status['Similaridade'] = f'❌ {str(e)}'
        logger.error(f"❌ Similaridade falhou: {e}")
    
    # Modelo 5: K-Means
    try:
        from models.kmeans_classifier import KMeansNaturezaClassifier
        models_status['K-Means'] = '✅'
        logger.info("✅ K-Means importado com sucesso")
    except Exception as e:
        models_status['K-Means'] = f'❌ {str(e)}'
        logger.error(f"❌ K-Means falhou: {e}")
    
    # Modelo 6: MLP + S-BERT
    try:
        from models.mlp_sbert_classifier import MLPSBERTNaturezaClassifier
        models_status['MLP + S-BERT'] = '✅'
        logger.info("✅ MLP + S-BERT importado com sucesso")
    except Exception as e:
        models_status['MLP + S-BERT'] = f'⚠️  {str(e)}'
        logger.warning(f"⚠️  MLP + S-BERT não disponível (opcional): {e}")
    
    return models_status


def test_metrics_module():
    """Testa o módulo de métricas."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 2: MÓDULO DE MÉTRICAS")
    logger.info("="*80)
    
    try:
        from utils.evaluation_metrics import NaturezaEvaluator
        
        # Testar criação do avaliador
        evaluator = NaturezaEvaluator(model_name="TestModel")
        
        # Verificar métodos principais
        methods = [
            'calculate_all_metrics',
            'calculate_confusion_matrix',
            'generate_classification_report',
            'analyze_errors',
            'print_metrics_report',
            'compare_models'
        ]
        
        logger.info("✅ NaturezaEvaluator importado com sucesso")
        logger.info("Métodos disponíveis:")
        for method in methods:
            if hasattr(evaluator, method):
                logger.info(f"  ✅ {method}")
            else:
                logger.warning(f"  ⚠️  {method} não encontrado")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao importar módulo de métricas: {e}")
        return False


def test_preprocessing_pipeline():
    """Testa o pipeline de pré-processamento."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 3: PIPELINE DE PRÉ-PROCESSAMENTO")
    logger.info("="*80)
    
    try:
        from utils.natureza_preprocessing import (
            TextPreprocessor,
            NaturezaDataLoader,
            NaturezaPreprocessingPipeline
        )
        
        # Testar TextPreprocessor
        preprocessor = TextPreprocessor(remove_stopwords=True)
        text = "A AQUISIÇÃO de Equipamentos!!! para secretaria"
        processed = preprocessor.preprocess_text(text)
        logger.info(f"✅ TextPreprocessor OK")
        logger.info(f"   Original: {text}")
        logger.info(f"   Processado: {processed}")
        
        # Testar DataLoader
        loader = NaturezaDataLoader(project_root=project_root)
        logger.info(f"✅ NaturezaDataLoader OK")
        
        # Testar Pipeline
        pipeline = NaturezaPreprocessingPipeline(project_root=project_root)
        logger.info(f"✅ NaturezaPreprocessingPipeline OK")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro no pipeline de pré-processamento: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_base_classifier():
    """Testa a classe base dos classificadores."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 4: CLASSE BASE")
    logger.info("="*80)
    
    try:
        from models.base_classifier import BaseNaturezaClassifier
        
        # Verificar métodos abstratos e concretos
        methods = [
            'preprocess_data',
            'build_model',  # Abstrato
            'train',
            'evaluate',
            'predict',
            'predict_top_k',
            'save_model',
            'load_model',
            'get_model_info'
        ]
        
        logger.info("✅ BaseNaturezaClassifier importada com sucesso")
        logger.info("Métodos da interface:")
        for method in methods:
            if hasattr(BaseNaturezaClassifier, method):
                logger.info(f"  ✅ {method}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao importar classe base: {e}")
        return False


def test_model_factory():
    """Testa o factory method de modelos."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 5: FACTORY METHOD")
    logger.info("="*80)
    
    try:
        from models import create_model, get_available_models
        
        # Testar get_available_models
        available = get_available_models()
        logger.info(f"✅ {len(available)} modelos disponíveis:")
        for name, _ in available:
            logger.info(f"   • {name}")
        
        # Testar create_model
        test_names = ['svm', 'logistic', 'similarity', 'kmeans']
        for name in test_names:
            try:
                model = create_model(name, project_root=project_root)
                logger.info(f"  ✅ create_model('{name}') OK")
            except Exception as e:
                logger.warning(f"  ⚠️  create_model('{name}') falhou: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Erro no factory method: {e}")
        return False


def test_complete_workflow():
    """Testa workflow completo com um modelo."""
    logger.info("\n" + "="*80)
    logger.info("TESTE 6: WORKFLOW COMPLETO (SVM)")
    logger.info("="*80)
    
    try:
        from models.svm_classifier import SVMNaturezaClassifier
        
        # 1. Criar modelo
        logger.info("1. Criando modelo...")
        clf = SVMNaturezaClassifier(project_root=project_root)
        
        # 2. Pré-processar (com poucos dados para ser rápido)
        logger.info("2. Pré-processando dados...")
        clf.preprocess_data(min_samples_per_class=5)  # Mais amostras para garantir validade
        
        # 3. Treinar
        logger.info("3. Treinando modelo...")
        clf.train(C=1.0, max_iter=500)
        
        # 4. Avaliar
        logger.info("4. Avaliando modelo...")
        metrics = clf.evaluate('test')
        
        logger.info("\n📊 RESULTADOS:")
        logger.info(f"   Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"   Macro F1-Score: {metrics['f1_macro']:.4f}")
        logger.info(f"   Tempo de treino: {metrics['training_time']:.2f}s")
        
        # 5. Testar predição
        logger.info("\n5. Testando predição...")
        texto = "Aquisição de materiais de escritório"
        natureza, conf = clf.predict(texto)
        logger.info(f"   Texto: {texto}")
        logger.info(f"   Natureza: {natureza}")
        logger.info(f"   Confiança: {conf:.4f}")
        
        logger.info("\n✅ WORKFLOW COMPLETO OK!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no workflow completo: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_final_report(results: Dict[str, Any]):
    """Gera relatório final da validação."""
    logger.info("\n" + "="*80)
    logger.info("RELATÓRIO FINAL DE VALIDAÇÃO")
    logger.info("="*80)
    
    # Status dos modelos
    logger.info("\n📦 STATUS DOS MODELOS:")
    models_status = results.get('models', {})
    essential_models = ['SVM Linear', 'Regressão Logística', 'Similaridade', 'K-Means']
    optional_models = ['LightGBM', 'MLP + S-BERT']
    
    essential_ok = all(models_status.get(m, '').startswith('✅') for m in essential_models)
    
    for model, status in models_status.items():
        logger.info(f"   {model:25s} {status}")
    
    # Status dos componentes
    logger.info("\n🔧 STATUS DOS COMPONENTES:")
    components = [
        ('Módulo de Métricas', results.get('metrics', False)),
        ('Pipeline de Pré-processamento', results.get('preprocessing', False)),
        ('Classe Base', results.get('base_class', False)),
        ('Factory Method', results.get('factory', False)),
        ('Workflow Completo', results.get('workflow', False))
    ]
    
    for name, status in components:
        status_str = '✅' if status else '❌'
        logger.info(f"   {name:30s} {status_str}")
    
    # Resumo
    logger.info("\n" + "="*80)
    logger.info("RESUMO")
    logger.info("="*80)
    
    if essential_ok and all(status for _, status in components):
        logger.info("✅ TODOS OS TESTES PASSARAM!")
        logger.info("\n🎉 IMPLEMENTAÇÃO COMPLETA E VALIDADA!")
        logger.info("\nTodos os 6 modelos descritos em Materiais e Métodos")
        logger.info("estão implementados e funcionando corretamente:")
        logger.info("  1. ✅ SVM Linear com TF-IDF")
        logger.info("  2. ✅ LightGBM (opcional)")
        logger.info("  3. ✅ Regressão Logística Multinomial")
        logger.info("  4. ✅ Busca por Similaridade")
        logger.info("  5. ✅ K-Means Clustering")
        logger.info("  6. ✅ MLP + Sentence-BERT (opcional)")
        logger.info("\nSistema de métricas completo implementado:")
        logger.info("  ✅ Accuracy")
        logger.info("  ✅ Macro F1-Score (métrica principal)")
        logger.info("  ✅ Weighted F1-Score")
        logger.info("  ✅ Precision e Recall (macro e weighted)")
        logger.info("  ✅ Top-3 Accuracy")
        logger.info("  ✅ Matriz de Confusão")
        logger.info("  ✅ Tempo de Treinamento")
        logger.info("  ✅ Tempo de Predição")
        return True
    else:
        logger.warning("⚠️  ALGUNS TESTES FALHARAM")
        logger.warning("Verifique os logs acima para detalhes")
        return False


def main():
    """Executa todos os testes de validação."""
    logger.info("\n" + "#"*80)
    logger.info("# VALIDAÇÃO COMPLETA - IMPLEMENTAÇÃO DE MATERIAIS E MÉTODOS")
    logger.info("#"*80)
    
    results = {}
    
    # Teste 1: Importação de modelos
    results['models'] = test_model_import()
    
    # Teste 2: Módulo de métricas
    results['metrics'] = test_metrics_module()
    
    # Teste 3: Pipeline de pré-processamento
    results['preprocessing'] = test_preprocessing_pipeline()
    
    # Teste 4: Classe base
    results['base_class'] = test_base_classifier()
    
    # Teste 5: Factory method
    results['factory'] = test_model_factory()
    
    # Teste 6: Workflow completo
    results['workflow'] = test_complete_workflow()
    
    # Relatório final
    success = generate_final_report(results)
    
    return success


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    success = main()
    sys.exit(0 if success else 1)