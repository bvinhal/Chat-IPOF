"""
Script de Comparação de Modelos para Classificação de Natureza de Despesa

Treina e avalia todos os modelos descritos em Materiais e Métodos,
comparando suas performances usando as métricas especificadas.

Conforme metodologia, a métrica principal é o Macro F1-Score.

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
from typing import Dict, List

# Adicionar diretório raiz ao path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importar modelos
from models.svm_classifier import SVMNaturezaClassifier
from models.lightgbm_classifier import LightGBMNaturezaClassifier
from models.logistic_classifier import LogisticRegressionNaturezaClassifier
from models.similarity_classifier import SimilarityNaturezaClassifier
from models.kmeans_classifier import KMeansNaturezaClassifier

try:
    from models.mlp_sbert_classifier import MLPSBERTNaturezaClassifier
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("MLP + S-BERT não disponível (sentence-transformers não instalado)")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Classe para comparar múltiplos modelos de classificação.
    """
    
    def __init__(self, project_root: str = None):
        """
        Inicializa o comparador.
        
        Args:
            project_root: Diretório raiz do projeto
        """
        self.project_root = project_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.models = {}
        self.results = {}
        
    def add_model(self, name: str, model_class, params: Dict = None):
        """
        Adiciona um modelo para comparação.
        
        Args:
            name: Nome descritivo do modelo
            model_class: Classe do modelo
            params: Parâmetros para o modelo
        """
        self.models[name] = {
            'class': model_class,
            'params': params or {}
        }
        logger.info(f"Modelo adicionado: {name}")
    
    def train_and_evaluate_all(self, min_samples_per_class: int = 3,
                               random_state: int = 42):
        """
        Treina e avalia todos os modelos.
        
        Args:
            min_samples_per_class: Mínimo de amostras por classe
            random_state: Seed para reprodutibilidade
        """
        logger.info("\n" + "="*80)
        logger.info("INICIANDO COMPARAÇÃO DE MODELOS")
        logger.info("="*80)
        logger.info(f"Total de modelos: {len(self.models)}")
        
        for i, (name, model_info) in enumerate(self.models.items(), 1):
            logger.info("\n" + "="*80)
            logger.info(f"MODELO {i}/{len(self.models)}: {name}")
            logger.info("="*80)
            
            try:
                # Criar instância do modelo
                model = model_info['class'](project_root=self.project_root)
                
                # Pré-processar dados (compartilhado entre modelos)
                if i == 1 or name == "MLP + S-BERT":
                    # Primeiro modelo ou S-BERT (que precisa de embeddings)
                    logger.info("Pré-processando dados...")
                    data = model.preprocess_data(
                        min_samples_per_class=min_samples_per_class
                    )
                else:
                    # Reutilizar dados processados
                    logger.info("Reutilizando dados pré-processados...")
                    if hasattr(self, 'shared_data'):
                        model.data = self.shared_data
                        model.pipeline.preprocessor = self.shared_preprocessor
                        model.pipeline.vectorizer = self.shared_vectorizer
                        model.pipeline.label_encoder = self.shared_label_encoder
                    else:
                        data = model.preprocess_data(
                            min_samples_per_class=min_samples_per_class
                        )
                
                # Salvar dados compartilhados
                if not hasattr(self, 'shared_data') and name != "MLP + S-BERT":
                    self.shared_data = model.data
                    self.shared_preprocessor = model.pipeline.preprocessor
                    self.shared_vectorizer = model.pipeline.vectorizer
                    self.shared_label_encoder = model.pipeline.label_encoder
                
                # Treinar
                logger.info("Treinando modelo...")
                start_train = time.time()
                model.train(**model_info['params'])
                train_time = time.time() - start_train
                
                # Avaliar
                logger.info("Avaliando modelo...")
                test_metrics = model.evaluate('test')
                
                # Salvar resultados
                self.results[name] = {
                    'model': model,
                    'metrics': test_metrics,
                    'train_time': train_time
                }
                
                logger.info(f"✅ {name} concluído com sucesso!")
                logger.info(f"   Macro F1-Score: {test_metrics['f1_macro']:.4f}")
                logger.info(f"   Accuracy: {test_metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao treinar {name}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.results[name] = {
                    'model': None,
                    'metrics': None,
                    'error': str(e)
                }
        
        logger.info("\n" + "="*80)
        logger.info("COMPARAÇÃO CONCLUÍDA!")
        logger.info("="*80)
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Gera tabela comparativa dos modelos.
        
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for name, result in self.results.items():
            if result.get('metrics') is None:
                comparison_data.append({
                    'Modelo': name,
                    'Accuracy': np.nan,
                    'F1-Macro': np.nan,
                    'F1-Weighted': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'Top-3 Acc': np.nan,
                    'Tempo Treino (s)': np.nan,
                    'Tempo Pred (s)': np.nan,
                    'Status': f"❌ Erro: {result.get('error', 'Unknown')}"
                })
            else:
                metrics = result['metrics']
                comparison_data.append({
                    'Modelo': name,
                    'Accuracy': metrics['accuracy'],
                    'F1-Macro': metrics['f1_macro'],
                    'F1-Weighted': metrics['f1_weighted'],
                    'Precision': metrics['precision_macro'],
                    'Recall': metrics['recall_macro'],
                    'Top-3 Acc': metrics.get('top_3_accuracy', np.nan),
                    'Tempo Treino (s)': metrics['training_time'],
                    'Tempo Pred (s)': metrics['prediction_time'],
                    'Status': '✅ OK'
                })
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar por F1-Macro (métrica principal)
        df = df.sort_values('F1-Macro', ascending=False)
        
        return df
    
    def print_comparison_report(self):
        """
        Imprime relatório completo da comparação.
        """
        df = self.generate_comparison_table()
        
        print("\n" + "="*80)
        print("TABELA COMPARATIVA DE MODELOS")
        print("Métrica Principal: Macro F1-Score")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        # Melhor modelo
        if not df['F1-Macro'].isna().all():
            best_model = df.iloc[0]
            print(f"\n🏆 MELHOR MODELO: {best_model['Modelo']}")
            print(f"   Macro F1-Score: {best_model['F1-Macro']:.4f}")
            print(f"   Accuracy: {best_model['Accuracy']:.4f}")
            if not pd.isna(best_model['Top-3 Acc']):
                print(f"   Top-3 Accuracy: {best_model['Top-3 Acc']:.4f}")
        
        # Análise por categoria
        print("\n" + "="*80)
        print("ANÁLISE POR PARADIGMA")
        print("="*80)
        
        paradigms = {
            'Supervisionados Clássicos': ['SVM Linear', 'LightGBM', 'Regressão Logística'],
            'Não Supervisionados': ['Similaridade', 'K-Means'],
            'Neural': ['MLP + S-BERT']
        }
        
        for paradigm, models in paradigms.items():
            print(f"\n{paradigm}:")
            for model in models:
                if model in self.results and self.results[model].get('metrics'):
                    metrics = self.results[model]['metrics']
                    print(f"  {model:25s} - F1: {metrics['f1_macro']:.4f}, Acc: {metrics['accuracy']:.4f}")
    
    def save_results(self, output_dir: str = None):
        """
        Salva resultados da comparação.
        
        Args:
            output_dir: Diretório para salvar resultados
        """
        if output_dir is None:
            output_dir = os.path.join(self.project_root, 'data', 'comparison_results')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar tabela comparativa
        df = self.generate_comparison_table()
        csv_path = os.path.join(output_dir, 'model_comparison.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Tabela comparativa salva em: {csv_path}")
        
        # Salvar métricas detalhadas
        import json
        for name, result in self.results.items():
            if result.get('metrics'):
                metrics_path = os.path.join(output_dir, f'{name.replace(" ", "_").lower()}_metrics.json')
                
                # Converter numpy types para Python types
                metrics_serializable = {}
                for key, value in result['metrics'].items():
                    if isinstance(value, (np.integer, np.floating)):
                        metrics_serializable[key] = float(value)
                    else:
                        metrics_serializable[key] = value
                
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_serializable, f, indent=2)


def main():
    """
    Função principal para executar comparação de modelos.
    """
    print("="*80)
    print("COMPARAÇÃO DE MODELOS - CLASSIFICAÇÃO DE NATUREZA DE DESPESA")
    print("Conforme Materiais e Métodos")
    print("="*80)
    
    # Criar comparador
    comparator = ModelComparator()
    
    # Adicionar modelos conforme metodologia
    print("\nAdicionando modelos...")
    
    # 1. Modelos Supervisionados
    comparator.add_model(
        "SVM Linear",
        SVMNaturezaClassifier,
        {'C': 1.0, 'max_iter': 1000}
    )
    
    try:
        comparator.add_model(
            "LightGBM",
            LightGBMNaturezaClassifier,
            {'n_estimators': 100, 'num_leaves': 31, 'learning_rate': 0.1}
        )
    except:
        logger.warning("LightGBM não disponível, pulando...")
    
    comparator.add_model(
        "Regressão Logística",
        LogisticRegressionNaturezaClassifier,
        {'C': 1.0, 'max_iter': 1000}
    )
    
    # 2. Modelos Não Supervisionados
    comparator.add_model(
        "Similaridade",
        SimilarityNaturezaClassifier,
        {}
    )
    
    comparator.add_model(
        "K-Means",
        KMeansNaturezaClassifier,
        {'n_init': 10}
    )
    
    # 3. Modelo Neural
    if SBERT_AVAILABLE:
        comparator.add_model(
            "MLP + S-BERT",
            MLPSBERTNaturezaClassifier,
            {'hidden_layer_sizes': (512, 256), 'max_iter': 200}
        )
    else:
        logger.warning("MLP + S-BERT não disponível (sentence-transformers não instalado)")
    
    # Treinar e avaliar todos
    comparator.train_and_evaluate_all(min_samples_per_class=3)
    
    # Gerar relatórios
    comparator.print_comparison_report()
    
    # Salvar resultados
    comparator.save_results()
    
    print("\n" + "="*80)
    print("✅ COMPARAÇÃO CONCLUÍDA!")
    print("="*80)
    print("\nResultados salvos em: data/comparison_results/")
    print("Consulte model_comparison.csv para tabela completa.")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    main()