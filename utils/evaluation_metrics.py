"""
Módulo de Métricas de Avaliação para Classificação de Naturezas de Despesa

Implementa todas as métricas descritas em Materiais e Métodos conforme
Faceli et al. (2011) e Monard e Baranauskas (2003).

Autor: Bruno Rudyard Mendes Vinhal
Data: Janeiro 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NaturezaEvaluator:
    """
    Avaliador completo para modelos de classificação de natureza de despesa.
    
    Implementa todas as métricas descritas na metodologia:
    - Accuracy
    - Macro F1-Score (métrica principal)
    - Weighted F1-Score
    - Precision e Recall (macro)
    - Top-3 Accuracy
    - Matriz de confusão
    - Tempo de treinamento e predição
    """
    
    def __init__(self, model_name: str = "modelo"):
        """
        Inicializa o avaliador.
        
        Args:
            model_name: Nome do modelo para relatórios
        """
        self.model_name = model_name
        self.metrics_history = {}
        
    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                              y_proba: Optional[np.ndarray] = None,
                              training_time: float = 0.0,
                              prediction_time: float = 0.0,
                              dataset_name: str = "test") -> Dict[str, Any]:
        """
        Calcula todas as métricas de avaliação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            y_proba: Probabilidades preditas (opcional, para Top-k accuracy)
            training_time: Tempo de treinamento em segundos
            prediction_time: Tempo de predição em segundos
            dataset_name: Nome do conjunto de dados
            
        Returns:
            Dicionário com todas as métricas
        """
        metrics = {}
        
        # Métricas básicas (conforme Faceli et al., 2011)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # Macro F1-Score - Métrica principal conforme metodologia
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Weighted F1-Score
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Micro F1-Score (para referência)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Precision e Recall (média macro) conforme metodologia
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Precision e Recall (média weighted)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Top-3 Accuracy (conforme metodologia)
        if y_proba is not None:
            metrics['top_3_accuracy'] = self._calculate_top_k_accuracy(y_true, y_proba, k=3)
            metrics['top_5_accuracy'] = self._calculate_top_k_accuracy(y_true, y_proba, k=5)
        
        # Tempos de execução
        metrics['training_time'] = training_time
        metrics['prediction_time'] = prediction_time
        metrics['samples'] = len(y_true)
        
        # Informações adicionais
        metrics['dataset'] = dataset_name
        metrics['n_classes'] = len(np.unique(y_true))
        
        # Salvar no histórico
        self.metrics_history[dataset_name] = metrics
        
        return metrics
    
    def _calculate_top_k_accuracy(self, y_true: np.ndarray, 
                                   y_proba: np.ndarray, k: int = 3) -> float:
        """
        Calcula Top-K Accuracy.
        
        Verifica se a classe correta está entre as k mais prováveis.
        Relevante para o uso prático esperado (metodologia).
        
        Args:
            y_true: Labels verdadeiros
            y_proba: Matriz de probabilidades (n_samples, n_classes)
            k: Número de top predições a considerar
            
        Returns:
            Top-K accuracy
        """
        # Obter os índices das k classes mais prováveis para cada amostra
        top_k_preds = np.argsort(y_proba, axis=1)[:, -k:]
        
        # Verificar se a classe verdadeira está entre as top-k
        correct = np.array([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        
        return np.mean(correct)
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   labels: Optional[List[str]] = None,
                                   normalize: bool = False) -> np.ndarray:
        """
        Calcula matriz de confusão para análise de erros sistemáticos.
        
        Conforme Monard e Baranauskas (2003) e Faceli et al. (2011).
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            labels: Nomes das classes (opcional)
            normalize: Se True, normaliza a matriz
            
        Returns:
            Matriz de confusão
        """
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        return cm
    
    def generate_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       target_names: Optional[List[str]] = None,
                                       output_dict: bool = False) -> Any:
        """
        Gera relatório completo de classificação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            target_names: Nomes das classes
            output_dict: Se True, retorna dicionário
            
        Returns:
            Relatório de classificação
        """
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            output_dict=output_dict,
            zero_division=0
        )
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                       X_raw: np.ndarray, label_encoder,
                       n_examples: int = 10) -> pd.DataFrame:
        """
        Analisa os erros de classificação.
        
        Args:
            y_true: Labels verdadeiros
            y_pred: Labels preditos
            X_raw: Textos originais
            label_encoder: Encoder de labels
            n_examples: Número de exemplos a retornar
            
        Returns:
            DataFrame com análise de erros
        """
        # Encontrar erros
        errors = y_pred != y_true
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            logger.info("Nenhum erro encontrado!")
            return pd.DataFrame()
        
        # Criar DataFrame com análise
        error_data = []
        for idx in error_indices[:n_examples]:
            error_data.append({
                'texto': X_raw[idx][:100] + '...' if len(X_raw[idx]) > 100 else X_raw[idx],
                'natureza_real': label_encoder.inverse_transform([y_true[idx]])[0],
                'natureza_predita': label_encoder.inverse_transform([y_pred[idx]])[0],
                'indice': idx
            })
        
        df_errors = pd.DataFrame(error_data)
        
        logger.info(f"\nTotal de erros: {errors.sum()} de {len(y_true)} ({errors.sum()/len(y_true)*100:.2f}%)")
        
        return df_errors
    
    def print_metrics_report(self, metrics: Dict[str, Any], detailed: bool = True):
        """
        Imprime relatório formatado das métricas.
        
        Args:
            metrics: Dicionário de métricas
            detailed: Se True, mostra métricas detalhadas
        """
        print("\n" + "="*80)
        print(f"MÉTRICAS DE AVALIAÇÃO - {self.model_name.upper()}")
        print(f"Dataset: {metrics.get('dataset', 'N/A')}")
        print("="*80)
        
        # Métricas principais
        print("\n📊 MÉTRICAS PRINCIPAIS:")
        print(f"  ✓ Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  ✓ Macro F1-Score:     {metrics['f1_macro']:.4f} [MÉTRICA PRINCIPAL]")
        print(f"  ✓ Weighted F1-Score:  {metrics['f1_weighted']:.4f}")
        
        if 'top_3_accuracy' in metrics:
            print(f"  ✓ Top-3 Accuracy:     {metrics['top_3_accuracy']:.4f} ({metrics['top_3_accuracy']*100:.2f}%)")
        
        if detailed:
            print("\n📈 MÉTRICAS DETALHADAS:")
            print(f"  • Precision (Macro):  {metrics['precision_macro']:.4f}")
            print(f"  • Recall (Macro):     {metrics['recall_macro']:.4f}")
            print(f"  • Precision (Wtd):    {metrics['precision_weighted']:.4f}")
            print(f"  • Recall (Wtd):       {metrics['recall_weighted']:.4f}")
            print(f"  • F1 (Micro):         {metrics['f1_micro']:.4f}")
        
        print("\n⏱️  PERFORMANCE:")
        print(f"  • Training Time:      {metrics['training_time']:.4f}s")
        print(f"  • Prediction Time:    {metrics['prediction_time']:.4f}s")
        print(f"  • Samples:            {metrics['samples']}")
        print(f"  • Classes:            {metrics['n_classes']}")
        
        print("="*80)
    
    def compare_models(self, models_metrics: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compara métricas de múltiplos modelos.
        
        Args:
            models_metrics: Dicionário {nome_modelo: métricas}
            
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for model_name, metrics in models_metrics.items():
            comparison_data.append({
                'Modelo': model_name,
                'Accuracy': metrics['accuracy'],
                'F1-Macro': metrics['f1_macro'],
                'F1-Weighted': metrics['f1_weighted'],
                'Top-3 Acc': metrics.get('top_3_accuracy', np.nan),
                'Tempo Treino': metrics['training_time'],
                'Tempo Pred': metrics['prediction_time']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('F1-Macro', ascending=False)
        
        return df
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                             class_names: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (12, 10),
                             save_path: Optional[str] = None):
        """
        Plota matriz de confusão.
        
        Args:
            cm: Matriz de confusão
            class_names: Nomes das classes
            figsize: Tamanho da figura
            save_path: Caminho para salvar figura
        """
        plt.figure(figsize=figsize)
        
        # Se há muitas classes, mostrar apenas heatmap sem labels
        if len(cm) > 50:
            sns.heatmap(cm, cmap='Blues', cbar=True, 
                       square=True, linewidths=0)
            plt.title(f'Matriz de Confusão - {self.model_name}\n({len(cm)} classes)')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names, yticklabels=class_names,
                       cbar=True, square=True)
            plt.title(f'Matriz de Confusão - {self.model_name}')
            plt.xlabel('Predito')
            plt.ylabel('Real')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusão salva em: {save_path}")
        
        plt.close()
    
    def save_metrics_to_file(self, metrics: Dict[str, Any], filepath: str):
        """
        Salva métricas em arquivo JSON.
        
        Args:
            metrics: Dicionário de métricas
            filepath: Caminho do arquivo
        """
        import json
        
        # Converter numpy types para Python types
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        logger.info(f"Métricas salvas em: {filepath}")


def calculate_metrics_summary(metrics_dict: Dict[str, Any]) -> str:
    """
    Gera sumário textual das métricas.
    
    Args:
        metrics_dict: Dicionário de métricas
        
    Returns:
        String formatada com sumário
    """
    summary = []
    summary.append("SUMÁRIO DAS MÉTRICAS")
    summary.append("=" * 50)
    summary.append(f"Accuracy:          {metrics_dict['accuracy']:.4f}")
    summary.append(f"Macro F1-Score:    {metrics_dict['f1_macro']:.4f} ⭐")
    summary.append(f"Weighted F1-Score: {metrics_dict['f1_weighted']:.4f}")
    
    if 'top_3_accuracy' in metrics_dict:
        summary.append(f"Top-3 Accuracy:    {metrics_dict['top_3_accuracy']:.4f}")
    
    summary.append(f"\nTempo de treino:   {metrics_dict['training_time']:.2f}s")
    summary.append(f"Tempo de predição: {metrics_dict['prediction_time']:.4f}s")
    summary.append("=" * 50)
    
    return "\n".join(summary)


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de métricas de avaliação carregado com sucesso!")
    print("Use NaturezaEvaluator para avaliar seus modelos.")