import numpy as np
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns



def evaluate_multiclass(prob_all_classes, actual, n_classes, hours_after_trach, decimals=5, model_name='', df_name='', eval_type=''):
    """
    Evaluate multiclass classifier.
    prob_all_classes: shape (n_samples, n_classes) - probabilities for each class
    actual: true labels (integers 0, 1, 2, ...)
    """
    
    # Get predicted class
    y_pred = np.argmax(prob_all_classes, axis=1)
    
    # === Overall Metrics ===
    
    # 1. Multiclass AUC (one-vs-rest or one-vs-one)
    auc_ovr = np.round(roc_auc_score(actual, prob_all_classes, 
                                      multi_class='ovr',  # one-vs-rest
                                      average='macro'), decimals)  # or 'weighted'
    
    # 2. Accuracy
    accuracy = np.round(accuracy_score(actual, y_pred), decimals)
    
    # 3. Macro/Weighted averages
    f1_macro = np.round(f1_score(actual, y_pred, average='macro'), decimals)
    f1_weighted = np.round(f1_score(actual, y_pred, average='weighted'), decimals)
    precision_macro = np.round(precision_score(actual, y_pred, average='macro', zero_division=0), decimals)
    recall_macro = np.round(recall_score(actual, y_pred, average='macro', zero_division=0), decimals)
    
    # 4. Per-class metrics
    per_class_metrics = {}
    for i in range(n_classes):
        # Binary metrics for class i vs rest
        actual_binary = (actual == i).astype(int)
        prob_class_i = prob_all_classes[:, i]
        
        auc_i = roc_auc_score(actual_binary, prob_class_i)
        # Can also calculate precision, recall for each class
        mask_pred_i = (y_pred == i)
        
        tp = np.sum((actual == i) & mask_pred_i)
        fp = np.sum((actual != i) & mask_pred_i)
        fn = np.sum((actual == i) & ~mask_pred_i)
        tn = np.sum((actual != i) & ~mask_pred_i)
        
        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        per_class_metrics[f'class_{i}'] = {
            'auc': np.round(auc_i, decimals),
            'precision': np.round(precision_i, decimals),
            'recall': np.round(recall_i, decimals)
        }
    
    # 5. Confusion Matrix
    cm = confusion_matrix(actual, y_pred) ##remove
    
    # === Plotting ===
    if model_name and df_name:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Per-class ROC curves
        from sklearn.metrics import roc_curve, auc
        for i in range(n_classes):
            actual_binary = (actual == i).astype(int)
            fpr, tpr, _ = roc_curve(actual_binary, prob_all_classes[:, i])
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')
        
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves (One-vs-Rest)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_title('Confusion Matrix')  #### remove
        
        # Plot 3: Per-class Precision-Recall
        from sklearn.metrics import precision_recall_curve, average_precision_score
        for i in range(n_classes):
            actual_binary = (actual == i).astype(int)
            precision, recall, _ = precision_recall_curve(actual_binary, prob_all_classes[:, i])
            ap = average_precision_score(actual_binary, prob_all_classes[:, i])
            axes[2].plot(recall, precision, lw=2, label=f'Class {i} (AP = {ap:.3f})')
        
        axes[2].set_xlabel('Recall')
        axes[2].set_ylabel('Precision')
        axes[2].set_title('Precision-Recall Curves')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_{eval_type}_multiclass_curves.png'
        os.makedirs('plots_' + str(hours_after_trach) + 'hrs', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        'auc_ovr_macro': auc_ovr,
        'accuracy': accuracy,
        'f1_macro': f1_macro,#
        'f1_weighted': f1_weighted,#
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'per_class_metrics': per_class_metrics,#
        'confusion_matrix': cm.tolist()#
    }
#specificity macro, npv macro, brier score 
# per class calibration curves