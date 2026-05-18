import numpy as np
from sklearn.metrics import (
    accuracy_score, hamming_loss, jaccard_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_score, recall_score,
    roc_curve, precision_recall_curve, brier_score_loss, average_precision_score
)
from sklearn.metrics import auc as sk_auc
from sklearn.calibration import calibration_curve
import os
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate_multilabel(prob_all_labels, actual, n_labels, hours_after_trach, decimals=4, model_name='', df_name='', eval_type=''):
    """
    Evaluate multilabel classifier.
    
    Parameters:
    -----------
    prob_all_labels : array of shape (n_samples, n_labels)
        Predicted probabilities for each label (probability of positive class)
    actual : array of shape (n_samples, n_labels)
        Binary ground truth for each label
    n_labels : int
        Number of labels
    hours_after_trach : int
        Hours after tracheostomy (for saving plots)
    decimals : int
        Decimal places for rounding
    model_name : str
        Model name (for saving plots)
    df_name : str
        Dataset name (for saving plots)
    eval_type : str
        Evaluation type ('timestep-level' or 'patient-level')
    
    Returns:
    --------
    dict : Dictionary with overall and per-label metrics
    """
    
    # Get binary predictions (threshold at 0.5)
    y_pred = (prob_all_labels > 0.5).astype(int)
    
    # === Overall Multilabel Metrics ===
    
    # 1. Hamming Loss (lower is better - fraction of wrong labels)
    hamming = np.round(hamming_loss(actual, y_pred), decimals)
    
    # 2. Jaccard Score (intersection over union per sample, averaged)
    jaccard = np.round(jaccard_score(actual, y_pred, average='macro'), decimals)
    
    # 3. F1-Score (macro - treats all labels equally)
    f1_macro = np.round(f1_score(actual, y_pred, average='macro', zero_division=0), decimals)

    # 4. AUC-ROC (macro - average of per-label AUCs)
    auc_macro = np.round(roc_auc_score(actual, prob_all_labels, average='macro'), decimals)

    # 5. Average Precision (AUPRC) (macro - average of per-label APs)
    ap_macro = np.round(average_precision_score(actual, prob_all_labels, average='macro'), decimals)

    # 6. Brier Score (macro - average of per-label Brier scores)
    brier_macro = np.round(np.mean([brier_score_loss(actual[:, i], prob_all_labels[:, i]) for i in range(n_labels)]), decimals)
    
    # F1-Score (weighted - weighted by support per label)
    # f1_weighted = np.round(f1_score(actual, y_pred, average='weighted', zero_division=0), decimals)
    
    # Subset Accuracy (strictest - all labels must match)
    # subset_accuracy = np.round(accuracy_score(actual, y_pred), decimals)
    
    # === Per-Label Metrics ===
    
    auc_per_label = []
    ap_per_label = []
    f1_per_label = []
    precision_per_label = []
    recall_per_label = []  # Sensitivity
    specificity_per_label = []
    npv_per_label = []  # Negative Predictive Value
    brier_per_label = []
    
    for i in range(n_labels):
        # AUC-ROC for label i (one-vs-rest)
        try:
            auc_score = roc_auc_score(actual[:, i], prob_all_labels[:, i])
            auc_per_label.append(np.round(auc_score, decimals))
        except:
            auc_per_label.append(np.nan)
        
        # Brier score for label i
        try:
            brier = brier_score_loss(actual[:, i], prob_all_labels[:, i])
            brier_per_label.append(np.round(brier, decimals))
        except:
            brier_per_label.append(np.nan)

        # Average Precision (AUPRC) for label i
        try:
            ap = average_precision_score(actual[:, i], prob_all_labels[:, i])
            ap_per_label.append(np.round(ap, decimals))
        except:
            ap_per_label.append(np.nan)

        # F1-Score for label i
        f1 = f1_score(actual[:, i], y_pred[:, i], zero_division=0)
        f1_per_label.append(np.round(f1, decimals))
        
        # Precision for label i
        prec = precision_score(actual[:, i], y_pred[:, i], zero_division=0)
        precision_per_label.append(np.round(prec, decimals))
        
        # Recall (Sensitivity) for label i
        rec = recall_score(actual[:, i], y_pred[:, i], zero_division=0)
        recall_per_label.append(np.round(rec, decimals))
        
        # Specificity for label i (TN / (TN + FP))
        cm = confusion_matrix(actual[:, i], y_pred[:, i], labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_label.append(np.round(spec, decimals))
        
        # Negative Predictive Value (NPV) for label i (TN / (TN + FN))
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        npv_per_label.append(np.round(npv, decimals))
    

    # === Plotting ===
    if model_name and df_name:
        fig, axes = plt.subplots(n_labels, 3, figsize=(18, 5*n_labels))
        
        # If only 1 label, axes is 1D; reshape to 2D for consistency
        if n_labels == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_labels):
            # --- Plot 1: ROC Curve ---
            fpr, tpr, _ = roc_curve(actual[:, i], prob_all_labels[:, i])
            roc_auc = sk_auc(fpr, tpr)
            axes[i, 0].plot(fpr, tpr, lw=2.5, color='steelblue', label=f'AUC = {roc_auc:.3f}')
            axes[i, 0].plot([0, 1], [0, 1], 'k--', lw=1.5)
            axes[i, 0].fill_between(fpr, tpr, alpha=0.2, color='steelblue')
            axes[i, 0].set_xlabel('False Positive Rate', fontsize=20)
            axes[i, 0].set_ylabel('True Positive Rate', fontsize=20)
            axes[i, 0].set_title(f'Label {i}: ROC Curve', fontsize=20, fontweight='bold')
            axes[i, 0].legend(fontsize=20, loc='lower right')
            axes[i, 0].grid(True, alpha=0.3)
            axes[i, 0].set_xlim([0, 1])
            axes[i, 0].set_ylim([0, 1])
            
            # --- Plot 2: Precision-Recall Curve ---
            precision, recall, _ = precision_recall_curve(actual[:, i], prob_all_labels[:, i])
            ap = average_precision_score(actual[:, i], prob_all_labels[:, i])
            axes[i, 1].plot(recall, precision, lw=2.5, color='coral', label=f'AP = {ap:.3f}')
            axes[i, 1].fill_between(recall, precision, alpha=0.2, color='coral')
            axes[i, 1].set_xlabel('Recall (Sensitivity)', fontsize=20)
            axes[i, 1].set_ylabel('Precision', fontsize=20)
            axes[i, 1].set_title(f'Label {i}: Precision-Recall Curve', fontsize=20, fontweight='bold')
            axes[i, 1].legend(fontsize=20, loc='best')
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_xlim([0, 1])
            axes[i, 1].set_ylim([0, 1])
            
            # --- Plot 3: Calibration Curve ---
            prob_true, prob_pred = calibration_curve(actual[:, i], prob_all_labels[:, i], n_bins=10, strategy='uniform')
            axes[i, 2].plot(prob_pred, prob_true, 'o-', lw=2.5, markersize=8, color='mediumpurple', label='Model')
            axes[i, 2].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Perfectly Calibrated')
            axes[i, 2].fill_between(prob_pred, prob_true, alpha=0.2, color='mediumpurple')
            axes[i, 2].set_xlabel('Mean Predicted Probability', fontsize=20)
            axes[i, 2].set_ylabel('Fraction of Positives', fontsize=20)
            axes[i, 2].set_title(f'Label {i}: Calibration Curve', fontsize=20, fontweight='bold')
            axes[i, 2].legend(fontsize=20, loc='upper left')
            axes[i, 2].grid(True, alpha=0.3)
            axes[i, 2].set_xlim([0, 1])
            axes[i, 2].set_ylim([0, 1])

            axes[i, 0].tick_params(axis='both', labelsize=20)
            axes[i, 1].tick_params(axis='both', labelsize=20)
            axes[i, 2].tick_params(axis='both', labelsize=20)
        
        plt.tight_layout()
        plot_path = f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_{eval_type}_multilabel_curves.png'
        os.makedirs(f'plots_{hours_after_trach}hrs', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")

    return {
        'hamming_loss': hamming,
        'jaccard_macro': jaccard,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro,
        'ap_macro': ap_macro,
        'brier_macro': brier_macro,
        'f1_per_label': f1_per_label,
        'auc_per_label': auc_per_label,
        'ap_per_label': ap_per_label,
        'brier_per_label': brier_per_label,
        'precision_per_label': precision_per_label,
        'recall_per_label': recall_per_label,
        'specificity_per_label': specificity_per_label,
        'npv_per_label': npv_per_label,
        'y_pred': y_pred
    }
