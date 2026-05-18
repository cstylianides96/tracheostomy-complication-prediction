import numpy as np
from sklearn.metrics import accuracy_score
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate(prob, actual, hours_after_trach, decimals=2, model_name='', df_name='', eval_type=''):
    """
    Evaluate classifier at two thresholds:
    1. Youden index (max TPR - FPR)
    2. Sensitivity >= 90% (choose threshold with max specificity at sen >= 0.90)
    
    Returns dict with metrics for both thresholds, plus AUPRC and Brier score.
    Also generates and saves ROC, PRC, and calibration plots.
    """
    
    # AUC & AUPRC
    auc = np.round(roc_auc_score(actual, prob), decimals)
    auprc = np.round(average_precision_score(actual, prob), decimals)
    brier = np.round(brier_score_loss(actual, prob), decimals)
    
    # ROC curve: compute Youden index
    fpr, tpr, thresholds = roc_curve(actual, prob)
    youden = tpr - fpr
    optimal_idx = int(np.nanargmax(youden))
    threshold_youden = thresholds[optimal_idx]
    
    # Sensitivity >= 90% threshold (max specificity subject to sen >= 0.90)
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(actual, prob)
    # recall = sensitivity
    idx_sen_90 = np.where(recall_vals >= 0.90)[0]
    if len(idx_sen_90) > 0:
        # Among thresholds with sen >= 0.90, pick one with highest specificity (or precision)
        # Use the last valid index (highest threshold that still achieves sen >= 0.90)
        best_idx_sen90 = idx_sen_90[-1]
        threshold_sen90 = pr_thresholds[best_idx_sen90] if best_idx_sen90 < len(pr_thresholds) else pr_thresholds[-1]
    else:
        # Fallback: use threshold that maximizes sensitivity (closest to 0.90)
        best_idx_sen90 = np.argmin(np.abs(recall_vals - 0.90))
        threshold_sen90 = pr_thresholds[best_idx_sen90] if best_idx_sen90 < len(pr_thresholds) else pr_thresholds[-1]
    
    # --- Metrics at Youden threshold ---
    pred_youden = (np.array(prob) >= threshold_youden)
    cm_youden = confusion_matrix(actual, pred_youden, labels=[0, 1])
    tn_y, fp_y, fn_y, tp_y = cm_youden.ravel()
    
    sen_youden = np.round(tp_y / (tp_y + fn_y) if (tp_y + fn_y) > 0 else 0, decimals)
    spec_youden = np.round(tn_y / (tn_y + fp_y) if (tn_y + fp_y) > 0 else 0, decimals)
    prec_youden = np.round(precision_score(actual, pred_youden, zero_division=0), decimals)
    npv_youden = np.round(tn_y / (tn_y + fn_y) if (tn_y + fn_y) > 0 else 0, decimals)
    acc_youden = np.round(accuracy_score(actual, pred_youden), decimals)
    
    # --- Metrics at Sen >= 90% threshold ---
    pred_sen90 = (np.array(prob) >= threshold_sen90)
    cm_sen90 = confusion_matrix(actual, pred_sen90, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_sen90.ravel()
    
    sen_sen90 = np.round(tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0, decimals)
    spec_sen90 = np.round(tn_s / (tn_s + fp_s) if (tn_s + fp_s) > 0 else 0, decimals)
    prec_sen90 = np.round(precision_score(actual, pred_sen90, zero_division=0), decimals)
    npv_sen90 = np.round(tn_s / (tn_s + fn_s) if (tn_s + fn_s) > 0 else 0, decimals)
    acc_sen90 = np.round(accuracy_score(actual, pred_sen90), decimals)
    
    # Set font sizes for better readability
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['axes.titlesize'] = 24
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 22

    # --- Plotting (only if model_name and df_name provided) ---
    if model_name and df_name:
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # ROC curve
        fpr_plot, tpr_plot, _ = roc_curve(actual, prob)
        axes[0, 0].plot(fpr_plot, tpr_plot, label=f'AUC = {auc}', lw=2)
        axes[0, 0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0, 0].scatter(fpr_plot[optimal_idx], tpr_plot[optimal_idx], color='red', s=100, label='Youden', zorder=5)
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall curve
        prec_plot, rec_plot, _ = precision_recall_curve(actual, prob)
        axes[0, 1].plot(rec_plot, prec_plot, label=f'AUPRC = {auprc}', lw=2)
        axes[0, 1].axvline(x=0.90, color='red', linestyle='--', label='Sen ≥ 90%')
        axes[0, 1].set_xlabel('Recall (Sensitivity)')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration curve # fix calibration curve
        prob_true, prob_pred = calibration_curve(actual, prob, n_bins=10, strategy='uniform')
        axes[1, 0].plot(prob_pred, prob_true, 'o-', label='Model', lw=2)
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title(f'Calibration Curve (Brier = {brier})')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        

        # Decision Curve Analysis
        thresholds_dca = np.linspace(0, 1, 100)
        net_benefit_model = []
        net_benefit_treat_all = []

        # Calculate net benefit at specific thresholds (0.1, 0.2, 0.3)
        specific_thresholds = [0.1, 0.2, 0.3]
        nb_specific = []
        
        for threshold in specific_thresholds:
            pred_specific = (np.array(prob) >= threshold).astype(int)
            tp = np.sum((pred_specific == 1) & (np.array(actual) == 1))
            fp = np.sum((pred_specific == 1) & (np.array(actual) == 0))
            n = len(actual)
            
            nb = (tp - fp * (threshold / (1 - threshold))) / n if threshold < 1 else 0
            nb_specific.append(np.round(nb, decimals))

        for threshold in thresholds_dca:
            pred_dca = (prob >= threshold).astype(int)
            tp = np.sum((pred_dca == 1) & (actual == 1))
            fp = np.sum((pred_dca == 1) & (actual == 0))
            n = len(actual)
            
            if threshold < 1:
                nb_model = (tp - fp * (threshold / (1 - threshold))) / n
                nb_treat_all = (np.sum(actual) - np.sum(actual == 0) * (threshold / (1 - threshold))) / n
            else:
                nb_model = 0
                nb_treat_all = 0
            
            net_benefit_model.append(nb_model)
            net_benefit_treat_all.append(nb_treat_all)
        
        # Find threshold where model net benefit exceeds treat-all
        for i, threshold in enumerate(thresholds_dca):
            if net_benefit_model[i] > net_benefit_treat_all[i]:
                print(f"Model exceeds treat-all at threshold: {threshold:.2f}")
                break

        axes[1, 1].plot(thresholds_dca, net_benefit_model, label='Model', lw=2)
        axes[1, 1].plot(thresholds_dca, net_benefit_treat_all, label='Treat All', lw=2, linestyle='--')
        axes[1, 1].axhline(y=0, color='k', linestyle='-', label='Treat None', lw=1)
        axes[1, 1].set_xlabel('Threshold Probability')
        axes[1, 1].set_ylabel('Net Benefit')
        axes[1, 1].set_title('Decision Curve Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].set_ylim([-3, 1])

        plt.tight_layout()
        plot_path = f'plots_{hours_after_trach}hrs/{model_name}_{df_name}_{eval_type}_curves.png'
        os.makedirs('plots_' + str(hours_after_trach) + 'hrs', exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_path}")
    
    return {
        'auc': auc,
        'auprc': auprc,
        'brier': brier,
        'nb_specific': nb_specific,
        'threshold_youden': np.round(threshold_youden, decimals),
        'sen_youden': sen_youden,
        'spec_youden': spec_youden,
        'prec_youden': prec_youden,
        'npv_youden': npv_youden,
        'acc_youden': acc_youden,
        'threshold_sen90': np.round(threshold_sen90, decimals),
        'sen_sen90': sen_sen90,
        'spec_sen90': spec_sen90,
        'prec_sen90': prec_sen90,
        'npv_sen90': npv_sen90,
        'acc_sen90': acc_sen90,
    }