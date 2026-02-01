from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import pandas as pd
import seaborn as sns
def plot_predictions_gt(y, predictions, base_predictions, speakers, feature_type):
    mae = mean_absolute_error(y, predictions)
    print("Mean Absolute Error:", mae)

    base_mae = mean_absolute_error(y, base_predictions)
    print("Baseline mean Absolute Error:", base_mae)

    # Get unique speaker names
    unique_speakers = sorted(set(speakers))
    num_colors = len(unique_speakers)
    color_map = plt.cm.get_cmap('tab20', num_colors)
    # Scatter plot with color coded points for each speaker
    for i, speaker in enumerate(unique_speakers):
        indices = [i for i, s in enumerate(speakers) if s == speaker]
        plt.scatter(np.array(y)[indices], np.array(predictions)[indices], label=speaker, color=color_map(i))

    # Plot predictions and ground truths
    # plt.scatter(y, predictions, c=speakers, cmap='tab10')
    plt.xlabel('Ground Truth')
    plt.ylabel('Predictions')

    plt.title(f'Features: {feature_type}, MAE: {mae:.2f}, Baseline MAE: {base_mae:.2f}')

    plt.legend(title='Speakers')
    plt.plot([np.array(y).min(), np.array(y).max()], [np.array(y).min(), np.array(y).max()], 'k--',
             lw=2)  # Plot the y=x line
    plt.savefig(f'../new_figures/features_{feature_type}_model_regressor.png')

    plt.show()

def plot_auc(y_test, probs, base_probs, feature_type, f1, base_f1):
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    b_fpr, b_tpr, b_thresholds = roc_curve(y_test, base_probs)
    base_roc_auc = auc(b_fpr, b_tpr)
    print("Roc AUC:", roc_auc)
    print("Baseline Roc AUC:", base_roc_auc)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'Features: {feature_type}, AUC = %0.2f, F1: %0.2f, '
                                                 f'Base AUC = %0.2f, Base F1: %0.2f' % (roc_auc, f1, base_roc_auc, base_f1))
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'../new_figures/features_{feature_type}_roc_model_classifier.png')
    plt.show()
    # Compute Precision-Recall pairs
    precision, recall, thr = precision_recall_curve(y_test, probs)
    print("thresholds:", thr)
    print("precision:", precision)
    print("recall:", recall)
    # Compute PR AUC
    pr_auc = auc(recall, precision)
    b_recall = np.arange(0, 1.1, 0.1)
    # calculate how many positive samples are in the test set
    n_pos = np.sum(y_test) / len(y_test)
    b_precision = np.repeat(n_pos, len(b_recall))
    print("base probs:", base_probs)
    print("baseline precision:", b_precision)
    print("baseline recall:", b_recall)
    base_pr_auc = auc(b_recall, b_precision)
    print("PR AUC:", pr_auc)
    print("Baseline PR AUC:", base_pr_auc)
    # Plot the Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, label=f'Features: {feature_type}, AUC = %0.2f, F1: %0.2f, '
                                      f'Base AUC = %0.2f, Base F1: %0.2f' % (pr_auc, f1, base_pr_auc, base_f1))
    plt.plot(b_recall, b_precision, color='gray', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.savefig(f'../new_figures/features_{feature_type}_pr_model_classifier.png')
    plt.show()
def vizualize_posteriors_classification(X, y, feature_names, feature_type):
    X_df = pd.DataFrame(X, columns=feature_names)
    y_classification_series = pd.Series(y, name='binary_label')
    data_classification = pd.concat([X_df, y_classification_series], axis=1)

    # Create a pair plot
    sns.pairplot(data_classification, hue='binary_label')
    plt.suptitle("Pair Plot for Binary Classification Task", y=1.02)
    plt.savefig(f'../new_figures/{feature_type}_classifier.png')
    plt.show()

def vizualize_posteriors_regression(X, y, feature_names, feature_type):
    # Convert lists to DataFrame and Series
    X_df = pd.DataFrame(X, columns=feature_names)
    y_regression_series = pd.Series(y, name='ground_truth')

    # Combine the data into a single DataFrame
    data_regression = pd.concat([X_df, y_regression_series], axis=1)

    # Create a pair plot
    sns.pairplot(data_regression, kind='reg', plot_kws={'line_kws': {'color': 'red'}})
    plt.suptitle("Pair Plot for Regression Task", y=1.02)
    plt.savefig(f'../new_figures/{feature_type}_regressor.png')
    plt.show()

def vizualize_embeddings_regression(X, y, feature_type):
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X, y=y)
    plt.title("Embeddings vs. Regression Targets")
    plt.xlabel("Embedding Features")
    plt.ylabel("Regression Targets")
    plt.grid(True)
    plt.savefig(f'../new_figures/{feature_type}_regressor.png')
    plt.show()

def vizualize_embeddings_classifier(X, y, feature_type):
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X, y=y, hue=y, palette="Set1",
                    legend=False)
    plt.title("Embeddings vs. Binary Labels")
    plt.xlabel("Embedding Features")
    plt.ylabel("Binary Labels")
    plt.grid(True)
    plt.savefig(f'../new_figures/{feature_type}_classifier.png')
    plt.show()