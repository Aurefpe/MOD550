# ========================================================================
# --- PLOTTING FUNCTION ---
# ========================================================================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

def plot_svm_results(X_data, y_data, title, plot_type='raw', clf=None, X_test=None, y_test=None, kernel_type=None):
    """

    :param X_data: The feature data (either unscaled or scaled).
    :param y_data: The target labels.
    :param title: Main title for the plot/figure.
    :param plot_type: 'raw' for unscaled data plot, 'result' for SVM decision boundary.
    :param clf: Fitted SVM classifier (required if plot_type is 'result').
    :param X_test: Test features for accuracy calculation (required if plot_type is 'result').
    :param y_test: Test labels for accuracy calculation (required if plot_type is 'result').
    :param kernel_type: The kernel used for the SVM (required if plot_type is 'result').
    """
    if plot_type == 'raw':
        plt.figure(figsize=(5, 4))
        plt.suptitle(f"{title} - Raw Data", fontsize=14)
        ax = plt.subplot(1, 1, 1)
        ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, s=20, edgecolors='k', cmap=plt.cm.coolwarm)
        ax.set_title("Original (Unscaled) Data")
        ax.set_xticks(()); ax.set_yticks(())
        plt.tight_layout()
        time.sleep(5)
        plt.show()

    elif plot_type == 'result':
        # --- SVM Result Plot (Decision Boundary) ---
        if clf is None or X_test is None or y_test is None or kernel_type is None:
            raise ValueError("For plot_type='result', clf, X_test, y_test, and kernel_type must be provided.")

        accuracy = accuracy_score(y_test, clf.predict(X_test))

        # Create a new figure for the result plot
        plt.figure(figsize=(5, 4))
        plt.suptitle(f"{title} - Kernel: {kernel_type}", fontsize=14)
        ax = plt.subplot(1, 1, 1)

        # Plot the decision boundary
        x_min, x_max = X_data[:, 0].min() - 0.5, X_data[:, 0].max() + 0.5
        y_min, y_max = X_data[:, 1].min() - 0.5, X_data[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        
        # Predict on the grid points
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        
        # Plot the decision region
        ax.contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.coolwarm)
        
        # Plot the data points (scaled data)
        ax.scatter(X_data[:, 0], X_data[:, 1], c=y_data, s=20, edgecolors='k', cmap=plt.cm.coolwarm)
        
        ax.set_title(f"Kernel: {kernel_type}\nAccuracy: {accuracy:.2f}")
        ax.set_xticks(()); ax.set_yticks(())
        plt.tight_layout()
        time.sleep(5)
        plt.show()

    else:
        raise ValueError("Invalid plot_type. Must be 'raw' or 'result'.")