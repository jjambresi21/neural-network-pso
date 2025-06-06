import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from neural_network import NeuralNetwork

class Evaluation:
    def __init__(self, nn: NeuralNetwork):
        self.nn = nn

    def calculate_metrics(self, y_true, y_pred):
        mse = self.nn.calculate_mse_loss(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }

    def print_metrics(self, metrics, dataset_name="Test"):
        print(f"\n=== METRIKE NA {dataset_name.upper()} SKUPU ===")
        print(f"MSE: {metrics['MSE']:.6f}")
        print(f"RMSE: {metrics['RMSE']:.6f}")
        print(f"MAE: {metrics['MAE']:.6f}")
        print(f"R² Score: {metrics['R2']:.6f}")

    def print_results(self, mse_train, best_score, improvement):
        print(f"\n=== REZULTATI ===")
        print(f"MSE (nasumične težine): {mse_train:.6f}")
        print(f"Najbolji MSE nakon optimizacije: {best_score:.6f}")
        print(f"Poboljšanje: {improvement:.2f}%")

    def detailed_performance_analysis(self, y_true, y_pred, dataset_name="Test"):
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        errors = y_true_flat - y_pred_flat
        abs_errors = np.abs(errors)

        print(f"\n=== DETALJNA ANALIZA - {dataset_name.upper()} SKUP ===")
        print(f"Broj primjera: {len(y_true_flat)}")
        print(f"Srednja apsolutna greška: {np.mean(abs_errors):.4f}")
        print(f"Medijan apsolutne greške: {np.median(abs_errors):.4f}")
        print(f"Standardna devijacija grešaka: {np.std(errors):.4f}")
        print(f"Maksimalna apsolutna greška: {np.max(abs_errors):.4f}")
        print(f"Minimalna apsolutna greška: {np.min(abs_errors):.4f}")
        print(f"95. percentil apsolutne greške: {np.percentile(abs_errors, 95):.4f}")

        q1, q2, q3 = np.percentile(y_true_flat, [25, 50, 75])
        mask_q1 = y_true_flat <= q1
        mask_q2 = (y_true_flat > q1) & (y_true_flat <= q2)
        mask_q3 = (y_true_flat > q2) & (y_true_flat <= q3)
        mask_q4 = y_true_flat > q3

        print(f"\nAnaliza po kvartilima stvarnih vrijednosti:")
        print(f"Q1 (≤{q1:.2f}): MAE = {np.mean(abs_errors[mask_q1]):.4f}")
        print(f"Q2 ({q1:.2f}-{q2:.2f}): MAE = {np.mean(abs_errors[mask_q2]):.4f}")
        print(f"Q3 ({q2:.2f}-{q3:.2f}): MAE = {np.mean(abs_errors[mask_q3]):.4f}")
        print(f"Q4 (>{q3:.2f}): MAE = {np.mean(abs_errors[mask_q4]):.4f}")

    def create_visualizations(self, y_train_original, y_pred_train_original, y_test_original, y_pred_test_original, errors, convergence_history):
        fig1, axs1 = plt.subplots(1, 2, figsize=(12, 5))
        axs1[0].scatter(y_test_original, y_pred_test_original, alpha=0.5)
        axs1[0].plot([y_test_original.min(), y_test_original.max()],
                     [y_test_original.min(), y_test_original.max()], 'r--')
        axs1[0].set_title('Test skup: Stvarne vs Predviđene')
        axs1[0].set_xlabel('Stvarne')
        axs1[0].set_ylabel('Predviđene')

        axs1[1].scatter(y_train_original, y_pred_train_original, alpha=0.5, color='orange')
        axs1[1].plot([y_train_original.min(), y_train_original.max()],
                     [y_train_original.min(), y_train_original.max()], 'r--')
        axs1[1].set_title('Trening skup: Stvarne vs Predviđene')
        axs1[1].set_xlabel('Stvarne')
        axs1[1].set_ylabel('Predviđene')

        fig2, axs2 = plt.subplots(1, 2, figsize=(12, 5))
        axs2[0].hist(errors, bins=40, color='green', edgecolor='black', alpha=0.7)
        axs2[0].set_title('Distribucija grešaka')
        axs2[0].set_xlabel('Greška')
        axs2[0].set_ylabel('Frekvencija')

        axs2[1].plot(convergence_history)
        axs2[1].set_title('Konvergencija PSO algoritma')
        axs2[1].set_xlabel('Iteracija')
        axs2[1].set_ylabel('Najbolji MSE')
        axs2[1].set_yscale('log')

        fig3, axs3 = plt.subplots(1, 2, figsize=(12, 5))
        axs3[0].boxplot(errors)
        axs3[0].set_title('Boxplot grešaka')
        axs3[0].set_ylabel('Greška')

        axs3[1].scatter(y_pred_test_original, errors, alpha=0.5)
        axs3[1].axhline(0, color='red', linestyle='--')
        axs3[1].set_title('Rezidualna analiza')
        axs3[1].set_xlabel('Predviđene vrijednosti')
        axs3[1].set_ylabel('Greške')

        plt.tight_layout()
        plt.show()
