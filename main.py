import numpy as np
from dataset import data_processing
from neural_network import NeuralNetwork
from pso import PSO
from evaluation import Evaluation

def main():
    X_train, X_test, y_train, y_test, norm_params = data_processing('Taxi_Trip_Data.csv')

    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    e = Evaluation(nn)

    W1, b1, W2, b2 = nn.initialize_parameters()
    y_pred_train = nn.forwardpropagation(X_train, W1=W1, b1=b1, W2=W2, b2=b2)
    mse_train = nn.calculate_mse_loss(y_train, y_pred_train)
    
    pso = PSO(nn, X_train, y_train)
    best_weights, best_score, convergence_history = pso.optimize()

    e.print_results(mse_train, best_score, ((mse_train - best_score) / mse_train * 100))

    y_pred_test = nn.forwardpropagation(X_test, weights=best_weights)
    y_pred_train_original = nn.forwardpropagation(X_train, weights=best_weights) * norm_params['y_max']
    y_train_original = y_train * norm_params['y_max']
    y_pred_test_original = y_pred_test * norm_params['y_max']
    y_test_original = y_test * norm_params['y_max']
    errors = y_test_original.flatten() - y_pred_test_original.flatten()

    metrics = e.calculate_metrics(y_test_original, y_pred_test_original)
    e.print_metrics(metrics, dataset_name="Test")

    e.detailed_performance_analysis(y_test_original, y_pred_test_original)

    print("\nPrimjeri predikcija:")
    for i in range(min(5, len(y_test))):
        print(f"Stvarno: {y_test_original[i][0]:.2f} | Predviđeno: {y_pred_test_original[i][0]:.2f} | Razlika: {abs(y_test_original[i][0] - y_pred_test_original[i][0]):.2f}")

    novi_ulaz = np.array([[1, 1.97, 12]])
    novi_ulaz_norm = novi_ulaz / norm_params['X_max']
    pred_norm = nn.predict(novi_ulaz_norm, weights=best_weights)
    pred_original = pred_norm * norm_params['y_max']

    print(f"\nNovi ulaz: {novi_ulaz}")
    print(f"Predviđena cijena: {pred_original[0][0]:.2f}")

    e.create_visualizations(y_train_original, y_pred_train_original, y_test_original, y_pred_test_original, errors, convergence_history)

if __name__ == "__main__":
    main()