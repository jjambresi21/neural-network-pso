import numpy as np
from dataset import data_processing
from neural_network import NeuralNetwork
from pso import PSO

def main():
    X_train, X_test, y_train, y_test, norm_params = data_processing('Taxi_Trip_Data.csv')

    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1
    nn = NeuralNetwork(input_size, hidden_size, output_size)

    W1, b1, W2, b2 = nn.initialize_parameters()
    y_pred_train = nn.forwardpropagation(X_train, W1=W1, b1=b1, W2=W2, b2=b2)
    mse_train = nn.calculate_mse_loss(y_train, y_pred_train)
    
    pso = PSO(nn, X_train, y_train)
    best_weights, best_score = pso.optimize()

    print(f"MSE na treniranju s nasumičnim težinama: {mse_train:.6f}")
    print(f"Najbolji MSE: {best_score:.6f}")

    y_pred_test = nn.forwardpropagation(X_test, weights=best_weights)
    y_pred_original = y_pred_test * norm_params['y_max']
    y_test_original = y_test * norm_params['y_max']

    print("\nPrimjeri predikcija:")
    for i in range(min(5, len(y_test))):
        print(f"Stvarno: {y_test_original[i][0]:.2f}   Predviđeno: {y_pred_original[i][0]:.2f}")

    novi_ulaz = np.array([[1, 1.97, 12]])
    novi_ulaz_norm = novi_ulaz / norm_params['X_max']
    pred_norm = nn.predict(novi_ulaz_norm, weights=best_weights)
    pred_original = pred_norm * norm_params['y_max']

    print(f"\nNovi ulaz: {novi_ulaz}")
    print(f"Predviđena cijena: {pred_original[0][0]:.2f}")

if __name__ == "__main__":
    main()