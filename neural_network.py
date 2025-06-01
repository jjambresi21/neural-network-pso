import numpy as np
from dataset import data_processing

def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def forwardpropagation(X, W1, b1, W2, b2):
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    output_output = output_input
    return output_output

def calculate_mse_loss(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Funkcija za dekodiranje vektora težina za PSO
def decode_weights(weights, input_size, hidden_size, output_size):
    w1_end = input_size * hidden_size
    b1_end = w1_end + hidden_size
    w2_end = b1_end + hidden_size * output_size
    b2_end = w2_end + output_size

    W1 = weights[:w1_end].reshape((input_size, hidden_size))
    b1 = weights[w1_end:b1_end].reshape((1, hidden_size))
    W2 = weights[b1_end:w2_end].reshape((hidden_size, output_size))
    b2 = weights[w2_end:b2_end].reshape((1, output_size))

    return W1, b1, W2, b2

# Evaluacijska funkcija za PSO
def evaluate_weights(weights, X, y, input_size, hidden_size, output_size):
    W1, b1, W2, b2 = decode_weights(weights, input_size, hidden_size, output_size)
    y_pred = forwardpropagation(X, W1, b1, W2, b2)
    return calculate_mse_loss(y, y_pred)

# Funkcija za predikciju
def predict(new_input, W1, b1, W2, b2):
    hidden_input = np.dot(new_input, W1) + b1
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, W2) + b2
    return output_input

def main():
    X_train, X_test, y_train, y_test, norm_params = data_processing('Taxi_Trip_Data.csv')

    input_size = X_train.shape[1]
    hidden_size = 5
    output_size = 1

    # promijeniti
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # Primjer evaluacije trenutnih težina
    y_pred_train = forwardpropagation(X_train, W1, b1, W2, b2)
    mse_train = calculate_mse_loss(y_train, y_pred_train)
    print(f"MSE na treniranju s nasumičnim težinama: {mse_train:.6f}")

    # Predikcija na test skupu
    y_pred_test = forwardpropagation(X_test, W1, b1, W2, b2)
    y_pred_original = y_pred_test * norm_params['y_max']
    y_test_original = y_test * norm_params['y_max']

    print("\nPrimjeri predikcija:")
    for i in range(min(5, len(y_test))):
        print(f"Stvarno: {y_test_original[i][0]:.2f}   Predviđeno: {y_pred_original[i][0]:.2f}")

    # Novi ulaz
    novi_ulaz = np.array([[1, 1.97, 12]])
    novi_ulaz_norm = novi_ulaz / norm_params['X_max']
    pred_norm = predict(novi_ulaz_norm, W1, b1, W2, b2)
    pred_original = pred_norm * norm_params['y_max']

    print(f"\nNovi ulaz: {novi_ulaz}")
    print(f"Predviđena cijena: {pred_original[0][0]:.2f}")

if __name__ == "__main__":
    main()
