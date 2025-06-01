import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def data_processing(putanja_do_csv, test_size=0.2):
    # Učitavanje podataka
    df = pd.read_csv(putanja_do_csv)

    # Izbacivanje stupca payment_type
    if 'payment_type' in df.columns:
        df = df.drop(columns=['payment_type'])

    df = df.dropna()

    X = df.drop(columns=['fare_amount']).values
    y = df['fare_amount'].values.reshape(-1, 1)

    # Normalizacija ulaza i izlaza (Min-Max normalizacija)
    X_max = X.max(axis=0)
    y_max = y.max()

    X_norm = X / X_max
    y_norm = y / y_max

    normalizacijski_parametri = {
        'X_max': X_max,
        'y_max': y_max
    }

    # Podjela na trening i testni skup
    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_norm, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test, normalizacijski_parametri