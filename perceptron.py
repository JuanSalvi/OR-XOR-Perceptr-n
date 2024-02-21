import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1):
        self.weights = np.random.rand(num_inputs)
        self.learning_rate = learning_rate

    def train(self, X, y, max_epochs=100):
        epoch = 0
        while epoch < max_epochs:
            error_count = 0
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    error_count += 1
            if error_count == 0:
                print("Entrenamiento completado en la época", epoch)
                break
            epoch += 1
        print("Entrenamiento finalizado")

    def predict(self, inputs):
        activation = np.dot(inputs, self.weights)
        return 1 if activation >= 0 else 0

def read_data(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
    X = []
    y = []
    for line in lines:
        data = line.strip().split(',')
        X.append([float(x) for x in data[:-1]])
        y.append(int(data[-1]))
    return np.array(X), np.array(y)

def plot_decision_boundary(X, y, perceptron):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Patrones y recta de separación')
    x_values = np.linspace(-1.5, 1.5, 100)
    y_values = -(perceptron.weights[0] * x_values) / perceptron.weights[1]
    plt.plot(x_values, y_values, label='Recta de separación')
    plt.legend()
    plt.show()

def main():
    # Paso 1: Leer los datos de entrenamiento y prueba desde los archivos CSV
    X_train, y_train = read_data("C:/Users/Salvi/Documents/Universidad/9 SEMESTRE/SEM Inteligencia Artificial II/Practica 1/XOR_trn.csv")
    X_test, y_test = read_data("C:/Users/Salvi/Documents/Universidad/9 SEMESTRE/SEM Inteligencia Artificial II/Practica 1/XOR_tst.csv")
    
    # X_train, y_train = read_data("C:/Users/Salvi/Documents/Universidad/9 SEMESTRE/SEM Inteligencia Artificial II/Practica 1/OR_trn.csv")
    # X_test, y_test = read_data("C:/Users/Salvi/Documents/Universidad/9 SEMESTRE/SEM Inteligencia Artificial II/Practica 1/OR_tst.csv")
    
    # Paso 2: Crear y entrenar el perceptrón
    perceptron = Perceptron(num_inputs=len(X_train[0]))
    perceptron.train(X_train, y_train)

    # Paso 3: Visualizar la separación de clases
    plot_decision_boundary(X_train, y_train, perceptron)

    # Paso 4: Probar el perceptrón en los datos de prueba
    correct_predictions = 0
    for i in range(len(X_test)):
        prediction = perceptron.predict(X_test[i])
        if prediction == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test)
    print("Precisión en los datos de prueba:", accuracy)

    # Paso 5: Mostrar gráficamente la separación de clases después de la prueba 
    plot_decision_boundary(X_train, y_train, perceptron)

if __name__ == "__main__":
    main()
