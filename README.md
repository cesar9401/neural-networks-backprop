# neural-networks-backprop
Final practice of the artificial intelligence course


# Manual Técnico: Red Neuronal en Python

## Índice

1. [Introducción](#introducción)
2. [Instalación](#instalación)
3. [Estructura de la Clase Network](#estructura-de-la-clase-network)
4. [Funciones de Activación](#funciones-de-activación)
5. [Métodos Principales](#métodos-principales)
   - [feedforward](#feedforward)
   - [SGD](#sgd)
   - [update_mini_batch](#update_mini_batch)
   - [backprop](#backprop)
   - [evaluate](#evaluate)
6. [Ejemplo de Uso](#ejemplo-de-uso)
7. [Consideraciones Adicionales](#consideraciones-adicionales)
8. [Conclusión](#conclusión)

## Introducción

Este manual técnico describe la implementación y uso de una red neuronal en Python. La red neuronal se construye utilizando matrices para representar los pesos y los sesgos, y se entrenará mediante el método de descenso de gradiente estocástico (SGD). Esta implementación permite el uso de diferentes funciones de activación para las capas ocultas y la capa de salida.

## Instalación

Antes de comenzar, asegúrate de tener las siguientes dependencias instaladas:
- `numpy`
- Las funciones de activación, que se encuentran en el archivo `neural/functions.py`.

Puedes instalarlas usando pip:

```bash
pip install numpy
```

Asegúrate también de tener el archivo `functions.py` con las siguientes funciones de activación: `sigmoid`, `sigmoid_prime`, `step_function`, `hyperbolic_tangent`, `hyperbolic_tangent_prime`, `identity_function`, `step_function_prime`, `identity_function_prime`.

## Estructura de la Clase Network

La clase `Network` es el núcleo de nuestra red neuronal. La clase se inicializa con los tamaños de las capas, las funciones de activación para las capas ocultas y la capa de salida.

```python
class Network:
    def __init__(self, sizes, hidden_activation_func: str, output_activation_func: str):
        self.layer_sizes = sizes
        self.layers = len(self.layer_sizes)

        self.biases = []
        for y in self.layer_sizes[1:]:
            self.biases.append(np.random.randn(y, 1))

        self.weights = []
        for x, y in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            self.weights.append(np.random.randn(y, x))

        self.hidden_activation, self.hidden_activation_prime = self.get_activation_function(hidden_activation_func)
        self.output_activation, self.output_activation_prime = self.get_activation_function(output_activation_func)
```

### Atributos

- `layer_sizes`: Lista con el número de neuronas en cada capa.
- `layers`: Número total de capas en la red.
- `biases`: Lista de matrices de sesgos para cada capa.
- `weights`: Lista de matrices de pesos para cada capa.
- `hidden_activation`, `hidden_activation_prime`: Funciones de activación y sus derivadas para las capas ocultas.
- `output_activation`, `output_activation_prime`: Funciones de activación y sus derivadas para la capa de salida.

## Funciones de Activación

Las funciones de activación y sus derivadas se definen en `neural/functions.py`. Aquí están las principales funciones disponibles:

- `sigmoid`
- `sigmoid_prime`
- `step_function`
- `step_function_prime`
- `hyperbolic_tangent`
- `hyperbolic_tangent_prime`
- `identity_function`
- `identity_function_prime`

## Métodos Principales

### feedforward

Calcula la salida de la red neuronal para una entrada dada.

```python
def feedforward(self, activation):
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        activation = self.hidden_activation(z)
    return self.output_activation(z)
```

### SGD

Implementa el algoritmo de descenso de gradiente estocástico.

```python
def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for epoch in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, learning_rate)
        if test_data:
            accuracy = self.evaluate(test_data)
            print(f'Epoch {epoch + 1}: {accuracy} / {len(test_data)}')
        else:
            print(f'Epoch {epoch + 1} complete')
```

### update_mini_batch

Actualiza los pesos y sesgos de la red utilizando retropropagación para un mini-lote de datos.

```python
def update_mini_batch(self, mini_batch, learning_rate):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

    self.weights = [(w - learning_rate * nw) for w, nw, in zip(self.weights, nabla_w)]
    self.biases = [(b - learning_rate * nb) for b, nb in zip(self.biases, nabla_b)]
```

### backprop

Calcula el gradiente de la función de costo utilizando retropropagación.

```python
def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]

    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = self.hidden_activation(z)
        activations.append(activation)

    delta = self.cost_prime(activations[-1], y)

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for layer in range(2, self.layers):
        z = zs[-layer]
        sp = self.hidden_activation_prime(z)
        delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
        nabla_b[-layer] = delta
        nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
    return nabla_b, nabla_w
```

### evaluate

Evalúa el rendimiento de la red neuronal en un conjunto de prueba.

```python
def evaluate(self, test_data):
    test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
    return sum(int(x == y) for x, y in test_results)
```

### cost_prime

Calcula la derivada de la función de costo.

```python
@staticmethod
def cost_prime(actual, predicted):
    return actual - predicted
```

### get_activation_function

Devuelve la función de activación y su derivada correspondiente según el nombre proporcionado.

```python
@staticmethod
def get_activation_function(name):
    if name == 'sigmoid':
        return sigmoid, sigmoid_prime
    elif name == 'tanh':
        return hyperbolic_tangent, hyperbolic_tangent_prime
    elif name == 'step':
        return step_function, step_function_prime
    elif name == 'identity':
        return identity_function, identity_function_prime
```

## Ejemplo de Uso

```python
import numpy as np
from neural.functions import sigmoid, sigmoid_prime, hyperbolic_tangent, hyperbolic_tangent_prime

# Definimos los tamaños de las capas: 2 neuronas en la entrada, 3 en la capa oculta y 1 en la salida
sizes = [2, 3, 1]

# Creamos la red neuronal con funciones de activación 'sigmoid' para la capa oculta y 'identity' para la salida
network = Network(sizes, hidden_activation_func='sigmoid', output_activation_func='identity')

# Datos de entrenamiento: lista de tuplas (entrada, salida esperada)
training_data = [
    (np.array([[0.1], [0.9]]), np.array([[1.0]])),
    (np.array([[0.8], [0.2]]), np.array([[0.0]]))
]

# Entrenamos la red neuronal
network.SGD(training_data, epochs=10, mini_batch_size=2, learning_rate=0.1)
```

## Consideraciones Adicionales

- Asegúrate de que las dimensiones de los datos de entrada y salida coincidan con las dimensiones esperadas por la red.
- Puedes ajustar los hiperparámetros como el número de épocas, el tamaño del mini-lote y la tasa de aprendizaje para mejorar el rendimiento del modelo.
- Las funciones de activación y sus derivadas deben estar correctamente definidas en `neural/functions.py`.

## Conclusión

Este manual técnico proporciona una guía completa para la implementación y uso de una red neuronal en Python. Siguiendo los ejemplos y las instrucciones, podrás entrenar y evaluar modelos de redes neuronales personalizados para diferentes tareas de aprendizaje automático.
