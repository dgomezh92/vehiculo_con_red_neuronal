#include <vector>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Posibles funciones de activación
enum class ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    LINEAR
};

// Implementaciones de las funciones de activación
inline float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float tanh_custom(float x) {
    return std::tanh(x);
}

inline float linear(float x) {
    return x;
}

// Helper para aplicar función de activación según el enum
inline float applyActivation(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:    return relu(x);
        case ActivationFunction::SIGMOID: return sigmoid(x);
        case ActivationFunction::TANH:    return tanh_custom(x);
        case ActivationFunction::LINEAR:  return linear(x);
        default:                          return x; // Por si no se especifica
    }
}

// Clase que representa la red neuronal
class NeuralNetwork {
public:
    // Constructor:
    //  - layers: vector con el número de neuronas en cada capa, 
    //    e.g. {4,8,3} => 4 entradas, 8 neuronas ocultas, 3 salidas.
    //  - hiddenAct y outputAct: funciones de activación para capas ocultas y salida.
    NeuralNetwork(const std::vector<int>& layers,
                  ActivationFunction hiddenAct = ActivationFunction::RELU,
                  ActivationFunction outputAct = ActivationFunction::SIGMOID)
        : m_layers(layers),
          m_hiddenActivation(hiddenAct),
          m_outputActivation(outputAct) 
    {
        // Validamos que haya al menos 2 capas (entradas y salidas)
        if (m_layers.size() < 2) {
            throw std::runtime_error("Se necesitan al menos 2 capas (entrada y salida).");
        }

        // Inicializamos vectores de pesos y sesgos según layers
        initWeights();
    }

    // Permite establecer manualmente los pesos y sesgos, 
    // útil si se quiere cargar un modelo entrenado.
    // weights[i] debe ser un vector de longitud (layers[i] * layers[i+1]),
    // biases[i]  debe ser de longitud (layers[i+1]).
    void setWeights(const std::vector<std::vector<float>>& weights,
                    const std::vector<std::vector<float>>& biases)
    {
        if (weights.size() != m_layers.size() - 1 ||
            biases.size()  != m_layers.size() - 1)
        {
            throw std::runtime_error("La dimensión de pesos o biases no coincide con la topología de la red.");
        }
        m_weights = weights;
        m_biases  = biases;
    }

    // forward: realiza la propagación hacia delante con una entrada (input)
    // y devuelve el vector de salidas resultante.
    // input.size() debe ser = layers[0] (número de neuronas de la capa de entrada).
    std::vector<float> forward(const std::vector<float>& input) {
        if (input.size() != static_cast<size_t>(m_layers[0])) {
            throw std::runtime_error("El tamaño del vector de entrada no coincide con la capa de entrada.");
        }

        // activations será el vector que vaya “viajando” capa a capa
        std::vector<float> activations = input;

        // Recorremos cada capa (excepto la última, pues no tiene pesos hacia nada posterior)
        for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
            int inSize  = m_layers[layerIndex];     // neuronas de entrada para esta capa
            int outSize = m_layers[layerIndex + 1]; // neuronas de salida para esta capa

            // Calculamos la salida de la capa actual
            std::vector<float> newActivations(outSize, 0.0f);

            // Multiplicación vector (activations) x matriz de pesos
            for (int j = 0; j < outSize; j++) {
                for (int i = 0; i < inSize; i++) {
                    // m_weights[layerIndex] es un vector aplanado de dimensión inSize*outSize
                    // en el que i*outSize + j indexa la posición de (i,j)
                    newActivations[j] += activations[i] * m_weights[layerIndex][i*outSize + j];
                }
                // Suma del sesgo
                newActivations[j] += m_biases[layerIndex][j];
            }

            // Determinamos qué función de activación usar:
            // - Si no es la última capa, usamos la activación "hidden"
            // - Si es la última capa, usamos la activación de "output"
            ActivationFunction func = (layerIndex < m_layers.size() - 2)
                                      ? m_hiddenActivation
                                      : m_outputActivation;

            // Aplicamos la función de activación neurona por neurona
            for (auto &val : newActivations) {
                val = applyActivation(val, func);
            }

            // Ahora newActivations pasa a ser la entrada para la siguiente capa
            activations = newActivations;
        }

        // El vector activations ahora contiene la salida final de la red
        return activations;
    }

private:
    // Inicializa los vectores de pesos y sesgos
    // Aquí, por simplicidad, los llenamos con un valor fijo (0.1f).
    // En un caso real se podrían inicializar aleatoriamente o con
    // valores cargados desde un modelo entrenado.
    void initWeights() {
        m_weights.resize(m_layers.size() - 1);
        m_biases.resize(m_layers.size() - 1);

        for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
            int inSize  = m_layers[layerIndex];
            int outSize = m_layers[layerIndex + 1];

            // Reserva para inSize x outSize pesos
            m_weights[layerIndex].resize(inSize * outSize, 0.1f);

            // Reserva para outSize biases
            m_biases[layerIndex].resize(outSize, 0.1f);
        }
    }

private:
    std::vector<int> m_layers;                      // n_i: neuronas por capa
    std::vector<std::vector<float>> m_weights;      // m_weights[i]: pesos de la capa i
    std::vector<std::vector<float>> m_biases;       // m_biases[i]: sesgos de la capa i

    ActivationFunction m_hiddenActivation;          // Activación para capas intermedias
    ActivationFunction m_outputActivation;          // Activación para la capa de salida
};

// Ejemplo de uso de la red
int main() {
    // Definimos la topología: 4 entradas, 8 en la capa oculta, 3 salidas.
    // (Podrías añadir más capas ocultas, p.ej. {4, 8, 8, 8, 3} y así sucesivamente)
    std::vector<int> layerSizes = { 4, 8, 3 };

    // Creamos la red con activación ReLU en capas ocultas y Sigmoid en la salida
    NeuralNetwork nn(layerSizes, ActivationFunction::RELU, ActivationFunction::SIGMOID);

    // Ejemplo de entrada con 4 valores
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};

    // Realizamos la propagación hacia adelante
    std::vector<float> output = nn.forward(input);

    // Mostramos la salida
    std::cout << "Salidas de la red: ";
    for (auto val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
