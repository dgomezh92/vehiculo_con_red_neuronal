#include "NeuralNetwork.h"
#include <cmath>
#include <stdexcept>

/**
 * @brief Función ReLU (Rectified Linear Unit).
 */
static inline float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * @brief Función Sigmoid.
 */
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Función Tanh.
 */
static inline float tanh_custom(float x) {
    return std::tanh(x);
}

/**
 * @brief Función Linear.
 */
static inline float linear(float x) {
    return x;
}

/**
 * @brief Helper para aplicar la función de activación según el enum.
 */
static inline float applyActivation(float x, ActivationFunction func) {
    switch (func) {
        case ActivationFunction::RELU:
            return relu(x);
        case ActivationFunction::SIGMOID:
            return sigmoid(x);
        case ActivationFunction::TANH:
            return tanh_custom(x);
        case ActivationFunction::LINEAR:
        default:
            return linear(x);
    }
}


NeuralNetwork::NeuralNetwork(const std::vector<int>& layers,
                             ActivationFunction hiddenAct,
                             ActivationFunction outputAct)
    : m_layers(layers),
      m_hiddenActivation(hiddenAct),
      m_outputActivation(outputAct)
{
    // Validación mínima: se necesitan al menos 2 capas (entrada y salida)
    if (m_layers.size() < 2) {
        throw std::runtime_error("Se requieren al menos 2 capas (entrada y salida).");
    }

    // Inicializamos los pesos y sesgos con valores fijos (0.1f)
    initWeights();
}

void NeuralNetwork::initWeights() {
    m_weights.resize(m_layers.size() - 1);
    m_biases.resize(m_layers.size() - 1);

    for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
        int inSize  = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        // Tamaño de la matriz (aplanada) de pesos
        m_weights[layerIndex].resize(inSize * outSize, 0.1f);

        // Tamaño de sesgos
        m_biases[layerIndex].resize(outSize, 0.1f);
    }
}


float NeuralNetwork::calculateMSE(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("El tamaño de output y target no coinciden.");
    }

    if (output.empty()) {
        return 0.0f; // Evita división por cero
    }

    float mse = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
       float error = output[i] - target[i];
       mse += error * error;
    }
    return mse / output.size();
}

float calculateMAE(const std::vector<float>& output, const std::vector<float>& target) {
    if (output.size() != target.size()) {
        throw std::runtime_error("El tamaño de output y target no coinciden.");
    }

    if (output.empty()) {
        return 0.0f;
    }

    float mae = 0.0f;
    for (size_t i = 0; i < output.size(); ++i) {
        mae += std::abs(output[i] - target[i]);
    }
    return mae / output.size(); // Promedio del error absoluto
}



float NeuralNetwork:: calculateError(const std::vector<float>& output, 
                     const std::vector<float>& target, 
                     ErrorFunction errorType) {
    
    if (output.size() != target.size()) {
        throw std::runtime_error("El tamaño de output y target no coinciden.");
    }

    if (output.empty()) {
        return 0.0f; // Evita división por cero
    }

    float errorValue = 0.0f;

    switch (errorType) {
        case ErrorFunction::MSE:
            // Error Cuadrático Medio (MSE)
            for (size_t i = 0; i < output.size(); ++i) {
                float error = output[i] - target[i];
                errorValue += error * error;
            }
            errorValue /= output.size();
            break;

        case ErrorFunction::MAE:
            // Error Absoluto Medio (MAE)
            for (size_t i = 0; i < output.size(); ++i) {
                errorValue += std::abs(output[i] - target[i]);
            }
            errorValue /= output.size();
            break;

        case ErrorFunction::CROSS_ENTROPY:
            // Entropía Cruzada (Cross-Entropy)
            for (size_t i = 0; i < output.size(); ++i) {
                float y = target[i];  // Valor real (0 o 1 para binario)
                float p = output[i];  // Probabilidad predicha (entre 0 y 1)

                // Evita log(0) reemplazando p = 0 con un valor muy pequeño
                p = std::max(p, 1e-9f);
                p = std::min(p, 1.0f - 1e-9f);

                errorValue -= y * std::log(p) + (1 - y) * std::log(1 - p);
            }
            errorValue /= output.size();
            break;

        default:
            throw std::runtime_error("Tipo de función de error no reconocido.");
    }

    return errorValue;
}


void NeuralNetwork::setWeights(const std::vector<std::vector<float>>& weights,
                               const std::vector<std::vector<float>>& biases)
{
    if (weights.size() != m_layers.size() - 1 ||
        biases.size()  != m_layers.size() - 1)
    {
        throw std::runtime_error("Pesos o sesgos no coinciden con la topología de la red.");
    }

    // Asigna los pesos y sesgos externos
    m_weights = weights;
    m_biases  = biases;
}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) {
    // Validamos que la entrada tenga el tamaño de la capa de entrada
    if (input.size() != static_cast<size_t>(m_layers[0])) {
        throw std::runtime_error("El vector de entrada no coincide con la capa de entrada.");
    }

    // Vector que almacena las activaciones de la capa actual
    std::vector<float> activations = input;

    // Recorremos cada capa, salvo la última (que no tiene pesos salientes)
    for (size_t layerIndex = 0; layerIndex < m_layers.size() - 1; layerIndex++) {
        int inSize  = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        std::vector<float> newActivations(outSize, 0.0f);

        // Multiplicación activations x pesos
        for (int j = 0; j < outSize; j++) {
            for (int i = 0; i < inSize; i++) {
                newActivations[j] += activations[i] * m_weights[layerIndex][i * outSize + j];
            }
            // Suma de sesgos
            newActivations[j] += m_biases[layerIndex][j];

            // Determinamos si es una capa oculta o la capa final
            ActivationFunction currentAct = (layerIndex < (m_layers.size() - 2))
                                            ? m_hiddenActivation
                                            : m_outputActivation;

            // Aplicamos la función de activación
            newActivations[j] = applyActivation(newActivations[j], currentAct);
        }

        // Actualizamos las activaciones para la siguiente capa
        activations = newActivations;
    }

    return activations;
}

void NeuralNetwork::computeGradients(const std::vector<float>& input,
                                     const std::vector<float>& target,
                                     std::vector<std::vector<float>>& grad_weights,
                                     std::vector<std::vector<float>>& grad_biases) {
    // Inicializar estructuras
    grad_weights = std::vector<std::vector<float>>(m_weights.size());
    grad_biases = std::vector<std::vector<float>>(m_biases.size());

    for (size_t i = 0; i < m_weights.size(); i++) {
        grad_weights[i] = std::vector<float>(m_weights[i].size(), 0.0f);
        grad_biases[i] = std::vector<float>(m_biases[i].size(), 0.0f);
    }

    // 1️ Forward pass
    std::vector<float> activations = input;
    std::vector<std::vector<float>> layer_activations = {activations};

    for (size_t layerIndex = 0; layerIndex < m_weights.size(); layerIndex++) {
        int inSize = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        std::vector<float> newActivations(outSize, 0.0f);

        for (int j = 0; j < outSize; j++) {
            for (int i = 0; i < inSize; i++) {
                newActivations[j] += activations[i] * m_weights[layerIndex][i * outSize + j];
            }
            newActivations[j] += m_biases[layerIndex][j];

            ActivationFunction func = (layerIndex < m_layers.size() - 2) ? m_hiddenActivation : m_outputActivation;
            newActivations[j] = applyActivation(newActivations[j], func);
        }
        activations = newActivations;
        layer_activations.push_back(activations);
    }

    //  Backpropagation
    std::vector<float> delta = layer_activations.back();
    for (size_t i = 0; i < delta.size(); i++) {
        delta[i] = delta[i] - target[i]; // Derivada de MSE: (output - target)
    }

    for (int layerIndex = m_weights.size() - 1; layerIndex >= 0; layerIndex--) {
        int inSize = m_layers[layerIndex];
        int outSize = m_layers[layerIndex + 1];

        std::vector<float> prev_delta = std::vector<float>(inSize, 0.0f);

        for (int j = 0; j < outSize; j++) {
            for (int i = 0; i < inSize; i++) {
                grad_weights[layerIndex][i * outSize + j] = layer_activations[layerIndex][i] * delta[j];
                prev_delta[i] += m_weights[layerIndex][i * outSize + j] * delta[j];
            }
            grad_biases[layerIndex][j] = delta[j];
        }

        delta = prev_delta;
    }
}


void NeuralNetwork::updateWeights(const std::vector<std::vector<float>>& gradients_weights,
                                  const std::vector<std::vector<float>>& gradients_biases,
                                  float learningRate) {
    if (gradients_weights.size() != m_weights.size() || 
        gradients_biases.size() != m_biases.size()) {
        throw std::runtime_error("Los gradientes no coinciden con la estructura de la red.");
    }

    // Recorremos todas las capas y actualizamos pesos y sesgos
    for (size_t layerIndex = 0; layerIndex < m_weights.size(); ++layerIndex) {
        for (size_t i = 0; i < m_weights[layerIndex].size(); ++i) {
            m_weights[layerIndex][i] -= learningRate * gradients_weights[layerIndex][i];
        }

        for (size_t j = 0; j < m_biases[layerIndex].size(); ++j) {
            m_biases[layerIndex][j] -= learningRate * gradients_biases[layerIndex][j];
        }
    }
}

