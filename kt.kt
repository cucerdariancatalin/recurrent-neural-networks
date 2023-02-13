class SimpleRNN(inputDim: Int, hiddenDim: Int, outputDim: Int) {
    val U = randomMatrix(inputDim, hiddenDim)  // weights for input to hidden layer
    val W = randomMatrix(hiddenDim, hiddenDim)  // weights for hidden to hidden layer
    val V = randomMatrix(hiddenDim, outputDim)  // weights for hidden to output layer

    fun forward(x: List<Double>, h: List<Double>) : List<Double> {
        val hidden = tanh(add(multiply(U, x), multiply(W, h)))  // update hidden state
        return sigmoid(multiply(V, hidden))  // calculate output
    }
}
