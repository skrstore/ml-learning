import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-3, 3, 100)
plt.plot(line, np.tanh(line), label="tanh")  # tanh - tanh saturates to –1 for low input values and +1 for high input values
plt.plot(line, np.maximum(line, 0), label="relu")  # relu - The relu function cuts off values below zero i.e remove value below zero
plt.legend(loc="best")
plt.xlabel("x")
plt.ylabel("relu(x), tanh(x)")
plt.show()


# np.maximum - Compare two arrays and returns a new array containing the element-wise maxima. 
# Example - 
np.maximum([1,2,3, -1], 0)
np.maximum([1,9,3, -1], [5,6,7,8])

# Sigmoid finction
x = np.arange(-8, 8, 0.1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.show()


# Softmax Function -  convert the values to probabilities distribution, sum of the probabilities is 1
def softmax(inputs):
    return np.exp(inputs) / float(sum(np.exp(inputs)))
 
 
softmax_inputs = [2, 3, 5, 6]
np.exp(softmax_inputs)
print ("Softmax Function Output :: {}".format(softmax(softmax_inputs)))

np.exp(softmax_inputs)

# Softmax Graph

def line_graph(x, y, x_title, y_title):
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.show()


graph_x = range(0, 21)
graph_y = softmax(graph_x)

print("Graph X readings: {}".format(graph_x))
print("Graph Y readings: {}".format(graph_y))

line_graph(graph_x, graph_y, "Inputs", "Softmax Scores")