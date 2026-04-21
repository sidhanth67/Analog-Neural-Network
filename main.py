import numpy as np


X =  np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
Correct = np.array([

    [0],
    [1],
    [1],
    [0]
])

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    
    return np.where(x > 0, 1, 0)

np.random.seed(20)


hidden_weights = np.random.uniform(-1, 1, (2, 2))


hidden_biases = np.random.uniform(-1, 1, (1, 2))


output_weights = np.random.uniform(-1, 1, (2, 1))
output_bias = np.random.uniform(-1, 1, (1, 1))

epochs = 10000 #also change





learning_rate = 0.05 #for now, change later on if neded


for epoch in range(epochs):
    
  
    hidden_layer_sum = np.dot(X, hidden_weights) + hidden_biases


    hidden_layer_output = relu(hidden_layer_sum)
    
    output_layer_sum = np.dot(hidden_layer_output, output_weights) + output_bias
    
    final_prediction = output_layer_sum 
    
    
    error = Correct - final_prediction
    
    
    d_predicted_output = error * 1 
    #cuz the slope of function is 1 only
    error_hidden_layer = np.dot(d_predicted_output, output_weights.T)



    d_hidden_layer = error_hidden_layer * relu_derivative(hidden_layer_output)#see if voltage was wrong or weight
    
    
    output_weights += np.dot(hidden_layer_output.T, d_predicted_output) * learning_rate

    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    hidden_weights += np.dot(X.T, d_hidden_layer) * learning_rate #final


    hidden_biases += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate #final




print("Final Trained Outputs")
print(np.round(final_prediction, 3))

print("The New Trained Weights")
print("Hidden Weights:\n", np.round(hidden_weights, 3))
print("Output Weights:\n", np.round(output_weights, 3))
