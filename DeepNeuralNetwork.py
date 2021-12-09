#!/usr/bin/env python
# coding: utf-8

# *   Abdullah Naveed   abdullahnaveedahmed@gmail.com

# In[ ]:


class neural_net:
  
  def __init__(self):

    '''
        variables:
            epochs = number of epochs
            learning_rate = alpha
            loss = name of loss function
            optimizer = name of optimizer
            layers = weight matrices and bias of every layer stored here

    '''
    self.epochs = None
    self.learning_rate = None
    self.loss = None
    self.optimizer = None
    self.layers = []

  ##########################
  #                        #
  #      Architecture      #
  #                        #
  ##########################

  #For Addition of Input Layer
  def flatten(self, input_shape):
    self.layers.append([])
    self.layers[0].append(input_shape)

  #For Addition of Hidden Layers
  def dense(self, number_of_neurons=0, activation_function="relu", dropout=0):

    parameters = {}

    prev_layer_index = len(self.layers)-1
    prev_layer_shape = self.layers[prev_layer_index][0][0]

    #initializing weights for first hidden layer
    parameters["W"+str(len(self.layers))] = np.random.randn(number_of_neurons, prev_layer_shape) * 0.01
    parameters["b"+str(len(self.layers))] = np.zeros((number_of_neurons,1))

    #Adding layer to neural net
    self.layers.append([(number_of_neurons, 0), activation_function, dropout, parameters])

  #For Addition of Output Layer
  def output(self, number_of_neurons=0, activation_function="sigmoid", dropout=0):

    parameters = {}

    prev_layer_index = len(self.layers)-1
    prev_layer_shape = self.layers[prev_layer_index][0][0]

    #initializing weights for layer
    parameters["W"+str(len(self.layers))] = np.random.randn(number_of_neurons, prev_layer_shape) * 0.01
    parameters["b"+str(len(self.layers))] = np.zeros((number_of_neurons,1))

    #Adding layer to neural net
    self.layers.append([(number_of_neurons, 0), activation_function, dropout, parameters])

  ##########################
  #                        #
  #      Compilation       #
  #                        #
  ##########################

  #For defining optimizers and loss functions and other hyperparameters
  def compile(self, optimizer="", loss="", learning_rate=0.001, epochs=0):
    self.epochs = epochs
    self.learning_rate = learning_rate
    self.loss = loss
    self.optimizer = optimizer

  #To start training neural net
  def fit(self, X, y):

    losses=[]

    #Iterating over epochs
    for index in range(self.epochs):

      #Applying forward pass
      y_hat, cache = self.forwardPass(X)

      #Computing Gradients
      gradients = self.backpropagation(y_hat, cache, y)

      #Updating weights
      parameters = self.update_paramaters(gradients)

      #Computing Loss
      loss = self.compute_loss(y_hat, y, loss=self.loss)

      #Printing Loss
      if(index%100 == 0):
          print("Epoch:",index, "\t\t Loss:", np.squeeze(loss))
          losses.append(loss)

    #Printing Performance
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

  ##########################
  #                        #
  #      Training          #
  #                        #
  ##########################

  #For Activation functions
  def activation(self, dot, activation_function=""):

    #For Storing activations
    Z = None

    #Sigmoid Activation
    if(activation_function=="sigmoid"):
      sigma = np.exp(-dot)
      Z = 1/(1+sigma)

    #Relu Activation
    elif(activation_function=="relu"):
      Z = np.maximum(0,dot)

    return Z

  #For Loss function
  def compute_loss(self, y_hat, Y, loss = ""):

    #Cross Entropy Loss
    if(loss == "cross_entropy"):
      tmp = (Y * np.log(y_hat)) + ((1-Y) * np.log(1-y_hat)) 
      cost = (-1/Y.shape[0]) * np.sum(tmp, axis=1)
      
      return cost
    
    #MSE Loss
    elif(loss== "MSE"):
      cost = np.square(np.subtract(Y,y_hat)).mean()
    
      return cost
    
    #Hinge Loss
    elif (loss == "hinge_loss"):
      margins = np.maximum(0, y_hat - Y + 1)
      cost =  np.sum(margins)

      return cost

  #Forward Pass
  def forwardPass(self, X):
    
    cache = {}

    #Iterating over Each Layer
    for index, layer in enumerate(self.layers):

      #If Input Layer
      if(index == 0):
        cache["A"+str(index)] = X

      #If Hidden layers or output layer
      elif(index > 0):
        Z = np.dot(layer[3]["W"+str(index)], cache["A"+str(index-1)])
        Z = np.add(Z, layer[3]["b"+str(index)])
        cache["Z"+str(index)] = Z

        A = self.activation(Z, layer[1])
        cache["A"+str(index)] = A

    #Returning y_hat and all activations
    return A, cache
  

  
  #Backward Pass
  def backpropagation(self, y_hat, cache, Y):
    
    totalLayers = len(self.layers) - 1
    gradients = {}

    #Derivating Cost
    cost_Derivative = -(np.divide(Y, y_hat) - np.divide(1-Y, 1-y_hat))

    #Output layer gradients
    gradients["dA"+str(totalLayers)], gradients["dW"+str(totalLayers)], gradients["db"+str(totalLayers)] = self.single_back_prop(cost_Derivative, cache, totalLayers, activation="sigmoid")
  
    #Hidden layers gradients
    for i in list(reversed(range(totalLayers))): 

      #If not input layer                      
      if(i>0):

        #Computing Gradients
        a, w, b = self.single_back_prop(gradients["dA"+str(i+1)], cache, i, activation="relu")
        gradients["dA"+str(i)] = a
        gradients["dW"+str(i)] = w
        gradients["db"+str(i)] = b

    return gradients
  
  def update_paramaters(self, gradients):
      
    for index, layer in enumerate(self.layers):

        #Updating Parameters: Vanilla Gradient Descent
        if(index > 0):
          layer[3]["W"+str(index)] = layer[3]["W"+str(index)] - self.learning_rate * gradients["dW"+str(index)]
          layer[3]["b"+str(index)] = layer[3]["b"+str(index)] - self.learning_rate * gradients["db"+str(index)]
  

  ##########################
  #                        #
  #      Helpers           #
  #                        #
  ##########################

  def summary(self):

    print("----------------Model Summary-----------------------")

    for index, layer in enumerate(self.layers):

      #input layer
      if(index == 0):
        print("Input Layer:\t\t\t", "shape", layer[0])

      #output layer
      elif(index == len(self.layers)-1):
        print("Output Layer:\t\t\t", "shape", (layer[0]))
        
      #hidden layers
      else:
          print("Hidden Layer #", index, "\t\t", "total parameters", layer[3]["W"+str(index)].size + layer[3]["b"+str(index)].size)

      print("----------------------------------------------------")
  
  def single_back_prop(self, dA, cache, cache_index, activation):
    
    #Fetching important variables
    Z = cache["Z"+str(cache_index)]
    APrime = cache["A"+str(cache_index-1)]
    W = self.layers[cache_index][3]["W"+str(cache_index)]
    b = self.layers[cache_index][3]["b"+str(cache_index)]

    m = APrime.shape[1]
        
    if activation == "sigmoid":
        s = 1/(1+np.exp(-Z))
        dZ = dA * s * (1-s)

    elif activation == "relu":
        dZ = np.array(dA, copy=True)
        dZ[Z<=0] = 0
         
    # Computing derivative wrt A & W  & b
    dAPrime = np.dot(W.transpose(), dZ)
    dW = (1/m) * np.dot(dZ, APrime.transpose())
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
   
    
    return dAPrime, dW, db

  ##########################
  #                        #
  #      Testing           #
  #                        #
  ##########################

  # Defining predict function
  def predict(self, X, Y):

    # forward propagation
    y_hat, caches = self.forwardPass(X)
    
    y_hat = y_hat.reshape(-1)
    predicted = np.where(y_hat>0.5, 1, 0)

    m = X.shape[1]
    accuracy = np.sum(predicted == Y) / m
     
    return accuracy

