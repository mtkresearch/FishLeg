#This should be read from a Json file
MLPs_dict = {'MNIST':  {'layer_sizes': [784, 512, 512, 10],
                          'activation_funs':['relu', 'relu', 'linear']},
                          }

Autoencoders_dict = {
    'MNIST':  {'layer_sizes': [784, 1000, 500, 250, 30, 250, 500, 1000, 784],
                'activation_funs':['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear'],
                'loss':'BCE' },
    'CURVES': {'layer_sizes': [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784], 
                'activation_funs':['relu', 'relu', 'relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],       
                'loss':'BCE'},
    'FACES':  {'layer_sizes': [625, 2000, 1000, 500, 30, 500, 1000, 2000, 625], 
                'activation_funs': ['relu', 'relu', 'relu', 'linear', 'relu', 'relu', 'relu', 'linear'],
                'loss':'MSE'}}