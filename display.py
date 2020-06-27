#External Modules
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import torch
#Internal Modules
import matrix
#Display Function
def Display2D(adjacenyMatrix,latentMapping):
    #Plot latent space
    X = list()
    Y = list()
    for node in range(len(adjacenyMatrix)):
        (x,y) = latentMapping[node]
        X.append(x)
        Y.append(y)
    #Plot Nodes
    plt.plot(X, Y, 'ro')
    #Draw edges
    for n in range(len(adjacenyMatrix)):
        row = adjacenyMatrix[n]
        point0 = latentMapping[n]
        for neigh in range(len(row)):
            point1 = latentMapping[neigh]
            x_values = [point0[0], point1[0]]
            y_values = [point0[1], point1[1]]
            c = 'k'
            a = adjacenyMatrix[n][neigh]*.5
            if a < 0:
                a = 0 #abs(a)*.99
                c = 'r'
            plt.plot(x_values, y_values, color=c, alpha=a)
    plt.show()

def Display3D(adjacenyMatrix,latentMapping):
    #Plot latent space
    X = list()
    Y = list()
    Z = list()
    ax = plt.axes(projection='3d')
    for node in range(len(adjacenyMatrix)):
        (x,y,z) = latentMapping[node]
        X.append(x)
        Y.append(y)
        Z.append(z)
    #Plot Nodes
    plt.plot(X, Y, Z, 'ro')
    #Draw edges
    for n in range(len(adjacenyMatrix)):
        row = adjacenyMatrix[n]
        point0 = latentMapping[n]
        for neigh in range(len(row)):
            point1 = latentMapping[neigh]
            x_values = [point0[0], point1[0]]
            y_values = [point0[1], point1[1]]
            z_values =  [point0[2], point1[2]]
            plt.plot(x_values, y_values, z_values, color='k', alpha=adjacenyMatrix[n][neigh]*.5)
    plt.show()

def Plot(adjacenyMatrix,encoder,doPrint=True):
    adjacenyTensor = torch.Tensor(adjacenyMatrix)
    latentMapping = encoder(adjacenyTensor).tolist()
    if doPrint:
        #Print matrix
        matrix.printMat(adjacenyMatrix)
        #Print latent mapping
        for node in range(len(latentMapping)):
            vector =  latentMapping[node]
            print(f"Node{node} -> {vector}")
    dimension = encoder.output_size
    if dimension == 2:
        Display2D(adjacenyMatrix,latentMapping)
    elif dimension== 3:
        Display3D(adjacenyMatrix,latentMapping)
    else:
        print("Unable to plot {dimension}-dimensional figure!")
