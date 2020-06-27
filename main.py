#External modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
#Internal modules
import similarity
import graphs
import matrix

#Encoders
class SimpleEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleEncoder,self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size,output_size,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

#Custom Losses
def latent_orthogonality_loss(output, similarityTensor):
    #Latent Orthoganlity
    outputT = torch.transpose(output,0,1)
    latentOrthogonality = torch.mm(output,outputT)
    loss = torch.mean((latentOrthogonality - similarityTensor)**2)
    return loss

def latent_distance_loss(output, similarityTensor):
    #Latent Orthoganlity
    size = output.size()[0]

    n = output.size(0)
    m = output.size(0)
    d = output.size(1)

    x = output.unsqueeze(1).expand(n, m, d)
    y = output.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    loss = torch.mean(((1 - dist) - similarityTensor)**2)
    return loss

#Creater function
def CreateEncoder(adjacenyMatrix,model=SimpleEncoder,similarity_function=similarity.directConnections,output_size=2,max_steps=10000):
    #Generate Similairty Matrix
    similarityTensor = torch.Tensor(similarity.getSimilarityMatrix(adjacenyMatrix))
    #training cycle
    adjacenyTensor = torch.Tensor(adjacenyMatrix)
    encoder = model(len(adjacenyMatrix),output_size)
    #Define optimizer & Criterion
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    criterion = latent_distance_loss
    loss_list = list()
    for i in range(max_steps):
        #zero gradient
        optimizer.zero_grad()
        #forward, backward, optimize
        output = encoder(adjacenyTensor)
        #print("Output:",output)
        loss = criterion(output,similarityTensor)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (i/max_steps*100) % 10 == 0:
            average_loss = sum(loss_list)/len(loss_list)
            loss_list = []
            print("Loss:",average_loss)
            if average_loss < 1e-9:
                print(f"Early Break! [{i}/{max_steps}]")
                break
    return encoder

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
    #plt.axis([0, 1, 0, 1])
    #Draw edges
    for n in range(len(adjacenyMatrix)):
        row = adjacenyMatrix[n]
        point0 = latentMapping[n]
        for neigh in range(len(row)):
            point1 = latentMapping[neigh]
            x_values = [point0[0], point1[0]]
            y_values = [point0[1], point1[1]]
            plt.plot(x_values, y_values, color='k', alpha=adjacenyMatrix[n][neigh]*.5)
    plt.show()
def Display3D(adjacenyMatrix,encoder):
    pass
def Display(adjacenyMatrix,encoder,doPrint=True):
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

#Call
graph = graphs.randomGraph(10,weighted=True)
encoder = CreateEncoder(graph,similarity_function=similarity.directConnections)
Display(graph,encoder)
