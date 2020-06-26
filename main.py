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
        self.fc1 = nn.Linear(input_size,output_size)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x

#Custom Loss
def my_loss(output, similarityTensor):
    #print("Running Loss!")
    #print("Output:",output.size())
    outputT = torch.transpose(output,0,1)
    #print("Output Transpose:",outputT.size())
    latentDistance = torch.mm(output,outputT)
    #print("Latent Distance:",latentDistance.size())
    #print("Similarity Tensor:",similarityTensor.size())
    loss = torch.mean((latentDistance - similarityTensor)**2)
    return loss

#Creater function
def CreateEncoder(adjacenyMatrix,model=SimpleEncoder, output_size=2,max_steps=10000):
    #Generate Similairty Matrix
    similarityTensor = torch.Tensor(similarity.getSimilarityMatrix(adjacenyMatrix))
    #training cycle
    adjacenyTensor = torch.Tensor(adjacenyMatrix)
    encoder = model(len(adjacenyMatrix),output_size)
    #Define optimizer & Criterion
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    criterion = my_loss
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
def Display(adjacenyMatrix,encoder):
    #Print matrix
    matrix.printMat(adjacenyMatrix)
    #Plot latent space
    adjacenyTensor = torch.Tensor(adjacenyMatrix)
    latentMapping = encoder(adjacenyTensor).tolist()
    X = list()
    Y = list()
    for node in range(len(adjacenyMatrix)):
        (x,y) = latentMapping[node]
        X.append(x)
        Y.append(y)
        print(f"Node: {node} -> ({x},{y})")
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
            plt.plot(x_values, y_values, color='k', alpha=adjacenyMatrix[n][neigh])
    plt.show()
graph = graphs.randomGraph(5,weighted=True)
encoder = CreateEncoder(graph)
Display(graph,encoder)
