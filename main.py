#External modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Internal modules
import similarity
import graphs
import matrix
import display

#Encoders
class SimpleEncoder(nn.Module):
    """
    This is a simple encoder with no hidden layers
    """
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
    """
    This will make more similar nodes more parallel, and more disimilar nodes
    more orthogonal.

    This is better for capturing information (I think), but not as pretty.
    """
    outputT = torch.transpose(output,0,1)
    latentOrthogonality = torch.mm(output,outputT)
    loss = torch.mean((latentOrthogonality - similarityTensor)**2)
    return loss

def latent_distance_loss(output, similarityTensor):
    """
    This will make more similar nodes closer to each other and disimilar nodes
    farther from each other in the latent space, using Euclidian Distance.

    This is more useful for visualizing.
    """
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
    """
    This will train and return a specified model on a specific graph.
    The encoder will assign nodes vectors in a latent space of a specified dimension (output_size)
    """
    #Generate Similairty Matrix
    similarityTensor = torch.Tensor(similarity.getSimilarityMatrix(adjacenyMatrix,similarity_function))
    #training cycle
    adjacenyTensor = torch.Tensor(adjacenyMatrix)
    encoder = model(len(adjacenyMatrix),output_size)
    #Define optimizer & Criterion
    optimizer = optim.Adam(encoder.parameters(), lr=0.001, weight_decay=1e-9)
    criterion = latent_distance_loss
    loss_list = list()
    last_loss = None
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
        last_loss = loss.item()
        if (i/max_steps*100) % 10 == 0:
            average_loss = sum(loss_list)/len(loss_list)
            loss_list = []
            print("Loss:",average_loss)
            if average_loss < 1e-9:
                print(f"Early Break! [{i}/{max_steps}]")
                break
    return encoder,last_loss

#Call
graph = graphs.DoubleCrossedCycle(12)
encoder,loss = CreateEncoder(graph,output_size=3,similarity_function=similarity.directConnections)
display.Plot(graph,encoder)
