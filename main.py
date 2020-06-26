#External modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
def CreateEncoder(adjacenyMatrix,model=SimpleEncoder, output_size=2,max_steps=100000):
    #Generate Similairty Matrix
    similarityTensor = torch.Tensor(similarity.getSimilarityMatrix(adjacenyMatrix))
    #training cycle
    batch = torch.Tensor(adjacenyMatrix)
    encoder = model(len(adjacenyMatrix),output_size)
    #Define optimizer & Criterion
    optimizer = optim.Adam(encoder.parameters(), lr=0.001)
    criterion = my_loss
    loss_list = list()
    for i in range(max_steps):
        #zero gradient
        optimizer.zero_grad()
        #forward, backward, optimize
        output = encoder(batch)
        #print("Output:",output)
        loss = criterion(output,similarityTensor)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if i % 10000 == 0:
            average_loss = sum(loss_list)/len(loss_list)
            loss_list = []
            print("Loss:",average_loss)
            if average_loss < 1e-9:
                print(f"Early Break! [{i}/{max_steps}]")
                break
    return encoder

graph = graphs.randomGraph(3)
encoder = CreateEncoder(graph)
