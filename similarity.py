import matrix
from numpy.random import choice

#Similiarity functions
def directConnections(node0,node1,AdjacenyMatrix):
    """
    Return a ratio between 0 and 1 representing how connected the nodes are
    """
    return AdjacenyMatrix[node0][node1]

def sharedNeighbors(node0,node1,AdjacenyMatrix):
    """
    Returns a ratio between 1 and 0 representing how shared their neighborly
    connections are.
    """
    total = 0
    size = len(AdjacenyMatrix)
    #Calculates how "shared" a neighbor edge is
    f = lambda x,y:1-(x-y)**2
    #Sum all the "sharedness" edges
    for node in range(size):
        total += f(AdjacenyMatrix[node0][node],AdjacenyMatrix[node1][node])
    #Return average sharedness
    return total/size

def randomWalk(node0,node1,AdjacenyMatrix,steps=100,walks=10):
    """
    Does a random walk through graph to determiante similarity between two nodes
    """
    visits = [0]*len(AdjacenyMatrix)
    for w in range(walks):
        pos = node0
        for s in range(steps):
            options = list(range(len(AdjacenyMatrix)))
            distribution = list(AdjacenyMatrix[pos])
            distribution[node0] = 0
            distribution = [x / sum(distribution) for x in distribution]
            next_node = choice(options, 1, p=distribution)[0]
            visits[next_node] += 1
            pos = next_node
    return visits[node1]/sum(visits)

#Similarity Matrix generators
def getSimilarityMatrix(AdjacenyMatrix,similarityFunction=sharedNeighbors,directed=False):
    """
    Returns a similiarity matrix of a given AdjacenyMatrix.
    """
    size = len(AdjacenyMatrix)
    #Create Blank Similiarity Matrix
    similarityMatrix = matrix.squareMatrix(size)
    #Populate similiarty Matrix appropriately
    for y in range(size):
        for x in range(0 if directed else y,size):
            if x == y:
                #Identical nodes have similiart of 1
                similarityMatrix[x][y] = 1
            else:
                #Generate Random Edge weight
                similarityMatrix[y][x] = similarityFunction(x,y,AdjacenyMatrix)
                if directed == False:
                    similarityMatrix[x][y] = similarityMatrix[y][x]
    return similarityMatrix
