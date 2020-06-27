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

def randomWalk(root,AdjacenyMatrix,steps=100,walks=10):
    """
    Does a random walk through graph to determiante similarity between two nodes
    """
    visits = [0]*len(AdjacenyMatrix)
    for w in range(walks):
        pos = root
        for s in range(steps):
            options = list(range(len(AdjacenyMatrix)))
            distribution = list(AdjacenyMatrix[pos])
            distribution[pos] = 0
            distribution = [x / sum(distribution) for x in distribution]
            next_node = choice(options, 1, p=distribution)[0]
            visits[next_node] += 1
            pos = next_node
    visits[root] = 0
    print( [int(100*v / sum(visits)) for v in visits] )
    return [v / sum(visits) for v in visits]

#Similarity Matrix generators
def getSimilarityMatrix(AdjacenyMatrix,similarityFunction=sharedNeighbors,directed=False):
    """
    Returns a similiarity matrix of a given AdjacenyMatrix.
    """
    size = len(AdjacenyMatrix)
    similarityMatrix = list()
    if similarityFunction == randomWalk:
        #Do a random walk for each node to determine similiarity
        for node in range(size):
            row = randomWalk(node,AdjacenyMatrix,steps=size**2)
            similarityMatrix.append(row)
        #Make diagonal all ones
        for i in range(size):
            similarityMatrix[i][i] = 1
    else:
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
