#Helper Matrix Functions
def printMatrix(matrix):
    """
    Prints a matrix (list of lists) in a readable format
    For debugging purposes only
    """
    txt = "["
    height = len(matrix)
    for i in range(height):
        row = str(matrix[i])
        if i > 0:
            row = " "+row
        txt += row
        if i < height-1:
            txt += "\n"
    print(txt+"]")

def createBlankSquareMatrix(size):
    """
    Returns a blank square matrix (list of lists) where all the elements are None
    """
    mat = []
    for i in range(size):
        mat.append([None]*size)
    return mat

#Similiarity functions
def directConnect(node0,node1,AdjacenyMatrix):
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

def randomWalk(node0,node1,AdjacenyMatrix):
    pass

#Similarity Matrix generators
def getDirectedSimiliarityMatrix(AdjacenyMatrix,similiarityFunction):
    """
    Returns a similiarity matrix of a given DIRECTED AdjacenyMatrix.
    """
    size = len(AdjacenyMatrix)
    #Create Blank Similiarity Matrix
    similarityMatrix = createBlankSquareMatrix(size)
    #Populate similiarty Matrix appropriately
    for y in range(size):
        for x in range(size):
            if x == y:
                #Identical nodes have similiart of 1
                similarityMatrix[x][y] == 1
            else:
                #Calculate similiarity for node pairs
                similarity = similiarityFunction(x,y,AdjacenyMatrix)
                similarityMatrix[y][x] = similarity
                similarityMatrix[x][y] = similarity
    return similarityMatrix

def getUndirectedSimiliarityMatrix(AdjacenyMatrix,similiarityFunction):
    """
    Returns a similiarity matrix of a given DIRECTED AdjacenyMatrix.
    """
    size = len(AdjacenyMatrix)
    #Create Blank Similiarity Matrix
    similarityMatrix = createBlankSquareMatrix(size)
    #Populate similiarty Matrix appropriately
    for y in range(size):
        #Similiarity between the same nodes will be 1
        similarityMatrix[y][y] = 1
        #Calculate similiarity for other nodes
        for x in range(y+1,size):
            similarity = similiarityFunction(x,y,AdjacenyMatrix)
            similarityMatrix[y][x] = similarity
            similarityMatrix[x][y] = similarity
    return similarityMatrix

printMatrix(getUndirectedSimiliarityMatrix([[1,1,1,0],[1,1,0,0],[1,0,1,0],[0,0,0,1]],sharedNeighbors))
