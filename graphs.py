import random
import matrix

#Pre-made graphs
def readGraph(fileName):
    """
    Returns the adajency matrix of a given graph stored at data/file_name
    """
    pass

def KarateClub():
    return readGraph("KarateClub")

#Random graphs
def Random(nodeCount,weighted=False,directed=False):
    """
    Generates and returns a random adjaceny matrix
    """
    adjacenyMatrix = matrix.squareMatrix(nodeCount)
    for y in range(nodeCount):
        for x in range(0 if directed else y,nodeCount):
            if x == y:
                #Identical nodes have similiart of 1
                adjacenyMatrix[x][y] = 1
            else:
                #Generate Random Edge weight
                adjacenyMatrix[y][x] = random.random() if weighted else round(random.random())
                if directed == False:
                    adjacenyMatrix[x][y] = adjacenyMatrix[y][x]
    return adjacenyMatrix

#Snakes
def Snake(nodeCount):
    adjacenyMatrix = matrix.squareMatrix(nodeCount,value=0)
    for n in range(nodeCount):
        adjacenyMatrix[n][n] = 1
        if n+1 < nodeCount:
            adjacenyMatrix[n][n+1] = 1
            adjacenyMatrix[n+1][n] = 1
    return adjacenyMatrix

def DoubleCrossSnake(nodeCount):
    pass

def TripleCrossSnake(nodeCount):
    pass

#Cycle Structures
def Cycle(nodeCount):
    adjacenyMatrix = Snake(nodeCount)
    adjacenyMatrix[nodeCount-1][0] = 1
    adjacenyMatrix[0][nodeCount-1] = 1
    return adjacenyMatrix

def CrossedCycle(nodeCount):
    mat = Cycle(nodeCount)
    mat[0][nodeCount//2] = 1
    mat[nodeCount//2][0] = 1
    return mat

def DoubleCrossedCycle(nodeCount):
    mat = Cycle(nodeCount)
    #First Cross
    mat[0][nodeCount//2] = 1
    mat[nodeCount//2][0] = 1
    #Second Cross
    mat[nodeCount//4][nodeCount//2+nodeCount//4] = 1
    mat[nodeCount//2+nodeCount//4][nodeCount//4] = 1
    return mat

def Connected(nodeCount):
    return matrix.squareMatrix(nodeCount,value=1)

#Molecules
def Benzine(Hydrogen=True):
    #Create base carbon center
    mat = Cycle(6)
    if Hydrogen:
        #Make carbons pull towards each other more
        #This is to motivate hydrogens to be on "outside" of cycle
        """for i in range(6):
            for j in range(6):
                if mat[i][j] == 0:
                    mat[i][j] = 0.2"""
        #Add Hydrogens
        for i in range(6):
            #Add Hydrogden to existing rows
            additional = [0]*6
            additional[i] = 1
            mat[i].extend(additional)
            #Add the extra rows
            new_row = [0]*6 + [0]*6
            new_row[i] = 1
            mat.append(new_row)
        #Make Diagonal Ones
        for i in range(12):
            mat[i][i] = 1
    print(mat)
    return mat
