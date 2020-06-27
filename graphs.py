import random
import matrix

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

def readGraph(fileName):
    """
    Returns the adajency matrix of a given graph stored at data/file_name
    """
    pass

def KarateClub():
    return readGraph("KaraClub")

def SocialNetwork():
    return readGraph("SocialNetwork")


def Cycle(nodeCount):
    adjacenyMatrix = matrix.squareMatrix(nodeCount,value=0)
    for n in range(nodeCount):
        adjacenyMatrix[n][n] = 1
        if n+1 < nodeCount:
            adjacenyMatrix[n][n+1] = 1
            adjacenyMatrix[n+1][n] = 1
        else:
            adjacenyMatrix[n][0] = 1
            adjacenyMatrix[0][n] = 1
    return(adjacenyMatrix)


def Connected(nodeCount):
    return matrix.squareMatrix(nodeCount,value=1)

def Benzine(Hydrogen=True):
    #Create base carbon center
    mat = Cycle(6)
    if Hydrogen:
        for i in range(6):
            #Add Hydrogden to existing rows
            additional = [0]*6
            additional[i] = 1
            mat[i].extend(additional)
            #Add the extra rows
            new_row = [0]*12
            new_row[i] = 1
            mat.append(new_row)
        #Make Diagonal Ones
        for i in range(12):
            mat[i][i] = 1
    return mat
