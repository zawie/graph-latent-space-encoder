import random
import matrix

def randomGraph(nodeCount,weighted=False,directed=False):
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
