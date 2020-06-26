#Helper Matrix Functions
def printMat(matrix):
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

def squareMatrix(size,value=None):
    """
    Returns a blank square matrix (list of lists) where all the elements are None
    """
    mat = []
    for i in range(size):
        mat.append([value]*size)
    return mat

def vectorDistance(u,v):
    """
    Returns the distance between two vectors
    """
    assert(len(u) == len(v))
    n = len(u)
    total = 0
    for i in range(n):
        total += (u[i] - v[i])**2
    return total**(1/2)

def distanceMatrix(matrix):
    n = len(matrix)
    distanceMatrix = squareMatrix(n)
    for i in range(n):
        v0 = matrix[i]
        for j in range(n):
            v1 = matrix[j]
            distanceMatrix[i][j] = vectorDistance(v0,v1)
    return distanceMatrix
