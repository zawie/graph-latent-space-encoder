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

def BlankSquareMatrix(size):
    """
    Returns a blank square matrix (list of lists) where all the elements are None
    """
    mat = []
    for i in range(size):
        mat.append([None]*size)
    return mat
