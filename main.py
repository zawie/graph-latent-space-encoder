import similarity
import graphs
import matrix

g = graphs.randomGraph(3)
s = similarity.getSimilarityMatrix(g)
matrix.printMat(g)
matrix.printMat(s)
