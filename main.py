""""
Rob Jansen
Data Mining Project 3
"""

import numpy
from numpy import linalg
from scipy.cluster.vq import whiten
import sys

'''Begin MY functions'''
def inner_prod_of_cov(myData):
    length, width = numpy.shape(myData)
    #center the datamatrix
    mean_vec = numpy.mean(myData, axis = 0, dtype = float)
    ones_array = numpy.ones((length,width))
    center_data = myData - ones_array * mean_vec.T
    #normalize inner prod
    cov_prod = 1/(float(length)) * numpy.dot(center_data.T, center_data)
    return cov_prod
def pca_algorithm(myData, perserve_val):
    partial_variance = 0
    count = 0
    covariance = inner_prod_of_cov(myData)
    numpy_covariance = numpy.array(covariance)
    basis_eigen_value, basis_eigen_vector = linalg.eig(numpy_covariance)
    total_variance = numpy.sum(basis_eigen_value)

    for i in basis_eigen_value:
        partial_variance += i
        count +=  1
        if((partial_variance/float(total_variance)) > float(perserve_val)):
            break

    sorted_data_eigenval = numpy.argsort(basis_eigen_value)
    basis_eigen_vector_transpose = basis_eigen_vector.T

    #get N eigenvectors
    max = sorted_data_eigenval[len(sorted_data_eigenval)-1]
    largest =  basis_eigen_vector_transpose[max]
    vec_com = numpy.array([])
    vec_com = numpy.hstack((vec_com, largest))
    for i in range(2, len(sorted_data_eigenval)+1):
        max = sorted_data_eigenval[len(sorted_data_eigenval)-i]
        largest = basis_eigen_vector_transpose[max]
        vec_com = numpy.vstack((vec_com, largest))

    vec_com_transpose = vec_com.T
    projection = myData.dot(vec_com_transpose)

    #get the first 10
    proj_data = projection[:10,:]
    round_proj_data = numpy.round(proj_data, decimals = 3)
    print round_proj_data

    #return the reduced dimensionality
    return proj_data
def computeQuestion1(myData):
    covariance_matrix = inner_prod_of_cov(myData)
    data_transpose = myData.T
    covariance_matrix_numpy = numpy.cov(data_transpose, bias = 1)
    difference = numpy.subtract(covariance_matrix, covariance_matrix_numpy)
    print "QUESTION 1 - My covariance matrix subtracted from numpy's"
    print numpy.round(difference, decimals=3)
    print "  "
    print "  "
def computeQuestion2(myData):
    data_transpose = myData.T
    covariance_matrix_numpy = numpy.cov(data_transpose, bias = 1)
    eig_value, eig_vector = linalg.eig(numpy.array(covariance_matrix_numpy))

    #Sort the data
    sorted_data = numpy.argsort(eig_value)
    eig_vector_transpose = eig_vector.T

    #Get the two largest eigen vectors
    min = sorted_data[0]
    max = sorted_data[len(sorted_data) - 1]
    smallest = eig_vector_transpose[min]
    largest = eig_vector_transpose[max]

    #covariance_matrix_numpy_round = numpy.round(covariance_matrix_numpy,decimals = 3)
    #print covariance_matrix_numpy_round

    print "Yes, this covariance matrix is the matrix along the diagnol"
    print "Q2 - Print largest and smallest and dimension"
    print "Minimum Value: ", smallest
    print "Dimension: ",min
    print "Max Value: ", largest
    print "Dimension: ",max
    print "  "
    print "  "
    return
def computeQuestion3(myData):
    data_transpose = myData.T
    covariance_matrix_numpy = numpy.cov(data_transpose, bias = 1)
    eig_value, eig_vector = linalg.eig(numpy.array(covariance_matrix_numpy))

    #Sort the data
    sorted_data = numpy.argsort(eig_value)
    eig_vector_transpose = eig_vector.T

    #Get the two largest eigen vectors
    max1 = sorted_data[len(sorted_data) - 1]
    max2 = sorted_data[len(sorted_data) - 2]
    largest1 = eig_vector_transpose[max1]
    largest2 = eig_vector_transpose[max2]

    #combine the two largest eigenvectors
    combined_vec = numpy.vstack((largest1, largest2)).T
    projection = myData.dot(combined_vec)

    projected_variance = inner_prod_of_cov(projection)
    rounded_project_variance = numpy.round(projected_variance, decimals = 3)

    print "Question 3 - Use linalg.eig"
    print "Variance of datapoints in subspace: "
    print rounded_project_variance
    print "Largest eigen values: "
    print numpy.trace(rounded_project_variance)
    print "  "
    print "  "
    return eig_vector
def computeQuestion4(vector):
    round_vector = numpy.round(vector,decimals=3)
    print "Q4 = Covariance matrix E"
    print round_vector
    print "  "
    print "  "
def computeQuestion5():
    print "Q5 - see pca(data) method"
    print "  "
    print "  "
def computeQuestion6(myData):
    print "Q6 - using pc, preserving 90%"
    preserving = .9
    pca_arr = pca_algorithm(myData,preserving)
    print "  "
    print "  "
    return pca_arr
def computeQuestion7(reduced_dim_data):
    whitened_data = whiten(reduced_dim_data)
    print "Q7 - Whitening Transformation"
    print whitened_data
    print "  "
    print "  "
    return whitened_data
def computeQuestion8(w_data):
    print "Q8 - Special Matrix Class?"
    print "YES - THIS IS THE IDENTITY MATRIX"
    cov_white_data = inner_prod_of_cov(w_data)
    print cov_white_data
    print "  "
    print "  "
'''End UserDefine Fucntions'''

#input file, get only x number of cols

file = "magic04.data.txt"

if len(sys.argv) == 2 :
    points_file = sys.argv[1]

input_file = numpy.loadtxt(file, delimiter = ",", usecols = (0,1,2,3,4,5,6,7,8,9))
data = input_file

#Each method does a certain question
computeQuestion1(data)
computeQuestion2(data)
eigen_vector = computeQuestion3(data)
computeQuestion4(eigen_vector)
computeQuestion5()
pca_data = computeQuestion6(data)
white_data = computeQuestion7(pca_data)
computeQuestion8(white_data)






