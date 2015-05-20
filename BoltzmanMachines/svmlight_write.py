
import numpy as np

def svmlight_write(targets,features,filename):
	# svmlight_write(targets,features,filename)

	# Description: This function takes a vector of classification 
	# targets, a matrix of feature values, and a file name and
	# write the targets and features to the specified file in
	# svmlight format.

	# targets: A size (N,1) numpy matrix of class labels. Class labels must
	#          be integers between 1 and C where C is the number of
	#          classes.
	# features: A size (N,D) numpy matrix containing D feature values for 
	#           each data case
	# filename: Name of file to write to

	N = features.shape[0]
	D = features.shape[1]

	with open(filename,'w+') as fout:

		for n in range(N):
			fout.write('%d' % targets[n])

			for d in range(D):
				if abs(features[n,d] > 1e-3):
					fout.write(' %d:%f' % (d+1,features[n,d]))

			fout.write('\n')


if __name__ == '__main__':

	targets = np.random.randint(1,20,size=(100,1))
	features = np.random.random(size=(100,500))
	svmlight_write(targets,features,'test.txt')



  
