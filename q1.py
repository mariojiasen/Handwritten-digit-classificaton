'''
Question	1	Skeleton	Code

Here	you	should	implement	and	evaluate	the	Conditional	Gaussian	classifier.
'''

import data
import numpy as np
#import	pyplot as plt
import	matplotlib.pyplot as plt
from PIL import Image 

def	compute_mean_mles(train_data,train_labels):
	'''
	Compute	the	mean	estimate	for	each	digit	class

	Should	return	a	numpy	array	of	size	(10,64)
	The	ith	row	will correspond	to the mean	estimate for digit class i
	'''
	
	means =	np.zeros((10,64))
	for i in range(len(train_labels)):
		k=int(train_labels[i])
		means[k]=means[k]+train_data[i]
	means=means/700
	
	return means

def	compute_sigma_mles(train_data,train_labels,means):
	'''
	Compute	the	covariance	estimate	for	each	digit	class

	Should	return	a	three	dimensional	numpy	array	of	shape	(10,64,	64)
	consisting	of	a	covariance	matrix	for	each	digit	class	
	'''
	covariances	=	np.zeros((10,64,64))
	
	for i in range(10):
		val=train_data[i*700:(i+1)*700][:]
		covariances[i]=np.cov(val.T)
		
		
				
	#covariances=covariances
	#	Compute	covariances
	
	return	covariances

def	generative_likelihood(digits,means,covariances):
	'''
	Compute	the	generative	log-likelihood:
		log	p(x|y,mu,Sigma)

	Should	return	an	n	x	10	numpy	array	
	'''
	out=np.zeros((1,10))
	for i in range(10):
		diffT=(digits-means[i]).reshape(64,1)
		diff=diffT.reshape(1,64)
		
		out[0][i]=-0.5*np.linalg.det(covariances[i])-0.5*diff.dot(np.linalg.inv(covariances[i])).dot(diffT)
	
	return	out

def	conditional_likelihood(digits,means,covariances):
	'''
	Compute	the	conditional	likelihood:

		log	p(y|x,	mu,	Sigma)

	This	should	be	a	numpy	array	of	shape	(n,	10)
	Where	n	is	the	number	of	datapoints	and	10	corresponds	to	each	digit	class
	'''
	
	return	None

def	avg_conditional_likelihood(digits,labels,means,covariances):
	'''
	Compute	the	average	conditional	likelihood	over	the	true	class	labels

		AVG(	log	p(y_i|x_i,	mu,	Sigma)	)

	i.e.	the	average	log	likelihood	that	the	model	assigns	to	the	correct	class	label
	'''
	cond_likelihood	=conditional_likelihood(digits,means,covariances)

	#	Compute	as	described	above	and	return
	return	None

def	classify_data(digits,means,covariances):
	'''
	Classify	new	points	by	taking	the	most	likely	posterior	class
	'''
	cond_likelihood=conditional_likelihood(digits,means,covariances)
	#	Compute	and	return	the	most	likely	class
	pass
def test_accuracy(test_data,test_labels,means,covariances):
	count=0
	for i in range(len(test_labels)):
		a=generative_likelihood(test_data[i],means,covariances)
		if np.argmax(a,axis=1)[0]==test_labels[i]:
			count=count+1
	return count/len(test_labels)	
	

def	main():
	train_data,	train_labels,test_data,	test_labels	=data.load_all_data('D:\\digit')
	#print(int(train_labels[0]))
	#	Fit	the	model
	means=compute_mean_mles(train_data,train_labels)
	#print(means[0])
	covariances	=compute_sigma_mles(train_data,train_labels,means)
	#print(covariances[0])
	#	Evaluation
	#a=np.array([1,2,3])
	#b=np.array([[1,2,3],[4,5,6]])
	#print(a-b)
	#a=np.array([[9,4],[3,6]])
	#a=a[a[:,1].argsort()]
	
	#print(generative_likelihood(test_data[0],means,covariances))
	print(test_accuracy(test_data,test_labels,means,covariances))
	#print()
	#a=np.array([1,2]).reshape(1,2)
	#b=np.array([1,2]).reshape(2,1)
	#print(b.dot(a))
	#hasi=Image.open('six.png').convert('L')
	#img=np.array(hasi.getdata(),np.uint8).reshape(1,64)/255*0.8
	
	#a=generative_likelihood(img,means,covariances)
	#print(img)
	#print(np.argmax(a,axis=1)[0])
	#plt.imshow(img.reshape(8,8))
	#plt.show()
if	__name__	==	'__main__':
	main()
