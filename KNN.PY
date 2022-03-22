import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np


def distance(element,x):

	dist = 0

	for i in range(len(element)):

		dist += (element[i]-x[i])**2


	return dist**(0.5)


def most_close_element(element,x_train,y_train,y_verification):


	min_distance = float('inf')
	class_most_close = 0
	idx_most_close = 0


	for i in range(np.shape(x_train)[0]):

		current_distance = distance(element,x_train[i])

		if current_distance < min_distance and y_verification[i]:
			min_distance = current_distance
			class_most_close = y_train[i]
			idx_most_close = i


	y_verification[idx_most_close] = False

	return class_most_close,y_verification


def most_common(most_classes_close):
	return max(set(most_classes_close),key=most_classes_close.count)

def main():

	data = pd.read_csv('films.csv', names=['id', 'violencia','romance','acao','comedia'])

	x_train,x_test =  train_test_split(data,test_size=0.9,random_state=32,shuffle=True)

	y_train = list(x_train['violencia'])
	del x_train['violencia']

	y_test = list(x_test['violencia'])
	del x_test['violencia']


	x_train = normalize(x_train,axis=0)

	x_test = normalize(x_test,axis=0)

	k = 15
	acc = 0

	for i in range(np.shape(x_test)[0]):

		most_classes_close = [0]*k

		y_verification = [True]*len(y_train)

		for j in range(k):

			most_classes_close[j],y_verification = most_close_element(x_test[i],x_train,y_train,y_verification)

		if most_common(most_classes_close) == y_test[i]:
			acc += 1

	print('Acc: ',acc,'Rate: ',acc/np.shape(y_test)[0])

main()