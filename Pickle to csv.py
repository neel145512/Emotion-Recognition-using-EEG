import pickle
import numpy

with open('dataset/train', 'rb') as fo:         #Open file
    d = pickle.load(fo, encoding='bytes')       #Load pickle data
    train_data = d['data']                      #Load train data
    train_label = d['label']   
    
with open('dataset/test', 'rb') as fo:         #Open file
    d = pickle.load(fo, encoding='bytes')       #Load pickle data
    test_data = d['data']                      #Load train data
    test_label = d['label']   
    

numpy.savetxt("train_data.txt", train_data, delimiter=",")
numpy.savetxt("train_label.txt", train_label, delimiter=",")
numpy.savetxt("test_data.txt", test_data, delimiter=",")
numpy.savetxt("test_label.txt", test_label, delimiter=",")