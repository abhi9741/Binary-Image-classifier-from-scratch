import numpy as np
import h5py
import matplotlib.pyplot as plt
    
def load_dataset():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

#reshaping the training/test dataset into rowvectors
train_set_x = train_set_x_orig.reshape(64*64*3,209)
test_set_x = test_set_x_orig.reshape(64*64*3,50)
train_set_x.shape
test_set_x.shape
test_set_y.shape

m_train = len(train_set_x) #number of training examples
m_test  = len(test_set_x) #number of test examples
height = train_set_x_orig.shape[2] #height / width of each image

# print("number of training examples - ",m_train)
# print("number of test examples -",m_test)
# print("height/width of each image -",height)

#pre proccesing a data set
#figure out dimensions and shapes of the  data set
#reshpe datasets for convinience
#"standardize" or normalise the data
max_p=[]
for x in train_set_x :
    max_p.append(max(x))
maxp=max(max_p)
# print("max -",maxp)

min_p=[]
for x in train_set_x :
    min_p.append(min(x))
minp=min(max_p)
# print("min -",minp)

train_set_x = train_set_x/(maxp-minp)


def init_with_zero(dim) :
    w = np.zeros(shape=(dim,1))
    b = 0
    return w,b
init_with_zero(2)


def sigmoid(z):
    return 1/(1+np.exp(-z))

sigmoid(np.array([1,2,3,4]))

#calculating gradient , cost
def propagation(w,b,X,Y) :
    m = X.shape[1]
    Z = np.dot(w.T,X)+b
    A = sigmoid(Z)
    L = (Y*np.log(A)+(1-Y)*(np.log(1-A)))
    cost=(np.sum(L))*(-1/m) #m number of training examples
    
    #d/dw of cost function
    dw =(1/m)*(np.dot(X,(A-Y).T))
    #d/db of cost function
    db = (1/m)*(np.sum(A-Y))
#     print(dw)
#     print("---")
#     print(db)
    return dw,db,cost


#optimising cost function , Gradient Descent , upgrading parameters
def optimise(w,b,X,Y,max_iterations,learning_rate) :
    costs = []
    for iteration in range(max_iterations) :
        dw,db,cost = propagation(w,b,X,Y) 
        w = w-learning_rate*dw
        b = b-learning_rate*db
        costs.append(cost)
#         print("w shape -",w.shape)
        if iteration%100 == 0:
            print("cost at iteration ",iteration," is -",cost)
#     print("w -",w)
#     print("b -",b)
#     print("dw -",dw)
#     print("db -",db)
    return w,b,costs

dim=int(train_set_x.shape[0])

# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
w,b = init_with_zero(dim)
max_iterations = 2000
learning_rate = 0.005
# w,b,cost=optimise(w,b,train_set_x,train_set_y,max_iterations,learning_rate)
learning_rate = float(input("enter learning rate : "))
max_iterations = int(input("enter max num of iterations :"))
# w,b,cost=optimise(w, b, train_set_x, train_set_y,max_iterations , learning_rate )



dim=int(train_set_x.shape[0])

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
w,b = init_with_zero(dim)
# max_iterations = 2000
# learning_rate = 0.005
# w,b,cost=optimise(w,b,train_set_x,train_set_y,max_iterations,learning_rate)
w,b,costs=optimise(w, b, train_set_x, train_set_y,max_iterations , learning_rate )


def predict(w,b,X) :
    m = X.shape[1]
    y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]) :
        y_prediction[0,i] = 1 if A[0,i]>0.5 else 0
    
    return y_prediction

Y_prediction_test = predict(w,b,test_set_x) 
Y_prediction_train = predict(w,b,train_set_x)

print("-"*100)

print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))


costs = np.squeeze(costs)
xvalues=[]
for x  in range(1,max_iterations) :
    xvalues.append(x)
xvalues = np.squeeze(xvalues)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations in hundreds')
plt.title("learning rate = "+str(learning_rate))
plt.show()
