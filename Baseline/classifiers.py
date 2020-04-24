import numpy as np
# You need to build your own model here instead of using well-built python packages such as sklearn
from collections import defaultdict

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZeor(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Add your code here!
        self.prior_log = defaultdict(float)     
        self.count_dict = defaultdict(float)
        self.MLH = defaultdict(float)
        self.num_features = None
        self.classes = None
        

    def fit(self, X, Y):
        # Add your code here!
        self.num_features = X.shape[1]
        num_train_samples = X.shape[0]
        self.classes = np.unique(Y)
        for i in self.classes: 
            class_samples = X[Y==i]
            class_feature_counts = np.sum(class_samples, axis=0)
            num_class_samples = class_samples.shape[0]
            self.prior_log[i] = np.log(num_class_samples/num_train_samples)

            #All V are the different feature indices
            for j in range(self.num_features):
                self.count_dict[(j,i)] = class_feature_counts[j]
                self.MLH[(j,i)] = (self.count_dict[(j,i)] + 1)/(np.sum(class_feature_counts) + self.num_features)       
    
    def predict(self, X):
        # Add your code here!
        args = {}
        predictions = np.zeros(X.shape[0])

        for index in range(X.shape[0]):
            sample = X[index,:]
            for i in self.classes:
                word_sum = self.prior_log[i]
                for j in range(X.shape[1]):
                    if sample[j] > 0:
                        word_sum += np.log(self.MLH[(j,i)])
                args[i] = word_sum

            argmax = self.classes[0]
            for key in args.keys():
                if args[key] > args[argmax]:
                    argmax = key

            predictions[index] = argmax

        return predictions

    def find_words(self, unigram):
        largest_words = []
        for i in range(self.num_features):
            largest_words.append((self.MLH[(i, 1)]/self.MLH[(i,0)], i))

        largest_words = sorted(largest_words, key=lambda x: -x[0])

        
        new_dict = {}
        for i in unigram.keys():
            new_dict[unigram[i]] = i

        words = largest_words[0:10]

        top_words = []
        for i in words:
            top_words.append(new_dict[i[1]])

        botwords = largest_words[-10:]

        bot_words = []
        for i in botwords:
            bot_words.append(new_dict[i[1]])

        print(top_words)
        print(bot_words)


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self, num_steps=1000, alpha=0.1, regularization=0.1):
        # Add your code here!
        self.num_steps = num_steps
        self.alpha = alpha
        self.regularization = regularization #chosen from [0.0001, 0.001, 0.01, 0.1, 1, 10]

        self.weights = None
    
    def log_likelihood(self, X, Y, weights):
        predictions = np.dot(X, weights)
        log_like = np.sum( Y*predictions - np.log(1 + np.exp(predictions)) )
        return log_like

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
                

    def fit(self, X, Y):
        # Add your code here!
        weights = np.ones(X.shape[1]) #starting weights
        
        for step in range(self.num_steps):
            prediction = np.dot(X, weights)
            predictions = self.sigmoid(prediction)

            error = Y - predictions
            grad = np.dot(X.T, error)
            weights += self.alpha * grad - 2 * self.regularization/self.num_steps * weights
            
        self.weights_vector = weights
        
    
    def predict(self, X):
        # Add your code here!
        pred = np.dot(X, self.weights_vector)
        final_predictions = self.sigmoid(pred)
        final_predictions = np.around(final_predictions,0)

        return final_predictions



