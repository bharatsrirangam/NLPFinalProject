import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse
from sklearn.model_selection import train_test_split


####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def accuracy(pred, labels):
    tp = 1
    fp = 1
    fn = 1
    tn = 1
    nppred = np.array(pred)
    nplabels = np.array(labels)
    for i in range(len(pred)):
        # print((nplabels[i]), nppred[i], int(nplabels[i]), int(nppred[i]), (int(nplabels[i]) == int(nppred[i])), (int(nppred[i]) is 1))
        if int(nplabels[i]) == 1:
            if int(nppred[i]) == 1:
                tp += 1
            else: 
                fn += 1
        else:
            if int(nppred[i]) == 1:
                fp += 1
            else: 
                tn += 1

    p = tp/(tp + fn)
    r = tp/(tp + fp)
    f1 = 2 * (p * r)/(p + r) 
    # correct = (np.array(pred) == np.array(labels)).sum()
    # accuracy = correct/len(pred)
    # print(pred, len(pred), tp + fp + fn + tn)
    # print(np.array(labels))

    print("Accuracy: %i / %i = %.4f " %(tp + tn, len(pred), (tp+tn)/len(pred)))
    print("Precision: %.4f" %(p))
    print("Recall: %.4f" %(r))
    print("F1_Score: %.4f" %(f1))


def read_data(path):
    train_frame = pd.read_csv(path + 'clean_train_data.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'test.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame


def main():
    split_ratio = 0.75
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='LogisticRegression',
                        choices=['AlwaysPredictZero', 'NaiveBayes', 'LogisticRegression'])
    parser.add_argument('--feature', '-f', type=str, default='unigram',
                        choices=['unigram', 'bigram', 'customized'])
    parser.add_argument('--path', type=str, default = './data/', help='path to datasets')
    args = parser.parse_args()
    print(args)

    train, test_frame = read_data(args.path)
    train['text'].fillna("*", inplace=True)

    split = int(len(train)*split_ratio)

    test_frame = train.iloc[split:, :]
    test_frame = test_frame.reset_index(drop=True)
    train_frame = train.loc[:split, :]


    # print(train_frame)

    # Convert text into features
    if args.feature == "unigram":
        feat_extractor = UnigramFeature()
    elif args.feature == "bigram":
        feat_extractor = BigramFeature()
    elif args.feature == "customized":
        feat_extractor = CustomFeature()
    else:
        raise Exception("Pass unigram, bigram or customized to --feature")


    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(train_frame['text'][i]))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']

    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']


    if args.model == "AlwaysPredictZero":
        model = AlwaysPreditZeor()
    elif args.model == "NaiveBayes":
        model = NaiveBayesClassifier()
    elif args.model == "LogisticRegression":
        model = LogisticRegressionClassifier()
    else:
        raise Exception("Pass AlwaysPositive, NaiveBayes, LogisticRegression to --model")


    start_time = time.time()
    model.fit(X_train,Y_train)
    print("===== Train Accuracy =====")
    accuracy(model.predict(X_train), Y_train)
    
    print("===== Test Accuracy =====")
    accuracy(model.predict(X_test), Y_test)

    print("Time for training and test: %.2f seconds" % (time.time() - start_time))



if __name__ == '__main__':
    main()