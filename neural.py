# -*- coding: utf-8 -*-
"""
Created on Fri May  6 03:55:46 2022

@author: Ghada Raee
"""


from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import time


#reading training data file
data_file = pd.ExcelFile('Emails2.xlsx')
data_file = data_file.parse()
data_values = data_file.values


def removearticles(text):
    nonkey = {'a': '', 'an':'', 'and':'', 'the':'',
              'you': '', 'your':'', 'regards':'', 'wishes':'',
              'is': '', 'are':'', 'dear':'', 'can':'',
              'be': '', 'but':'', 'to':'', 'as':'',
              'if': '', 'of':'', 'in':'', 'on':'',
              'at': '', 'we':'', 'has':'', 'have':'',
              'by': '', 'for':'', 'please':'', 'that':'',
              'ourselves': '', 'hers':'', 'between':'', 'yourself':'',
              'again': '', 'there':'', 'about':'', 'once':'',
              'during': '', 'out':'', 'very':'', 'having':'',
              'with': '', 'they':'', 'own':'', 'some':'',
              'do': '', 'its':'', 'yours':'', 'such':'',
              'into': '', 'most':'', 'itself':'', 'other':'',
              'off': '', 'am':'', 'or':'', 'who':'',
              'from': '', 'him':'', 'each':'', 'themselves':'',
              'until': '', 'below':'', 'these':'', 'his':'',
              'through': '', 'nor':'', 'me':'', 'were':'',
              'her': '', 'more':'', 'himself':'', 'before':'',
              'this': '', 'down':'', 'should':'', 'our':'',
              'their': '', 'while':'', 'above':'', 'both':'',
              'up': '', 'ours':'', 'had':'', 'she':'',
              'all': '', 'no':'', 'when':'', 'any':'',
              'them': '', 'same':'', 'been':'', 'will':'',
              'does': '', 'yourselves':'', 'then':'', 'that':'',
              'because': '', 'what':'', 'over':'', 'why':'',
              'so': '', 'did':'', 'not':'', 'now':'',
              'under': '', 'he':'', 'herself':'', 'just':'',
              'where': '', 'too':'', 'only':'', 'myself':'',
              'which': '', 'those':'', 'i':'', 'after':'',
              'few': '', 'whom':'', 'being':'', 'hope':'',
              'theirs': '', 'my':'', 'doing':'', 'it':'',
              'how': '', 'further':'', 'here':'', 'than':'',
              
              }
    rest = []
    for word in text.split():
        if word not in nonkey:
            rest.append(word)
    return ' '.join(rest)
def removeNonKey(email):
    emailS = email
    emailS = emailS.replace(',','')
    emailS = emailS.replace('.','')
    emailS = emailS.replace(':','')
    emailS = emailS.replace(';','')
    emailS = emailS.replace('\"','')
    emailS = emailS.replace('”','')
    emailS = emailS.replace('”','')
    emailS = emailS.replace('!','')
    emailS = emailS.replace('?','')
    emailS = emailS.replace('(','')
    emailS = emailS.replace(')','')
    emailS = emailS.replace('*','')
    emailS = removearticles(emailS)
    emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    emailS = emoji_pattern.sub(r'', emailS) # no emoji
    return emailS

training_data = []
training_labels = []

for i in range(160):
    emailS = data_values[i,0]
    #make every word into lowercase and remove non-key words
    emailS = str(emailS).lower()
    emailS = removeNonKey(emailS)
    training_data.append(emailS)
    training_labels.append(data_values[i,1])

total_training= len(training_data)


#The frequency of occurrence of each word in the email
def get_count(text):
    wordCounts = dict()
    textArr = text.split()
    for word in textArr:
        if word in wordCounts:
            wordCounts[word] += 1
        else:
            wordCounts[word] = 1
    return wordCounts

#The euclidean distance is used to determine the similarity between two emails; the smaller the distance, the more similar.
#we're seeing how much the test email is similar to a training email
def euclidean_difference(test_WordCounts, training_WordCounts):
    total = 0
   
    for word in test_WordCounts:
        # if word is in both emails, calculate count difference, square it, and add to total
        if word in test_WordCounts and word in training_WordCounts:
            total += (test_WordCounts[word] - training_WordCounts[word])**2
          
            # to remove common words, to speed up processing in next for loop
            del training_WordCounts[word] 
            
        # if word in test email only, square the count and add to total
        elif word in test_WordCounts and word not in training_WordCounts:
            total += test_WordCounts[word]**2
        elif word not in test_WordCounts and word in training_WordCounts:
            total += training_WordCounts[word]**2

         
    return total**0.5

def get_class(selected_Kvalues):
    uninteresting_count = 0
    interesting_count = 0
    for value in selected_Kvalues:
        if value[0] == False or  value[0] == 0.0 or  value[0] == 'uninteresting':
            uninteresting_count += 1
        else:
            interesting_count += 1
    if uninteresting_count > interesting_count:
        return "uninteresting"
    else:
        return "interesting"
def knn_classifier(training_data, training_labels, test_data, K,tsize):
    result = []
    counter = 1
    
    # word counts for training email
    training_WordCounts = [] 
    
    for training_text in training_data:
            training_WordCounts.append(get_count(training_text))  
       
      
    for test_text in test_data:
        similarity = [] # List of euclidean distances
        
        test_WordCounts = get_count(test_text)  # word counts for test email
        
        # Getting euclidean difference 
        for index in range(len(training_data)):
           euclidean_diff = euclidean_difference(test_WordCounts, training_WordCounts[index])
           similarity.append([training_labels[index], euclidean_diff])
        
        # Sort list in ascending order based on euclidean difference
        similarity = sorted(similarity, key = lambda i:i[1])  
        
        # Select K nearest neighbours
        selected_Kvalues = [] 
        for i in range(K):
            selected_Kvalues.append(similarity[i])
        # Predicting the class of email
        result.append(get_class(selected_Kvalues))
        counter += 1
    return result
def accuracy_score(result,tests_labels):
    corr=0
    for i in range(len(tests_labels)):
        classi='interesting'
        if tests_labels[i] == False or tests_labels[i] == 0.0 or tests_labels[i] == 'interesting':
            classi='uninteresting'
        res ='interesting'
        if result[i] == False or result[i] == 0.0 or result[i] == 'uninteresting':
            res='uninteresting'
        
        if res == classi:
            corr+=1
    return (corr / len(tests_labels))
def get_k():

    # sample size of test emails to be tested.
    tsize = len(tests)
    
    K_accuracy = []
    for K in range(1,50, 2):
        result = knn_classifier(training_data, training_labels, tests, K, tsize) 
        accuracy = accuracy_score(result, tests_labels)
        K_accuracy.append([K, accuracy*100])
        #print("Training data size\t: ",len(training_data))
        #print("Test data size\t\t: ",len(tests))
        #print("K value\t\t\t\t: " ,K)
        #print("Number correct\t\t: ",int(accuracy * tsize))
        #print("Number wrong\t\t: ",int((1 - accuracy) * tsize))
        #print("Accuracy\t\t\t: ",(accuracy * 100))
        #print("\n\n")
    K_accuracy_sorted = sorted(K_accuracy, key = lambda i:i[1])
    #print(K_accuracy_sorted)
    BestKArr = max(K_accuracy_sorted, key = lambda i:i[1])
    #print("MAX: " + str(BestKArr))
    print("Best K to use: ", BestKArr[0] )
    print("\n\n")
    return BestKArr[0]




#reading test data
tests = []
tests_labels=[]
for i in range(160,200):
    emailS = data_values[i,0]
    #make every word into lowercase and remove non-key words
    emailS = str(emailS).lower()
    emailS = removeNonKey(emailS)
    tests.append(emailS)
    tests_labels.append(data_values[i,1])
start= time.time()
K = int(get_k())
result = knn_classifier(training_data, training_labels, tests, K, len(tests)) 
accuracy = accuracy_score(result, tests_labels)
end= time.time()
"""
for i in range(len(result)):
    actual='interesting'
    if tests_labels[i] == False:
        actual='uninteresting'
    print("Classified: ", result[i], "Actual: ", actual)
"""
print("Training data size\t: ",len(training_data))
print("Test data size\t\t: ",len(tests))
correct = accuracy * len(tests)
correct = round(correct,1)
print("Number correct\t\t: ", int(correct))
wrong = (1 - accuracy) * len(tests)
wrong = round(wrong,1)
print("Number wrong\t\t: ",int(wrong))
print("Accuracy\t\t\t: ", (accuracy * 100),"%")
print("Time of execution\t:", (end-start))

     
 
    
