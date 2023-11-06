# -*- coding: utf-8 -*-
"""
Created on Thu May  5 23:34:22 2022

@author: Ghada Raee
"""

import pandas as pd
#!/usr/bin/env python
import re
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

train_uninteresting = []
train_interesting = []

for i in range(160):
    emailS = data_values[i,0]
    #make every word into lowercase and remove non-key words
    emailS = str(emailS).lower()
    emailS = removeNonKey(emailS)
    if data_values[i,1]== 1.0 : #if intersting
        train_interesting.append(emailS)
    else:
        train_uninteresting.append(emailS)

total_uninteresting = len(train_uninteresting)
total_interesting = len(train_interesting)

#make a vocabulary of unique words that occur in known uninteresting emails
vocab_words_uninteresting = []
for emailss in train_uninteresting:
    emailss_as_list = emailss.split()
    for word in emailss_as_list:
        vocab_words_uninteresting.append(word)    
    
#to delete duplicates
vocab_unique_words_uninteresting = list(dict.fromkeys(vocab_words_uninteresting))

# make a vocabulary of unique words that occur in known interesting emails
vocab_words_interesting = []
for emailss in train_interesting:
    emailss_as_list = emailss.split()
    for word in emailss_as_list:
        vocab_words_interesting.append(word)
#to delete duplicates
vocab_unique_words_interesting = list(dict.fromkeys(vocab_words_interesting))

#number of uninteresting emails that has a specidic word
dict_unintersticity = {}
for w in vocab_unique_words_uninteresting:
    emails_with_w = 0    # counter
    for emailss in train_uninteresting:
        if w in emailss:
            emails_with_w+=1       
    unintersticity = (emails_with_w+1)/(total_uninteresting+1)
    dict_unintersticity[w] = unintersticity
#number of interesting emails that has a specidic word
dict_interecity = {}
total_interesting = len(train_interesting)
for w in vocab_unique_words_interesting:
    emails_with_w = 0     # counter
    for emailss in train_interesting:
        if w in emailss:
            emails_with_w+=1
    interecity = (emails_with_w+1)/(total_interesting+1)  # Smoothing applied
    dict_interecity[w.lower()] = interecity 

#compute probality of uninteresting
prob_uninteresting = len(train_uninteresting) / (len(train_uninteresting)+(len(train_interesting)))

#compute probality of interesting
prob_interesting = len(train_interesting) / (len(train_uninteresting)+(len(train_interesting)))

#reading test data
tests = []
for i in range(160,200):
    emailS = data_values[i,0]
    #make every word into lowercase and remove non-key words
    emailS = str(emailS).lower()
    emailS = removeNonKey(emailS)
    tests.append(emailS)

# split emails into distinct words
distinct_words_as_emailsss_test = []
for emailss in tests:
    emailss_as_list = str(emailss).split()
    senten = []
    for word in emailss_as_list:
        senten.append(word)
    distinct_words_as_emailsss_test.append(senten)

#Ignore the words that you haven’t seen in the labelled training data:
reduced_emailsss_test = []
for emailss in distinct_words_as_emailsss_test:
    words_ = []
    for word in emailss:
        if word in vocab_unique_words_uninteresting:
            words_.append(word)
        elif word in vocab_unique_words_interesting:
            words_.append(word)
    reduced_emailsss_test.append(words_)


def mult(list_) :        # function to multiply all word probs together 
    total_prob = 1
    for i in list_: 
         total_prob = total_prob * i  
    return total_prob
def Bayes(email):
    probsuninteresting=[]
    probsinteresting=[]
    for word in email:
        Pr_S = prob_uninteresting
        try:
            pr_WS = dict_unintersticity[word]
        except KeyError:
            pr_WS = 1/(total_uninteresting+1)  # Apply smoothing for word not seen in uninteresting training data, but seen in interesting training    
        Pr_H = prob_interesting
        try:
            pr_WH = dict_interecity[word]
        except KeyError:
            pr_WH = (1/(total_interesting+1))  # Apply smoothing for word not seen in interesting training data, but seen in uninteresting training
        prob_word_is_uninteresting_BAYES = (pr_WS*Pr_S)/((pr_WS*Pr_S)+(pr_WH*Pr_H))
        probsuninteresting.append(prob_word_is_uninteresting_BAYES)
        prob_word_is_interesting_BAYES = (pr_WH*Pr_H)/((pr_WS*Pr_S)+(pr_WH*Pr_H))
        probsinteresting.append(prob_word_is_interesting_BAYES)
    
    prob_of_being_uninteresting = mult(probsuninteresting)
    prob_of_being_interesting = mult(probsinteresting)
    if prob_of_being_uninteresting > prob_of_being_interesting:
        return 'uninteresting'
    elif prob_of_being_uninteresting <= prob_of_being_interesting:
        return 'interesting'
    prob_word_is_uninteresting_BAYES=0
    prob_word_is_interesting_BAYES =0

interestingActualToReality=0
uninterestingActualToReality=0
actualToReality=0
count=160
start = time.time()
for email in reduced_emailsss_test:
    if len(email) !=0: 
        all_word_probs = Bayes(email)
        #print("Classification result: ", all_word_probs)
        actual=''
        if data_values[count,1]== 1.0 :
            #print("Actual label: interesting")
            actual='interesting'
        else:
            #print("Actual label: uninteresting")
            actual='uninteresting'
        if actual ==  all_word_probs :  
            actualToReality+=1
    count+=1
accuracy = actualToReality/len(reduced_emailsss_test)
end = time.time()

print("Training data size\t: ",(len(train_uninteresting)+len(train_interesting)))
print("Test data size\t\t: ",len(tests))
print("Number correct\t\t: ", actualToReality)
print("Number wrong\t\t: ",(len(tests)-actualToReality))
print("Accuracy\t\t\t: ", (accuracy * 100),"%")
print("Time of execution\t:", (end-start))
    
