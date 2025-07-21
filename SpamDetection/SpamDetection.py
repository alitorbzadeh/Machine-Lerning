pip install parsivar
from parsivar import Normalizer , Tokenizer , FindStems
import csv
from string import *
import xlwt
stop = []
spam=[]

with open("stop-words.txt", encoding="utf8") as data:
  for line in data:
    stop.append(word)
for u in range(0, len(stop)):
    p = stop[u]
    w = str(p).replace('\\u200c', ' ')
    s = str(w).replace('\\u200f', ' ')
    stop[u] = str(s).replace('\n', '')
    
with open("spam-words.txt", encoding="utf8") as q1:
  for word1 in q1:
    spam.append(word1)
for u1 in range(0, len(spam)):
    p1 = spam[u1]
    r1 = str(p1).replace('\\u200c', ' ')
    w1 = str(r1).replace('\u200c', ' ')
    s1 = str(w1).replace('\\u200f', ' ')
    spam[u1] = str(s1).replace('\n', '')
print(len(spam))

with open("sms_data.txt", "r+", encoding="utf16") as K:
  for i in K:
    spamcounter=0
    my_norm=Normalizer()
    my_token=Tokenizer()
    sms0=my_norm.normalize(i)
    sms1=my_token.tokenize_words(sms0)
    for i7 in stop:
      if i7 not in sms1:
        continue
      else:
        sms1.remove(i7)
    for i in spam:
      if i is  sms1:
        spamcounter=spamcounter+1
    prob_sp= spamcounter/len(sms1)
    if prob_sp>.5 :
      sms1.append("spam")
    else:
      sms1.append("not_spam")
    result=xlwt.Workbook()
    my_sheet=result.add_sheet('my data')
    for i in range(len(sms1)):
      my_sheet.write(0, i, sms1[i])
    result.save('classification1.xlsx')