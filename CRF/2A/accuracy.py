import sys
import os

count  = 1
for word in open("data/test_words.txt", "r"):
     
    os.system("python sumProduct.py data/test_img"+str(count)+".txt "+str(word.strip('\n')))
    count+=1
    if count == 6:
        break

