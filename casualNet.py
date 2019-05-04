import xml.etree.cElementTree as ET
import pickle
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)
# tree = ET.ElementTree(file='copa-dev.xml')
# root = tree.getroot()
# for child_of_root in root:
#     print(child_of_root.tag, child_of_root.attrib)
class str2(str):
    def __repr__(self):
        # Allow str.__repr__() to do the hard work, then
        # remove the outer two characters, single quotes,
        # and replace them with double quotes.
        return ''.join(('"', super().__repr__()[1:-1], '"'))

class casualNet:
    def __init__(self, fileName, isload):
        self.lam = 0.8
        self.lemmatizer = WordNetLemmatizer()
        if isload == False:
            self.fileName = fileName
            self.posRecDict = dict()

            with open(self.fileName) as f:
                self.txtLines = f.readlines()
            f.close()

            if isload is False:
                self.edgesRec = list()
                count = 0
                for idx, line in enumerate(self.txtLines):
                    comp = line.split('\t')
                    if comp[0] in self.posRecDict:
                        numa = self.posRecDict[comp[0]]
                    else:
                        self.posRecDict[comp[0]] = count
                        count = count + 1
                        numa = self.posRecDict[comp[0]]

                    if comp[1] in self.posRecDict:
                        numb = self.posRecDict[comp[1]]
                    else:
                        self.posRecDict[comp[1]] = count
                        count = count + 1
                        numb = self.posRecDict[comp[1]]
                    numc = int(comp[2])
                    self.edgesRec.append(np.array([numa, numb, numc]))
                    if np.mod(idx, 1000000) == 0:
                        print("%f\% finished", idx / 62100000 * 100)
                self.edgesRec = np.stack(self.edgesRec, axis=0)
                self.savePickle()
        else:
            self.txtLines = pickle.load(open("textLines.p", "rb"))
            self.posRecDict = pickle.load(open("posRecDict.p", "rb"))
            self.edgesRec = pickle.load(open("edgesRec.p", "rb"))
        self.totNum = np.sum(self.edgesRec[:,2])
    def savePickle(self):
        with open('textLines.p', 'wb') as handle:
            pickle.dump(self.txtLines, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('posRecDict.p', 'wb') as handle:
            pickle.dump(self.posRecDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('edgesRec.p', 'wb') as handle:
            pickle.dump(self.edgesRec, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def calculateCS(self, word1c, word2c):
        if word1c[0] in self.posRecDict:
            word1 = word1c[0]
        else:
            word1 = word1c[1]

        if word2c[0] in self.posRecDict:
            word2 = word2c[0]
        else:
            word2 = word2c[1]
        if not ((word1 in self.posRecDict) and (word2 in self.posRecDict)):
            return 0
        num1 = self.posRecDict[word1]
        num2 = self.posRecDict[word2]
        seletor1 = self.edgesRec[:,0] == num1
        seletor2 = self.edgesRec[:,1] == num2
        p12 = np.sum(self.edgesRec[np.logical_and(seletor1, seletor2), 2]) / self.totNum
        alpha = 0.66
        if p12 == 0:
            return 0
        p1 = np.sum(self.edgesRec[seletor1, 2]) / self.totNum
        p2 = np.sum(self.edgesRec[seletor2, 2]) / self.totNum

        csnecl = np.log(p12 / np.power(p1, alpha) / p2)
        cssufl = np.log(p12 / p1 / np.power(p2, alpha))
        re = np.exp(self.lam * csnecl + (1-self.lam) * cssufl)
        return re
    def calculateSentenceCS(self, s1, s2):
        comp1 = s1.split(' ')
        comp1Process = [self.lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s1)]
        comp2 = s2.split(' ')
        comp2Process = [self.lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(s2)]
        re = 0
        for idx1, ele1 in enumerate(comp1):
            for idx2, ele2 in enumerate(comp2):
                # re = re + self.calculateCS([ele1, comp1Process[idx1]], [ele2, comp2Process[idx2]])
                re = re + self.calculateCS([comp1Process[idx1], comp1Process[idx1]], [comp2Process[idx2], comp2Process[idx2]])
        return re / (len(comp1) + len(comp2))
    def lookUpEntry(self, word):
        if word in self.posRecDict:
            print(self.posRecDict[word])
        else:
            print("Missing")

causalNetf = "CausalNet.txt"
cn = casualNet(causalNetf, False)
tree = ET.ElementTree(file='copa-test.xml')
root = tree.getroot()
predict = list()
gt = list()
for child_of_root in root:
    que = child_of_root[0].text
    ans1 = child_of_root[1].text
    ans2 = child_of_root[2].text

    que = re.sub(r'[^\w\s]','',que).lower()
    ans1 = re.sub(r'[^\w\s]', '', ans1).lower()
    ans2 = re.sub(r'[^\w\s]', '', ans2).lower()
    gt.append(int(child_of_root.attrib[ 'most-plausible-alternative']))
    if child_of_root.attrib['asks-for'] == 'effect':
        str1 = cn.calculateSentenceCS(que, ans1)
        str2 = cn.calculateSentenceCS(que, ans2)
    else:
        str1 = cn.calculateSentenceCS(ans1, que)
        str2 = cn.calculateSentenceCS(ans2, que)
    if str1 > str2:
        predict.append(1)
    else:
        predict.append(2)
    print(child_of_root.attrib['id'] + ' finished')
    predictT = np.array(predict)
    gtT = np.array(gt)
    print("correctness: %f" % (np.sum(predictT == gtT) / predictT.shape[0]))
predict = np.array(predict)
gt = np.array(gt)
print("correctness: %f" % (np.sum(predict == gt) / predict.shape[0]))
