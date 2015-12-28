import math
from os import listdir
from os.path import isfile, isdir, join
from os import getcwd
import copy
import matplotlib.pyplot as plt
import numpy as np
import random

#Note: Some of the above libraries are not in the standard python libraries.

#Vectorizes a string by the "trigram" method.
def trivectorize(string):
    trigram = ''
    docvector = dict()
    for index in range(0, len(string)-2, 1):
        trigram = string[index] + string[index + 1] + string[index + 2]
        if trigram in docvector:
            docvector[trigram] += 1
        else:
            docvector[trigram] = 1
    return docvector

#Vectorizes a string by word frequency.
def wordvectorize(string):
    word = ''
    docvector = dict()
    ignoredpunct = ['!', '"', '#', '$', '%', '(', ')', '*', '+', ',', '.', '?', '/', ':', ';', '<', '>', '=', '[', ']', '\\', '{', '}', '|', '^', '~', '_', '`']
    for index in range(0, len(string)):
        if string[index] != ' ':
            if string[index] in ignoredpunct:
                pass
            else:
                word += string[index]
        else:
            if word != '':
                if word in docvector:
                    docvector[word] += 1
                else:
                    docvector[word] = 1
            word = ''
    return docvector

#Vectorizes a string by the "letterhop" algorithm:
#   Each character is a key in the dictionary, and is assigned a number based on
#   the average distance between each instance of that character in the string.
def lhvectorize(string):
    docvector = dict()
    letterlist = []
    start = 0
    startlist = []
    for letter in string:
        if letter not in letterlist:
            letterlist.append(letter)
            startlist.append(start)
        start += 1

    stringlen = len(string)
    elnum = 0
    for element in letterlist:
        index = 0
        count = 0
        total = 0
        for x in range(startlist[elnum], stringlen):
            if string[x] == element:
                total += index
                count += 1
                index = 0
            else:
                index += 1
        total = float(total) // float(count)
        docvector[element] = total
    return docvector

#Files are vectorized using one or a mixture of vectorizations defined.
def vectorizefile(method, filename, newfilename = None):
    if method == "trigram":
        filevector = trivectorize(filename.read())
        filevector = HTnormalize(filevector)
    elif method == "word":
        filevector = wordvectorize(filename.read())
        filevector = HTnormalize(filevector)
    elif method == "tri+word":
        trigramvector = trivectorize(filename.read())
        filevector = wordvectorize(filename.read())
        filevector.update(trigramvector)
        filevector = HTnormalize(filevector)
        #
        #note that in this implementation, any three-letter entries shared
        #by both methods are resolved to the trigram method's value
        #
    elif method == "letterhop":
        filevector = lhvectorize(filename.read())
        filevector = HTnormalize(filevector)
    elif method == "tri+lh":
        trigramvector = trivectorize(filename.read())
        filevector = lhvectorize(filename.read())
        filevector.update(trigramvector)
        filevector = HTnormalize(filevector)
    try:
        if newfilename != None:
            newfilename.write(str(filevector))
    except (ValueError, NameError, TypeError):
        print("Error with method entered, please pick from the following methods: 'trigram', 'word', 'tri+word', 'letterhop', 'tri+lh'")
    return filevector

#This is an interface function used by the method loadFormattedFile of
#profileDataSet that loads into the structure an "HT" type file: many
#documents with the format: ID, tab(\t), document content, newline(\n).
#Each document with the same ID will be loaded into the same dataProfile.
class HTFormatter():

    def myformat(self, string):
        formattedlist = dict()
        formatted = ""
        currentID = ""
        mystery = ""
        valid = ['0','1','2','3','4','5','6','7','8','9']
        for index in range(0, len(string)-1):
            if string[index] == "\t":
                if (mystery[0] in valid) & (mystery[1] in valid):
                    currentID = mystery
                    mystery = ""
                    if currentID in formattedlist:
                        formattedlist[currentID] = formattedlist[currentID] + formatted
                    else:
                        formattedlist[currentID] = formatted
                    currentID = ""
                    formatted = ""
                else:
                    mystery = mystery + string[index]
            elif string[index] == "\n":
                formatted = formatted + mystery
                mystery = ""
            else:
                mystery = mystery + string[index]
        return formattedlist

#Removes unwanted characters from keys in a dictionary, and removes keys with
#capitalization by forcing lowercase, adding their values to the lowercase
#version of the key.
def HTnormalize(sampledict):
    rejectionlist = ['@', '!', '"', '#', '$', '%', '(', ')', '*', '+', ',', '.', '?', '/', ':', ';', '<', '>', '=', '[', ']', '\\', '{', '}', '|', '^', '~', '_', '`', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    newdict = copy.deepcopy(sampledict)
    for sample in sampledict:
        samplereject = 0
        normsample = ''
        for letter in sample:
            normletter = letter
            if samplereject == 1:
                break
            elif letter in rejectionlist:
                samplereject = 1
                break
            elif (ord(letter) > 64) & (ord(letter) < 91):
                normletter = chr(ord(letter)+32) #check this
            normsample = normsample + normletter
        if samplereject == 1:
            del newdict[sample]
        else:
            if normsample != sample:
                try:
                    tempval = newdict[normsample]
                except KeyError:
                    tempval = 0
                newdict[normsample] = tempval + newdict[sample]
                del newdict[sample]
    return newdict


#The dataProfile class wraps different file vectors for passing into
#profileDataSet, and performs relevant computations on vectors.
class dataProfile:

    def __init__(self, method, filename, pvector = ''):  #second argument allows passing directly as vector
        if filename == 'pass as vector':
            self._vector = pvector
            self._name = 'RAW_VECTOR'
        else:
            filetemp = open(filename, 'r', encoding="utf8")
            self._vector = vectorizefile(method, filetemp)
            self._name = filename

    def __str__(self):
        return self._name

    def setName(self, newname):
        self._name = newname

    #Retrieves the vector of this instance.
    def getVector(self):
        vectorgotten = self._vector
        return vectorgotten

    #Retrieves value for a given key; if the key is not in this instance's,
    #vector returns 0.
    def getCoord(self, key):
        if key not in self._vector:
            return 0
        else:
            return self._vector[key]

    #Computes the Euclidean distance between this instance and another instance
    #of dataProfile.
    def dist(self, other):
        keylist = list(set(self._vector.keys()) | set(other._vector.keys()))
        distancescalar = 0
        for key in keylist:
            dimen = self.getCoord(key) - other.getCoord(key)
            dimen = dimen**2
            distancescalar += dimen
        distancescalar = math.sqrt(distancescalar)
        return distancescalar

#The profileDataSet class contains references to a set of dataProfile
#instances, and provides tools for their analysis.
class profileDataSet:

    def __init__(self):
        self._profileList = []
        self._masterKeyList = []

    #Adds an instance of dataProfile to this set.
    def addProfile(self, profile):
        self._profileList.append(profile)
        self._masterKeyList = list(set(self._masterKeyList) | set(profile.getVector().keys()))

    #Vectorizes all files (not of extension .py) and loads them into this
    #instance of profileDataSet.
    def loadCurrentDir(self, method):
        #Available methods: "trigram", "word", "tri+word", "letterhop"
        currentdir = getcwd()
        files = []
        for f in listdir(currentdir):
            if isfile(join(currentdir, f)):
                files.append(f)
            elif isdir(f):
                for e in listdir(f):
                    if isfile(join(f, e)):
                        files.append(e)

        #Files in the directory are chosen and added.
        for filename in files:
            if '.py' in filename:  #This means it only skips over .py files; this can be modified to fit different sorts of directories.
                pass
            else:
                self.addProfile(dataProfile(method, filename))

    #Loads a file into the data set with a certain formatting method
    #interfaced as formatter.
    def loadFormattedFile(self, formatter, filename, method): #Inefficient to comply with dataProfile initialization implementation.
        unformattedfile = open(filename, 'r', encoding = "utf8")
        formattedlist = formatter.myformat(unformattedfile.read())
        unformattedfile.close()
        print(len(formattedlist))
        for ID in formattedlist:
            temp = open('temp.txt', 'w+', encoding = "utf8")
            temp.write(formattedlist[ID])
            temp.close()
            profile = dataProfile(method, 'temp.txt')
            profile.setName(ID)
            self.addProfile(profile)

    #Returns a displayable list of the current profiles as strings.
    def __str__(self):
        strlist = []
        for profile in self._profileList:
            strlist.append(str(profile))
        return str(strlist)

    #Returns the current number of profiles (length of profileList).
    def getLength(self): #check necessity
        return len(self._profileList)

    #Retrieves the list of current profiles.
    def getprofileList(self):
        listgotten = copy.deepcopy(self._profileList)
        return listgotten

    def getmasterKeyList(self):
        listgotten = copy.deepcopy(self._masterKeyList)
        return listgotten

    #Computes the average of all current profiles and returns this as a profile.
    def getMean(self):
        #Running time proportional to (number of unique characters overall) *
        #(number of profiles in data set).
        meanVector = dict((key, 0) for key in self._masterKeyList)
        for meankey in meanVector:
            for vector in self._profileList:
                if meankey in vector.getVector():
                    meanVector[meankey] += vector.getVector()[meankey]
            meanVector[meankey] = float(meanVector[meankey]) / float(self.getLength())
        meanWrapper = dataProfile("vector", 'pass as vector', meanVector)
        return meanWrapper

    #Performs a k-means clustering algorithm on the profiles in this set, and
    #returns list containing first a list of the centroid assignments for each
    #profile, and second the average Euclidean distance between profiles and
    #their assigned centroid mean.
    def kCluster(self, k):
        #Streamlining of the k=1 case...
        if k == 1:
            allmean = self.getMean()
            ADCC = 0
            index = 0
            for vector in self._profileList:
                vdist = vector.dist(allmean)
                ADCC += vdist * vdist
                index += 1
            ADCCk = float(ADCC) / float(index)
            return ["k set to 1; All vectors are in the same cluster.", ADCCk]
        #And here the k>1 case.
        else:
            IDList = []
            for vector in self._profileList:
                IDList.append(1)
            meanList = []
            index = 0
            rintlist = []
            while len(meanList) < k:  #setting initial mean vectors
                rint = random.randint(0, len(self._profileList)-1)
                if rint in rintlist: #Is this check necessary? Can I skip this line?
                    while rint in rintlist:
                        rint = random.randint(0, len(self._profileList)-1)
                rintlist.append(rint)
                meanList.append(self._profileList[rint])
                index += 1
            index = 0
            for vector in self._profileList:  #setting initial clusters
                lowestdist = vector.dist(meanList[0])
                for mean in meanList:
                    tempdist = vector.dist(mean)
                    if tempdist < lowestdist:
                        lowestdist = tempdist
                        IDList[index] = meanList.index(mean) + 1
                    else:
                        pass
                index += 1

            loopiterator = 0
            completion = 0
            while completion == 0:  #primary clustering loop
                oldIDList = copy.deepcopy(IDList)
                oldMeanList = copy.deepcopy(meanList)
                for clusterID in range(1, k+1, 1):  #computing new meanList
                    clusterSet = profileDataSet()
                    for index in range(len(IDList)):
                        if IDList[index] == clusterID:
                            clusterSet.addProfile(self._profileList[index])
                        else:
                            pass
                    meanList[clusterID-1] = clusterSet.getMean()

                index = 0
                ADCCk = 0
                for vector in self._profileList:  #assigning new IDs
                    lowestdist = vector.dist(meanList[0])
                    for mean in meanList:
                        tempdist = vector.dist(mean)
                        if tempdist <= lowestdist:  #<= for lowestdist == case
                            lowestdist = tempdist
                            IDList[index] = meanList.index(mean) + 1
                        else:
                            pass
                    index += 1
                    ADCCk += lowestdist * lowestdist
                ADCCk = float(ADCCk) / float(len(IDList))

                oldmeans = []
                for profile in oldMeanList:
                    oldmeans.append(profile.getVector())
                newmeans = []
                for profile in meanList:
                    newmeans.append(profile.getVector())

                loopiterator += 1
                if (oldmeans == newmeans) & (oldIDList == IDList):
                    completion = 1
                print(loopiterator)
                print(ADCCk)

            Assignments = dict()
            counter = 0
            for profileID in IDList:
                Assignments[str(self._profileList[counter])] = profileID
                counter += 1
                
            return [Assignments, ADCCk]


    #Performs a series of kCluster trials for k = 1 to k = maxk, maxk being
    #the input argument. Prints a list of the average distances between
    #vectors and their centroid mean, and plots these according to k value.
    def kAnalysis(self, maxk):
        averagelist = []
        for k in range(1, maxk+1):
            clusteringlist = self.kCluster(k)
            clusterID = clusteringlist[1]
            averagelist.append(clusterID)
            print(k, ' done.')

        print(averagelist)
        plt.plot(np.arange(1, maxk + 1), np.array(averagelist))
        plt.xlabel('k')
        plt.ylabel('Average Distance from Cluster Center')
        plt.show()

    #Computes the entropies of each sample in the data set with respect to the
    #individual groups determined by earlier assignment. Each sample entropy
    #value is contained in a dictionary for its group, and these dictionaries
    #are each contained in a list "entropylist" indexed by the list "grouplist".
    def sampleEntropyAnalysis(self, assignments, samples):
        entropylist = []
        grouplist = []

        #builds a list of group IDs
        for ID in assignments:
            if assignments[ID] not in grouplist:
                grouplist.append(assignments[ID])

        #computes the entropies for samples in each group
        for group in grouplist:
            sample_ent = dict()
            currentset = profileDataSet()
            for profile in self._profileList:
                if assignments[str(profile)] == group:
                    currentset.addProfile(profile)

            #check frequency of each sample in the given group
            total = currentset.getLength()
            for sample in samples:
                has_sample = 0
                for profile in currentset._profileList:
                    if sample in profile._vector:
                        has_sample += 1
                prob = float(has_sample) / float(total)
                if (prob > 0) & (prob < 1):
                    has = prob * math.log(prob, 2) * -1
                    hasnt = (1 - prob) * math.log((1 - prob), 2) * -1
                    entropy = has + hasnt
                else:
                    entropy = 0
                sample_ent[sample] = entropy

            print(group)    
            entropylist.append(sample_ent)

        return [entropylist, grouplist]
                        
    #returns the x most frequent values in each profile of the set (allowed being x).
    def getTopValues(self, allowed):
        topList = []
        count = 0
        for profile in self._profileList:
            count = count + 1
            vals = list(profile.getVector().values())
            maxval = 0
            for i in range(0, allowed+1):
                maxval = max(vals)
                vals.remove(maxval)
            for key in profile.getVector():
                if profile.getVector()[key] > maxval:
                    topList.append(key)
            
        #eliminating repeats
        compList = []
        for sample in topList:
            if sample not in compList:
                compList.append(sample)
        return compList

    #Computes the percent of documents each cluster that possesses each sample,
    #in list form organized by sample.
    def getDistrib(self, assignments, sample):

        #creating dict of possible assignments and inverse dict,
        #to catalogue occurrence.
        possassign = dict()
        for num in assignments.values():
            if num not in possassign:
                possassign[num] = 0
        nosample = copy.deepcopy(possassign)

        #incrementing possassign and nosample keys based on sample occurrence
        #--remember that calls to the assignments dictionary depends on unique
        #to_str for each profile
        for profile in self._profileList:
            if sample in profile._vector:
                oldvalue = possassign[assignments[str(profile)]]
                possassign[assignments[str(profile)]] = oldvalue + 1
            else:
                oldvalue = nosample[assignments[str(profile)]]
                nosample[assignments[str(profile)]] = oldvalue + 1

        #returning percent of each cluster in possession of the sample
        anslist = []
        for num in possassign:
            percent = (possassign[num]/(possassign[num] + nosample[num])) * 100
            anslist.append(percent)
        return anslist

    #Returns the samples from a list that can differentiate between groups
    #by at least a minimum value of percentage. This is for sets divided into 3
    #groups only.
    def findSignal3(self, assignments, samples, minimum):
        returns = []
        for sample in samples:
            maxdiff = 0
            [a, b, c] = self.getDistrib(assignments, sample)
            diffs = [0, 0, 0]
            diffs[0] = abs(a-b) + abs(a-c)
            diffs[1] = abs(b-a) + abs(b-c)
            diffs[2] = abs(c-a) + abs(c-b)
            if diffs[0] == max(diffs):
                maxdiff = diffs[0]
            elif diffs[1] == max(diffs):
                maxdiff = diffs[1]
            else:
                maxdiff = diffs[2]
            if maxdiff >= minimum:
                returns.append(sample)
        return returns

    #Multiplies together the getDistrib percents for each sample given in
    #signals. This is one way to try to predict a most likely group for
    #a given document.
    def computeLikely(self, assignments, signals):
        likelyhoods = []
        for sample in signals:
            percents = self.getDistrib(assignments, sample)
            if len(likelyhoods) == 0:
                likelyhoods = [1, 1, 1]
            for i in range(0, len(percents)):
                likelyhoods[i] = likelyhoods[i] * (percents[i] / 100)
        return likelyhoods

    #Uses the computeLikely method to make grouping guesses for the current set
    def testLikely(self, assignments, signals):
        accuracy = 0
        for profile in self._profileList:
            relsignals = []
            for sample in signals:
                if sample in profile._vector:
                    relsignals.append(sample)
            likelyhoods = self.computeLikely(assignments, relsignals)
            guess = likelyhoods.index(max(likelyhoods)) + 1 #problems if two probability are the same value
            if guess == assignments[str(profile)]:
                accuracy += 1
        return (accuracy / self.getLength())

    #Tool to 1) not make me type so much and 2) remove extraneous signals from
    #data computed in testLikely
    def mydriver(self):
        self.loadFormattedFile(HTFormatter(), 'HT2.txt', 'trigram')
        top = self.getTopValues(50)
        assign = {'218182525': 3, '161382077': 2, '139502884': 2, '228805380': 3, '221416764': 3, '213449543': 1, '207976054': 1, '127904846': 3, '154943520': 3, '192278205': 1, '30802458': 3, '214534063': 3, '186108835': 1, '182074125': 2, '219465014': 3, '269579419': 3, '282723907': 2, '205774290': 1, '18762952': 2, '185279493': 1, '141344968': 2, '255212821': 3, '176788137': 1, '259726736': 3, '415938467': 3, '89711195': 3, '118487240': 3, '105899913': 3, '48609272': 3, '102422722': 3, '124843030': 3, '230023205': 3, '122969617': 3, '233961489': 3, '202275056': 3, '164331380': 2, '170885603': 3, '190091698': 1, '139459981': 3, '165989800': 3, '223045155': 2, '220007368': 1, '79773639': 2, '62705931': 3, '250284844': 1, '187711461': 3, '221105713': 3, '141630586': 3, '171591427': 3, '226832004': 3, '147672196': 3, '239921610': 3, '236070767': 3, '48418841': 3, '220393904': 3, '240355281': 1, '123769512': 3, '209558466': 3, '176546944': 2, '98259023': 3, '226761031': 3, '187922653': 3, '119217451': 3, '204180553': 1, '257554587': 1, '82291724': 3, '138553666': 3, '28672096': 3, '52629824': 2, '197951258': 3, '126310173': 3, '140163817': 2, '268421034': 3, '257687913': 3, '135596812': 2, '187179654': 1, '153141908': 2, '90265127': 3, '176664846': 1, '246850296': 2, '34129443': 3, '202240295': 3, '238277118': 3, '138562652': 2, '189699355': 1, '249944621': 1, '194785234': 2, '195184884': 2, '98463900': 3, '220818172': 2, '61662990': 3, '222166439': 3, '134956309': 3, '208205566': 1, '243416988': 1, '28611078': 3, '185764381': 3, '230523166': 2, '171569064': 2, '187686902': 3, '191105493': 3, '84501819': 3, '132560260': 3, '33301714': 3, '104352762': 3, '183698999': 2, '49236623': 3, '89446943': 3, '177321164': 3, '174725769': 2, '72484829': 3, '28364705': 3, '224754440': 3, '197972928': 3, '136755833': 2, '58108118': 3, '179656269': 3, '246518777': 1, '106619132': 3, '254212960': 2, '222528340': 1, '271796255': 3, '216103963': 1, '136557826': 3, '29458765': 3, '234968098': 2, '188915754': 2, '137205156': 3, '166418693': 2, '244892319': 1, '167029227': 3, '141610108': 3, '50917497': 3, '136759075': 3, '90066689': 3, '177695411': 3, '126773377': 3, '184276173': 3, '195077874': 2, '152023346': 3, '229597300': 2, '88122291': 3, '208204270': 1, '259125062': 2, '264175932': 1, '138943502': 3, '26346384': 3, '199062708': 3, '169601723': 3, '270866115': 3, '16126957': 3, '202609585': 1, '185820095': 1, '180655564': 3, '145832019': 1, '200710519': 2, '179329818': 2, '202803893': 3, '273028092': 3, '95679174': 3, '199925054': 1, '198905568': 1, '198579239': 3, '31723427': 3, '40550745': 3, '178058483': 1, '23329121': 3, '173875896': 2, '168854546': 3, '240975316': 1, '73154051': 3, '186093802': 3, '169972102': 1, '94791278': 3, '197501420': 3, '170449314': 3, '223653724': 2, '191323557': 3, '197086832': 3, '243887510': 3, '95186765': 3, '135278424': 2, '199465213': 3, '34827206': 3, '50666510': 3, '179256309': 2, '169175129': 3, '175580206': 1, '216897958': 2, '155727191': 2, '138316913': 3, '84996571': 3, '149690324': 3, '180805461': 2, '131653602': 3, '129269635': 2, '156256404': 1, '186579698': 2, '150688536': 3, '159999253': 2, '112843363': 3, '187695212': 3, '33364286': 3, '250309713': 3, '137869308': 3, '39441410': 3, '143166405': 2, '263696400': 2, '111620030': 3, '120961545': 3, '149313361': 2, '203984188': 3, '138401081': 2, '187278530': 2, '148520243': 3, '137522792': 3, '187653235': 1, '233490625': 2, '178930563': 2, '168020694': 3, '104699406': 3, '63499606': 3, '52468929': 3, '336067401': 3, '246023501': 3, '189654069': 2, '207795016': 1, '104568901': 3, '245618567': 3, '138265683': 1, '94569119': 3, '178566607': 3, '146046425': 3, '151689740': 3, '172989094': 2, '238029196': 2, '33493437': 2, '190389510': 3, '123682350': 3, '142504028': 2, '137953782': 3, '194545979': 1, '118912142': 3, '100187406': 3, '223647742': 2, '18488486': 2, '201094560': 1, '140132814': 2, '241360726': 1, '98941300': 3, '195382733': 2, '178184405': 1, '27987139': 3, '226405236': 3, '21497761': 3, '55878127': 3, '126352267': 3, '164011123': 3, '165099653': 3, '24481831': 2, '204561252': 3, '194202360': 3, '94361148': 3, '104602508': 3, '46017426': 3, '108990996': 3, '41239795': 3, '20112541': 3, '82497092': 3, '224439537': 2, '203319399': 3, '221058808': 1, '41681173': 3, '83175775': 3, '138262280': 1, '247680398': 1, '122463447': 3, '146120073': 3, '15826674': 3, '28040839': 3, '192017951': 2, '192578918': 2, '181595938': 2, '212374721': 1, '123702315': 3, '159285975': 3, '135208814': 3, '139776683': 3, '241618321': 3, '165903498': 1, '35031566': 3, '216922104': 1, '34030288': 3, '232718590': 3, '117786796': 3, '213498638': 3, '165684558': 3, '106125681': 3, '34163092': 3, '257622203': 2, '101343118': 3, '212317604': 1, '162027920': 3, '252436755': 1, '61070608': 3, '42431248': 2, '183913701': 3, '252734595': 2, '163712823': 2, '59650018': 3, '217469777': 3, '199001643': 3, '46746701': 3, '152387191': 2, '73950809': 3, '149790980': 3, '222162007': 1, '122240583': 3, '179159469': 2, '59140724': 3, '236540639': 1, '73576113': 3, '218722291': 3, '41941523': 3, '124584052': 3, '139793294': 2, '212383867': 2, '222223684': 1, '236368233': 1, '38109483': 3, '139768816': 2, '123062127': 3, '214993382': 2, '190348161': 3, '197470104': 1, '244645705': 2, '174083361': 1, '16231382': 3, '184645944': 2, '225671822': 1, '171352877': 2, '170501927': 2, '245452263': 1, '81114040': 3, '55956413': 3, '230508458': 3, '198243230': 3, '43617549': 3, '219708600': 2, '102861672': 3, '211522541': 3, '260581228': 3, '222551411': 1, '210203747': 2, '48850962': 3, '145788224': 3, '47760041': 3, '167019258': 2, '190326616': 3, '186156890': 1, '177997781': 1, '111870690': 3, '173538583': 3, '243215968': 1, '221005930': 1, '164929603': 1, '144883550': 3, '237938652': 1, '206475854': 3, '256281854': 2, '96356008': 3, '206926717': 3, '43515977': 3, '227543256': 2, '184484607': 3, '241540947': 3, '60219398': 3, '160683249': 2, '188727346': 1, '146288347': 2, '160948019': 3, '42272681': 3, '160720239': 2, '124948942': 3, '273390607': 2, '167841773': 3, '139200422': 3, '63965081': 3, '131919140': 3, '180546484': 2, '151665512': 3, '160328454': 2, '253090922': 2, '186590562': 2, '177216573': 3, '186614776': 3, '203114717': 2, '23389150': 2, '203322585': 1, '49084129': 3, '143276013': 3, '214053474': 1, '169930231': 2, '88750991': 3, '258933531': 3, '73574169': 3, '149461242': 3, '37701937': 3, '138604451': 2, '15471915': 3, '41720417': 2, '96138658': 3, '24913952': 3, '223621186': 1, '24629050': 3, '188208557': 3, '210557450': 3, '89726964': 3, '55984294': 3, '167047063': 2, '208874322': 3, '107620195': 3, '18114143': 3, '209149917': 2, '116589436': 3, '128544354': 3, '83859377': 3, '227103472': 2, '38691508': 3, '128007548': 3, '213482966': 3, '102702396': 3, '245455420': 3, '66255718': 3, '179238237': 1, '123736214': 3, '140615689': 2, '231644178': 2, '17877400': 3, '151540692': 3, '178422740': 1, '184973087': 3, '156404845': 3, '41876674': 2, '150913668': 3, '229135279': 2, '106131115': 3, '211655493': 1, '61869629': 3, '257649313': 1, '151886173': 3, '205574042': 3, '140946303': 3, '170552002': 3, '105163808': 3, '26590865': 3, '55337279': 3, '195458660': 2, '209956386': 3, '27311659': 3, '171514301': 3, '153172815': 3, '25560426': 3, '92790230': 3, '231558761': 1, '191351843': 2, '146257656': 1, '184960484': 1, '200156193': 1, '175612668': 2, '262852923': 2, '20343068': 3, '154319166': 3, '80612086': 3, '126735823': 3, '187332666': 2, '139204359': 2, '255371843': 3, '271185998': 3, '33735474': 2, '118911528': 3, '163516905': 3, '18127981': 3, '140081830': 3, '70814512': 3, '211718672': 1, '225487433': 3, '212055317': 1, '143688773': 3, '209686519': 1, '222726236': 3, '126352565': 3, '31917136': 3, '125059588': 3, '157767580': 3, '31048934': 3, '254739262': 2, '131147824': 3, '161141519': 3, '32245245': 3, '135585657': 3, '141041329': 3, '182036982': 3, '161714485': 1, '162332120': 3, '187229709': 3, '86992418': 3, '55121168': 3, '211557352': 2, '240900136': 2, '221417389': 1, '260788630': 3, '84177710': 3, '220685546': 2, '120468873': 3, '179362870': 3, '175809088': 3, '57060621': 2, '163178646': 3, '235249813': 3, '185854688': 3, '232273078': 1, '93074927': 3, '167525306': 3, '223566209': 3, '120657922': 3, '138041506': 2, '168658616': 2, '57399222': 3, '206380496': 3, '110748278': 3, '85748568': 3, '216181537': 2, '189013085': 2, '16288014': 3, '237026635': 2, '174061818': 3, '15885983': 3, '30818990': 3, '197099153': 3, '73285089': 3, '148748153': 3, '149864709': 3, '176626131': 3, '97675435': 2, '211668979': 2, '163364821': 3, '196479488': 3, '179585600': 2, '179471047': 2, '170330495': 3, '114617885': 3, '222887796': 2, '246569366': 2, '184501024': 1, '217105849': 2, '138490563': 3, '172439095': 2, '257996313': 3, '167588865': 3, '198948142': 2, '188086752': 3, '52754636': 3, '185875032': 1, '171985739': 2, '200237729': 1, '230255902': 3, '232024421': 1, '138260439': 3, '163271110': 2, '154709863': 3, '204651705': 3, '86339761': 2, '229096547': 3, '222241978': 2, '113997899': 3, '149336370': 1, '35014380': 2, '22362043': 3, '62952901': 3, '168776165': 3, '205483363': 2, '236258840': 2, '195896174': 2, '197914903': 3, '195379186': 2, '185506002': 2, '106005814': 3, '219498167': 1, '60148899': 3, '240749811': 2, '230869728': 3, '69374799': 3, '206022482': 2, '35291103': 3, '41628247': 2, '171736623': 3, '214542649': 2, '27932506': 3, '182520748': 2, '243448512': 3, '138821028': 3, '217494535': 3, '87119587': 3, '228937371': 2, '157255024': 3, '21399864': 3, '182893614': 2, '181472044': 3, '186327313': 3, '196737992': 2, '73947331': 2, '231850859': 3, '201122496': 2, '165231735': 3, '40517965': 3, '171266293': 3, '236043929': 2, '204502572': 3, '138528110': 3, '236674010': 2, '194412478': 3, '210072665': 2, '125121680': 3, '210885786': 2, '187635209': 3, '223601829': 1, '22010465': 3, '187526484': 2, '60326763': 3, '148562581': 2, '182399395': 3, '245435613': 3, '240255068': 1, '191878831': 2, '87603508': 2, '240326567': 3, '229187854': 2, '136531053': 3, '190515760': 3, '28595345': 3, '86360431': 3, '141407214': 1, '185258363': 2, '123596384': 3, '99992211': 3, '212943649': 1, '251183211': 3, '20461177': 3, '198922049': 1, '224476520': 3, '245689796': 3, '233687406': 3, '142886195': 3, '140518559': 3, '69790205': 3, '59702847': 3, '173517450': 1, '24655254': 3, '246628103': 3, '164043761': 1, '90446707': 3, '178326373': 3, '186929659': 3, '224619049': 2, '197630574': 3, '96997373': 3, '216437939': 2, '140499756': 3, '230065476': 2, '219688451': 3, '201857015': 1, '234462939': 2, '223524551': 2, '28581142': 3, '75074952': 3, '63674792': 3, '221404588': 3, '222538056': 3, '54544971': 3, '126581860': 3, '147489444': 3, '242632117': 2, '137105423': 3, '18129371': 3, '73985632': 3, '135690536': 3, '236041528': 3, '223312341': 3, '199672676': 2, '178344309': 3, '210907863': 2, '23785863': 3, '128327267': 3, '257903285': 1, '26651032': 3, '53530789': 3, '236187880': 1, '194791624': 1, '178111458': 3, '174685099': 3, '186290212': 3, '191165871': 3, '225046603': 2, '176538310': 2, '106555176': 3, '43793607': 3, '119905863': 2, '198963308': 2, '129243991': 3, '167385883': 2, '205340365': 3, '174359966': 2, '229848368': 3, '137155290': 3, '244589170': 1, '365691173': 3, '44608187': 3, '196801232': 2, '42429110': 3, '179766289': 1, '193840711': 3, '193128740': 2, '242348687': 3, '191566859': 3, '186621504': 2, '87799278': 3, '179664631': 1, '197768308': 1, '221482020': 3, '150010871': 3, '178081878': 3, '90711550': 3, '208348286': 3, '39203730': 3, '197888509': 3, '204210440': 3, '166424921': 2, '198185859': 1, '153005753': 3, '139112841': 3, '50008870': 3, '173497593': 1, '224863456': 3, '128722091': 2, '23342437': 3, '148394231': 3, '104669751': 3, '194990747': 3, '121983817': 3, '132068159': 3, '90401916': 3, '164414706': 3, '228112060': 1, '131496039': 3, '192910624': 1, '81861341': 3, '213752405': 3, '32404039': 3, '168200483': 1, '41486430': 3, '84089335': 3, '193479001': 3, '184934269': 1, '119829134': 3, '201388634': 2, '35316651': 3, '185134721': 2, '134980267': 3, '29304563': 3, '263250930': 3, '173836109': 1, '34155241': 3, '47874071': 3, '144677344': 2, '51160789': 3, '163242130': 3, '202283825': 3, '239844393': 3, '48508100': 3, '169500151': 3, '171495127': 2, '139211129': 1, '126758930': 3, '102737698': 3, '147791025': 3, '40108269': 3, '148148887': 3, '183336921': 2, '46910352': 3, '191243838': 2, '108978658': 3, '223660066': 3, '193277715': 1, '259427187': 1, '165406920': 2, '141348921': 2, '150869281': 3, '21309068': 3, '41304261': 3, '20726495': 3, '211673288': 3, '18402067': 3, '173190042': 3, '165098768': 3, '220658491': 3, '176956851': 2, '192991011': 2, '17777600': 3, '174468680': 2, '199808438': 3, '199850847': 3, '182105461': 2, '225174900': 1, '137787295': 3, '213779796': 1, '106299786': 3, '220410961': 3, '182553594': 2, '116915624': 3, '259046537': 3, '73046116': 3, '247970517': 3, '120175417': 3, '104397936': 3, '179227847': 2, '27067684': 3, '193913642': 3, '223198874': 2, '78870377': 3, '155692632': 3, '71247422': 3, '200976481': 1, '248121628': 1, '204367459': 1, '186671746': 3, '140123105': 1, '222592035': 3, '25737532': 3, '255116221': 3, '101924437': 3, '16332498': 3, '181543641': 2, '220197881': 2, '74832616': 2, '46793500': 2, '147708554': 2, '186485260': 1, '210625720': 3, '257248019': 1, '167206360': 3, '175137679': 1, '224285988': 2, '184279727': 1, '184538686': 1, '128619844': 3, '240898511': 2, '220195876': 3, '213018223': 3, '170406551': 3, '269342966': 2, '216451824': 1, '198879580': 1, '85968608': 3, '164001850': 3, '126035571': 3, '149327536': 3, '144664246': 3, '74434137': 2, '177716351': 3, '204760936': 1, '16614089': 2, '153429706': 3, '109764961': 3, '258854424': 2, '190055537': 3, '42142966': 3, '199708243': 3, '236622517': 1, '167639169': 3, '191314878': 3, '136763775': 3, '152003447': 3, '152890786': 3, '180224095': 2, '119022784': 3, '167776491': 2, '35354815': 3, '57440442': 2, '106307528': 3, '195852028': 3, '170398119': 2, '180180631': 1, '187033992': 1, '205866862': 2, '225134494': 3, '165101991': 3, '211260450': 2, '193876250': 3, '110836933': 3, '180839850': 3, '211035534': 3, '40743560': 3, '134100467': 2, '187083165': 1, '266149914': 3, '192241822': 2, '83855716': 3, '35272811': 3, '181239603': 3, '151272650': 3, '129322011': 3, '236491992': 3, '39198376': 3, '194938001': 2, '74339329': 3, '155763609': 2, '182555004': 2, '260326865': 3, '186294757': 1, '42890617': 2, '168858250': 2, '105489759': 3, '106601966': 3, '217790301': 3, '77622049': 3, '190462475': 1, '194763119': 3, '170583874': 2, '169968089': 1, '140189498': 3, '115267089': 2, '184039130': 3, '215485677': 1, '249684405': 3, '219358937': 1, '98436935': 3, '149295172': 3, '196708632': 3, '34816039': 3, '19896168': 3, '211361298': 2, '148032719': 3, '185479769': 3, '177303771': 3, '199002693': 3, '171250405': 3, '235235080': 2, '167935551': 1, '215297881': 1, '161905608': 3, '110539973': 3, '246142482': 3, '155745520': 3, '253456632': 3, '170888213': 3, '248440805': 3, '116476443': 3, '228564270': 3, '228153471': 2, '60964956': 3, '76702449': 3, '35598508': 3, '490826545': 3, '151272799': 3, '140117927': 3, '24487996': 3, '191961773': 3, '190685483': 1, '217089956': 1, '106887653': 2, '224266295': 3, '103967566': 3, '124837829': 3, '214779734': 2, '36454917': 2, '169247917': 1, '195889284': 2, '210673384': 1, '165048628': 3, '195375499': 2, '205974878': 1, '122963435': 3, '201381597': 1, '185706657': 1, '188558351': 3, '230063027': 2, '169525586': 2, '183036731': 3, '224987724': 3, '184200246': 2, '168510999': 3, '188772459': 2, '41795534': 2, '170892886': 2, '101842356': 3, '106140601': 3, '247125407': 2, '217172670': 3, '143504230': 3}
        signalling = self.findSignal3(assign, top, 71)
        ans = self.testLikely(assign, signalling)
        stop_sig = 0
        while stop_sig == 0:
            stop_sig = 1
            for sample in signalling:
                temp = copy.deepcopy(signalling)
                temp.remove(sample)
                trial = self.testLikely(assign, temp)
                if trial >= ans:
                    signalling.remove(sample)
                    stop_sig = 0
                    ans = trial
                    print(trial)
        return [self.testLikely(assign, signalling), signalling]
                    


