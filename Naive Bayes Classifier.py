import math
import random
import itertools

def random_generator(low, high):
    while True:
        yield random.randrange(low, high)

# Loading Data
file = open('download.txt', 'r').read().split('\n')
dataset = [line.split() for line in file]
for i in range(len(dataset)):
    for j in range(len(dataset[i])-1):
        dataset[i][j]=float(dataset[i][j])
    dataset[i][len(dataset[i])-1]=int(dataset[i][len(dataset[i])-1])     


# Splitting into training and testing
split_ratio = 0.8
training = []
testing = []
gen = random_generator(1, len(dataset)-1)
test_pos = set()    
for x in itertools.takewhile(lambda x: len(test_pos) < len(dataset)*(1-split_ratio), gen):
    test_pos.add(x)
 
for i in range(0, len(dataset)):
    if i in test_pos:
        testing.append(dataset[i])
    else:
        training.append(dataset[i])
print(len(test_pos), '/', len(dataset), 'in testing')


# Aggregating over class
seg_data = dict()
model = dict()
for i in range(len(training)):
    cl = training[i][-1]
    if cl in seg_data:
        seg_data[cl].append(training[i])
    else:
        seg_data[cl] = list()

# Stacking class data by columns, and computing mean, standard deviation
for cl, rows in seg_data.items():
    
    stacks = dict()    
    for j in range(len(rows[0])):
        stacks[j]=list()
    for i in range(len(rows)):
        for j in range(len(rows[i])):
            stacks[j].append(rows[i][j])
    
    col_descriptor = []
    for col_stack in stacks:
        col = tuple(stacks[col_stack])
        col_mean = (1.0*sum(col))/len(col)
        col_len = len(stacks[col_stack])
        var = 0
        for entry in stacks[col_stack]:
            var+=math.pow((entry-col_mean),2)
        var *= 1.0
        var/=(col_len-1)
        col_std = math.sqrt(var)
        if col_std == 0:
            continue
        col_descriptor.append((col_mean, col_std, col_len))
    model[cl] = col_descriptor

# Model Depiction
for cl in model:
    print('Class: feature-wise [Mean, Standard Deviation, Length]\n',cl, model[cl])

# Testing the model
tp = 0
total=len(testing)
for entry in testing:
    # Computing Posteriors
    length = 0
    priors = dict()
    for cl in model:
        priors[cl] = model[cl][0][2]
        length+= model[cl][0][2]        # length component
    posteriors = dict()
    for cl in model:
        posteriors[cl] = 1.0*priors[cl]/length
        for i in range(len(model[cl])):
            mu, sigma, _ = model[cl][i]
            
            # Computing Gausssian Probability
            variance = math.pow(sigma,2)
            denom = math.sqrt(2*math.pi*variance)
            exp = math.exp(-math.pow((float(entry[i])-float(mu)),2)/(2*variance))
            posteriors[cl] *= exp/denom

    # Finding class with highest posterior
    label = ('1', -1)
    for cl in posteriors:
        if posteriors[cl] > label[1]:
           label = (cl, posteriors[cl]) 
           
    # Accuracy Computation
    if label[0]==entry[-1]:
        tp+=1
print('\nFinal Accuracy:', 100*tp/total)
