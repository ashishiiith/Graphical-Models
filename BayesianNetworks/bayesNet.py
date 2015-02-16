#Author: Ashish Jain
#This script loads the data file for bayes net and learns the join probabilities.

import sys
import os
import itertools

""" stores values a particular variable can take """
variable_values = {} 
""" stores order in which variable is read from file """
variable_index = [] 
""" stores parent-child relationship """ 
graph = {}          
""" main structure which contains counts corresponding to each child-parent setting """
node_count = {}     


def initialize():


    #initializing values associated with each variable
    global variable_values
    variable_values['A'] = ['1', '2', '3']
    variable_values['G'] = ['1', '2']
    variable_values['CP'] = ['1', '2', '3', '4']
    variable_values['BP'] = ['1', '2']
    variable_values['CH'] = ['1', '2']
    variable_values['ECG'] = ['1', '2']
    variable_values['HR'] = ['1', '2']
    variable_values['EIA'] = ['1', '2']
    variable_values['HD'] = ['1', '2']
   
    #initializing graph with child:parent relations    
    global graph
    graph['G'] = []
    graph['A'] = []
    graph['BP'] = ['G']
    graph['CH'] = ['G', 'A']
    graph['HD'] = ['BP','CH']
    graph['CP'] = ['HD']
    graph['EIA'] = ['HD']
    graph['ECG'] = ['HD']
    graph['HR'] = ['HD', 'A']


    #defining order in which variables from data files would be read
    global variable_index
    variable_index = ['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR', 'EIA', 'HD']

    global node_count   
    for variable in variable_index:
        node_count[variable] = {}
        parents = get_parents(variable)
        if len(parents) == 0:
            for values in variable_values[variable]:
                node_count[variable][values] = 0
        else:
           list_parents = []

           for parent in parents:
                list_parents.append(variable_values[parent])
           
           for pval in itertools.product(*list_parents):
                node_count[variable][pval] = 0
                for nval in variable_values[variable]:
                    node_count[variable][(nval,)+pval] = 0 
    #print node_count

def get_variable(var):
        
    return variable_index[var]

def get_parents(var):
        
    return graph[var]


def compute_counts(data):

    global node_count
    for variable, val in data.items():
    
        parents = get_parents(variable)
        if len(parents) == 0:
            node_count[variable][val] += 1
        else:
            pval = tuple([data[parent] for parent in parents])
            node_count[variable][pval] += 1
            node_count[variable][(data[variable],)+pval]+=1         

def learn_graph():

    lines =  open('Data/data-train-1.txt', "r").readlines()
    
    for index in xrange(0, len(lines), 1):
        data = {}
        tokens = lines[index].strip().split(",")
        for i in xrange(0, len(tokens)):
            data[variable_index[i]] = tokens[i]
        compute_counts(data)        
    print node_count

def findCPT(variable):

    parents = get_parents(variable)
        
    if len(parents) == 0:
        total_count = 0
        for key, val in node_count[variable].items():
            total_count += val
        check = 0 
        for values in variable_values[variable]:
        
            print variable + " " + str(values) + " " + str((node_count[variable][values]/(total_count*1.0)))
    else:
        list_parents=[]
        for parent in parents:
            list_parents.append(variable_values[parent])
        for pval in itertools.product(*list_parents):
            for nval in variable_values[variable]:
                print variable + " " + str((nval,)+pval) + " " + str(float(node_count[variable][(nval,)+pval]/(1.0*node_count[variable][pval])))

def solveQuery(query):

    probability = 1.0
    for variable, parents in graph.items():
  
        if len(parents) == 0:    
            total_count = 0
            for key, val in node_count[variable].items():
                total_count += val
            probability*=float(node_count[variable][query[variable]]/(total_count*1.0))
        else:
            pval = tuple([query[parent] for parent in parents])
            probability*=float(node_count[variable][(query[variable],)+pval]/(1.0*node_count[variable][pval]))
    return probability

def findQuery():

    """ Solving the first query"""

    Query11 = {'A': '2', 'G':'2', 'CH':'1', 'CP':'4', 'BP' : '1', 'ECG' : '1', 'HR' : '1', 'EIA' : '1', 'HD' : '1'} 
    numerator = solveQuery(Query11)
    #print numerator
    Query12 = {'A': '2', 'G':'2', 'CH':'2', 'CP':'4', 'BP' : '1', 'ECG' : '1', 'HR' : '1', 'EIA' : '1', 'HD' : '1'}
    denominator =  solveQuery(Query12)
    #print denominator
    print float(numerator/(numerator+denominator))

    """ Solving the second query """

    numerator=0.0
    denominator=0.0
    Query2 = {'A': '2', 'CH':'2', 'CP':'1', 'BP' : '1', 'ECG' : '1', 'HR' : '2', 'EIA' : '2', 'HD' : '1'}
    for value in variable_values['G']:
        Query2['G'] = value
        numerator+=solveQuery(Query2)
    for bp in variable_values['G']:
        for g in variable_values['BP']:
            Query2['BP'] = bp
            Query2['G'] = g
            denominator+=solveQuery(Query2)
    print float(numerator/denominator)

def classification():

   lines =  open('Data/data-test-6.txt', "r").readlines()
 
   correct = 0
   total = 0
   for index in xrange(0, len(lines), 1): 
        
        data = {}       
        tokens = lines[index].strip().split(",")
        true_class = tokens[8]
        for i in xrange(0, len(tokens)):
            data[variable_index[i]] = tokens[i]
        data['HD'] = '1'
        prob1 = solveQuery(data)
        data['HD'] = '2'
        prob2 = solveQuery(data)
        if prob1 >= prob2 and true_class =='1':
            correct+=1
        elif prob2 > prob1 and true_class == '2':
            correct+=1
        total+=1
   print correct
   print total

def main():

    initialize()
    learn_graph()
    
    #findCPT('A')
    #findCPT('BP')
    #findCPT('HD')
    #findCPT('HR')

    findQuery()
    #classification()

if __name__ == "__main__":
    main()
