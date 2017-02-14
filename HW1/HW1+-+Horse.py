
# coding: utf-8

# In[1]:

import math
# BEST HEURISTIC
def heuristic(state_x, state_y, final_x, final_y):
    return min(math.ceil(abs(state_x - final_x)/2.0),math.ceil(abs(state_y - final_y)/2.0))

def heuristic_2(state_x, state_y, final_x, final_y):
    return min((abs(state_x - final_x)/2.0),(abs(state_y - final_y)/2.0))

def heuristic_3(state_x, state_y, final_x, final_y):
    if(abs(state_x - final_x) <= 2):
        #print "diff in x abs <= 2 " 
        if(abs(state_y - final_y) <=2):
            #print "diff in y abs <= 2 "
            return 1;
        else:
            return 2;
    else:
        return 2;    


# In[2]:

import datetime
import random

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[3]:

from collections import defaultdict;
class HorseNode:
    def __init__(self, x, y, final_x, final_y, path_cost):
        self.pos_x = x
        self.pos_y = y
        self.h_cost = heuristic(x, y, final_x, final_y)
        self.g_cost = path_cost
        self.f_cost = self.g_cost + self.h_cost

def appendToFrontier(newNode, Frontier):
    if(newNode.f_cost in Frontier):
        Frontier[newNode.f_cost].append(newNode)
    else:
        Frontier[newNode.f_cost] = [newNode]

def expandNode(Node, Frontier, final_x, final_y, VisitedHashSet):
    nodeGen = 0
    if((Node.pos_x+1, Node.pos_y+2) not in VisitedHashSet):
        h1 = HorseNode(Node.pos_x+1, Node.pos_y+2, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h1, Frontier)
        nodeGen+=1
    if((Node.pos_x+2, Node.pos_y+1) not in VisitedHashSet):
        h2 = HorseNode(Node.pos_x+2, Node.pos_y+1, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h2, Frontier)
        nodeGen+=1
    if((Node.pos_x+1, Node.pos_y-2) not in VisitedHashSet):
        h3 = HorseNode(Node.pos_x+1, Node.pos_y-2, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h3, Frontier)
        nodeGen+=1
    if((Node.pos_x+2, Node.pos_y-1) not in VisitedHashSet):
        h4 = HorseNode(Node.pos_x+2, Node.pos_y-1, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h4, Frontier)
        nodeGen+=1
    if((Node.pos_x-1, Node.pos_y+2) not in VisitedHashSet):
        h5 = HorseNode(Node.pos_x-1, Node.pos_y+2, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h5, Frontier)
        nodeGen+=1
    if((Node.pos_x-2, Node.pos_y+1) not in VisitedHashSet):
        h6 = HorseNode(Node.pos_x-2, Node.pos_y+1, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h6, Frontier)
        nodeGen+=1
    if((Node.pos_x-1, Node.pos_y-2) not in VisitedHashSet):
        h7 = HorseNode(Node.pos_x-1, Node.pos_y-2, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h7, Frontier)
        nodeGen+=1
    if((Node.pos_x-2, Node.pos_y-1) not in VisitedHashSet):
        h8 = HorseNode(Node.pos_x-2, Node.pos_y-1, final_x, final_y, Node.g_cost+1)
        appendToFrontier(h8, Frontier)
        nodeGen+=1
    return nodeGen
    
def popMinNode(Frontier):
    m = min(i for i in Frontier.keys() if len(Frontier[i]) > 0)
    minNode = Frontier[m].pop(0)
    return minNode;
    
def findBestNode(Frontier, VisitedHashSet):
    bestNode = popMinNode(Frontier)
    tries = 0
    while(((bestNode.pos_x,bestNode.pos_y) in VisitedHashSet)):
        bestNode = popMinNode(Frontier)
    return bestNode

def checkIfFCostIsGreaterThanBestFinalNode(bestFinalNode, node):
    if(bestFinalNode.f_cost <= node.f_cost):
        return False
    return True
    

def checkNoOtherBetterNodeToExpand(Frontier, finalNode, VisitedHashSet):
    bestNodeToExpand = findBestNode(Frontier, VisitedHashSet)
    return checkIfFCostIsGreaterThanBestFinalNode(finalNode, bestNodeToExpand)

def goalTest(pos_x, pos_y, final_x, final_y):
    if(pos_x == final_x and pos_y == final_y):
        return True
    return False

def solveAstar(start_x, start_y, final_x, final_y):
    sNode = HorseNode(start_x, start_y, final_x, final_y, 0)
    HashSet = {(start_x,start_y): 1}
    Frontier = defaultdict(list)
    Frontier[sNode.f_cost] = [sNode]
    isCompleted = False
    VisitedHashSet = {}
    bestFinalNode = None
    FoundCostToFinal = False;
    noOfNodesExpanded = 0
    noOfNodesGenerated = 0
    while(len(Frontier)!=0 or isCompleted):
        bestNode = findBestNode(Frontier, VisitedHashSet)
        isNodeGoalState = goalTest(bestNode.pos_x, bestNode.pos_y, final_x, final_y)
        if(FoundCostToFinal):
            if(not checkIfFCostIsGreaterThanBestFinalNode(bestFinalNode, bestNode)):
                break
        if(isNodeGoalState):
            if(not FoundCostToFinal):
                bestFinalNode = bestNode
                FoundCostToFinal = True
            else:
                if(bestFinalNode.g_cost > bestNode.g_cost):
                    bestFinalNode=bestNode
        if(isNodeGoalState and checkNoOtherBetterNodeToExpand(Frontier, bestNode, VisitedHashSet)):
            isCompleted = True
            break
        VisitedHashSet[(bestNode.pos_x, bestNode.pos_y)] = 1
        noOfNodesExpanded += 1
        # print "Expanding Node with co-ordinates (%s,%s) " %(bestNode.pos_x, bestNode.pos_y)
        noOfNodesGenerated+=expandNode(bestNode, Frontier, final_x, final_y, VisitedHashSet)
    return bestFinalNode, noOfNodesExpanded, noOfNodesGenerated


# In[4]:
start_x, start_y, end_x, end_y = (0, 0, 25,25)
minNode, noOfNodesExpanded, noOfNodesGenerated = solveAstar(start_x, start_y, end_x, end_y)
print "Minimum Number of Moves are %d to move from (%d,%d) to (%d,%d)" %(minNode.g_cost, start_x,start_y, end_x, end_y)


# In[5]:

import timeit

xMinMoves = []
y = []
xNodesExpanded = []
xNodesGenerated = []
def generateRandomStartFinalPositions(startRange,endRange):
    start_x = random.randint(startRange, endRange)
    start_y = random.randint(startRange, endRange)
    final_x = random.randint(startRange, endRange)
    final_y = random.randint(startRange, endRange)
    if(start_x == final_x and start_y == final_y):
        return generateRandomStartFinalPositions(startRange,endRange)
    else:
        return start_x, start_y, final_x, final_y

for i in range(0,100):
    startRange = 1
    endRange = 100
    start_x, start_y, final_x, final_y = generateRandomStartFinalPositions(startRange,endRange)
    start_time = timeit.default_timer()
    print "Start Position : (%d,%d) End Position : (%d,%d) " %(0, 0, final_x, final_y)
    minNode, noOfNodesExpanded, noOfNodesGenerated = solveAstar(0, 0, final_x, final_y)
    elapsed = timeit.default_timer() - start_time
    xMinMoves.append(minNode.g_cost)
    print "Solution min number of moves: %d" %(minNode.g_cost)
    xNodesExpanded.append(noOfNodesExpanded)
    xNodesGenerated.append(noOfNodesGenerated)
    y.append(elapsed)

#Please uncomment the following to draw the plots
# draw_plots()


def draw_plots():
    # # In[6]:

    # # plot
    plt.xlabel('min Number of Moves to solution')
    plt.ylabel('Time in Seconds')
    plt.scatter(xMinMoves,y)
    plt.show()


    # In[7]:

    plt.xlabel('Number of Nodes Expanded')
    plt.ylabel('Time in Seconds')
    plt.scatter(xNodesExpanded,y)
    plt.show()


    # In[8]:

    plt.xlabel('Number of Nodes Generated')
    plt.ylabel('Time in Seconds')
    plt.scatter(xNodesGenerated,y)
    plt.show()


    # In[9]:

    plt.xlabel('min Number of Moves to solution')
    plt.ylabel('Number of Nodes Expanded')
    plt.scatter(xMinMoves,xNodesExpanded)
    plt.show()


    # In[10]:

    plt.xlabel('min Number of Moves to solution')
    plt.ylabel('Number of Nodes Generated')
    plt.scatter(xMinMoves,xNodesGenerated)
    plt.show()

