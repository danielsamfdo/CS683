
# coding: utf-8

# In[3]:

import random
import math
import copy
from heapq import heappush, heappop
from collections import defaultdict
def isSorted(nodes):
    if(len(nodes)==0):
        return True
    for i in range(1,len(nodes)):
        if(nodes[i-1] > nodes[i]):
            return False
    return True

def EuclideanDistance(point1_x, point1_y, point2_x, point2_y):
    internalCalculation = math.pow((point1_x - point2_x),2) + math.pow((point1_y - point2_y),2)
    return math.sqrt(internalCalculation);

def getDistanceBetweenNodes(node, otherNode, AdjHash, Map, DistanceComputation):
    if(node == otherNode):
        return 0;
    if(otherNode in AdjHash and node in AdjHash[otherNode]):
        return AdjHash[otherNode][node]
    return DistanceComputation(Map[node][0], Map[node][1], Map[otherNode][0], Map[otherNode][1])

def computeAdjHashOfCompleteGraph(Map, DistanceComputation):
    # Input::  Map: {0: (0.13037375991631983, 0.17982980099790336), ... }
    nodes = Map.keys()
    assert isSorted(nodes)
    AdjHash= {}
    for node in nodes:
        AdjHash[node] = {}
        for otherNode in nodes:
            AdjHash[node][otherNode] = getDistanceBetweenNodes(node, otherNode, AdjHash, Map, DistanceComputation)
    return AdjHash

def TspGenerator(numberOfCities, lowRange=0.0, highRange=1.0, DistanceComputation=EuclideanDistance):
    Map = {}
    inverseMap = {}
    AdjHash = {}
    for x in range(0, numberOfCities):
        coordinate = (random.uniform(lowRange, highRange), random.uniform(lowRange, highRange))
        tries = 0;
        while(coordinate in inverseMap):
            coordinate = (random.uniform(lowRange, highRange), random.uniform(lowRange, highRange))
            tries+=1;
            if(tries==5):
                print "Unable to Generate Coordinates"
                return ;
        Map[x] = coordinate
        inverseMap[coordinate] = x
    return Map, inverseMap, computeAdjHashOfCompleteGraph(Map, DistanceComputation);


# In[2]:

def isAllVisited(VisitedHash, nodes):
    for node in nodes:
        if(node not in VisitedHash):
            return False
    return True

def PrimsAlgorithm(startNode, nodes, AdjHash):
#     print startNode, nodes
    #Input startNode : type int, nodes : type list of int's
    MSTCost = 0;
    h = []
    visitedHash = {}
    visitedHash[startNode] = True
    prevNode = startNode
    MstEdges = []
    while(not isAllVisited(visitedHash, nodes)):
        for node in nodes:
            if(node not in visitedHash.keys()):
#                 print h
                heappush(h, (AdjHash[prevNode][node], (prevNode, node)))
        cost, edge = heappop(h)
        parentNode, minNode = edge
        while(minNode in visitedHash):
            # if min node was alreadyVisited
            cost, edge = heappop(h)
            parentNode, minNode = edge
        MSTCost += cost
        MstEdges.append((parentNode, minNode))
        visitedHash[minNode] = True
    return MSTCost, MstEdges;

def MSTHeuristic(startNode, nodes, AdjHash):
    #Input :: AdjHash is a Adjacency map of the entire set of nodes with the value being the distance
    assert startNode in nodes;
    cost, edges = PrimsAlgorithm(startNode, nodes, AdjHash)
    return cost;
    

    
class TspNode:
    def __init__(self, x, y, nodeNumber, path_cost, parentNode, heuristicCostMap, listOfNodeNumbers, AdjHash, HeuristicFunction=MSTHeuristic):
        self.pos_x = x
        self.pos_y = y
        self.node_number = nodeNumber
        self.g_cost = path_cost
        self.parent = parentNode
        
        # If this is not the start node
        if(parentNode!=-1):
            self.nodes_visited = copy.copy(parentNode.nodes_visited)
            MSTSet = frozenset(set(listOfNodeNumbers) - set(self.nodes_visited.keys()))
            if(len(MSTSet)==1):
                heuristicCostMap[MSTSet] = 0;
            if(MSTSet not in heuristicCostMap):
                heuristicCostMap[MSTSet] = HeuristicFunction(self.node_number, list(MSTSet), AdjHash)
            heuristicCost = heuristicCostMap[MSTSet]
        else:
            heuristicCostMap[frozenset(listOfNodeNumbers)] = HeuristicFunction(self.node_number, listOfNodeNumbers, AdjHash)
            heuristicCost = heuristicCostMap[frozenset(listOfNodeNumbers)]
        self.h_cost = heuristicCost
        self.f_cost = self.g_cost + self.h_cost
        
def goalTest(node, listOfNodeNumbers):
    return isAllVisited(node.nodes_visited, list(set(listOfNodeNumbers) - set([node.node_number])))

def appendToFrontier(newNode, Frontier):
    if(newNode.f_cost in Frontier):
        Frontier[newNode.f_cost].append(newNode)
    else:
        Frontier[newNode.f_cost] = [newNode]

def printPathTraversed(node, printOutput=True):
    pathTraversed = []
    while(node.parent!=-1):
        pathTraversed.append(node.node_number)
        node=node.parent
    pathTraversed.append(node.node_number)
    pathTraversed = list(reversed(pathTraversed))
    pathTraversed.append(pathTraversed[0])
    if(printOutput):
        print pathTraversed
    return pathTraversed
    
def popMinNode(Frontier):
    isFrontierPresent = bool([a for a in Frontier.values() if a != []])
    if not isFrontierPresent:
        return None 
    m = min(i for i in Frontier.keys() if len(Frontier[i]) > 0)
    minNode = Frontier[m].pop(0)
    return minNode;
        
def hasTraversedPreviously(newNodeNumber, node):
    if(newNodeNumber in node.nodes_visited):
        return True
    return False

def findBestNode(Frontier):
    bestNode = popMinNode(Frontier)
    if bestNode == None:
        return None 
    while((hasTraversedPreviously(bestNode.node_number, bestNode))):
        bestNode = popMinNode(Frontier)
    return bestNode
        
def checkIfFCostIsGreaterThanBestFinalNode(bestFinalNode, node):
    if(bestFinalNode.f_cost <= node.f_cost):
        return False
    return True

def checkNoOtherBetterNodeToExpand(Frontier, finalNode):
    bestNodeToExpand = findBestNode(Frontier)
    return checkIfFCostIsGreaterThanBestFinalNode(finalNode, bestNodeToExpand)

def appendToFrontier(newNode, Frontier):
    if(newNode.f_cost in Frontier):
        Frontier[newNode.f_cost].append(newNode)
    else:
        Frontier[newNode.f_cost] = [newNode]

def expandNode(Node, Frontier, AdjHash, listOfNodeNumbers, Map, heuristicCostMap, HeuristicFunction):
    Node.nodes_visited[Node.node_number] = True
#     print Frontier
    toExpand = list(set(listOfNodeNumbers) - set(Node.nodes_visited.keys()))
    cntNodesGen = 0
    for nodeNumber in toExpand:
        if(nodeNumber != Node.node_number):
            latitude = Map[nodeNumber][0]
            longitude = Map[nodeNumber][1]
            tspNode = TspNode(latitude, longitude, nodeNumber, Node.g_cost+AdjHash[Node.node_number][nodeNumber], Node, heuristicCostMap, listOfNodeNumbers, AdjHash, HeuristicFunction)
            appendToFrontier(tspNode, Frontier)
            cntNodesGen+=1
#     print Frontier
    return cntNodesGen;
    
def solveAstar(startNode, AdjHash, listOfNodeNumbers, Map, HeuristicFunction=MSTHeuristic, printOutput=True):
    heuristicCostMap = {}
    #Input startNode : type int, listOfNodeNumbers : type list of int's
    latitude = Map[startNode][0]
    longitude = Map[startNode][1]
    sNode = TspNode(latitude, longitude, startNode, 0, -1, heuristicCostMap, listOfNodeNumbers, AdjHash, HeuristicFunction)
    sNode.nodes_visited = {}
    Frontier = defaultdict(list)
    Frontier[sNode.f_cost] = [sNode]
#     print Frontier
    isCompleted = False
    bestFinalNode = None
    FoundCostToFinal = False;
    noOfNodesExpanded = 0
    cntNodesGenerated = 0
    while(len(Frontier)!=0 or isCompleted):
        bestNode = findBestNode(Frontier)
        if(bestNode==None):
            break;
        isNodeGoalState = goalTest(bestNode, listOfNodeNumbers)
#         if(isNodeGoalState and checkNoOtherBetterNodeToExpand(Frontier, bestNode)):
#             isCompleted = True
#             break
        if(bestFinalNode != None and bestNode.f_cost > bestFinalNode.g_cost + AdjHash[bestFinalNode.node_number][sNode.node_number]):
            isCompleted = True
            break
        if(isNodeGoalState):
            if(printOutput):
                print " ******* reached Goal State ******* "
                print " Nodes Reached by Goal State of node number %d is %s" %(bestNode.node_number, bestNode.nodes_visited)
                print " ******* ****************** ******* "
            
#             if(bestFinalNode != None):
                #print bestFinalNode.f_cost
                #print bestFinalNode.g_cost + AdjHash[bestFinalNode.node_number][sNode.node_number]
            if(not FoundCostToFinal):
                bestFinalNode = bestNode
                FoundCostToFinal = True
            else:
                if(bestFinalNode.g_cost + AdjHash[bestFinalNode.node_number][sNode.node_number] > bestNode.g_cost  + AdjHash[bestNode.node_number][sNode.node_number]):
                    bestFinalNode=bestNode
        if(not isNodeGoalState):
            noOfNodesExpanded+=1
            if(printOutput):
                print "Expanding Node : %d and nodes visited %s" %(bestNode.node_number, bestNode.nodes_visited)
            cntNodesGenerated += expandNode(bestNode, Frontier, AdjHash, listOfNodeNumbers, Map, heuristicCostMap, HeuristicFunction)
    if(printOutput):
        print heuristicCostMap
    totalCost = bestFinalNode.g_cost + AdjHash[bestFinalNode.node_number][sNode.node_number]
    return bestFinalNode, totalCost, noOfNodesExpanded, cntNodesGenerated


# In[4]:
print "For map of Size 10"
Map, inverseMap, AdjHash = TspGenerator(10)
nodes = Map.keys()
startNode = 0
# print Map
# print AdjHash
bestFinalNode, TotalCost, noOfNodesExpanded, noOfNodesGenerated = solveAstar(startNode, AdjHash, nodes, Map, MSTHeuristic, False)


# In[5]:
printPathTraversed(bestFinalNode)
print "Total Cost %f , no of Nodes expanded %f" %(TotalCost,noOfNodesExpanded)


# In[5]:

import timeit
import datetime
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

startRange = 3
endRange = 15
nodes_size = []
time_taken = []
nodes_expanded = []
nodes_generated = []

# Data Used 
MapsUsed = []
inverseMapsUsed = []
AdjHashUsed = []
TotalCosts = []
PathsTraversed = []
for i in range(0,100):
    noOfNodes = random.randint(startRange, endRange)
    Map, inverseMap, AdjHash = TspGenerator(noOfNodes)
    nodes = Map.keys()
    nodes_size.append(noOfNodes)
    startNode = 0
#     print "###### STARTED #######"
#     print Map
    start_time = timeit.default_timer()
    bestFinalNode, TotalCost, noOfNodesExpanded, noOfNodesGenerated = solveAstar(startNode, AdjHash, nodes, Map, MSTHeuristic, False)
    elapsed = timeit.default_timer() - start_time
    nodes_generated.append(noOfNodesGenerated)
    nodes_expanded.append(noOfNodesExpanded)
    time_taken.append(elapsed)
    MapsUsed.append(Map)
    inverseMapsUsed.append(inverseMap)
    AdjHashUsed.append(AdjHash)
    TotalCosts.append(TotalCost)
    PathsTraversed.append(printPathTraversed(bestFinalNode, False))

# Please uncomment following for the graphs
# drawPlotsI()

# plot
def drawPlotsI():
    plt.xlabel('Number of Cities')
    plt.ylabel('Time in Seconds')
    plt.scatter(nodes_size,time_taken)


    # In[7]:

    # plot
    plt.xlabel('Number of Cities')
    plt.ylabel('Number of Nodes Expanded')
    plt.scatter(nodes_size,nodes_expanded)


    # In[8]:

    # plot
    plt.xlabel('Number of Cities')
    plt.ylabel('Number of Nodes Generated')
    plt.scatter(nodes_size,nodes_generated)


# In[9]:

# Testing Example TSP Algorithm
ExamplestartNode = 1
Examplenodes = [1,2,3,4]
ExampleAdjHash =  {
    1:{ 
        1: 0,
        2: 10,
        3: 15,
        4: 20
      },
    2:{ 
        1: 10,
        2: 0,
        3: 35,
        4: 25
      },
    3:{ 
        1: 15,
        2: 35,
        3: 0,
        4: 30
      },
    4:{ 
        1: 20,
        2: 25,
        3: 30,
        4: 0
      }
}
ExampleMap = {
    1:(0,0),
    2:(1,1),
    3:(2,2),
    4:(3,3)
}
ExampleheuristicCost, Edges = PrimsAlgorithm(ExamplestartNode, Examplenodes, ExampleAdjHash)


# In[10]:

ExamplebestNode, ExampleTotalCost, ExamplenoOfNodesExpanded, ExampleNoOfNodesGenerated = solveAstar(ExamplestartNode, ExampleAdjHash, Examplenodes, ExampleMap, MSTHeuristic, False)


# In[11]:
print "I have taken an example TSP Problem Path Traversed"
print printPathTraversed(ExamplebestNode)
print "Total Cost is " + str(ExampleTotalCost)


# In[12]:

def getInputFromFile(fileName):
    try:
        Map = {}
        inverseMap = {}
        in_file = open(fileName, 'r')
        lines = in_file.readlines()
        ind = 0;
        for line in lines:
            if(line[0]=='1'):
                break
            else:
                ind+=1
        for line in lines[ind:len(lines)-1]:
            data = line.split()
            nodeNo = int(data[0])
            latitude, longitude = float(data[1]), float(data[2])
            coordinate = (latitude, longitude)
            Map[nodeNo] = coordinate
            inverseMap[coordinate] = nodeNo
    finally:
        in_file.close()
    return Map, inverseMap, computeAdjHashOfCompleteGraph(Map, DistanceComputation=EuclideanDistance)


#GEOM-norm
# M_PI = 3.14159265358979323846264

# def geom_edgelen(xi, xj, yi, yj):
#     lati = M_PI * xi / 180.0;
#     latj = M_PI * xj / 180.0;

#     longi = M_PI * yi / 180.0;
#     longj = M_PI * yj / 180.0;

#     q1 = math.cos (latj) * math.sin(longi - longj);
#     q3 = math.sin((longi - longj)/2.0);
#     q4 = math.cos((longi - longj)/2.0);
#     q2 = math.sin(lati + latj) * q3 * q3 - math.sin(lati - latj) * q4 * q4;
#     q5 = math.cos(lati - latj) * q4 * q4 - math.cos(lati + latj) * q3 * q3;
#     return (int) (6378388.0 * math.atan2(sqrt(q1*q1 + q2*q2), q5) + 1.0);



# In[13]:

# Optional Question Answers : (Q6.4)
def OptimizedMSTHeuristic(startNode, nodes, AdjHash):
#     print "COMPUTING Optimized HEURISTIC"
    minCostBackToStartNode = 0.0
    for node in nodes:
        if(node!=startNode):
            minCostBackToStartNode = min(minCostBackToStartNode, AdjHash[node][startNode])
    return MSTHeuristic(startNode, nodes, AdjHash) + minCostBackToStartNode

WCityMap, WCityinverseMap, WCityAdjHash = getInputFromFile('wi29.tsp.txt')
WCitystartNode = 1
WCitynodes = WCityMap.keys()
# print WCityMap


# In[14]:
print "For the World TSP Problem Path Traversed" 
WCitybestNode, WCityTotalCost, WCitynoOfNodesExpanded, WCityNoOfNodesExpanded = solveAstar(WCitystartNode, WCityAdjHash, WCitynodes, WCityMap, OptimizedMSTHeuristic, False)


# In[15]:

printPathTraversed(WCitybestNode)
print "Total Cost was %f" %(WCityTotalCost)


# Q 6.6
# Implement and test a hill-climbing method to solve TSPs. Compare the results with the optimal solutions obtained using A*
# 

# In[16]:

#http://homepage.divms.uiowa.edu/~hzhang/c231/ch12.pdf
#https://www.cs.cmu.edu/afs/cs/academic/class/15381-s07/www/slides/012507searchlocal.pdf
class HillNode:
    def __init__(self, x, y, nodeNumber, path_cost, parentNode, AdjHash):
        self.pos_x = x
        self.pos_y = y
        self.node_number = nodeNumber
        self.g_cost = path_cost
        self.parent = parentNode

def CostTour(Tour, AdjHash):
    prev = Tour[0].node_number
    Cost = 0.0
    for ind in range(1,len(Tour)):
        node = Tour[ind]
        Cost+=AdjHash[prev][node.node_number]
        prev = node.node_number
    #add Cost For Start as well
    Cost+=AdjHash[prev][0]
    return Cost
       
def printTour(Tour):
    Traversal = []
    for ind in range(0,len(Tour)):
        node = Tour[ind]
        Traversal.append(node.node_number)
    Traversal.append(Tour[0].node_number)
    print str(Traversal)
    return (Traversal)

def findNeighbour(Tour,AdjHash,pathCost):
    prev = 0
    # I pick two nodes randomly except start and swap
    index1=1
    index2=1
    newTour = copy.copy(Tour)
    Cost = 0
    changed = False
    while(index1==index2):
        index1 = random.randint(1, len(Tour)-1)
        index2 = random.randint(1, len(Tour)-1)
    swapNode = newTour[index1]
    newTour[index1] = newTour[index2]
    newTour[index2] = swapNode
    for ind in range(1,len(newTour)):
        node = Tour[ind]
        Cost+=AdjHash[prev][node.node_number]
        node.g_cost = Cost
        prev = node.node_number
    #add Cost For Start as well
    Cost+=AdjHash[prev][0]
    newTourCost = CostTour(newTour,AdjHash)
    oldTourCost = CostTour(Tour,AdjHash)
#     print "new Tour %s Traversal has a cost of %f while old Tour %s has a cost of %f" %(printTour(newTour), newTourCost, printTour(Tour), oldTourCost)
    if(newTourCost<oldTourCost):
        changed = True
        Tour = newTour
    return newTour, changed, Cost
        
    
def HillClimbing(startNodeNumber, AdjHash, listOfNodeNumbers, Map):
    Tour = []
    prevNode = -1
    prevNodeNumber = 0
    pathCost = 0.0
    MAX_ITER = 100
    iterations = 0
    for ind in range(0,len(listOfNodeNumbers)):
        nodeNumber = listOfNodeNumbers[ind]
        latitude = Map[nodeNumber][0]
        longitude = Map[nodeNumber][1]
        node = HillNode(latitude, longitude, nodeNumber, pathCost, prevNode, AdjHash)
        if(ind<len(listOfNodeNumbers)-1):
            nextNodeNumber = listOfNodeNumbers[ind+1]
        else:
            nextNodeNumber = listOfNodeNumbers[0]
        pathCost+= AdjHash[nodeNumber][nextNodeNumber];
        Tour.append(node);
        prevNode = node
    changed=True
#     print printTour(Tour);
    while(iterations < MAX_ITER):
        newTour, changed, pathCost = findNeighbour(Tour,AdjHash,pathCost)
        if(changed):
            Tour = copy.copy(newTour)
#             printTour(Tour)
        if(iterations >= MAX_ITER):
            break
        iterations+=1
#     print printTour(Tour)
#     print AdjHash
    return CostTour(Tour, AdjHash), Tour
        
        
        
    
    


# In[17]:

# http://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
print "Hill Climbing "
HillClimbingTotalCosts = []
failedFindingOptimalSolutionInHillClimbing = 0
for i in range(0,100):
    Map = MapsUsed[i]
    inverseMap = inverseMapsUsed[i]
    AdjHash = AdjHashUsed[i]
    startNode = 0
    nodes = Map.keys()
    hcCost, Tour = HillClimbing(startNode, AdjHash, nodes, Map)
    HillClimbingTotalCosts.append(hcCost)
    print "Hill CLimbing Cost : %f, Heuristic Cost: %f, Is Same : %d" %(HillClimbingTotalCosts[i], TotalCosts[i], isclose(TotalCosts[i], HillClimbingTotalCosts[i]))
    if(not isclose(TotalCosts[i], HillClimbingTotalCosts[i])):
#         if(TotalCosts[i]>HillClimbingTotalCosts[i]):
#             print AdjHash, printTour(Tour), PathsTraversed[i]
#         print TotalCosts[i], HillClimbingTotalCosts[i]
#         assert TotalCosts[i]<HillClimbingTotalCosts[i]
        failedFindingOptimalSolutionInHillClimbing+=1;
print "Hill Climbing Failed to Find Similar Solutions in %d cases out of 100" %(failedFindingOptimalSolutionInHillClimbing)


# Propose an inadmissible heuristic for the TSP problem and compare the performance of A* (number of nodes expanded and solution quality) with the inadmissible heuristic versus the admissible one.
# 

# In[18]:

# An inadmissible Heurisitic could be 10 Times the Minimum Spanning Tree because it is an overestimate of the true cost to reaching the solution
def InadmissibleMSTHeuristic(startNode, nodes, AdjHash):
#     print "COMPUTING INADMISSIBLE HEURISTIC"
    return 10*MSTHeuristic(startNode, nodes, AdjHash)


# bestFinalNode, TotalCost, noOfNodesExpanded, noOfNodesGenerated = solveAstar(startNode, AdjHash, nodes, Map, InadmissibleMSTHeuristic, True)
# print TotalCost


# In[19]:

InAdmissibleHeuristictime_taken = []
InAdmissibleHeuristicnodes_expanded = []
InAdmissibleHeuristicnodes_generated = []
InAdmissibleHeuristicTotalCosts = []

for i in range(0,100):
    Map = MapsUsed[i]
    inverseMap = inverseMapsUsed[i]
    AdjHash = AdjHashUsed[i]
    startNode = 0
    nodes = Map.keys()
    start_time = timeit.default_timer()
    bestFinalNode, TotalCost, noOfNodesExpanded, noOfNodesGenerated = solveAstar(startNode, AdjHash, nodes, Map, InadmissibleMSTHeuristic, False)
    elapsed = timeit.default_timer() - start_time
    InAdmissibleHeuristicnodes_generated.append(noOfNodesGenerated)
    InAdmissibleHeuristicnodes_expanded.append(noOfNodesExpanded)
    InAdmissibleHeuristictime_taken.append(elapsed)
    InAdmissibleHeuristicTotalCosts.append(TotalCost)
    


FailedToFindOptimalSolutions = 0
for i in range(0,100):
    print "Using inadmissible heuristic, Cost : %f, Admissible Heuristic Cost: %f, Is Same : %d" %(InAdmissibleHeuristicTotalCosts[i], TotalCosts[i], isclose(TotalCosts[i], InAdmissibleHeuristicTotalCosts[i]))
    # print InAdmissibleHeuristicTotalCosts[i], TotalCosts[i], isclose(TotalCosts[i],InAdmissibleHeuristicTotalCosts[i])
    if(not isclose(TotalCosts[i],InAdmissibleHeuristicTotalCosts[i])):
#         assert TotalCosts[i] < InAdmissibleHeuristicTotalCosts[i]
        FailedToFindOptimalSolutions+=1;

print "THE INADMISSIBLE HEURISTIC WAS UNABLE TO FIND SIMILAR SOLUTION IN %d CASES OUT OF 100" %(FailedToFindOptimalSolutions)

# Please uncomment for heuristic Plots
# inadmissibleHeuristicPlots()

def inadmissibleHeuristicPlots():
    # plot
    plt.xlabel('Number of Cities')
    plt.ylabel('Time in Seconds For Inadmissible Heuristic')
    plt.scatter(nodes_size,InAdmissibleHeuristictime_taken)


    # In[22]:

    # plot
    plt.xlabel('Number of Cities')
    plt.ylabel('Nodes Expanded For Inadmissible Heuristic')
    plt.scatter(nodes_size,InAdmissibleHeuristicnodes_expanded)


    # In[23]:

    # plot
    plt.xlabel('Number of Cities')
    plt.ylabel('Nodes Generated For Inadmissible Heuristic')
    plt.scatter(nodes_size,InAdmissibleHeuristicnodes_generated)

