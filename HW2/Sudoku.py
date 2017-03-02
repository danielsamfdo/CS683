import copy

# Function : Takes a file name and returns a board
def returnBoard(filename, rows, columns):
    with open(filename) as f:
        content = f.readlines()
        content = [x.split() for x in content] 
    board = []
    for row in range(rows):
        currentRow = []
        for column in range(columns):
            if((content[row][column])=='-'):
                value = 0
            else:
                value = int(content[row][column])
            currentRow.append(value)
        board.append(currentRow)
    return board
    
    
class SudokuPuzzle:

    # Unassigned Variable have a domain associated with it which can be found in the self.variableValuesAvailable[nRow][nCol], Assigned variables are discovered if the board[nRow][nRow]!=0
    def __init__(self, possibleValues, filename, board=None, initialize=True, rows=9, columns=9, boxRows=3, boxColumns=3):
        if(initialize):
            self.variableSize = rows * columns
            self.possibleValues = possibleValues
            self.rows = rows
            self.columns = columns
            self.boxRows = boxRows
            self.boxColumns = boxColumns
            if(board==None):
                self.board = returnBoard(filename,rows,columns)
            else:
                self.board = board
            self.updateDomainVariables()
            
    # CHECKS the row constraints for a given row.
    def rowAllDiff(self, row):
        dictV = {}
        for column in range(self.columns):
            if(self.board[row][column] in dictV):
                return False
            else:
                if(self.board[row][column]!=0):
                    dictV[self.board[row][column]] = 1
        return True
    
    # CHECKS the Column constraints for a given Column.
    def columnAllDiff(self, column):
        dictV = {}
        for row in range(self.rows):
            if(self.board[row][column] in dictV):
                return False
            else:
                if(self.board[row][column]!=0):
                    dictV[self.board[row][column]] = 1
        return True
    
    # CHECKS the Box constraints for a given Column.                  
    def boxAllDiff(self, r, c):
        dictV = {}
        boxRowStart = self.boxRows*(r/self.boxRows)
        boxColumnStart = self.boxColumns*(c/self.boxColumns)
        boxRowEnd = boxRowStart + self.boxRows
        boxColumnEnd = boxColumnStart + self.boxColumns
        for row in range(boxRowStart, boxRowEnd):
            for column in range(boxColumnStart, boxColumnEnd):
                if(self.board[row][column] in dictV):
                    return False
                else:
                    if(self.board[row][column]!=0):
                        dictV[self.board[row][column]] = 1
        return True

    # Update the Domain After Assignment - Currently unused Function
    def updateDomainAfterAssignment(self, rcIndex, value):
        neighbours = self.findNeighbours(rcIndex)
        for neighbour in neighbours:
            nRow, nCol = neighbour
            if(self.board[nRow][nCol]==0):
                if(value in self.variableValuesAvailable[nRow][nCol]):
                    self.variableValuesAvailable[nRow][nCol].remove(value)
        
    
    # This is a minor version of Technique 1.3 in http://www.su-doku.net/tech.php
    # we try to remove values from domains of other variables. Say in a given box, we have in a particular column we have two unassigned variables and both have a domain of \({1,6}\), then we can remove 1 and 6 from the domain of the unassigned variables in that particular column excluding the ones in the box. The same can be done for removal of values from domain of  unassigned variables in a row, if two unassigned variables in a given row in a box have the same domain. 
    def boxRemoveColumnAndRowValues(self):
        boxrowstart=[0,3,6]
        boxcolstart=[0,3,6]
        
        for arrayrowidx in range(3):
            for arraycolidx in range(3):
                BoxMap = {}
                IndicesOfBox = []
                for i in range(boxrowstart[arrayrowidx],boxrowstart[arrayrowidx]+3):
                    for j in range(boxcolstart[arraycolidx],boxcolstart[arraycolidx]+3):
                        IndicesOfBox.append((i,j))
                        if self.board[i][j]==0:
                            setValues = copy.deepcopy(self.variableValuesAvailable[i][j])
                            setValues = frozenset(setValues)
                            if(setValues not in BoxMap):
                                BoxMap[setValues] = [(i,j)]
                            else:
                                BoxMap[setValues].append((i,j))
                for key in BoxMap:
                    if(len(BoxMap[key]) == len(key) and len(key)==2):
                        # If there are two unassigned variables that take on exactly the same domain values and its size is two, we try to remove it(domain values) from either the interacting row/ column. 
                        r1,c1 = BoxMap[key][0]
                        r2,c2 = BoxMap[key][1]
                        rowOnly = False
                        colOnly = False
                        if(r1 == r2):
                            rowOnly = True
                        if(c1 == c2):
                            colOnly = True
                        for rcIndex in BoxMap[key]:
                            neighbours = self.findNeighbours(rcIndex)
                            for neighbour in neighbours:
                                nRow, nCol = neighbour
                                if(self.board[nRow][nCol]==0 and (neighbour not in IndicesOfBox)):
                                    # Neighbour should be in the row if rowOnly is True
                                    if(rowOnly and nRow == r1):
                                        for value in key:
                                            if(value in self.variableValuesAvailable[nRow][nCol]):
                                                self.variableValuesAvailable[nRow][nCol].remove(value)
                                    elif(colOnly and nCol == c1): 
                                        for value in key:
                                            if(value in self.variableValuesAvailable[nRow][nCol]):
                                                self.variableValuesAvailable[nRow][nCol].remove(value)

    
    # Function to check all the constraints based on the value that was currently assigned.
    def checkConstraints(self, row, column):
        toCheckVal = self.board[row][column]
        return self.rowAllDiff(row) and self.columnAllDiff(column) and self.boxAllDiff(row, column)
    
    def nakedpairs(self):
        for row in range(self.rows):
            for k in range(1,10):
                found=False
                for column in range(self.columns):
                    if(self.board[row][column]==k):
                        found=True
                        break
                if found==False:
                    count=0
                    for column in range(self.columns):
                        if(self.board[row][column] == 0 and k in self.variableValuesAvailable[row][column]):
                            pos=column
                            count+=1
                    if count==1:
                        self.assignment(row,pos,k)
        for column in range(self.columns):
            for k in range(1,10):
                found=False
                for row in range(self.rows):
                    if(self.board[row][column]==k):
                        found=True
                        break
                if found==False:
                    count=0
                    for row in range(self.rows):
                        if(self.board[row][column] == 0 and k in self.variableValuesAvailable[row][column]):
                            pos=row
                            count+=1
                    if count==1:
                        self.assignment(pos,column,k)
    
        boxrowstart=[0,3,6]
        boxcolstart=[0,3,6]
        for arrayrowidx in range(3):
            for arraycolidx in range(3):
                
                for k in range(1,10):
                    found=False
                    for i in range(boxrowstart[arrayrowidx],boxrowstart[arrayrowidx]+3):
                        for j in range(boxcolstart[arraycolidx],boxcolstart[arraycolidx]+3):
                            if self.board[i][j]==k:
                                found=True
                    if found==False:
                        count=0
                        for i in range(boxrowstart[arrayrowidx],boxrowstart[arrayrowidx]+3):
                            for j in range(boxcolstart[arraycolidx],boxcolstart[arraycolidx]+3):
                                if self.board[i][j] == 0 and k in self.variableValuesAvailable[i][j]:
                                    posx,posy=i,j
                                    count+=1
                        if count==1:
                            self.assignment(posx,posy,k)
                        
                        
   
    # Function to check if it reached the GOAL
    def reachedGoal(self):
        for row in range(self.rows):
            for column in range(self.columns):
                if(self.board[row][column]==0):
                    return False
        for row in range(self.rows):
            for column in range(self.columns):
                if(self.checkConstraints(row,column)):
                    continue
                else:
                    return None
        return True
    
    # Function to assign board values and update domain variables
    def assignment(self, row, column, value):
        self.board[row][column] = value
        rcIndex = (row,column)
        self.updateDomainAfterAssignment(rcIndex, value)
        # self.updateDomainVariables()
    
    # Function specific to back track to assign board values and it doesn't update domain variables
    def backtrackAssignment(self, row, column, value):
        self.board[row][column] = value
        
    # Function to update the Domain Variables for each unassigned variable
    # Unassigned Variable have a domain associated with it which can be found in the self.variableValuesAvailable[nRow][nCol], Assigned variables are discovered if the board[nRow][nRow]!=0
    def updateDomainVariables(self):
        domain = {}
        for value in self.possibleValues:
            domain[value] = 1;
        boxAvailableValues = []
        rowAvailableValues = []
        columnAvailableValues = []
        for row in range(self.rows):
            rowAvailableValues.append(copy.deepcopy(domain))
        for column in range(self.columns):
            columnAvailableValues.append(copy.deepcopy(domain))
        
        totalBoxes = (self.rows * self.columns)/(self.boxRows*self.boxColumns)
        for box in range(totalBoxes):
            boxAvailableValues.append(copy.deepcopy(domain))
            
        variableDomain = []
        for row in range(self.rows):
            variableDomain.append([])
            for column in range(self.columns):
                if(self.board[row][column]!=0):
                    boxAvailableValues[((3*(row/3))+(column/3))][self.board[row][column]] = 0
                    rowAvailableValues[row][self.board[row][column]] = 0
                    columnAvailableValues[column][self.board[row][column]] = 0

        for row in range(self.columns):
            for column in range(self.columns):  
                if(self.board[row][column]==0):
                    variableDomain[row].append(self.findPositionsPossibleValues(row, column, boxAvailableValues, rowAvailableValues, columnAvailableValues))
                else:
                    variableDomain[row].append(self.board[row][column])
        self.variableValuesAvailable = variableDomain
        self.boxValuesAvailable = boxAvailableValues
        self.rowValuesAvailable = rowAvailableValues
        self.columnValuesAvailable = columnAvailableValues
        return variableDomain, boxAvailableValues, rowAvailableValues, columnAvailableValues
        
    # given a row column position on board, find out all the values that the position can take on.
    def findPositionsPossibleValues(self, row, column, boxAvailableValues, rowAvailableValues, columnAvailableValues):
        BoxIndex = ((3*(row/3))+(column/3))
        boxHash = boxAvailableValues[BoxIndex]
        boxValueSet = set({k: v for k, v in boxHash.iteritems() if boxHash[k]==1})
        rowHash = rowAvailableValues[row]
        rowValueSet = set({k: v for k, v in rowHash.iteritems() if rowHash[k]==1})
        columnHash = columnAvailableValues[column]
        columnValueSet = set({k: v for k, v in columnHash.iteritems() if columnHash[k]==1})
        return boxValueSet & columnValueSet & rowValueSet

        
    def printBoard(self):
        for row in range(self.rows):
            print self.board[row]
            
    def printAvailableValues(self):
        for row in range(self.rows):
            for column in range(self.columns):
                if(self.board[row][column]==0):
                    print "Values available for row %d column %d are " %(row, column) 
                    print self.variableValuesAvailable[row][column]
    
    def unAssignedValuesCnt(self):
        unassignedCnt = 0;
        for row in range(self.rows):
            for column in range(self.columns):
                if(self.board[row][column]==0):
                    unassignedCnt+=1;
        return unassignedCnt
    
    def unAssignedValuesBranchFactor(self):
        vCnt = 0.0;
        domLength = 0.0;
        for row in range(self.rows):
            for column in range(self.columns):
                if(self.board[row][column]==0): # IF a value is unassigned , board[row][column] = 0 & self.variableValuesAvailable[row][column] is a set of available values it can take on.
                    domLength += len(self.variableValuesAvailable[row][column])
                    vCnt += 1
        return domLength/vCnt;

    
    def reachedFailure(self):
        for row in range(self.rows):
            for column in range(self.columns):
                if( (self.board[row][column]==0) and len(self.variableValuesAvailable[row][column])==0):
                    return True
                if(self.checkConstraints(row,column)):
                    continue
                else:
                    return False
        return False
    
    # Function to do the following
    # The First Easy Tactic Would be that we figure out the entire domain and the values a variable can take given its neighbours values. Once we know what values a variable can take based on its neighbours, we try to assign values on the board. Then we make assign to the variables whose domain size is 1, if present ie, when the variable has only one value in its domain, we assign it.
    # Once assignment is made, the next move would be to remove the assigned value from the domain of the neighbours. We repeat this process till there is no more variable having a single sized domain.
    def updateBoardAndDomainValues(self):
        for row in range(self.rows):
            for column in range(self.columns):
                if((self.board[row][column]==0) and len(self.variableValuesAvailable[row][column])==1):
                    #print "Adding the only elt to the board for %d, %d" %(row,column)
                    self.board[row][column] = list(self.variableValuesAvailable[row][column])[0]
        self.updateDomainVariables()
        needToCallAgain = False
        for row in range(self.rows):
            for column in range(self.columns):
                if((self.board[row][column]==0) and len(self.variableValuesAvailable[row][column])==1):
                    needToCallAgain = True
                    break
        if(needToCallAgain):
            self.updateBoardAndDomainValues()
        return
        
                    
    def revise(self, Xi, Xj):
        revised = False
        row, column = Xi
        otherRow, otherColumn = Xj
        if((self.board[row][column])!=0):
            varValuesAvailbleForXi = [self.board[row][column]]
        else:
            varValuesAvailbleForXi = copy.deepcopy(self.variableValuesAvailable[row][column])
        # ASSIGNED VARIABLE
        if((self.board[otherRow][otherColumn])!=0):
            varValuesAvailbleForXj = [self.board[otherRow][otherColumn]]
        else:
            # UNASSIGNED VARIABLE
            varValuesAvailbleForXj = copy.deepcopy(self.variableValuesAvailable[otherRow][otherColumn]) 
    
        for valueAvailableForXi in varValuesAvailbleForXi:
            satisfiedValue = False
            
            for otherValueAvailableForXj in varValuesAvailbleForXj:
                # THERE HAS TO BE SOME OTHER VALUE IN X_J TO SATISFY THE CONSTRAINT, IF THERE IS ATLEAST ONE OTHER VALUE IT SATISFIES THE CONSTRAINT
                if(otherValueAvailableForXj!=valueAvailableForXi):
                    satisfiedValue = True
            # IF THERE IS NO VALUE THAT SATISFIES THE CONSTRAINT, we REVISE it.
            if(satisfiedValue == False):
                self.variableValuesAvailable[row][column].remove(valueAvailableForXi)
                revised = True
        return revised
        
                    
    def addAllConstraintsAvailable(self):
        Constraints = []
        boxRowStart = 0 
        for boxRow in range(self.boxRows):
            boxColumnStart = 0
            for boxColumn in range(self.boxColumns):
                for value in self.boxValueCombinations(boxRowStart, boxColumnStart):
                    Constraints.append(value)
                boxColumnStart+=self.boxColumns
            boxRowStart += self.boxRows
        
        for row in range(self.rows):
            ColumnVariables = []
            for column in range(self.columns):
                ColumnVariables.append((row,column))
            for value in itertools.combinations(ColumnVariables, 2):
                Constraints.append(value)
        
        for column in range(self.columns):
            RowVariables = []
            for row in range(self.rows):
                RowVariables.append((row,column))
            for value in itertools.combinations(RowVariables, 2):
                Constraints.append(value)
        
        
        return Constraints 
    
    # FUNCTION : USED FOR FINDING OUT ALL THE CONSTRAINTS OF THE BOX
    def boxValueCombinations(self, r, c):
        boxVariables = []
        boxRowStart = self.boxRows*(r/self.boxRows)
        boxColumnStart = self.boxColumns*(c/self.boxColumns)
        boxRowEnd = boxRowStart + self.boxRows
        boxColumnEnd = boxColumnStart + self.boxColumns
        for row in range(boxRowStart, boxRowEnd):
            for column in range(boxColumnStart, boxColumnEnd):
                boxVariables.append((row, column))
        return itertools.combinations(boxVariables, 2)

    # FUNCTION : Return list of a rdInd(r,c) neighbours
    def findNeighbours(self, rcInd):
        r,c = rcInd
        Neighbours = set()
        for row in range(self.rows):
            Neighbours.add((row, c))
        for column in range(self.columns):
            Neighbours.add((r, column))
        
        boxRowStart = self.boxRows*(r/self.boxRows)
        boxColumnStart = self.boxColumns*(c/self.boxColumns)
        boxRowEnd = boxRowStart + self.boxRows
        boxColumnEnd = boxColumnStart + self.boxColumns
        for row in range(boxRowStart, boxRowEnd):
            for column in range(boxColumnStart, boxColumnEnd):
                Neighbours.add((row, column))

        # REMOVE THE INDEX OF THE CURRENT ONE BEING INSPECTED
        Neighbours.remove((r,c))
        return list(Neighbours)


# - ----------------------------------  END OF CLASS FUNCTIONS ------------------------------------------- -

#Simple Backtrack Recursive Code
def SIMPLEBACKTRACK(puzzle):
    global count
    global searchcount
   
    
    reached = puzzle.reachedGoal()
    if(reached == None):
        return False
    if(reached):
       
        return puzzle
    if(puzzle.reachedFailure()):
     
        return False
    
    # SELECT UNASSIGNED VARIABLE
    for row in range(puzzle.rows):
        for column in range(puzzle.columns):
            
            if(puzzle.board[row][column]==0):
                Row = row
                Column = column
                break;
    row = Row
    column = Column

                

    ValueFromRowColumnSatisfied = False
    possibleValuesInDomain = puzzle.possibleValues


    if len(possibleValuesInDomain)!=0:
        count+=len(possibleValuesInDomain)-1
    # GO Through each value Possible
    ValueFromRowColumnSatisfied = False
    for value in possibleValuesInDomain:

        board = copy.deepcopy(puzzle.board)
        new_puzzle = SudokuPuzzle(possibleValues, filename, board)
        new_puzzle.backtrackAssignment(row, column, value)
        # IF VALUE IS CONSISTENT GIVEN CONSTRAINTS
        if(not new_puzzle.checkConstraints(row,column)):
            
            continue # assignment was a bad assignment
        
        result = SIMPLEBACKTRACK(new_puzzle)
        
        if(result == False):
            continue
        else:
            ValueFromRowColumnSatisfied = True
            return result
    if(not ValueFromRowColumnSatisfied):
        
        return False
count=0
searchcount=0
#Function to find the unassigned square with smallest domain size 
def mrv(puzzle):
    minval=100
    for row in range(puzzle.rows):
        for column in range(puzzle.columns):
            if((puzzle.board[row][column]==0) and len(puzzle.variableValuesAvailable[row][column])<minval ):
                minval=len(puzzle.variableValuesAvailable[row][column])
                rowvalue=row
                columnvalue=column
    return rowvalue,columnvalue

#MRV Backtrack Algorithm
def BACKTRACKINGSEARCH(puzzle): 
    return BACKTRACK(puzzle)
#MRV Recursive Backtrack Code
def BACKTRACK(puzzle, simple_backtracking=False):
    global count
    global searchcount

    # puzzle.updateDomainVariables()
    reached = puzzle.reachedGoal()
    if(reached == None):
        return False
    if(reached):
        return puzzle
    if(puzzle.reachedFailure()):
        
        return False
    
    ValueFromRowColumnSatisfied = False
    rowvalues,columnvalues=mrv(puzzle)
    # We have the first mrv position (rowvalues, columnvalues) on the board, so the following tries to make an assignment to one mrv position on the board
    for row in [rowvalues]:
        for column in [columnvalues]:
            if(puzzle.board[row][column]==0):
                
                ValueFromRowColumnSatisfied = False
                possibleValuesInDomain = puzzle.possibleValues
                if len(puzzle.variableValuesAvailable[row][column])!=0:
                    count+=len(puzzle.variableValuesAvailable[row][column])-1
                for value in puzzle.variableValuesAvailable[row][column]:
                    
                    board = copy.deepcopy(puzzle.board)
                    new_puzzle = SudokuPuzzle(possibleValues, filename, board)
                    
                    # THIS ASSIGNMENT METHOD ALSO DOES FORWARD CHECKING in the Method updateDomainVariableAfterAssignment(), 
                    new_puzzle.assignment(row, column, value)

                    if(not new_puzzle.checkConstraints(row,column)):
                        continue # assignment was a bad assignment
                    
                    result = BACKTRACK(new_puzzle)
                    if(result == False):
                        continue
                    else:
                        ValueFromRowColumnSatisfied = True
                        return result
                if(not ValueFromRowColumnSatisfied):
                    return False

from collections import deque
import itertools
#AC3 algorithm
def AC_3(puzzle, assignedIndex):
    row, column = assignedIndex
    result = puzzle.addAllConstraintsAvailable()

    Deque = deque()
    for value in result:
        Deque.append(value)
    while(len(Deque)>0):
        Xi, Xj = Deque[0]
        Deque.popleft()
        if(puzzle.revise(Xi,Xj)):
            # IF UNASSIGNED VARIABLE SET of DOMAIN VALUES IS EMPTY
            if(puzzle.board[Xi[0]][Xi[1]] ==0 and len(puzzle.variableValuesAvailable[Xi[0]][Xi[1]])==0):
                return False
            Xi_neighbours = puzzle.findNeighbours(Xi)
           
            if(len(Xi_neighbours)>0):
                for value in (Xi_neighbours):
                    if(value != Xj):
                        Deque.append((value, Xi))

    return True

count=0
searchcount=0

#Recursive Backtrack with AC3 and MRV
def SolveUsingAC3(puzzle, simple_backtracking=False):
    global count
    global searchcount
   
    # puzzle.updateDomainVariables()
    reached = puzzle.reachedGoal()
    if(reached == None):
        return False
    if(reached):
   
        return puzzle
    if(puzzle.reachedFailure()):
    
        return False
    
    rowvalues,columnvalues=mrv(puzzle)
    for row in [rowvalues]:
        for column in [columnvalues]:
            if(puzzle.board[row][column]==0):
      
                ValueFromRowColumnSatisfied = False
                
                if len(puzzle.variableValuesAvailable[row][column])!=0:
                    count+=len(puzzle.variableValuesAvailable[row][column])-1
                for value in puzzle.variableValuesAvailable[row][column]:
                    
                    board = copy.deepcopy(puzzle.board)
                    new_puzzle = SudokuPuzzle(possibleValues, filename, board)
                    new_puzzle.assignment(row, column, value)
                    if(not AC_3(new_puzzle, (row, column))):
                     
                        continue # assignment was a bad assignment
                 
                    result = SolveUsingAC3(new_puzzle)
                    if(result == False):
                        continue
                    else:
                        ValueFromRowColumnSatisfied = True
                        return result
                if(not ValueFromRowColumnSatisfied):
                    return False
#Final Waterfall Algorithm
def WATERFALLSEARCH(puzzle): 
    return WATERFALL(puzzle)

#Recursive Code for Final Backtrack Algorithm
def WATERFALL(puzzle, simple_backtracking=False):
    global count
    global searchcount
   
  
    puzzle.updateBoardAndDomainValues()
    puzzle.nakedpairs()
    puzzle.boxRemoveColumnAndRowValues();
    reached = puzzle.reachedGoal()
    if(reached == None):
        return False
    if(reached):
    
        return puzzle
    if(puzzle.reachedFailure()):
        
        
        return False
    
    rowvalues,columnvalues=mrv(puzzle)
    for row in [rowvalues]:
        for column in [columnvalues]:
            if(puzzle.board[row][column]==0):
                
         
                ValueFromRowColumnSatisfied = False
                
                if len(puzzle.variableValuesAvailable[row][column])!=0:
                    count+=len(puzzle.variableValuesAvailable[row][column])-1
                for value in puzzle.variableValuesAvailable[row][column]:
                    
                    board = copy.deepcopy(puzzle.board)
                    new_puzzle = SudokuPuzzle(possibleValues, filename, board)
                    new_puzzle.assignment(row, column, value)
                    if(not AC_3(new_puzzle, (row, column))):
                        
                        continue # assignment was a bad assignment

                    result = WATERFALL(new_puzzle)
                    if(result == False):
                        continue
                    else:
                        ValueFromRowColumnSatisfied = True
                        return result
                if(not ValueFromRowColumnSatisfied):
                    return False

possibleValues = [1,2,3,4,5,6,7,8,9]

#Function to call SimpleBacktrack
def solveSudokuUsingSimplebackTrack(filename):
    possibleValues = [1,2,3,4,5,6,7,8,9]
    global count
    global searchcount
    count=0
    searchcount=0
    puzzle = SudokuPuzzle(possibleValues, filename)
   
    
    return SIMPLEBACKTRACK(puzzle)

#Function to call MRV
def solveSudokuUsingMRV(filename):
    possibleValues = [1,2,3,4,5,6,7,8,9]
    global count
    global searchcount
    count=0
    searchcount=0
    puzzle = SudokuPuzzle(possibleValues, filename)
    
    return BACKTRACKINGSEARCH(puzzle)

#Function to call AC3+MRV
def solveSudokuUsingAC3(filename):
    possibleValues = [1,2,3,4,5,6,7,8,9]
    global count
    global searchcount
    count=0
    searchcount=0
    puzzle = SudokuPuzzle(possibleValues, filename)
    return SolveUsingAC3(puzzle)

#Function to call Final Waterfall Model
def solveSudokuUsingWATERFALL(filename):
    possibleValues = [1,2,3,4,5,6,7,8,9]
    global count
    global searchcount
    count=0
    searchcount=0
    puzzle = SudokuPuzzle(possibleValues, filename)
    
    return WATERFALLSEARCH(puzzle)

#Function to display guesscounts
def countlister(filename):
    puzzle = solveSudokuUsingSimplebackTrack(filename)
    puzzle.printBoard()
    print "GUESSES FOR SIMPLE BACKTRACK = ",count
    puzzle = solveSudokuUsingMRV(filename)
    print "GUESSES FOR MRV = ",count
    puzzle = solveSudokuUsingAC3(filename)
    print "GUESSES FOR AC3 = ",count
    puzzle = solveSudokuUsingWATERFALL(filename)
    print "GUESSES FOR WATERFALL = ",count
filename=""
numbers=["puzzle/puz-001.txt","puzzle/puz-002.txt","puzzle/puz-010.txt","puzzle/puz-015.txt","puzzle/puz-025.txt","puzzle/puz-026.txt","puzzle/puz-048.txt","puzzle/puz-051.txt","puzzle/puz-062.txt","puzzle/puz-076.txt","puzzle/puz-081.txt","puzzle/puz-082.txt","puzzle/puz-090.txt","puzzle/puz-095.txt","puzzle/puz-099.txt","puzzle/puz-100.txt"]
for i in numbers:
    print i
    global filename
    filename=i
    countlister(i)
#BONUS: Function to classify difficulty of puzzles
def diffSolver(filename):
    puzzle = puzzle = SudokuPuzzle(possibleValues, filename)
    puzzle.updateBoardAndDomainValues()
    reached = puzzle.reachedGoal()
    # puzzle.print
    if(reached):
        return filename + "Light and Easy"
    i = 0
    while True:
        unassignedCount = puzzle.unAssignedValuesCnt()
        puzzle.nakedpairs();
        puzzle.updateBoardAndDomainValues()
        if(unassignedCount == puzzle.unAssignedValuesCnt()):
            break
    
    unassignedCount = puzzle.unAssignedValuesCnt()
    reached = puzzle.reachedGoal()
    if(reached):
        return filename + "Moderate"
    unassignedCount = puzzle.unAssignedValuesCnt()
  
    count=0
    searchcount=0
    puzzle = SudokuPuzzle(possibleValues, filename)
    bfFactor = puzzle.unAssignedValuesBranchFactor()
    print bfFactor*unassignedCount
    if(bfFactor*unassignedCount <= 125):
        return "Demanding"
    puzzle = SolveUsingAC3(puzzle)
   
    return "beware! very challenging"



filename=""
numbers=["puzzle/puz-001.txt","puzzle/puz-002.txt","puzzle/puz-010.txt","puzzle/puz-015.txt","puzzle/puz-025.txt","puzzle/puz-026.txt","puzzle/puz-048.txt","puzzle/puz-051.txt","puzzle/puz-062.txt","puzzle/puz-076.txt","puzzle/puz-081.txt","puzzle/puz-082.txt","puzzle/puz-090.txt","puzzle/puz-095.txt","puzzle/puz-099.txt","puzzle/puz-100.txt"]
for i in numbers:
    print i
    global filename
    print diffSolver(i)
