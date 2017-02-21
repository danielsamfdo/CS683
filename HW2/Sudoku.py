
# coding: utf-8

# In[167]:

import copy
def returnBoard(filename, rows, columns):
    # SRC: http://stackoverflow.com/questions/3277503/how-to-read-a-file-line-by-line-into-a-list
    with open(filename) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
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
        # print currentRow
    return board
    
    
class SudokuPuzzle:
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
            # print "Creating domainVariables"
            self.updateDomainVariables()
            # print self.domainVariables
            
    def rowAllDiff(self, row):
        dictV = {}
        for column in range(self.columns):
            if(self.board[row][column] in dictV):
                return False
            else:
                if(self.board[row][column]!=0):
                    dictV[self.board[row][column]] = 1
        return True
    
    def columnAllDiff(self, column):
        dictV = {}
        for row in range(self.rows):
            if(self.board[row][column] in dictV):
                return False
            else:
                if(self.board[row][column]!=0):
                    dictV[self.board[row][column]] = 1
        return True
    
    def boxAllDiff(self, r, c):
        dictV = {}
        boxRowStart = self.boxRows*(r/self.boxRows)
        boxColumnStart = self.boxColumns*(c/self.boxColumns)
        boxRowEnd = boxRowStart + self.boxRows
        boxColumnEnd = boxColumnStart + self.boxColumns
        # print "CHECKING BOX (%d, %d) " %(r,c) 
        for row in range(boxRowStart, boxRowEnd):
            for column in range(boxColumnStart, boxColumnEnd):
                if(not (row == r and column==c)):
                    # print (row,column)
                    if(self.board[row][column] in dictV):
                        # self.printBoard()
                        return False
                    else:
                        if(self.board[row][column]!=0):
                            dictV[self.board[row][column]] = 1
        return True
    
    def checkConstraints(self, row, column):
        toCheckVal = self.board[row][column]
        return self.rowAllDiff(row) and self.columnAllDiff(column) and self.boxAllDiff(row, column)
    
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
    
    def assignment(self, row, column, value):
        self.board[row][column] = value
        self.updateDomainVariables()
    
    def updateDomainVariables(self):
        domain = {}
        for value in self.possibleValues:
            domain[value] = 1;
        boxAvailableValues = []
        rowAvailableValues = []
        columnAvailableValues = []
        # copy.copy(domain)
        for row in range(self.rows):
            rowAvailableValues.append(copy.deepcopy(domain))
        for column in range(self.columns):
            columnAvailableValues.append(copy.deepcopy(domain))
        
        totalBoxes = (self.rows * self.columns)/(self.boxRows*self.boxColumns)
        for box in range(totalBoxes):
            boxAvailableValues.append(copy.deepcopy(domain))
            
        variableDomain = []
        # Add used values 
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
        # print variableDomain, boxAvailableValues, rowAvailableValues, columnAvailableValues
        self.variableValuesAvailable = variableDomain
        self.boxValuesAvailable = boxAvailableValues
        self.rowValuesAvailable = rowAvailableValues
        self.columnValuesAvailable = columnAvailableValues
        return variableDomain, boxAvailableValues, rowAvailableValues, columnAvailableValues
        
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
    
    def updateBoardAndDomainValues(self):
        for row in range(self.rows):
            for column in range(self.columns):
                if((self.board[row][column]==0) and len(self.variableValuesAvailable[row][column])==1):
                    #print "Adding the only elt to the board for %d, %d" %(row,column)
                    self.board[row][column] = list(self.variableValuesAvailable[row][column])[0]
        self.updateDomainVariables()
        #print "Updated Domain Variables"
        #print self.variableValuesAvailable
        needToCallAgain = False
        for row in range(self.rows):
            for column in range(self.columns):
                if((self.board[row][column]==0) and len(self.variableValuesAvailable[row][column])==1):
                    needToCallAgain = True
                    break
        if(needToCallAgain):
            self.updateBoardAndDomainValues()
        return
        
                    
    def OrderedDomainValues(self, row, column):
        BoxIndex = ((3*(row/3))+(column/3))
        boxHash = self.boxAvailableValues[BoxIndex]
        boxValueSet = set({k: v for k, v in boxHash.iteritems() if boxHash[k]==1})
        rowHash = self.rowAvailableValues[row]
        rowValueSet = set({k: v for k, v in rowHash.iteritems() if rowHash[k]==1})
        columnHash = self.columnAvailableValues[column]
        columnValueSet = set({k: v for k, v in columnHash.iteritems() if columnHash[k]==1})
        
        for value in self.variableValuesAvailable[row][column]:
            value not in self.rowAvailableValues[row] and value not in self.columnAvailableValues[column]
            


# In[225]:

def BACKTRACKINGSEARCH(puzzle): 
    return BACKTRACK(puzzle)

def BACKTRACK(puzzle):
    # print "**********"
    # puzzle.printBoard()
    # print "**********"
    # puzzle.updateBoardAndDomainValues()
    reached = puzzle.reachedGoal()
    if(reached == None):
        return False
    if(reached):
        print "SOLUTION REACHED"
        puzzle.printBoard()
        # print puzzle.columnAllDiff(1);
        return puzzle
    if(puzzle.reachedFailure()):
        print "Failure Reached"
        puzzle.printBoard()
        
        return False
    
    
    for row in range(puzzle.rows):
        for column in range(puzzle.columns):
            if(puzzle.board[row][column]==0):
                print "Reached %d %d Pos" %(row, column)
                # orderedBestPossibleValues = OrderedDomainValues(puzzle, row, column)
                puzzle.printAvailableValues()
                # if(len(puzzle.variableValuesAvailable[row][column])==0):
                #    return False
                ValueFromRowColumnSatisfied = False
                for value in puzzle.variableValuesAvailable[row][column]:
                    board = copy.copy(puzzle.board)
                    new_puzzle = SudokuPuzzle(possibleValues, filename, board)
                    new_puzzle.assignment(row, column, value)
                    print "Assigning position (%d %d) a value of %d" %(row, column, value)
                    if(not new_puzzle.checkConstraints(row,column)):
                        print "CONSTRAINTS FAILED For value : %d" %(value)
                        new_puzzle.printBoard()
                        continue # assignment was a bad assignment
                    # print "value : %d" %(value)
                    
                    new_puzzle.printBoard()
                    result = BACKTRACK(new_puzzle)
                    # ValueFromRowColumnSatisfied = result or ValueFromRowColumnSatisfied
                    if(result == False):
                        continue
                    else:
                        ValueFromRowColumnSatisfied = True
                        return result
                if(not ValueFromRowColumnSatisfied):
                    print "BackTracking for ROW %d COLUMN %d" %(row, column)
                    return False
                
                
            # inferences ←INFERENCE(csp,var,value) 
            # if inferences!= failure then
                # add inferences to assignment
                # result ← BACKTRACK(assignment, csp) 
                # if result != failure then
                #    return result
        # remove {var = value} and inferences from assignment
    # return failure


# In[228]:

possibleValues = [1,2,3,4,5,6,7,8,9]
filename = "sudoku_sample3.txt"
#filename = "puzzle/puz-001.txt"
def solveSudoku():
    puzzle = SudokuPuzzle(possibleValues, filename)
    # puzzle.printBoard()
    # puzzle.printAvailableValues()
#     return puzzle
    return BACKTRACKINGSEARCH(puzzle)
    
puzzle = solveSudoku()
#puzzle.printBoard()

