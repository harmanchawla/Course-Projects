pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



''' 
Notes to navigate and understand the code: 
1. I have added one line comments over all major lines of code in BFSAgent to communicate their use. 
2. Since DFS is pretty similar in approach except a few caveats, I have refrained for cluttering the codes.
3. A* is a bit different so I do explain it. 
4. I did try implementing my own stack, queue and priority queue functions, 
   but I have gone around it by using built-in list functions in Python
'''

from pacman import Directions
from game import Agent
from heuristics import *
import random


# class PriorityQueue(object):
#     def __init__(self):
#         self.queue = []
#         self.cost = []

#     def __str__(self):
#         return ' '.join([str(i) for i in self.queue])

#         # for checking if the queue is empty

#     def empty(self):
#         return len(self.queue) == []

#         # for inserting an element in the queue

#     def insert(self, data):
#         self.queue.append(data)
#         self.cost.append(data)

#     # for popping an element based on Priority
#     def delete(self):
#         try:
#             max = 0
#             for i in range(len(self.queue)):
#                 if self.queue[i] > self.queue[max]:
#                     max = i
#             item = self.queue[max]
#             del self.queue[max]
#             del self.cost[max]
#             return item
#         except IndexError:
#             print()
#             exit()


class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0, len(actions) - 1)]


class OneStepLookAheadAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        legal = state.getLegalPacmanActions()
        print('Legal actions: ', legal)
        # get all the successor state for these actions
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]

        print('Successors: ', successors, 'Game state: ', successors[0][0].generatePacmanSuccessor('EAST'))
        future_states = successors.generatePacmanSuccessor(legal[0])
        print('Successor 1 trial', future_states)
        # evaluate the successor states using admissibleHeuristic heuristic
        scored = [(admissibleHeuristic(state), action) for state, action in successors]
        print('Scores: ', scored)
        # get best choice
        bestScore = min(scored)[0]
        print('Best score: ', bestScore)
        # get all actions that lead to the highest score
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        print('Best Actions: ', bestActions)
        # return random action from the list of the best actions
        print(random.choice(bestActions))
        return random.choice(bestActions)


class BFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):

        # using python list as a queue for BFS
        queue = []

        # State Information is a dict which store key value pairs of states for each move (frame)
        stateInfo = {}

        # Get legal actions for the move
        legal = state.getLegalPacmanActions()

        # Iterate over legal actions to get corresponding successor states
        # results in something like list of [(<Game Instance>, 'MOVE')]
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]

        # for each successive possible state, calculate cost and add it to the queue as a element. 
        for successor in successors:
            # store the game State (key) and move leading to it (value)
            # just makes it easy to return later
            stateInfo[successor[0]] = successor[1]
            queue.append((successor[0], admissibleHeuristic(successor[0])))

        while queue:

            # extract the MOVE 
            node = queue[0][0]

            # Taking everything but the first element (queue[0]), 
            # essentially deleting the first element. Ergo, queue. 
            queue = queue[1:] 

            # check if it is a terminal state i.e. win state or lose state
            if node.isLose():
                # because we need to find the win state
                continue

            elif node.isWin():
                # we found the win state so just return the action which lead to it. 
                return stateInfo[node]

            # same as I did above. For each move, get possible action 
            # and for all actions, all following states. 

            # Note how this time I have not used a element but just a list of game instances. 
            legal = node.getLegalPacmanActions()
            successors = [node.generatePacmanSuccessor(action) for action in legal]

            ''' 
            Else condition explained: is not None, simply add {successor: MOVE} to the dict
            IF condition explained: For each successor, if None successor is seen then go over the queue deleting 
            all lose states. If I find the win state in the process of traversing the queue, I return it. 
            '''
            for successor in successors:
                if successor is None:
                    for queueNode in queue:
                        if queueNode[0].isWin():
                            return stateInfo[queueNode]
                        elif queueNode[0].isLose():
                            queue.remove(queueNode)
                    return stateInfo[max(queue,key=lambda x : x[1])[0]]
                stateInfo[successor] = stateInfo[node]
                queue.append((successor,admissibleHeuristic(successor)))


        return Directions.STOP

class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        stack = []
        stateInfo = {}

        legal = state.getLegalPacmanActions()
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]
        for successor in successors:
            stateInfo[successor[0]] = successor[1]
            stack.append((successor[0], admissibleHeuristic(successor[0])))

        while stack:
            node = stack[0][0]

            # this is the only major change as I slice everything but the last element
            stack = stack[:-1] 

            # check if it is a terminal state i.e. win state or lose state
            if node.isLose():
                # because we need to find the win state
                continue

            elif node.isWin():
                # we found the win state so just return the action which lead to it. 
                return stateInfo[node]

            legal = node.getLegalPacmanActions()
            successors = [node.generatePacmanSuccessor(action) for action in legal]

            for successor in successors:
                if successor is None:
                    for stackNode in stack:
                        if stackNode[0].isWin():
                            return stateInfo[stackNode]
                        elif stackNode[0].isLose():
                            stack.remove(stackNode)
                    return stateInfo[max(stack, key=lambda x : x[1])[0]]
                stateInfo[successor] = stateInfo[node]
                stack.append((successor, admissibleHeuristic(successor)))

        return Directions.STOP

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # Computes f(x) = g(x) + h(x)
    def costEstimate(self,state,stateInfo,root):
        return int(stateInfo[state][1]) -(admissibleHeuristic(state) - admissibleHeuristic(root));

    # GetAction Function: Called with every frame
    def getAction(self, state):
        priorityQueue = []
        stateInfo = {}
        depth = 0

        # Get legal actions for the move
        legal = state.getLegalPacmanActions()
        # results in something like list of [(<Game Instance>, 'MOVE')]
        successors = [(state.generatePacmanSuccessor(action), action) for action in legal]

        # for each successive possible state, calculate cost and add it to the queue as a element. 
        for successor in successors:
            # store the game State (key) and move leading to it along with the depth (value = (MOVE, depth))
            # just makes it easy to return later
            stateInfo[successor[0]] = (successor[1],depth) 
            priorityQueue.append((successor[0], self.costEstimate(successor[0], stateInfo, state)))

        while priorityQueue:

            # writing priority queue logic. 
            element = min(priorityQueue, key= lambda x : x[1])
            node = element[0]
            priorityQueue.remove(element)

            # check if it is a terminal state i.e. win state or lose state
            if node.isLose():
                # because we need to find the win state
                continue

            elif node.isWin():
                # we found the win state so just return the action which lead to it. 
                return stateInfo[node][0]

            
            # Get legal actions for the move
            legal = node.getLegalPacmanActions()

            # Iterate over legal actions to get corresponding successor states
            successors = [node.generatePacmanSuccessor(action) for action in legal]

            ''' 
            Else condition explained: is not None, simply add {successor: (MOVE, new_depth)} to the dict
            IF condition explained: For each successor, if None successor is seen then go over the queue deleting 
            all lose states. If I find the win state in the process of traversing the queue, I return it. 
            '''

            for successor in successors:
                if successor is None:
                    for priorityQueueNode in priorityQueue:
                        if priorityQueueNode[0].isWin():
                            return stateInfo[priorityQueueNode][0]
                        elif priorityQueueNode[0].isLose():
                            priorityQueue.remove(priorityQueueNode)

                    return stateInfo[min(priorityQueue,key= lambda x : x[1])[0]][0]
                stateInfo[successor] = (stateInfo[node][0],stateInfo[node][1] + 1)
                queue.append((successor, self.costEstimate(successor, stateInfo, state)))
        return Directions.STOP