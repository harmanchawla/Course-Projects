# pacmanAgents.py
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



from pacman import Directions
from game import Agent
from heuristics import *
import random


# class Node(node, parent_node = None, cost, new_cost):
#     def __init__():
#         self.current_node = node
#         self.parent_node = parent_node
#         self.child_nodes = []
#         self.parent_cost = cost
#         self.current_cost = new_cost
#


class RandomAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # get all legal actions for pacman
        actions = state.getLegalPacmanActions()
        # returns random action from all the valide actions
        return actions[random.randint(0,len(actions)-1)]

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
        print('Successors: ', successors)
        # evaluate the successor states using scoreEvaluation heuristic
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
        # get all legal actions possible
        legal = state.getLegalPacmanActions()

        //possibleMoves
        //minCost and minChild
        //for (Node child : legal)
            //compare minPath for each child and update minCost and minChild accordingly

     def bfsgenMinPath (root):
            // Make a Queue
            //Add the root node to the queue


            //minCost
            //while (queue.size() != 0)
                //pop Head
                //check if win
                //if win update micost
                //else push root children in the Queue

            // return micCost

            def dfsminpath(root):
                if (root.children == 0) {
                return root.cost;
                }
                //minCost
                for (child: root.children) {
                    cost = dfsMin(child);
                    //update minCost and minChild
                }

                return mincost+root.cost;




class DFSAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write DFS Algorithm instead of returning Directions.STOP
        return Directions.STOP

class AStarAgent(Agent):
    # Initialization Function: Called one time when the game starts
    def registerInitialState(self, state):
        return;

    # GetAction Function: Called with every frame
    def getAction(self, state):
        # TODO: write A* Algorithm instead of returning Directions.STOP
        return Directions.STOP
