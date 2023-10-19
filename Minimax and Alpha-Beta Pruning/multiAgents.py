# multiAgents.py
# --------------
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


from pacman import GameState
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(gameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return gameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, agentIndex, depth, currentgameState):
            if currentgameState.isWin() or currentgameState.isLose() or depth == 0: #If terminal state, return evaluation
                return self.evaluationFunction(currentgameState)

            legalActions = currentgameState.getLegalActions(agentIndex) #Get legal actions for current agent
            if agentIndex == 0:  # Pacman's turn
                bestScore = float("-inf") 
                for action in legalActions:
                    successorState = currentgameState.generateSuccessor(agentIndex, action) #Get successor state for given action
                    score = self.minimax(1, depth, successorState) # Recursive step to get score for successor state
                    bestScore = max(bestScore, score)
                return bestScore
            else:  # Ghosts' turn 
                bestScore = float("inf")
                nextAgentIndex = (agentIndex + 1) % currentgameState.getNumAgents() # Run through the agents in a cyclic manner
                for action in legalActions:
                    successorState = currentgameState.generateSuccessor(agentIndex, action)
                    if agentIndex == currentgameState.getNumAgents() - 1:#If current agent is the final ghost, then increase depth
                        score = self.minimax(nextAgentIndex, depth - 1, successorState) 
                    else: #Else, continue with the same depth
                        score = self.minimax(nextAgentIndex, depth, successorState)
                    bestScore = min(bestScore, score)
                return bestScore

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        legalActions = gameState.getLegalActions(self.index)
        scores = []
        for action in legalActions:
            successorState = gameState.generateSuccessor(self.index, action)
            score = self.minimax(1, self.depth, successorState)
            scores.append(score)
        bestScoreIndex = scores.index(max(scores)) 
        return legalActions[bestScoreIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def alpha_beta_pruning(agentIndex, depth, gameState, alpha, beta): 
            if gameState.isWin() or gameState.isLose() or depth == 0: # If terminal state, return evaluation
                return self.evaluationFunction(gameState)

            legalActions = gameState.getLegalActions(agentIndex)
            if agentIndex == 0:  
                bestScore = float("-inf")
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    score = alpha_beta_pruning(1, depth, successorState, alpha, beta)  # Recursive step to get score for successor state
                    bestScore = max(bestScore, score)
                    alpha = max(alpha, bestScore)
                    if bestScore > beta: # Prune the tree, because the minimizing player will never choose this path
                        return bestScore
                return bestScore
            else:  
                bestScore = float("inf")
                nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents() # Run through the agents in a cyclic manner
                for action in legalActions:
                    successorState = gameState.generateSuccessor(agentIndex, action)
                    if agentIndex == gameState.getNumAgents() - 1: # If current agent is the final ghost, then increase depth
                        score = alpha_beta_pruning(nextAgentIndex, depth - 1, successorState, alpha, beta)
                    else: # Else, continue with the same depth
                        score = alpha_beta_pruning(nextAgentIndex, depth, successorState, alpha, beta)
                    bestScore = min(bestScore, score)
                    beta = min(beta, bestScore)
                    if bestScore < alpha:  # Prune the tree, because the maximizing player will never choose this path
                        return bestScore
                return bestScore

        legalActions = gameState.getLegalActions(0)
        bestScore = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        bestAction = None
        for action in legalActions:
            successorState = gameState.generateSuccessor(0, action)
            score = alpha_beta_pruning(1, self.depth, successorState, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
