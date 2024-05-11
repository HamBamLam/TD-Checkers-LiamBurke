import os
import copy
import time
import random
import numpy as np
from functools import reduce
import math

class Game(object):

    WIDTH = 4
    HEIGHT = 8
    PAWNS = ['x', 'o']
    KINGS = ['X', 'O']
    EMPTY = '-'
    BACKWARDS_PLAYER = 'o'

    def __init__(self, grid=None, players=None, prev_grid=None):
        """
        Define a new game object
        """
        if grid is not None:
            self.grid = copy.deepcopy(grid)
            self.prev_grid = prev_grid
            self.players = players
            self.kings = Game.KINGS
            return
        self.players = Game.PAWNS
        self.kings = Game.KINGS
        self.grid = [[j,j,j,j] for j in [Game.PAWNS[0], Game.PAWNS[0], Game.PAWNS[0], Game.EMPTY, Game.EMPTY, Game.PAWNS[1], Game.PAWNS[1], Game.PAWNS[1]]]
        self.prev_grid = copy.deepcopy(self.grid)
        self.cap = [0,0]
    @staticmethod
    def new():
        game = Game()
        game.reset()
        return game
    
    def get_piece_count(self,player):
        ind = 0
        if player == self.players[1]:
            ind = 1
        pawn = 0
        king = 0
        for row in range(Game.HEIGHT):
            for col in range(Game.WIDTH):
                if self.grid[row][col] == Game.PAWNS[ind]:
                    pawn+=1
                elif self.grid[row][col] == Game.KINGS[ind]:
                    king+=1
        return pawn, king

    def extract_features(self, player):
        features = []
        #features is board position for each player, pawn/king counts for each player, and player turn indicator
        if player == self.players[0]:
            for row in range(Game.HEIGHT):
                feats = [0.] * Game.WIDTH
                for col in range(Game.WIDTH):
                    if self.grid[row][col] == player:
                        feats[col]+=1
                features+=feats
            u, k = self.get_piece_count(player)
            features.append(u)
            features.append(k)
            opp = self.opponent(player)
            for row in range(Game.HEIGHT):
                feats = [0.] * Game.WIDTH
                for col in range(Game.WIDTH):
                    if self.grid[row][col] == opp:
                        feats[col]+=1
                features+=feats
            u_o,k_o = self.get_piece_count(opp)
            features.append(u_o)
            features.append(k_o)
            self.get_possible_next_moves(player)
            features.append(self.cap[0])
        elif player == self.players[1]:
            grid = copy.deepcopy(self.grid)
            grid.reverse()
            for row in range(Game.HEIGHT):
                feats = [0.] * Game.WIDTH
                for col in range(Game.WIDTH):
                    if grid[row][col] == player:
                        feats[col]+=1
                features+=feats
            u, k = self.get_piece_count(player)
            features.append(u)
            features.append(k)
            opp = self.opponent(player)
            for row in range(Game.HEIGHT):
                feats = [0.] * Game.WIDTH
                for col in range(Game.WIDTH):
                    if grid[row][col] == opp:
                        feats[col]+=1
                features+=feats
            u_o,k_o = self.get_piece_count(opp)
            features.append(u_o)
            features.append(k_o)
            self.get_possible_next_moves(player)
            features.append(self.cap[1])
        return np.array(features).reshape(1, -1)

    def play(self, players, draw = False):
        player_num = random.randint(0, 1)
        turn_count = 0
        while not self.is_over(player_num) and turn_count < 500:
            self.next_step(players[player_num], player_num)
            player_num = (player_num + 1) % 2
            turn_count+=1
            if draw:
                print("TURN: ", turn_count, "PLAYER: ", player_num)
                self.print_board()
                print("-------------------------------------------------------------------")
                
        if turn_count >= 500:
            return 2
        return (player_num+1)%2

    def next_step(self, player, player_num):
        self.take_turn(player)

    def take_turn(self, player):
        moves = self.get_actions(player.player)
        move = player.get_action(moves, self) if moves else None
        if move:
            self.take_action(move)
    
    def clone(self):
        """
        Return an exact copy of the game. Changes can be made
        to the cloned version without affecting the original.
        """
        return Game(grid=self.grid, players = self.players, prev_grid=self.prev_grid)

    def take_action(self, move):
        """
        Makes a given move on the board, and (as long as is wanted) switches the indicator for
        which players turn it is.
        """
        self.prev_grid = copy.deepcopy(self.grid)
        if abs(move[0][0] - move[1][0]) == 2:
            for j in range(len(move) - 1):
                if move[j][0] % 2 == 1:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j][1]
                    else:
                        middle_y = move[j + 1][1]
                else:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j + 1][1]
                    else:
                        middle_y = move[j][1]
                        
                self.grid[int((move[j][0] + move[j + 1][0]) / 2)][middle_y] = self.EMPTY
                
                
        self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.grid[move[0][0]][move[0][1]]
        if move[len(move) - 1][0] == self.HEIGHT - 1 and self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.players[0]:
            self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.kings[0]
        elif move[len(move) - 1][0] == 0 and self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.players[1]:
            self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.kings[1]
        else:
            self.grid[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.grid[move[0][0]][move[0][1]]
        self.grid[move[0][0]][move[0][1]] = Game.EMPTY

    def undo_action(self):
        self.grid = copy.deepcopy(self.prev_grid)

    def get_actions(self, player):
        """
        Get set of all possible move tuples
        """
        moves = self.get_possible_next_moves(player)

        return moves

    def not_spot(self, loc):
        """
        Finds out of the spot at the given location is an actual spot on the game board.
        """
        if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or loc[1] > self.WIDTH - 1:
            return True
        return False
    
    
    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        
        NOTE:
        Might want to not use this for the sake of computational time.
        """
        return self.grid[loc[0]][loc[1]]
    
    def get_spot_num(self, loc):
        spot = self.grid[loc[0]][loc[1]]
        if spot == Game.EMPTY:
            return 0
        elif spot == self.players[0] or spot == self.kings[0]:
            return 1
        elif spot == self.players[1] or spot == self.kings[1]:
            return 2
        return 0
    
    
    def forward_n_locations(self, start_loc, n, backwards=False):
        """
        Gets the locations possible for moving a piece from a given location diagonally
        forward (or backwards if wanted) a given number of times(without directional change midway).  
        """
        #even number of moves
        if n % 2 == 0:
            temp1 = 0
            temp2 = 0
        #even row
        elif start_loc[0] % 2 == 0:
            temp1 = 0
            temp2 = 1
        #odd row 
        else:
            temp1 = 1
            temp2 = 0

        answer = [[start_loc[0], start_loc[1] + math.floor(n / 2) + temp1], [start_loc[0], start_loc[1] - math.floor(n / 2) - temp2]]

        if backwards: 
            answer[0][0] = answer[0][0] - n
            answer[1][0] = answer[1][0] - n
        else:
            answer[0][0] = answer[0][0] + n
            answer[1][0] = answer[1][0] + n

        if self.not_spot(answer[0]):
            answer[0] = []
        if self.not_spot(answer[1]):
            answer[1] = []
            
        return answer
    

    def get_simple_moves(self, start_loc):
        """    
        Gets the possible moves a piece can make given that it does not capture any opponents pieces.
        
        PRE-CONDITION:
        -start_loc is a location with a players piece
        """
        if self.grid[start_loc[0]][start_loc[1]] == self.kings[0] or self.grid[start_loc[0]][start_loc[1]] == self.kings[1]:
            next_locations = self.forward_n_locations(start_loc, 1)
            next_locations.extend(self.forward_n_locations(start_loc, 1, True))
        elif self.grid[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next_locations = self.forward_n_locations(start_loc, 1, True)  # Switched the true from the statement below
        else:
            next_locations = self.forward_n_locations(start_loc, 1)
        

        possible_next_locations = []

        for location in next_locations:
            if len(location) != 0:
                if self.grid[location[0]][location[1]] == Game.EMPTY:
                    possible_next_locations.append(location)
            
        return [[start_loc, end_spot] for end_spot in possible_next_locations]      
           

     
    def get_capture_moves(self, start_loc, move_beginnings=None):
        """
        Recursively get all of the possible moves for a piece which involve capturing an opponent's piece.
        """
        if move_beginnings is None:
            move_beginnings = [start_loc]
            
        answer = []
        if self.grid[start_loc[0]][start_loc[1]] == self.KINGS[0] or self.grid[start_loc[0]][start_loc[1]] == self.KINGS[1]:  
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
            next1.extend(self.forward_n_locations(start_loc, 1, True))
            next2.extend(self.forward_n_locations(start_loc, 2, True))
        elif self.grid[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next1 = self.forward_n_locations(start_loc, 1, True)
            next2 = self.forward_n_locations(start_loc, 2, True)
        else:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
        
        
        for j in range(len(next1)):
            if (not self.not_spot(next2[j])) and (not self.not_spot(next1[j])) :  # if both spots exist
                if self.get_spot_info(next1[j]) != Game.EMPTY and self.get_spot_num(next1[j]) != self.get_spot_num(start_loc):  # if next spot is opponent
                    if self.get_spot_info(next2[j]) == Game.EMPTY:  # if next next spot is empty
                        temp_move1 = copy.deepcopy(move_beginnings)
                        temp_move1.append(next2[j])
                        
                        answer_length = len(answer)
                        
                        if self.get_spot_info(start_loc) != self.players[0] or next2[j][0] != self.HEIGHT - 1: 
                            if self.get_spot_info(start_loc) != self.players[1] or next2[j][0] != 0: 

                                temp_move2 = [start_loc, next2[j]]
                                
                                temp_board = self.clone()
                                temp_board.take_action(temp_move2)

                                answer.extend(temp_board.get_capture_moves(temp_move2[1], temp_move1))
                                
                        if len(answer) == answer_length:
                            answer.append(temp_move1)
                            
        return answer
    
        
    def get_possible_next_moves(self, player):
        """
        Gets the possible moves that can be made from the current board configuration.
        """
        piece_locations = []
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if (player == self.players[0] and (self.grid[j][i] == self.players[0] or self.grid[j][i] == self.kings[0])) or (player == self.players[1] and (self.grid[j][i] == self.players[1] or self.grid[j][i] == self.kings[1])):
                    piece_locations.append([j, i])
        if not piece_locations:
            return False
        capture_moves = list(reduce(lambda a, b: a + b, list(map(self.get_capture_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY

        if len(capture_moves) != 0:
            if player == self.players[0]:
                self.cap[0] = len(capture_moves)
            else:
                self.cap[1] = len(capture_moves)
            return capture_moves

        return list(reduce(lambda a, b: a + b, list(map(self.get_simple_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY
    
    def opponent(self, token):
        """
        Retrieve opponent players token for a given players token.
        """
        for t in self.players:
            if t != token:
                return t
        
    def is_over(self, player_num):
        """
        Finds out and returns whether the game currently being played is over or
        not.
        """
        t = self.players[player_num]
        if not self.get_possible_next_moves(t):
            return True
        return False

    def is_won(self, player_num):
        """
        If game is over and player won, return True, else return False
        """
        return not self.is_over(player_num)

    def is_lost(self, player_num):
        """
        If game is over and player lost, return True, else return False
        """
        return self.is_over(player_num)

    def reset(self):
        """
        Resets game to original layout.
        """
        self.grid = [[j,j,j,j] for j in [Game.PAWNS[0], Game.PAWNS[0], Game.PAWNS[0], Game.EMPTY, Game.EMPTY, Game.PAWNS[1], Game.PAWNS[1], Game.PAWNS[1]]]
        self.prev_grid = [copy.deepcopy(self.grid)]

    def print_board(self):
        """
        Prints a string representation of the current game board.
        """
        norm_line = "|---|---|---|---|---|---|---|---|"
        print(norm_line)
        for j in range(self.HEIGHT):
            if j % 2 == 1:
                temp_line = "|///|"
            else:
                temp_line = "|"
            for i in range(self.WIDTH):
                temp_line = temp_line + " " + self.grid[j][i] + " |"
                if i != 3 or j % 2 != 1:
                    temp_line = temp_line + "///|"
            print(temp_line)
            print(norm_line)   