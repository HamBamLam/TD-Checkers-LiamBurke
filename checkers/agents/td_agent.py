import numpy as np
import copy

class TDAgent(object):

    def __init__(self, player, model):
        self.player = player
        self.model = model
        self.name = 'TD-Checkers'
            
    def alpha_beta_search(self, game, depth, alpha, beta, maximizing_player, num):
        num = (num+1)%2
        player = game.players[num]
        if depth == 0 or game.is_over(num):
            features = game.extract_features(player)
            v = self.model.get_output(features)
            #game.undo_action()
            return v

        if maximizing_player:
            v = float('-inf')
            for a in game.get_actions(player):
                grid = copy.deepcopy(game.grid)
                game.take_action(a)
                v = max(v, self.alpha_beta_search(game, depth - 1, alpha, beta, False,num))
                alpha = max(alpha, v)
                game.grid = copy.deepcopy(grid)
                if beta <= alpha:
                    break
            return v
        else:
            v = float('inf')
            for a in game.get_actions(player):
                grid = copy.deepcopy(game.grid)
                game.take_action(a)
                v = min(v, self.alpha_beta_search(game, depth - 1, alpha, beta, True,num))
                beta = min(beta, v)
                game.grid = copy.deepcopy(grid)
                if beta <= alpha:
                    break
            return v

    def get_action(self, actions, game):
        """
        Return best action according to model output with alpha-beta search.
        """
        v_best = float('-inf')
        a_best = None
        alpha = float('-inf')
        beta = float('inf')
        num = 1
        if self.player == game.players[0]:
            num = 0
        for a in actions:
            grid = copy.deepcopy(game.grid)
            game.take_action(a)
            v = self.alpha_beta_search(game, 2, alpha, beta, True, num)
            if v > v_best:
                v_best = v
                a_best = a
            game.grid = copy.deepcopy(grid)

        return a_best
