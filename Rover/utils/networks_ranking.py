import numpy as np
from Rover.algos.ppo.ppo import PPO_Rover
import os
from os import path as osp
from typing import List

class RoverRankingSystem:
    '''
    :param num_rankings: amount of comparing criteria for the networks
    :param networks_limit_per_ranking: how many networks will be tracked per ranking
    :param save_path: if not None, will save the networks and the ranking at the given path
    :param model: the model used in training
    :param verbose: 0 for no information, 1 for network leaves all rankings or first time it enters the rankings,
    2 for all times network leaves/enters a ranking
    '''
    def __init__(self, model: PPO_Rover, networks_limit_per_ranking: int = 20, num_rankings: int = 3, save_path: str = None, networks_subfolder: str = 'saved_networks', verbose: int = 0):
        self.num_rankings = num_rankings
        self.networks_limit_per_ranking = networks_limit_per_ranking
        self.rankings = [[] for i in range(num_rankings)]
        self.save_path = save_path
        self.networks_subfolder = networks_subfolder
        self.model = model
        self.verbose = verbose

    class Network:
        def __init__(self, iteration: int, performance_list: List):
            self.iteration = iteration
            self.performance_list = performance_list
            self.numlists = 0
    def rank_new_network(self, new_network: Network):
        assert len(new_network.performance_list) == self.num_rankings, "Given network has {} performance parameters while ranking asks for {}".format(len(new_network.performance_list), self.num_rankings)
        ranks_changed = False
        for ranking in range(len(self.rankings)):
            len_list = len(self.rankings[ranking])
            if len_list > 0:  # if the list is not empty, compare
                last = 0
                for j in range(1, len_list + 1):
                    net = self.rankings[ranking][-j]
                    if net.performance_list[ranking] > new_network.performance_list[ranking]:
                        break
                    else:
                        last = j
                if not last == 0:  # The network is better than some previously stored
                    if self.verbose == 2:
                        print("Network {} entered in ranking {}".format(new_network.iteration, ranking))
                    elif self.verbose == 1 and ranks_changed == False:
                        print("Network {} entered in rankings".format(new_network.iteration))
                    self.rankings[ranking].insert(-last, new_network)
                    new_network.numlists += 1
                    ranks_changed = True
            else:  # if it is, just store
                self.rankings[ranking].append(new_network)
                new_network.numlists += 1
                ranks_changed = True
            if len(self.rankings[ranking]) > self.networks_limit_per_ranking:  # if the list size has been passed, deletes worst
                del_net = self.rankings[ranking].pop()
                del_net.numlists -= 1
                if self.verbose == 2 and del_net.numlists > 0:
                    print("Network {} left ranking {}. Still in {} ranking(s)".format(del_net.iteration, ranking,
                                                                                      del_net.numlists))
                if del_net.numlists == 0:
                    if self.verbose >= 1:
                        print("Network {} left all rankings.".format(del_net.iteration))
                    if self.save_path is not None:  # Network's file must be removed
                        print("Deleting saved network {}".format(del_net.numepoch))
                        os.remove(osp.join(self.save_path, self.networks_subfolder, '%.5i' % del_net.numepoch))
                if ranks_changed and self.save_path is not None:
                    self.save_current_network()
    def save_current_network(self):
        pass  #TODO: implement networks saving function


