import numpy as np
import os
from os import path as osp
from typing import List

class RoverRankingSystem:
    '''
    This class is responsible for ranking and organizing the networks of the experiment.
    The comparison is made according the networks' performance values.
    In the case of adding the network, it will be placed in the right position of the list, according to the
    performance value.
    At the end of the training, if the save_path is not None, a networks_rank.csv file will be generated with the
    performance in each of the best networks. The networks will also be saved.

    :param num_rankings: amount of comparing criteria for the networks
    :param networks_limit_per_ranking: how many networks will be tracked per ranking
    :param save_path: if not None, will save the networks and the ranking at the given path
    :param model: the model used in training
    :param verbose: 0 for no information, 1 for network leaves all rankings or first time it enters the rankings,
    2 for all times network leaves/enters a ranking
    '''
    def __init__(self, networks_limit_per_ranking: int = 20, num_rankings: int = 3,
                 save_path: str = None, networks_subfolder: str = 'saved_networks', verbose: int = 0):
        self.num_rankings = num_rankings
        self.networks_limit_per_ranking = networks_limit_per_ranking
        self.rankings = [[] for i in range(num_rankings)]
        self.save_path = save_path
        self.networks_subfolder = networks_subfolder or 'saved_networks'
        self.verbose = verbose
        if save_path is not None and networks_subfolder is not None:
            os.makedirs(os.path.join(self.save_path, self.networks_subfolder), exist_ok=True)

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
                        if self.verbose >= 1:
                            print("Deleting saved network {}".format(del_net.iteration))
                        os.remove(osp.join(self.save_path, self.networks_subfolder, '%.5i.zip' % del_net.iteration))
        if ranks_changed and self.save_path is not None:
            return True
        else:
            return False
    def print_rankings(self):  # TODO: expand to variable num_rankings. Current == 3
        print()
        fmt_size = '{:>' + str(18) + '}'
        descrip = ['', 'Non dying [%]', 'Avg/MEA [%]', 'Success [%]', '']
        print(' |'.join([fmt_size.format(d) for d in descrip]))
        print(fmt_size.format('Position') + ' |', end='')
        print(' |'.join(['{:>7}{:>11}'.format('Perf', ' Iteration') for _ in self.rankings]), '|')

        print((' ' * 10) + '-' * (4 * 18 - 3))

        rankings_sizes = [len(r) for r in self.rankings]
        for line_idx in range(np.max(rankings_sizes)):
            print(fmt_size.format(line_idx + 1) + ' |', end='')
            for rank_idx, rank in enumerate(self.rankings):
                if line_idx < len(rank):
                    print('{:>7.2f}{:>11}'.format(rank[line_idx].performance_list[rank_idx], rank[line_idx].iteration),
                          end='')
                else:
                    print(fmt_size.format(''), end='')
                print(' |', end='')
            print()
        print()

    def write_ranking_files(self):  # TODO: expand to variable num_rankings. Current == 3
        if self.save_path is not None:
            ranking_file1 = open(osp.join(self.save_path, 'networks_ranking1.txt'), 'w')
            ranking_file2 = open(osp.join(self.save_path, 'networks_ranking2.txt'), 'w')
            ranking_file3 = open(osp.join(self.save_path, 'networks_ranking3.txt'), 'w')

            ranking_file1.write('Non dying [%]\tIteration\n')
            ranking_file2.write('Avg/MEA [%]\tIteration\n')
            ranking_file3.write('Success [%]\tIteration\n')
            rf = [ranking_file1, ranking_file2, ranking_file3]

            rankings_sizes = [len(r) for r in self.rankings]
            for line_idx in range(np.max(rankings_sizes)):
                for rank_idx, rank in enumerate(self.rankings):
                    if line_idx < len(rank):
                        rf[rank_idx].write(
                            '{:0.2f}\t{}\n'.format(rank[line_idx].performance_list[rank_idx], rank[line_idx].iteration))

            ranking_file1.flush()
            ranking_file1.close()
            ranking_file2.flush()
            ranking_file2.close()
            ranking_file3.flush()
            ranking_file3.close()


