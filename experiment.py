from replication import ReplicationDestDispatch, ReplicationTraditional

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


class Experiment:
    def __init__(self):
        self.floor_counts = [3, 5, 10, 15]
        self.floor_pops = [25, 50, 100, 200, 300]
        self.fleet_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 18, 20, 25]
        self.base_path = '../experiments'

    def __call__(self):
        return self.main()

    @staticmethod
    def fleet_sizing_by_prob_threshold(replication_class,
                                       num_floors,
                                       pop_per_floor,
                                       fleet_sizes=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
                                       target=50 / 60,
                                       threshold=0.95,
                                       output_index=1):
        for fleet_size in fleet_sizes:
            replication = replication_class(run_length=60 * 24 * 2,
                                            num_floors=num_floors,
                                            pop_per_floor=pop_per_floor,
                                            num_cars=fleet_size)
            output_stats = replication.main(print_trace=False)  # 2 is waiting time
            output = output_stats[output_index]
            print(f"fleet size: {fleet_size}, prob: {output.probInRangeCI95([0, target])}")
            if output.probInRangeCI95([0, target])[1][0] >= threshold:
                return fleet_size

    @staticmethod
    def fleet_sizing_by_best_selection(replication_class,
                                       num_floors,
                                       pop_per_floor,
                                       fleet_sizes=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20],
                                       alpha=0.05,
                                       delta=0.25):  # delta of 0.0005 seems to give me an indifference zone of 5s...
        """Unused"""
        n_0 = math.inf
        Y = []
        I = fleet_sizes
        for fleet_size in I:
            replication = replication_class(run_length=60 * 24 * 2,
                                            num_floors=num_floors,
                                            pop_per_floor=pop_per_floor,
                                            num_cars=fleet_size)
            TimesInSystem, _, _ = replication.main(print_trace=False)
            n_0 = min(len(TimesInSystem.Observations), n_0)
            Y.append(TimesInSystem)
        n_0 = 20  # allow room for iteration when r gets incremented
        eta = 0.5 * ((2 * alpha / (len(I) - 1))**(-2 / (n_0 - 1)) - 1)
        t2 = 2 * eta * (n_0 - 1)
        s2 = []
        for i in range(len(I)):
            print("step 2 i:", i)
            s2.append([])
            for h in range(len(I)):
                if h <= i:
                    s2[-1].append(None)
                else:
                    sum = np.sum(np.square(np.array(Y[i].Observations[0:n_0]) - np.array(Y[h].Observations[0:n_0])
                                           - (np.mean(Y[i].Observations[0:n_0]) - np.mean(Y[h].Observations[0:n_0]))))
                    s2[-1].append(sum / (n_0 - 1))
        r = n_0
        while len(I) > 3:
            print("I:", I)
            I_old = I
            I = []
            for i in range(len(I_old)):
                print("step 3 i:", i)
                include = True
                for h in range(len(I_old)):
                    if h <= i:
                        pass
                    else:
                        W_ih = max(0, (delta / (2 * r)) * (t2 * s2[min(i, h)][max(i, h)] / delta**2 - r))
                        if np.mean(Y[i].Observations[0: r]) > (np.mean(Y[h].Observations[0:r]) + W_ih):
                            include = False
                if include:
                    I.append(I_old[i])
            r += 1

        return I

    def figures(self):
        tis_trad = np.empty(shape=(len(self.floor_counts), len(self.floor_pops)))
        tis_dd = np.empty(shape=(len(self.floor_counts), len(self.floor_pops)))
        wait_trad = np.empty(shape=(len(self.floor_counts), len(self.floor_pops)))
        wait_dd = np.empty(shape=(len(self.floor_counts), len(self.floor_pops)))
        for i, floor_count in enumerate(self.floor_counts):
            for j, floor_pop in enumerate(self.floor_pops):
                f = open(f"{self.base_path}/{floor_count}floors_{floor_pop}pflr.txt", "r")
                contents = f.read()
                tis, wait = tuple([(int(measure_fleetsizes.split(' ')[0]), int(measure_fleetsizes.split(' ')[-1]))
                                   for measure_fleetsizes in contents.split('\n')[-2:-4:-1][::-1]])
                tis_trad[i][j] = tis[0]
                tis_dd[i][j] = tis[1]
                wait_trad[i][j] = wait[0]
                wait_dd[i][j] = wait[1]

        sns.heatmap(tis_trad, cmap="YlGnBu")
        plt.title("Required number of elevator cars to achieve (3*(building height/ top speed)+50)s journey time with 95% probability at 95% confidence (Traditional Algorithm)")
        plt.xlabel("Number of patrons per floor")
        plt.ylabel("Number of floors")
        plt.show()
        sns.heatmap(tis_dd)
        plt.title("Required number of elevator cars to achieve (3*(building height/ top speed)+50)s journey time with 95% probability at 95% confidence (Destination Dispatch Algorithm)")
        plt.show()
        sns.heatmap(wait_trad)
        plt.title("Required number of elevator cars to achieve 50s wait time with 95% probability at 95% confidence (Traditional Algorithm)")
        plt.show()
        sns.heatmap(wait_dd)
        plt.title("Required number of elevator cars to achieve 50s wait time with 95% probability at 95% confidence (Destination Dispatch Algorithm)")
        plt.show()
        sns.heatmap(tis_dd - tis_trad)
        plt.title("Net improvement of Destination Dispatch for elevator car fleet sizing measured by journey time")
        plt.show()
        sns.heatmap(wait_dd - wait_trad)
        plt.title("Net improvement of Destination Dispatch for elevator car fleet sizing measured by wait time")
        plt.show()


    def main(self):
        waiting_time_threshold = 50 / 60
        floor_distance = 4.5
        top_speed = 3.0

        for floor_count in self.floor_counts:
            tis_threshold = (floor_distance * floor_count / top_speed * 3) / 60 + waiting_time_threshold
            for floor_pop in self.floor_pops:
                sys.stdout = open(f'{self.base_path}/{floor_count}floors_{floor_pop}pflr.txt', 'w+')
                fleet_size_trad_tis = self.fleet_sizing_by_prob_threshold(
                    replication_class=ReplicationTraditional,
                    num_floors=floor_count,
                    pop_per_floor=floor_pop,
                    target=tis_threshold,
                    fleet_sizes=self.fleet_sizes,
                    output_index=0)
                fleet_size_dd_tis = self.fleet_sizing_by_prob_threshold(
                    replication_class=ReplicationDestDispatch,
                    num_floors=floor_count,
                    pop_per_floor=floor_pop,
                    target=tis_threshold,
                    fleet_sizes=self.fleet_sizes,
                    output_index=0)
                fleet_size_trad_wait = self.fleet_sizing_by_prob_threshold(
                    replication_class=ReplicationTraditional,
                    num_floors=floor_count,
                    pop_per_floor=floor_pop,
                    target=waiting_time_threshold,
                    fleet_sizes=self.fleet_sizes,
                    output_index=1)
                fleet_size_dd_wait = self.fleet_sizing_by_prob_threshold(
                    replication_class=ReplicationDestDispatch,
                    num_floors=floor_count,
                    pop_per_floor=floor_pop,
                    target=waiting_time_threshold,
                    fleet_sizes=self.fleet_sizes,
                    output_index=1)
                print(fleet_size_trad_tis, fleet_size_dd_tis)
                print(fleet_size_trad_wait, fleet_size_dd_wait)


exp = Experiment()
# exp.main()
exp.figures()
