from replication import ReplicationDestDispatch, ReplicationTraditional

import numpy as np
import math
from scipy.stats import t


class Experiment:
    def __init__(self):
        pass

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

    @classmethod
    def main(cls):
        print(
            cls.fleet_sizing_by_prob_threshold(
                replication_class=ReplicationTraditional,
                num_floors=10,
                pop_per_floor=100,
                target=90 / 60,
                output_index=0))
        print(
            cls.fleet_sizing_by_prob_threshold(
                replication_class=ReplicationDestDispatch,
                num_floors=10,
                pop_per_floor=100,
                target=90 / 60,
                output_index=0))

        # print(
        #     cls.fleet_sizing_by_best_selection(
        #         replication_class=ReplicationTraditional,
        #         num_floors=10,
        #         pop_per_floor=100))
        # print(
        #     cls.fleet_sizing_by_best_selection(
        #         replication_class=ReplicationDestDispatch,
        #         num_floors=10,
        #         pop_per_floor=100))
        pass


Experiment.main()
