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
    def fleet_sizing_by_threshold(replication_class,
                                  num_floors,
                                  pop_per_floor,
                                  fleet_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
                                  threshold=0.95):
        for fleet_size in fleet_sizes:
            replication = replication_class(run_length=60 * 24 * 2,
                                            num_floors=num_floors,
                                            pop_per_floor=pop_per_floor,
                                            num_cars=fleet_size)
            _, WaitingTimes, _ = replication.main(print_trace=False)
            print(f"fleet size: {fleet_size}, prob: {WaitingTimes.probInRangeCI95([0, 50/60])}")
            if WaitingTimes.probInRangeCI95([0, 50/60])[1][0] >= threshold:
                return fleet_size

    @staticmethod
    def fleet_sizing_by_best_selection(replication_class,
                                       num_floors,
                                       pop_per_floor,
                                       fleet_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20],
                                       alpha=0.05,
                                       delta=15/60):
        n_0 = math.inf
        Y = []
        I = fleet_sizes
        for fleet_size in I:
            replication = replication_class(run_length=60 * 24 * 2,
                                            num_floors=num_floors,
                                            pop_per_floor=pop_per_floor,
                                            num_cars=fleet_size)
            TimesInSystem, _,  _ = replication.main(print_trace=False)
            n_0 = min(len(TimesInSystem.Observations), n_0)
            Y.append(TimesInSystem)
        eta = 0.5*((2*alpha/(len(I)-1))**(-2/(n_0-1))-1)
        t2 = 2*eta*(n_0-1)
        s2 = []
        for i in range(len(I)):
            s2.append([])
            for h in range(len(I)):
                if h <= i:
                    s2[-1].append(None)
                else:
                    sum = 0
                    for j in range(n_0):
                        sum += (Y[i].Observations[j] - Y[h].Observations[j] 
                                - (np.mean(Y[i].Observations[0:n_0]) - np.mean(Y[h].Observations[0:n_0]))) ** 2
                    s2[-1].append(sum / (n_0 - 1))
        r = n_0-100
        while len(I) > 1:
            I_old = I
            I = []
            for i in range(len(I_old)):
                include = True
                for h in range(len(I_old)):
                    if h <= i:
                        pass
                    else:
                        W_ih = max(0, (delta/ 2*r)*(t2*s2[min(i, h)][max(i, h)]/delta**2-r))
                        if np.mean(Y[i].Observations[0: r]) > np.mean(Y[i].Observations[0:r]) + W_ih:
                            include = False
                if include:
                    I.append(I_old[i])
            r += 1

        return I
    

    @classmethod
    def main(cls):
        # TimesInSystem, WaitingTimes, TravelTimes = ReplicationTraditional(run_length=60 * 24,
        #                                                                   num_floors=9,
        #                                                                   num_cars=6,
        #                                                                   pop_per_floor=100,
        #                                                                   write_to_csvs=False).main(print_trace=False)
        # print(np.mean(TimesInSystem.Observations),
        #       WaitingTimes.probInRangeCI95([0, 50/60]),
        #       TravelTimes.probInRangeCI95([0, 100/60]))
        print(cls.fleet_sizing_by_threshold(replication_class=ReplicationTraditional, num_floors=10, pop_per_floor=100))
        print(cls.fleet_sizing_by_threshold(replication_class=ReplicationDestDispatch, num_floors=10, pop_per_floor=100))

        pass


Experiment.main()
