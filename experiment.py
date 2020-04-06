from replication import ReplicationDestDispatch, ReplicationTraditional

import numpy as np
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

    def fleet_sizing_by_best_selection(self, replication: ReplicationTraditional):
        pass

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
