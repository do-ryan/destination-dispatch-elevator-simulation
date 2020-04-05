from replication import ReplicationDestDispatch, ReplicationTraditional

import numpy as np

class Experiment:
    def __init__(self,
                 batch_size,
                 quantile):
        self.batch_size = batch_size
        self.quantile = quantile
        pass

    def __call__(self):
        return self.main()

    def main(self):
        TimesInSystem, WaitingTimes, TravelTimes = ReplicationDestDispatch(run_length=60 * 24,
                                                                          num_floors=7,
                                                                          num_cars=6,
                                                                          pop_per_floor=300,
                                                                          write_to_csvs=False).main(print_trace=False)
        print(np.mean(TimesInSystem.Observations),
              WaitingTimes.probInRangeCI95([0, 50/60]),
              TravelTimes.probInRangeCI95([0, 100/60]))
        pass

Experiment(batch_size=20, quantile=0.99).main()