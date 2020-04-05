from replication import ReplicationDestDispatch, ReplicationTraditional


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
        TimesInSystem, WaitingTimes, TravelTimes = ReplicationTraditional(run_length=60 * 24,
                                                                          num_floors=5,
                                                                          num_cars=2,
                                                                          car_capacity=20,
                                                                          write_to_csvs=False).main(print_trace=False)
        # print(TimesInSystem.Mean, WaitingTimes.probInRangeCI95([0, 50/60]), TravelTimes.probInRangeCI95([0, 150/60]))
        pass

Experiment(batch_size=20, quantile=0.99).main()