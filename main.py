
import pythonsim.SimFunctions as SimFunctions
import pythonsim.SimRNG as SimRNG
import pythonsim.SimClasses as SimClasses
import numpy as np
import math

from collections import defaultdict

np.random.seed(1)

ZSimRNG = SimRNG.InitializeRNSeed()
s = 1  # constant s as parameter to sim


class ElevatorCar(SimClasses.Resource):
    def __init__(self):
        self.passengers = defaultdict(list)  # {destination floor: list_of_passengers}


class Passenger(SimClasses.Entity):
    def __init__(self,
                 num_floors: int):
        super().__init__()
        self.source_floor = math.floor(SimRNG.Uniform(0, num_floors, s))
        self.destination_floor = math.floor(SimRNG.Uniform(0, num_floors, s))


class Replication:
    def __init__(self,
                 run_length=120.0,
                 warm_up=0,
                 num_floors=12,
                 num_cars=2,
                 mean_passenger_interarrival=5
                 ):
        """All time units are in minutes.
        """
        self.run_length = run_length
        self.warm_up = warm_up
        self.num_floors = num_floors
        self.num_cars = num_cars
        self.mean_passenger_interarrival = mean_passenger_interarrival

        self.floor_queues = [SimClasses.FIFOQueue() for _ in range(0, self.num_floors)]
        self.cars = [ElevatorCar() for _ in range(0, self.num_cars)]
        self.calendar = SimClasses.EventCalendar()

        self.CTStats = []
        self.DTStats = []
        self.TheQueues = []
        self.TheResources = []
        SimFunctions.SimFunctionsInit(
            self.calendar,
            self.TheQueues,
            self.CTStats,
            self.DTStats,
            self.TheResources)
        # resets all, including SimClasses.Clock

        SimFunctions.Schedule(self.calendar, "PassengerArrival", 0)
        SimFunctions.Schedule(self.calendar, "EndSimulation", self.run_length)
        SimFunctions.Schedule(self.calendar, "ClearIt", self.warm_up)

    def __call__(self):
        return self.main()

    def passenger_arrival_event(self):
        new_passenger = Passenger(self.num_floors)
        self.floor_queues[new_passenger.source_floor].Add(new_passenger)
        SimFunctions.Schedule(self.calendar, "PassengerArrival", SimRNG.Expon(self.mean_passenger_interarrival, 1))
        pass

    def main(self):
        NextEvent = self.calendar.ThisCalendar[0]

        while NextEvent.EventType != "EndSimulation":
            NextEvent = self.calendar.Remove()
            SimClasses.Clock = NextEvent.EventTime
            if NextEvent.EventType == 'PassengerArrival':
                self.passenger_arrival_event()
            elif NextEvent.EventType == "ClearIt":
                SimFunctions.ClearStats(self.CTStats, self.DTStats)

    @classmethod
    def CI_95(cls, data):
        a = np.array(data)
        n = len(a)
        m = np.mean(a)
        sd = np.std(a, ddof=1)
        hw = 1.96 * sd / np.sqrt(n)
        return m, [m - hw, m + hw]


class Experiment:
    def __init__(self):
        pass

    def __call__(self):
        return self.main()

    def main(self):
        pass
