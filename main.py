from custom_library import FunctionalEventNotice
from entities import ElevatorCar, Passenger

import pythonsim.SimFunctions as SimFunctions
import pythonsim.SimRNG as SimRNG
import pythonsim.SimClasses as SimClasses
import numpy as np
import pprint

pp = pprint.PrettyPrinter()

np.random.seed(1)

ZSimRNG = SimRNG.InitializeRNSeed()
s = 1  # constant s as parameter to sim


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

        self.Calendar = SimClasses.EventCalendar()
        self.WaitingTimes = SimClasses.DTStat()
        self.TimesInSystem = SimClasses.DTStat()
        self.CTStats = []
        self.DTStats = [self.WaitingTimes, self.TimesInSystem]
        self.TheQueues = [] + self.floor_queues

        self.cars = [ElevatorCar(outer=self) for _ in range(0, self.num_cars)]

        self.TheResources = [] + self.cars
        SimFunctions.SimFunctionsInit(
            self.Calendar,
            self.TheQueues,
            self.CTStats,
            self.DTStats,
            self.TheResources)
        # resets all, including SimClasses.Clock

    def __call__(self):
        return self.main()

    class AssignRequestEvent(FunctionalEventNotice):
        def __init__(self, new_passenger: Passenger, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.new_passenger = new_passenger

        def event(self):
            assigned_car = None
            best_suitability = 0
            request_floor = self.new_passenger.source_floor
            # below, sometimes a car may not have a next floor but is not idle either. This occurs when a car is idle
            # and a passenger arrives on its current floor.
            for car in self.outer.cars:
                suitability = 0
                if car.requests[request_floor, int(self.new_passenger.direction)] == 1:
                    # if car is already allocated there
                    suitability = 5
                elif car.status == 0 and best_suitability <= 4:  # if car is idle
                    assert car.next_floor is None
                    suitability = 4
                elif (request_floor > (car.next_floor or car.floor) and car.direction)\
                        or (request_floor < (car.next_floor or car.floor) and not car.direction):
                    # if this is an incoming car
                    if self.new_passenger.direction == car.direction and best_suitability < 3:
                        # if intended direction is same
                        suitability = 3
                    elif self.new_passenger.direction != car.direction and best_suitability < 2:
                        # if intended direction is different
                        suitability = 2
                else:
                    suitability = 1
                # determine suitability of this car

                if suitability > best_suitability:
                    assigned_car = car
                    best_suitability = suitability
                # update most suitable so far

                elif suitability == best_suitability:
                    if abs(request_floor - (car.next_floor or car.floor)) \
                            < abs(request_floor - (assigned_car.next_floor or assigned_car.floor)):
                        assigned_car = car
                    # if this car's suitability is tied with best so far, use proximity as tie breaker
                    # if proximities tie, first index wins.

            assert assigned_car is not None
            assigned_car.requests[request_floor, int(self.new_passenger.direction)] = 1

            if assigned_car.status == 0:
                assigned_car.status = 3  # in decision process status. must not keep as 0
                assigned_car.next_action()

    class PassengerArrivalEvent(FunctionalEventNotice):
        def event(self):
            new_passenger = Passenger(self.outer.num_floors, stream=s)
            self.outer.floor_queues[new_passenger.source_floor].Add(new_passenger)

            # self.assign_request(new_passenger)
            self.outer.Calendar.Schedule(
                self.outer.PassengerArrivalEvent(
                    EventTime=SimRNG.Expon(
                        self.outer.mean_passenger_interarrival,
                        1), outer=self.outer))
            self.outer.Calendar.Schedule(self.outer.AssignRequestEvent(EventTime=0,
                                                                       outer=self.outer,
                                                                       new_passenger=new_passenger))

    class ClearItEvent(FunctionalEventNotice):
        def event(self):
            SimFunctions.ClearStats(self.outer.CTStats, self.outer.DTStats)

    class EndSimulationEvent(FunctionalEventNotice):
        def event(self):
            pass

    def main(self):
        self.Calendar.Schedule(self.PassengerArrivalEvent(EventTime=0, outer=self))
        self.Calendar.Schedule(self.ClearItEvent(EventTime=0, outer=self))
        # self.Calendar.Schedule(self.cars[0].PickupEvent(EventTime=50, outer=self.cars[0]))  # test
        # self.Calendar.Schedule(self.cars[0].MoveEvent(EventTime=60, outer=self.cars[0], destination_floor=6))  # test
        # self.Calendar.Schedule(self.cars[0].DropoffEvent(EventTime=70, outer=self.cars[0]))  # test
        self.Calendar.Schedule(self.EndSimulationEvent(EventTime=self.run_length, outer=self))

        NextEvent = self.Calendar.Remove()

        print(f"waiting passengers: [(source floor, create time, destination floor), ...] "
              f"car passengers: [{{floor: [passengers]}}, ...] car requests")
        while not isinstance(NextEvent, self.EndSimulationEvent):
            SimClasses.Clock = NextEvent.EventTime  # advance clock to start of next event
            NextEvent.event()

            print(NextEvent, SimClasses.Clock)
            # pp.pprint([(e, e.EventTime, e.outer) for e in self.Calendar.ThisCalendar])
            print([(i, p.CreateTime, p.destination_floor)
                   for i, q in enumerate(self.floor_queues) for p in q.ThisQueue])
            for car in self.cars:
                print(f"{[len(floor) for floor in car.dest_passenger_map]} floor: {car.floor} next floor: {car.next_floor} status: {car.status} direction: {car.direction}")
            print([car.requests for car in self.cars])
            # trace
            
            NextEvent = self.Calendar.Remove()

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


r0 = Replication(run_length=120, mean_passenger_interarrival=0.2, num_cars=2)
r0()
pass
