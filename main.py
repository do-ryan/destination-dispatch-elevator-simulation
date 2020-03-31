
import pythonsim.SimFunctions as SimFunctions
import pythonsim.SimRNG as SimRNG
import pythonsim.SimClasses as SimClasses
import numpy as np
import math

from collections import defaultdict
from typing import List
from abc import abstractmethod

np.random.seed(1)

ZSimRNG = SimRNG.InitializeRNSeed()
s = 1  # constant s as parameter to sim


class FunctionalEventNotice(SimClasses.EventNotice):

    def __init__(self,
                 EventTime: float,
                 outer):
        """outer refers to an outer scope class. This class is designed to be nested in another class"""
        super().__init__()
        self.EventTime = SimClasses.Clock + EventTime
        self.outer = outer

    @abstractmethod
    def event(self):
        pass


class ElevatorCar(SimClasses.Resource):
    def __init__(self,
                 floor_queues: List[SimClasses.FIFOQueue],  # index is floor number
                 Calendar: SimClasses.EventCalendar,
                 WaitingTimes: SimClasses.DTStat,
                 TimesInSystem: SimClasses.DTStat,

                 initial_floor=0,
                 capacity=20,
                 door_move_time=0.0417,  # 2.5 seconds
                 passenger_move_time=0.0167,  # 1 second
                 acceleration=1.0,  # m/s^2
                 top_speed=3.0,  # m/s
                 floor_distance=4.5  # metres
                 ):
        # resource states
        self.status = 0  # 0 for idle car, 1 for busy
        self.Busy = 0
        self.NumberOfUnits = capacity
        self.NumBusy = SimClasses.CTStat()
        self.Calendar = Calendar

        # elevator system states
        self.dest_passenger_map = defaultdict(list)  # {destination floor: list_of_passengers}
        self.floor_queues = floor_queues
        self.floor = initial_floor
        self.next_floor = None
        self.requests = [[0, 0] for _ in range(len(self.floor_queues))]
        # for each floor: index 0, 1 is down, up request respectively. value of 0 indicates no request.

        # kinematic properties
        self.door_move_time = door_move_time
        self.passenger_move_time = passenger_move_time
        self.acceleration = acceleration
        self.top_speed = top_speed
        self.floor_distance = floor_distance

        # stats
        self.WaitingTimes = WaitingTimes
        self.TimesInSystem = TimesInSystem

    def floor_dwell(self, num_passengers: int):
        """Return amount of time to spend from door open, pickup/ dropoff, door close"""
        return self.door_move_time + self.passenger_move_time * num_passengers + self.door_move_time

    class PickupEvent(FunctionalEventNotice):
        def event(self):
            """Pick-up as many passengers as possible from current floor, update self resource, add waiting time data,
            and return amount of time spent on floor."""
            num_passengers = 0
            while len(self.outer.floor_queues[self.outer.floor].ThisQueue) > 0 and self.outer.NumberOfUnits > 0:
                next_passenger = self.outer.floor_queues[self.outer.floor].Remove()
                assert isinstance(next_passenger, Passenger)
                self.outer.Seize(1)
                self.outer.dest_passenger_map[next_passenger.destination_floor].append(next_passenger)
                self.outer.WaitingTimes.Record(SimClasses.Clock - next_passenger.CreateTime)
                num_passengers += 1
            self.outer.Calendar.Schedule(self.outer.PickupEndEvent(EventTime=self.outer.floor_dwell(num_passengers),
                                                                   outer=self.outer))

    class PickupEndEvent(FunctionalEventNotice):
        def event(self):
            pass

    class DropoffEvent(FunctionalEventNotice):
        def event(self):
            """Drop-off all passengers on self with destination as current floor. Add time"""
            num_passengers = 0
            while len(self.outer.dest_passenger_map[self.outer.floor]) > 0:
                self.outer.Free(1)
                self.outer.TimesInSystem.Record(SimClasses.Clock -
                                                self.outer.dest_passenger_map[self.outer.floor].pop(0).CreateTime)
                num_passengers += 1
                self.outer.Calendar.Schedule(
                    self.outer.DropoffEndEvent(
                        EventTime=self.outer.floor_dwell(num_passengers),
                        outer=self.outer))

    class DropoffEndEvent(FunctionalEventNotice):
        def event(self):
            pass

    class MoveEvent(FunctionalEventNotice):
        def __init__(self, destination_floor: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.destination_floor = destination_floor

        def event(self):
            """Return amount of time required to move from current floor to destination floor after doors close."""
            time_to_top_speed = self.outer.top_speed / self.outer.acceleration
            distance_to_top_speed = self.outer.acceleration * time_to_top_speed**2 / 2
            total_distance = ((self.destination_floor - self.outer.floor) * self.outer.floor_distance)
            self.outer.next_floor = self.destination_floor
            if distance_to_top_speed > total_distance / 2:
                travel_time = (total_distance / self.outer.acceleration) ** (1 / 2)
            else:
                travel_time = time_to_top_speed \
                              + (total_distance - (2 * distance_to_top_speed)) / self.outer.top_speed \
                              + time_to_top_speed
            self.outer.Calendar.Schedule(
                self.outer.MoveEndEvent(EventTime=travel_time / 60, outer=self.outer))

    class MoveEndEvent(FunctionalEventNotice):
        def event(self):
            self.outer.floor = self.outer.next_floor
            self.outer.next_floor = None


class Passenger(SimClasses.Entity):
    def __init__(self,
                 num_floors: int):
        super().__init__()
        self.source_floor = math.floor(SimRNG.Uniform(0, num_floors, s))
        while True:
            self.destination_floor = math.floor(SimRNG.Uniform(0, num_floors, s))
            if self.destination_floor != self.source_floor:
                break


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

        self.cars = [ElevatorCar(floor_queues=self.floor_queues,
                                 Calendar=self.Calendar,
                                 WaitingTimes=self.WaitingTimes,
                                 TimesInSystem=self.TimesInSystem) for _ in range(0, self.num_cars)]

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
            passenger_direction = self.new_passenger.destination_floor > request_floor
            for car in self.outer.cars:
                if car.status == 0:  # if car is idle, choose it
                    assert car.next_floor is None
                    best_suitability = 4
                    assigned_car = car
                    break
                direction = car.next_floor > car.floor
                if (request_floor > car.next_floor and direction) or (request_floor < car.next_floor and not direction):
                    # if this is an incoming car
                    if passenger_direction != direction and best_suitability < 2:  # if intended direction is different
                        best_suitability = 2
                    elif passenger_direction == direction and best_suitability < 3:  # if intended direction is same
                        best_suitability = 3
                else:
                    best_suitability = 1
                assigned_car = car

            assert assigned_car is not None
            assigned_car.requests[request_floor][passenger_direction] = 1

    class PassengerArrivalEvent(FunctionalEventNotice):
        def event(self):
            new_passenger = Passenger(self.outer.num_floors)
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

    def main(self):
        self.Calendar.Schedule(self.PassengerArrivalEvent(EventTime=0, outer=self))
        self.Calendar.Schedule(self.ClearItEvent(EventTime=0, outer=self))
        self.Calendar.Schedule(self.cars[0].PickupEvent(EventTime=50, outer=self.cars[0]))  # test
        self.Calendar.Schedule(self.cars[0].MoveEvent(EventTime=60, outer=self.cars[0], destination_floor=6))  # test
        self.Calendar.Schedule(self.cars[0].DropoffEvent(EventTime=70, outer=self.cars[0]))  # test
        SimFunctions.Schedule(self.Calendar, "EndSimulation", self.run_length)

        NextEvent = self.Calendar.Remove()

        print(f"waiting passengers: [(source floor, create time, destination floor), ...] "
              f"car passengers: [{{floor: [passengers]}}, ...]")
        while NextEvent.EventType != "EndSimulation":
            SimClasses.Clock = NextEvent.EventTime  # advance clock to start of next event
            NextEvent.event()
            print(NextEvent, SimClasses.Clock)
            print([(i, p.CreateTime, p.destination_floor)
                   for i, q in enumerate(self.floor_queues) for p in q.ThisQueue],
                  [car.dest_passenger_map for car in self.cars])
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


r0 = Replication()
r0()
pass
