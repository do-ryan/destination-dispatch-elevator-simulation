
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
                 Calendar: SimClasses.EventCalendar,
                 EventTime: float,
                 outer):
        """outer refers to an outer scope class. This class is designed to be nested in another class"""
        super().__init__()
        self.Calendar = Calendar
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

    def pickup(self):
        """Pick-up as many passengers as possible from current floor, update self resource, add waiting time data,
        and return amount of time spent on floor."""
        num_passengers = 0
        while len(self.floor_queues[self.floor].ThisQueue) > 0 and self.NumberOfUnits > 0:
            next_passenger = self.floor_queues[self.floor].Remove()
            assert isinstance(next_passenger, Passenger)
            self.Seize(1)
            self.dest_passenger_map[next_passenger.destination_floor].append(next_passenger)
            self.WaitingTimes.Record(SimClasses.Clock - next_passenger.CreateTime)
            num_passengers += 1
        return self.floor_dwell(num_passengers)

    def dropoff(self):
        """Drop-off all passengers on self with destination as current floor. Add time"""
        num_passengers = 0
        while len(self.dest_passenger_map[self.floor]) > 0:
            self.Free(1)
            self.TimesInSystem.Record(SimClasses.Clock - self.dest_passenger_map[self.floor].pop(0).CreateTime)
            num_passengers += 1
            SimFunctions.Schedule(self.Calendar, self.after_dropoff, self.floor_dwell(num_passengers), 1)

    def after_dropoff(self):
        # search for next task
        pass

    def floor_dwell(self, num_passengers: int):
        """Return amount of time to spend from door open, pickup/ dropoff, door close"""
        return self.door_move_time + self.passenger_move_time * num_passengers + self.door_move_time

    def move(self, destination_floor: int):
        """Return amount of time required to move from current floor to destination floor after doors close."""
        time_to_top_speed = self.top_speed / self.acceleration
        distance_to_top_speed = self.acceleration * time_to_top_speed**2 / 2
        total_distance = ((destination_floor - self.floor) * self.floor_distance)
        self.next_floor = destination_floor
        if distance_to_top_speed > total_distance / 2:
            return (total_distance / self.acceleration) ** (1 / 2)
        else:
            return time_to_top_speed \
                + (total_distance - (2 * distance_to_top_speed)) / self.top_speed / \
                + time_to_top_speed


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

    #
    # def assign_request(self, new_passenger: Passenger):
    #     assigned_car = None
    #     for car in self.cars:
    #         if car.status == 0:
    #             assigned_car = car
    #             break
    #     if assigned_car is None:
    #
    #     self.requests[new_passenger.destination_floor][int(
    #         new_passenger.destination_floor > new_passenger.source_floor)] = 1

    class PassengerArrivalEvent(FunctionalEventNotice):
        def event(self):
            new_passenger = Passenger(self.outer.num_floors)
            self.outer.floor_queues[new_passenger.source_floor].Add(new_passenger)

            # self.assign_request(new_passenger)
            self.Calendar.Schedule(
                self.outer.PassengerArrivalEvent(
                    Calendar=self.Calendar,
                    EventTime=SimRNG.Expon(
                        self.outer.mean_passenger_interarrival,
                        1),
                    outer=self.outer))

    class ClearIt(FunctionalEventNotice):
        def event(self):
            SimFunctions.ClearStats(self.outer.CTStats, self.outer.DTStats)

    def main(self):
        self.Calendar.Schedule(self.PassengerArrivalEvent(Calendar=self.Calendar, EventTime=0, outer=self))
        self.Calendar.Schedule(self.ClearIt(Calendar=self.Calendar, EventTime=0, outer=self))
        SimFunctions.Schedule(self.Calendar, "EndSimulation", self.run_length)

        NextEvent = self.Calendar.Remove()

        print(f"waiting passengers: [(source floor, create time, destination floor), () ...]")
        while NextEvent.EventType != "EndSimulation":
            SimClasses.Clock = NextEvent.EventTime
            NextEvent.event()
            NextEvent = self.Calendar.Remove()
            # trace
            print(NextEvent)
            print([(i, p.CreateTime, p.destination_floor)
                   for i, q in enumerate(self.floor_queues) for p in q.ThisQueue])

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
