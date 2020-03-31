
import pythonsim.SimFunctions as SimFunctions
import pythonsim.SimRNG as SimRNG
import pythonsim.SimClasses as SimClasses
import numpy as np
import math
import itertools

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
                 outer,  # must be Replication class

                 initial_floor=0,
                 capacity=20,
                 door_move_time=0.0417,  # 2.5 seconds
                 passenger_move_time=0.0167,  # 1 second
                 acceleration=1.0,  # m/s^2
                 top_speed=3.0,  # m/s
                 floor_distance=4.5  # metres
                 ):
        self.outer = outer

        # resource states
        self.status = 0  # 0 for idle car, 1 for moving, 2 for transferring. only changes at the start of some action
        self.Busy = 0
        self.NumberOfUnits = capacity
        self.NumBusy = SimClasses.CTStat()
        self.Calendar = self.outer.Calendar

        # elevator system states
        self.floor_queues = self.outer.floor_queues
        self.floor = initial_floor
        self.next_floor = None
        self.dest_passenger_map = [[] for _ in range(len(self.floor_queues))]  # [destination floor][list_of_passengers]
        self.requests = np.zeros(shape=(len(self.floor_queues), 2))
        # for each floor: index 0, 1 is down, up request respectively. value of 0 indicates no request.
        self.direction = None
        # direction is only updated when car goes idle, or when car starts moving from idle.

        # kinematic properties
        self.door_move_time = door_move_time
        self.passenger_move_time = passenger_move_time
        self.acceleration = acceleration
        self.top_speed = top_speed
        self.floor_distance = floor_distance

        # stats
        self.WaitingTimes = self.outer.WaitingTimes
        self.TimesInSystem = self.outer.TimesInSystem

    def floor_dwell(self, num_passengers: int):
        """Return amount of time to spend transferring passengers. Door open/ close is lumped into move event."""
        return self.passenger_move_time * num_passengers

    class PickupEvent(FunctionalEventNotice):
        def event(self):
            """Pick-up as many passengers as possible from current floor, update self resource, add waiting time data,
            and return amount of time spent on floor."""
            self.outer.status = 2
            num_passengers = 0
            directions_served = set()
            while len(
                    self.outer.floor_queues[self.outer.floor].ThisQueue) > 0 and self.outer.Busy < self.outer.NumberOfUnits:
                if self.outer.requests[self.outer.floor,
                                       self.outer.floor_queues[self.outer.floor].ThisQueue[0].direction]:
                    # pickup passenger if this car was allocated to the request
                    next_passenger = self.outer.floor_queues[self.outer.floor].Remove()
                    assert isinstance(next_passenger, Passenger)

                    self.outer.Seize(1)
                    self.outer.dest_passenger_map[next_passenger.destination_floor].append(next_passenger)
                    directions_served |= {next_passenger.direction}

                    self.outer.WaitingTimes.Record(SimClasses.Clock - next_passenger.CreateTime)
                    num_passengers += 1

            for direction in directions_served:
                self.outer.requests[self.outer.floor, direction] = 0
                # update served requests

            self.outer.Calendar.Schedule(self.outer.PickupEndEvent(EventTime=self.outer.floor_dwell(num_passengers),
                                                                   outer=self.outer))

            for passenger in self.outer.floor_queues[self.outer.floor].ThisQueue:
                if passenger.direction in directions_served:
                    # for all passengers requests that were allocated to this car that couldn't get on
                    self.outer.Calendar.Schedule(self.outer.outer.AssignRequestEvent(new_passenger=passenger,
                                                                                     EventTime=0,
                                                                                     outer=self.outer.outer))

    class PickupEndEvent(FunctionalEventNotice):
        def event(self):
            self.outer.next_action()

    class DropoffEvent(FunctionalEventNotice):
        def event(self):
            """Drop-off all passengers on self with destination as current floor. Add time"""
            self.outer.status = 2
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
            self.outer.next_action()

    class MoveEvent(FunctionalEventNotice):
        def __init__(self, destination_floor: int, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.destination_floor = destination_floor

        def event(self):
            """Return amount of time required to move from current floor to destination floor after doors close."""
            time_to_top_speed = self.outer.top_speed / self.outer.acceleration
            distance_to_top_speed = self.outer.acceleration * time_to_top_speed**2 / 2
            total_distance = (abs(self.destination_floor - self.outer.floor) * self.outer.floor_distance)
            self.outer.next_floor = self.destination_floor
            if distance_to_top_speed > total_distance / 2:
                travel_time = (total_distance / self.outer.acceleration) ** (1 / 2)
            else:
                travel_time = time_to_top_speed \
                    + (total_distance - (2 * distance_to_top_speed)) / self.outer.top_speed \
                    + time_to_top_speed
            self.outer.Calendar.Schedule(
                self.outer.MoveEndEvent(
                    EventTime=self.outer.door_move_time +
                    travel_time /
                    60 +
                    self.outer.door_move_time,
                    outer=self.outer))

            self.outer.direction = int(self.outer.next_floor > self.outer.floor)
            self.outer.status = 1

    class MoveEndEvent(FunctionalEventNotice):
        def event(self):
            self.outer.floor = self.outer.next_floor
            # on MoveEnd, car.next_floor = car.floor
            self.outer.next_action()

    def next_action(self):
        """Triggered after moving, or done a transfer."""
        if np.sum(self.requests) == 0 and len(
                [queue for queue in itertools.chain.from_iterable(self.dest_passenger_map)]) == 0:
            # if no requests nor drop-offs, go idle.
            self.status = 0
            self.next_floor = None
            self.direction = None
        else:
            if len(self.dest_passenger_map[self.floor]) > 0:
                self.Calendar.Schedule(self.DropoffEvent(EventTime=0, outer=self))
            elif np.sum(self.requests[self.floor, :]) > 0:
                self.Calendar.Schedule(self.PickupEvent(EventTime=0, outer=self))
            else:  # if no more pickups nor drop-offs on current floor, take on the next request
                scan_start = self.floor  # scan start is the floor that "up-bound" and "down-bound" are relative to
                if self.status == 0:
                    #  if idle, take the next request if there is one- start closer to terminal ends.
                    # currently there is no direction so we need to decide which direction this car will proceed in and
                    # which request to start on.
                    if np.sum(self.requests[:, 1]) \
                            + len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map)])\
                            > np.sum(self.requests[:, 0]) \
                            + len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map)]):
                        #  if there are more up-bound requests than down-bound
                        self.direction = 1
                        scan_start = 0
                    else:
                        self.direction = 0
                        scan_start = self.requests.shape[0]-1
                        # go to highest down-bound request

                # from here downwards, if car is currently idle, then it will have chosen a direction.
                # it will start at the globally lowest upbound task, or highest downbound task, dropping off passengers
                # on the car along the way.
                if self.direction == 1:
                    # if going upwards
                    if np.max(self.requests[:, 1][scan_start:]) > 0:
                        # if there are up-bound requests above
                        next_pickup = np.argmax(self.requests[:, 1][scan_start:]) + scan_start
                    else:
                        next_pickup = None
                    if len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map[scan_start:])]):
                        next_dropoff = [queue for queue in itertools.chain.from_iterable(
                            self.dest_passenger_map[scan_start:])][0].destination_floor
                        # if there are drop-offs above
                    else:
                        next_dropoff = None

                    if next_pickup is not None and next_dropoff is not None:
                        next_destination = min(next_pickup, next_dropoff)
                    elif next_pickup is not None or next_dropoff is not None:
                        if next_pickup is None:
                            next_destination = next_dropoff
                        else:
                            next_destination = next_pickup
                    # from pool of allocated up-bound requests or drop-offs,
                    # pick the lowest floor that is higher than current floor
                    if next_pickup is None and next_dropoff is None:
                        # if there are no up-bound requests of drop-offs above, take next request
                        self.status = 0
                        self.next_floor = None
                        self.direction = None
                        self.next_action()
                    else:
                        self.Calendar.Schedule(self.MoveEvent(destination_floor=int(next_destination),
                                                              EventTime=0,
                                                              outer=self))
                else:  # if going downwards or currently directionless (previously idle, and just picked-up)
                    if np.max(self.requests[:, 0][0:scan_start + 1]) > 0:
                        # if there are down-bound requests below
                        next_pickup = scan_start - np.argmax(np.flip(self.requests[0:scan_start+1, 0], axis=0))
                        # take the highest down-bound request below
                    else:
                        next_pickup = None
                    if len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map[0:scan_start + 1])]):
                        next_dropoff = [queue for queue in itertools.chain.from_iterable(
                            self.dest_passenger_map[0:scan_start + 1])][-1].destination_floor
                        # if there are drop-offs below, save the highest floor drop-off
                    else:
                        next_dropoff = None

                    if next_pickup is not None and next_dropoff is not None:
                        next_destination = max(next_pickup, next_dropoff)
                    elif next_pickup is not None or next_dropoff is not None:
                        next_destination = next_pickup or next_dropoff
                    # from pool of allocated down-bound requests or drop-offs,
                    # pick the lowest floor that is higher than current floor

                    if next_pickup is None and next_dropoff is None:
                        # if there are no down-bound requests or drop-offs below, take next request
                        self.status = 0
                        self.next_floor = None
                        self.direction = None
                        self.next_action()
                    else:
                        self.Calendar.Schedule(self.MoveEvent(destination_floor=int(next_destination),
                                                              EventTime=0,
                                                              outer=self))


class Passenger(SimClasses.Entity):
    def __init__(self,
                 num_floors: int):
        super().__init__()
        self.source_floor = math.floor(SimRNG.Uniform(0, num_floors, s))
        while True:
            self.destination_floor = math.floor(SimRNG.Uniform(0, num_floors, s))
            if self.destination_floor != self.source_floor:
                break
        self.direction = int(self.destination_floor > self.source_floor)


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
            for car in self.outer.cars:
                #  TODO: still need to consider distance
                if car.requests[request_floor, int(self.new_passenger.direction)] == 1 and best_suitability < 5:
                    # if car is already allocated there
                    best_suitability = 5
                    assigned_car = car
                elif car.status == 0 and best_suitability <= 4:  # if car is idle
                    assert car.next_floor is None
                    best_suitability = 4
                    assigned_car = car
                elif (request_floor > car.next_floor and car.direction) or (request_floor < car.next_floor and not car.direction):
                    # if this is an incoming car
                    if self.new_passenger.direction != car.direction and best_suitability < 2:  # if intended direction is different
                        best_suitability = 2
                        assigned_car = car
                    elif self.new_passenger.direction == car.direction and best_suitability < 3:  # if intended direction is same
                        best_suitability = 3
                        assigned_car = car
                else:
                    if best_suitability < 1:
                        best_suitability = 1
                        assigned_car = car

            assert assigned_car is not None
            assigned_car.requests[request_floor, int(self.new_passenger.direction)] = 1

            if assigned_car.status == 0:
                assigned_car.next_action()

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
              f"car passengers: [{{floor: [passengers]}}, ...] car requests")
        while NextEvent.EventType != "EndSimulation":
            SimClasses.Clock = NextEvent.EventTime  # advance clock to start of next event
            NextEvent.event()
            print(NextEvent, SimClasses.Clock)
            print([(i, p.CreateTime, p.destination_floor)
                   for i, q in enumerate(self.floor_queues) for p in q.ThisQueue])
            print([f"{car.dest_passenger_map} floor: {car.floor} status: {car.status} direction: {car.direction}" for car in self.cars])
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


r0 = Replication(run_length=20, mean_passenger_interarrival=0.5)
r0()
pass
