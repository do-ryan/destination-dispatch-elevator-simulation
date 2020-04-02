import pprint
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_library import FunctionalEventNotice, DTStatPlus, FIFOQueuePlus
from traditional_elevator import ElevatorCarTraditional, Passenger
from destdispatch_elevator import ElevatorCarDestDispatch

import pythonsim.SimFunctions as SimFunctions
import pythonsim.SimRNG as SimRNG
import pythonsim.SimClasses as SimClasses

import seaborn as sns
import shutil
import os
sns.set()

pp = pprint.PrettyPrinter()

np.random.seed(1)

ZSimRNG = SimRNG.InitializeRNSeed()
s = 1  # constant s as parameter to sim


class ReplicationTraditional:
    def __init__(self,
                 run_length=120.0,
                 warm_up=0,
                 num_floors=12,
                 pop_per_floor=100,
                 num_cars=2,
                 car_capacity=20,
                 mean_passenger_interarrival=0.2,
                 write_to_csvs=False
                 ):
        """
        Models the operation of an elevator system for one replication. The elevator assignment logic also resides here.

        Args:
            All time units are in minutes.
        """
        self.run_length = run_length
        self.warm_up = warm_up
        self.num_floors = num_floors
        self.num_cars = num_cars
        assert self.num_floors > 1

        self.mean_passenger_interarrival = mean_passenger_interarrival  # for stationary option
        self.pop_per_floor = pop_per_floor

        # daily patterns for non-stationary. Values are % of building population per 5 minutes, by hour starting 12AM
        self.upbound_arrival_rate = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 8, 0.125, 0.125, 0.125, 2.75, 5.5, 0.125, 0.125, 0.125, 0.125, 0, 0, 0, 0, 0, 0])
        self.downbound_arrival_rate = np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0.125, 0.125, 0.125, 0.125, 5.5, 2.75, 0.125, 0.125, 0.125, 8, 0, 0, 0, 0, 0, 0])
        self.crossfloor_arrival_rate = \
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])

        self.write_to_csvs = write_to_csvs
        if self.write_to_csvs:
            self.figure_dir = '/Users/ryando/Dropbox/MEng/MIE1613/Project/figures/data'
            shutil.rmtree(self.figure_dir)
            os.mkdir(self.figure_dir)
        else:
            self.figure_dir = None

        self.floor_queues = [FIFOQueuePlus(id=i, figure_dir=self.figure_dir)
                             for i, _ in enumerate(range(0, self.num_floors))]

        self.Calendar = SimClasses.EventCalendar()
        self.WaitingTimes = DTStatPlus()
        self.TimesInSystem = DTStatPlus()
        self.CTStats = []
        self.DTStats = [self.WaitingTimes, self.TimesInSystem]
        self.TheQueues = [] + self.floor_queues
        self.AllArrivalTimes = []

        self.cars = [ElevatorCarTraditional(outer=self, capacity=car_capacity) for _ in range(0, self.num_cars)]

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

            # trigger an action from the idle vehicle. otherwise it will stay idle indefinitely
            if assigned_car.status == 0:
                assigned_car.status = 3  # in decision process status. must not keep as 0
                assigned_car.next_action()

    class PassengerArrivalEvent(FunctionalEventNotice):
        """Simple uniformly distributed source and destination floor, stationary poisson."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def event(self):
            source = math.floor(SimRNG.Uniform(0, self.outer.num_floors, s))
            while True:
                destination = math.floor(SimRNG.Uniform(0, self.outer.num_floors, s))
                if destination != source:
                    break
            new_passenger = Passenger(source_floor=source, destination_floor=destination)
            self.outer.floor_queues[self.floor_queue_index].Add(new_passenger)
            self.outer.AllArrivalTimes.append(SimClasses.Clock)

            self.outer.Calendar.Schedule(
                self.outer.PassengerArrivalEvent(
                    EventTime=SimRNG.Expon(self.outer.mean_passenger_interarrival, 1),
                    outer=self.outer))
            self.outer.Calendar.Schedule(self.outer.AssignRequestEvent(EventTime=0,
                                                                       outer=self.outer,
                                                                       new_passenger=new_passenger))

    class PassengerNonStationaryArrivalEvent(FunctionalEventNotice):
        """Non-stationary poisson arrival process based on daily patterns"""

        def __init__(self,
                     arrival_rates: np.array,
                     arrival_mode: int,
                     *args,
                     **kwargs):
            """
            Args:
               - arrival_rates - Must be a 24 length 1D array representing hourly rate for each hour
               - arrival_mode - 0 for ground-floor up-bound, 1 for random floor down to ground, 2 for cross-floor
            """
            super().__init__(*args, **kwargs)
            self.arrival_rates = arrival_rates
            self.arrival_mode = arrival_mode
            assert arrival_mode in [0, 1, 2]

        def nspp(self):
            max_rate = np.max(self.arrival_rates)
            possible_arrival = SimClasses.Clock + SimRNG.Expon(1 / (max_rate / 60), s)
            while SimRNG.Uniform(0, 1, 1) \
                    >= self.arrival_rates[int(possible_arrival / 60 % 24)] / max_rate:
                possible_arrival += SimRNG.Expon(1 / (max_rate / 60), s)
            return possible_arrival - SimClasses.Clock

        def enqueue_passenger(self, passenger: Passenger):
            self.outer.floor_queues[passenger.source_floor].Add(passenger)

        def event(self):
            # set source, destination of new passenger
            if self.arrival_mode == 0:
                source = 0
                destination = math.floor(SimRNG.Uniform(1, self.outer.num_floors, s))
            elif self.arrival_mode == 1:
                source = math.floor(SimRNG.Uniform(1, self.outer.num_floors, s))
                destination = 0
            elif self.arrival_mode == 2:
                source = math.floor(SimRNG.Uniform(1, self.outer.num_floors, s))
                while True:
                    destination = math.floor(SimRNG.Uniform(1, self.outer.num_floors, s))
                    if destination != source or self.outer.num_floors <= 2:
                        break
            new_passenger = Passenger(source_floor=source, destination_floor=destination)
            self.enqueue_passenger(new_passenger)
            self.outer.AllArrivalTimes.append(SimClasses.Clock)

            self.outer.Calendar.Schedule(
                self.outer.PassengerNonStationaryArrivalEvent(
                    arrival_rates=self.arrival_rates,
                    arrival_mode=self.arrival_mode,
                    EventTime=self.nspp(),
                    outer=self.outer))
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
        BuildingPopulation = self.pop_per_floor * self.num_floors
        self.Calendar.Schedule(
            self.PassengerNonStationaryArrivalEvent(
                arrival_rates=self.upbound_arrival_rate /
                100 *
                BuildingPopulation *
                12,
                arrival_mode=0,
                EventTime=0,
                outer=self))
        self.Calendar.Schedule(
            self.PassengerNonStationaryArrivalEvent(
                arrival_rates=self.downbound_arrival_rate /
                100 *
                BuildingPopulation *
                12,
                arrival_mode=1,
                EventTime=0,
                outer=self))
        self.Calendar.Schedule(
            self.PassengerNonStationaryArrivalEvent(
                arrival_rates=self.crossfloor_arrival_rate /
                100 *
                BuildingPopulation *
                12,
                arrival_mode=2,
                EventTime=0,
                outer=self))
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

            # ### TRACE ######
            print(f"Executed event: {NextEvent} Current time: {SimClasses.Clock}, Post-event state below")
            # # pp.pprint([(e, e.EventTime, e.outer) for e in self.Calendar.ThisCalendar])
            # print(
            #     "Passengers waiting (source floor, destination floor)", [
            #         (i, p.destination_floor) for i, q in enumerate(
            #             self.floor_queues) for p in q.ThisQueue])
            # for i, car in enumerate(self.cars):
            #     print(f"Car {i+1} - onboard:{[len(floor) for floor in car.dest_passenger_map]} floor: {car.floor} "
            #           f"next floor: {car.next_floor} status: {car.status} direction: {car.direction}")
            # print([car.requests for car in self.cars])
            # ################

            NextEvent = self.Calendar.Remove()

        self.callback()

    def callback(self):
        print(f"Mean time in system: {self.TimesInSystem.Mean()} Mean waiting time: {self.WaitingTimes.Mean()}")

        arrivals_in_hours = np.array(self.AllArrivalTimes) / 60
        sns.lineplot(arrivals_in_hours, list(range(1, len(self.AllArrivalTimes) + 1)))
        # plot cumulative arrivals
        plt.xlabel("Time (24H)")
        plt.ylabel("Cumulative passenger arrivals")
        plt.xticks(range(math.floor(min(arrivals_in_hours)), math.ceil(max(arrivals_in_hours)) + 1))
        plt.show()

        sns.distplot(self.TimesInSystem.Observations, norm_hist=True)  # plot histogram of times in system

        plt.xlabel("Time (minutes)")
        plt.show()

        sns.distplot(self.WaitingTimes.Observations, norm_hist=True)
        plt.xlabel("Time (minutes)")
        plt.show()

        if self.write_to_csvs:
            for queue in self.floor_queues:
                df = pd.read_csv(f"{queue.figure_dir}/queue{queue.id}_lengths.csv", names=['time', 'count'])
                df.time = df.time / 60
                sns.lineplot(x='time', y='count', label=f'floor {queue.id}', data=df)
            plt.xticks(range(math.floor(min(df.time)), math.ceil(max(df.time)) + 1))
            plt.xlabel("Time (24H)")
            plt.title('Number of Patrons Queuing for Elevators by Floor Over Time')
            plt.legend()
            plt.show()

    @classmethod
    def CI_95(cls, data):
        a = np.array(data)
        n = len(a)
        m = np.mean(a)
        sd = np.std(a, ddof=1)
        hw = 1.96 * sd / np.sqrt(n)
        return m, [m - hw, m + hw]


class ReplicationDestDispatch(ReplicationTraditional):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cars = [ElevatorCarDestDispatch(outer=self) for _ in range(0, self.num_cars)]
        self.source_destination_queue_matrix = np.array([[FIFOQueuePlus(id=(i, j), figure_dir=self.figure_dir)
                                                        for i, _ in enumerate(range(0, self.num_floors))]
                                                        for j, _ in enumerate(range(0, self.num_floors))])
        self.source_destination_matrix = np.frompyfunc(lambda x: len(x.ThisQueue), 1, 1)(self.source_destination_queue_matrix)

    class PassengerNonStationaryArrivalEvent(ReplicationTraditional.PassengerNonStationaryArrivalEvent):
        def enqueue_passenger(self, passenger: Passenger):
            self.outer.source_destination_queue_matrix[passenger.source_floor, passenger.destination_floor].Add(
                passenger)

    class AssignRequestEvent(ReplicationTraditional.AssignRequestEvent):
        def event(self):
            # trigger an action from the idle vehicle. otherwise it will stay idle indefinitely
            self.outer.source_destination_matrix[self.new_passenger.source_floor, self.new_passenger.destination_floor] += 1
            for car in self.outer.cars:
                if car.status == 0:
                    car.status = 3
                    car.next_action()
                    break

    def callback(self):
        print(f"Mean time in system: {self.TimesInSystem.Mean()} Mean waiting time: {self.WaitingTimes.Mean()}")

        arrivals_in_hours = np.array(self.AllArrivalTimes) / 60
        sns.lineplot(arrivals_in_hours, list(range(1, len(self.AllArrivalTimes) + 1)))
        # plot cumulative arrivals
        plt.xlabel("Time (24H)")
        plt.ylabel("Cumulative passenger arrivals")
        plt.xticks(range(math.floor(min(arrivals_in_hours)), math.ceil(max(arrivals_in_hours)) + 1))
        plt.show()

        sns.distplot(self.TimesInSystem.Observations, norm_hist=True)  # plot histogram of times in system

        plt.xlabel("Time (minutes)")
        plt.show()

        sns.distplot(self.WaitingTimes.Observations, norm_hist=True)
        plt.xlabel("Time (minutes)")
        plt.show()

        if self.write_to_csvs:
            for queue in np.nditer(self.source_destination_queue_matrix):
                df = pd.read_csv(f"{queue.figure_dir}/queue{queue.id}_lengths.csv", names=['time', 'count'])
                df.time = df.time / 60
                sns.lineplot(x='time', y='count', label=f'floor {queue.id}', data=df)
            plt.xticks(range(math.floor(min(df.time)), math.ceil(max(df.time)) + 1))
            plt.xlabel("Time (24H)")
            plt.title('Number of Patrons Queuing for Elevators by Floor Over Time')
            plt.legend()
            plt.show()

class Experiment:
    def __init__(self):
        pass

    def __call__(self):
        return self.main()

    def main(self):
        pass


r0 = ReplicationTraditional(run_length=60 * 24,
                            num_floors=5,
                            num_cars=2,
                            car_capacity=20,
                            write_to_csvs=True)
r0()
pass
