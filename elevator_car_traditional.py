from custom_library import FunctionalEventNotice

import pythonsim.SimClasses as SimClasses
import numpy as np
import itertools


class ElevatorCarTraditional(SimClasses.Resource):
    def __init__(self,
                 outer, # must be ReplicationTraditional class
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
        self.status = 0  # 0 for idle car, 1 for moving, 2 for transferring, 3 for in process of choosing next task
        # only changes at the start of some action
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

        # Note: must change status as soon as start action event is scheduled.
        # Otherwise it allows task allocated to trigger another action event if car is idle

    def floor_dwell(self, num_passengers: int):
        """Return amount of time to spend transferring passengers. Door open/ close is lumped into move event."""
        return self.passenger_move_time * num_passengers

    class PickupEvent(FunctionalEventNotice):
        def event(self):
            """Pick-up as many passengers as possible from current floor, update self resource, add waiting time data."""
            self.outer.status = 2
            num_passengers = 0
            directions_requested = set(np.nonzero(self.outer.requests[self.outer.floor])[0].tolist())
            num_to_pick_up = len(
                [p for p in self.outer.floor_queues[self.outer.floor].ThisQueue if p.direction in directions_requested])
            while num_passengers < num_to_pick_up and self.outer.Busy < self.outer.NumberOfUnits:
                next_passenger = self.outer.floor_queues[self.outer.floor].Remove()
                if self.outer.requests[self.outer.floor, next_passenger.direction]:
                    # if this car has a request equal to the first person in queue's intended direction
                    assert isinstance(next_passenger, Passenger)
                    self.outer.Seize(1)
                    self.outer.dest_passenger_map[next_passenger.destination_floor].append(next_passenger)
                    self.outer.outer.WaitingTimes.Record(SimClasses.Clock - next_passenger.CreateTime)
                    num_passengers += 1
                else:
                    # if the car does not have a request equal to first person's intended
                    # direction, put him to back of line
                    self.outer.floor_queues[self.outer.floor].ThisQueue.append(next_passenger)

            self.outer.requests[self.outer.floor, 0] = 0
            self.outer.requests[self.outer.floor, 1] = 0
            # ensure that there are no more requests on this floor for this car otherwise it will infinitely
            # try to pick them up if the car is full.

            self.outer.Calendar.Schedule(self.outer.PickupEndEvent(EventTime=self.outer.floor_dwell(num_passengers),
                                                                   outer=self.outer))

            for passenger in self.outer.floor_queues[self.outer.floor].ThisQueue:
                # Guaranteed O(2) per pickup.
                if passenger.direction in directions_requested:
                    directions_requested -= {passenger.direction}
                    # for all passengers requests that were allocated to this car that couldn't get on,
                    # re-allocate car to those requests
                    self.outer.Calendar.Schedule(self.outer.outer.AssignRequestEvent(new_passenger=passenger,
                                                                                     EventTime=0,
                                                                                     outer=self.outer.outer))
                if len(directions_requested) == 0:
                    break

    class PickupEndEvent(FunctionalEventNotice):
        def event(self):
            for dest in self.outer.dest_passenger_map:
                for passenger in dest:
                    passenger.entry_time = SimClasses.Clock
            self.outer.next_action()

    class DropoffEvent(FunctionalEventNotice):
        def event(self):
            """Drop-off all passengers on self with destination as current floor. Add time"""
            self.outer.status = 2
            num_passengers = 0
            while len(self.outer.dest_passenger_map[self.outer.floor]) > 0:
                self.outer.Free(1)
                next_passenger = self.outer.dest_passenger_map[self.outer.floor].pop(0)
                self.outer.outer.TimesInSystem.Record(SimClasses.Clock - next_passenger.CreateTime)
                self.outer.outer.TravelTimes.Record(SimClasses.Clock - next_passenger.entry_time)
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
        """Triggered after moving, or done a transfer. Schedules event representing next action."""
        if np.sum(self.requests) == 0 and len(
                [passenger for passenger in itertools.chain.from_iterable(self.dest_passenger_map)]) == 0:
            # if no requests nor drop-offs, go idle. this is the only case where the car should go idle.
            # if the car goes idle while it has a potential task, a new task may trigger a state of multiple actions,
            # because new passengers look for idle cars.
            self.status = 0
            self.next_floor = None
            self.direction = None
        else:
            if len(self.dest_passenger_map[self.floor]) > 0:
                self.Calendar.Schedule(self.DropoffEvent(EventTime=0, outer=self))
                self.outer.status = 2
            elif np.sum(self.requests[self.floor, :]) > 0:
                self.Calendar.Schedule(self.PickupEvent(EventTime=0, outer=self))
                self.outer.status = 2
            else:  # if no more pickups nor drop-offs on current floor, take on the next request
                scan_start = self.floor  # scan start is the floor that "up-bound" and "down-bound" are relative to
                if self.status == 3:
                    #  if in process, take the next request if there is one- start closer to terminal ends.
                    # currently there is no direction so we need to decide which direction this car will proceed in and
                    # which request to start on.
                    if np.sum(self.requests[:, 1]) \
                            + len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map)])\
                            > np.sum(self.requests[:, 0]) \
                            + len([queue for queue in itertools.chain.from_iterable(self.dest_passenger_map)]):
                        #  if there are more up-bound requests than down-bound, go up-bound.
                        self.direction = 1
                        scan_start = 0  # go to globally lowest up-bound request
                    else:
                        self.direction = 0
                        scan_start = self.requests.shape[0] - 1
                        # go to globally highest down-bound request

                # from here downwards, if car is currently idle, then it will have chosen a direction.
                # if idle,
                # it will start at the globally lowest upbound task, or highest downbound task, dropping off passengers
                # on the car along the way.
                # if the car already has a direction, it will sequentially take tasks in that direction until all
                # tasks are finished in that direction.
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
                        self.status = 3
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
                        next_pickup = scan_start - np.argmax(np.flip(self.requests[0:scan_start + 1, 0], axis=0))
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
                        self.status = 3
                        self.next_floor = None
                        self.direction = None
                        self.next_action()
                    else:
                        self.Calendar.Schedule(self.MoveEvent(destination_floor=int(next_destination),
                                                              EventTime=0,
                                                              outer=self))


class Passenger(SimClasses.Entity):
    def __init__(self,
                 source_floor: int,
                 destination_floor: int,
                 ):
        super().__init__()
        self.source_floor = source_floor
        self.destination_floor = destination_floor
        self.direction = int(self.destination_floor > self.source_floor)
        self.entry_time = None
