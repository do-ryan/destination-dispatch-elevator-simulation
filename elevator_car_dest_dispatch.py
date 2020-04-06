from custom_library import FunctionalEventNotice
from elevator_car_traditional import ElevatorCarTraditional

import pythonsim.SimClasses as SimClasses
import numpy as np
import itertools
import math


class ElevatorCarDestDispatch(ElevatorCarTraditional):
    def __init__(self, outer, max_wait_threshold=3, *args, **kwargs):
        """
        Args:
            - outer: ReplicationDestDispatch
            - max_wait_threshold: if anyone waits past this threshold, they get priority.
        """
        super().__init__(outer, *args, **kwargs)
        self.source_destination_matrix = self.outer.source_destination_matrix
        self.source_destination_queue_matrix = self.outer.source_destination_queue_matrix
        self.destination_dispatched = None  # the destination floor this car is dispatched to
        self.max_wait_threshold = max_wait_threshold

    class PickupEvent(FunctionalEventNotice):
        def event(self):
            """Pick-up as many passengers going to current destination as possible from current floor,
            update self resource, add waiting time data."""
            self.outer.status = 2
            num_passengers = 0
            directions_requested = np.nonzero(self.outer.requests[self.outer.floor])[0].tolist()
            assert len(directions_requested) == 1
            direction_requested = directions_requested[0]
            leftovers = []
            while (self.outer.requests[self.outer.floor, direction_requested] > 0
                    or self.outer.source_destination_matrix[self.outer.floor, self.outer.destination_dispatched] > 0):
                # while this car still need to pick up allocated tasks or if there are new ones
                next_passenger = self.outer.source_destination_queue_matrix[self.outer.floor,
                                                                            self.outer.destination_dispatched].Remove()
                # O(number of queued passengers/ num_floors)

                if self.outer.requests[self.outer.floor, direction_requested] <= 0:
                    self.outer.source_destination_matrix[self.outer.floor, self.outer.destination_dispatched] -= 1
                    # if the car is now picking up unallocated requests, deplete from the matrix.
                if self.outer.Busy < self.outer.NumberOfUnits:
                    self.outer.Seize(1)
                    self.outer.dest_passenger_map[next_passenger.destination_floor].append(next_passenger)
                    self.outer.outer.WaitingTimes.Record(SimClasses.Clock - next_passenger.CreateTime)
                    num_passengers += 1
                else:
                    self.outer.Calendar.Schedule(self.outer.outer.AssignRequestEvent(new_passenger=next_passenger,
                                                                                     EventTime=0,
                                                                                     outer=self.outer.outer))
                    leftovers.append(next_passenger)
                    # if there are more suitable passengers than space available. usually if
                    # too many are allocated in AssignRequestEvent
                self.outer.requests[self.outer.floor, direction_requested] -= 1
                # subtract from requests array whether or not the passsenger is allocated or non-allocated.

            self.outer.requests[self.outer.floor, direction_requested] = 0
            # must be 0 here. if there were leftover requests, there are now in leftovers. might have been negative
            # if there were rejects (more waiting passengers than expected).

            self.outer.source_destination_queue_matrix[self.outer.floor, self.outer.destination_dispatched].ThisQueue \
                = leftovers + self.outer.source_destination_queue_matrix[self.outer.floor,
                                                                         self.outer.destination_dispatched].ThisQueue
                # make sure people added back to the queue are put in the front
            self.outer.Calendar.Schedule(self.outer.PickupEndEvent(EventTime=self.outer.floor_dwell(num_passengers),
                                                                   outer=self.outer))

    def next_action(self):
        """Triggered after moving, or done a transfer. Schedules event representing next action."""
        if np.sum(self.requests) == 0 \
                and len([passenger for passenger in itertools.chain.from_iterable(self.dest_passenger_map)]) == 0:
            # If no requests nor dropoffs, check central pool for waiting requests.
            # Pick destination-direction combination with highest count of passengers.
            largest_count = 0
            exceed_max_wait_threshold = False
            chosen_destination_direction = ()
            chosen_destination_longest_wait = 0
            for i in range(2 * self.source_destination_matrix.shape[1]):
                destination_direction = (i // 2, i % 2)
                if destination_direction[1] == 1:
                    sum = np.sum(self.source_destination_matrix[
                                 0:destination_direction[0] + 1,
                                 destination_direction[0]])
                else:
                    sum = np.sum(self.source_destination_matrix[
                                 destination_direction[0]::,
                                 destination_direction[0]
                                 ])
                # Find count of passengers with this destination and direction

                destination_longest_wait = 0
                if destination_direction[1] == 0:
                    sweep = self.source_destination_queue_matrix[destination_direction[0]+1::,
                                                                 destination_direction[0]]
                else:
                    sweep = self.source_destination_queue_matrix[0:destination_direction[0],
                                                                 destination_direction[0]]

                for target_destination_queue in sweep:
                    # O(num floors)
                    if target_destination_queue.ThisQueue:
                        if sum >= largest_count:
                            destination_longest_wait = max(
                                destination_longest_wait,
                                SimClasses.Clock - target_destination_queue.ThisQueue[0].CreateTime)
                        # Find earliest arrival time out of all waiting to go to this destination in this direction
                if destination_longest_wait >= self.max_wait_threshold:
                        exceed_max_wait_threshold = True
                            # But if any queue has a person waiting past the allowed threshold,
                        # here on out prioritize only floors with people waiting past that threshold.

                if destination_longest_wait >= self.max_wait_threshold or not exceed_max_wait_threshold:
                    # if max wait threshold has not yet been exceeded, consider all. otherwise only consider those past threshold.
                    if sum > largest_count:
                        largest_count = sum
                        chosen_destination_direction = destination_direction
                        chosen_destination_longest_wait = destination_longest_wait
                    elif sum == largest_count:
                        if destination_longest_wait > chosen_destination_longest_wait:
                            chosen_destination_longest_wait = destination_longest_wait
                            chosen_destination_direction = destination_direction
                    # Choose the destination/direction with the largest count.
                    # In the case of a tie, choose the one with the longer wait

            if largest_count > 0:
                # in the state where all combinations are WIP (all negative count) and this car has no tasks,
                # this condition prevents this car from taking other cars' tasks.
                self.destination_dispatched = chosen_destination_direction[0]
                if chosen_destination_direction[1] == 1:
                    source_floors = np.nonzero(self.source_destination_matrix[
                        0:chosen_destination_direction[0] + 1,
                        chosen_destination_direction[0]])[0]
                else:
                    source_floors = np.nonzero(self.source_destination_matrix[
                                               chosen_destination_direction[0]::,
                                               chosen_destination_direction[0]])[0] + chosen_destination_direction[0]
                for floor in source_floors:
                    num_more_to_pickup = min((self.NumberOfUnits - self.Busy) - np.sum(self.requests),
                                             self.source_destination_matrix[floor, chosen_destination_direction[0]])
                    self.requests[floor, chosen_destination_direction[1]] += num_more_to_pickup
                    self.source_destination_matrix[floor, chosen_destination_direction[0]] -= num_more_to_pickup
                    # The source-destination matrix acts as a global tracker of number of waiting passengers
                    # without allocated car. This updates the matrix when car allocates itself.
                    # This also makes sure the car is not allocating itself to more than it can carry.

        super().next_action()
