import pythonsim.SimClasses as SimClasses

from abc import abstractmethod
from pathlib import Path
import csv


class DTStatPlus(SimClasses.DTStat):
    def __init__(self):
        super().__init__()
        self.Observations = []

    def Record(self, X):
        # Update the DTStat
        self.Sum = self.Sum + X
        self.SumSquared = self.SumSquared + X * X
        self.NumberOfObservations = self.NumberOfObservations + 1
        self.Observations.append(X)


class FIFOQueuePlus(SimClasses.FIFOQueue):
    def __init__(self, id, figure_dir=None):
        super().__init__()
        self.id = id
        self.figure_dir = figure_dir

    def output_size(self):
        if self.figure_dir is not None:
            with open(Path(self.figure_dir) / f'queue{self.id}_lengths.csv', 'a+', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([SimClasses.Clock, self.NumQueue()])

    def Add(self,X):
        # Add an entity to the end of the queue
        self.ThisQueue.append(X)
        numqueue = self.NumQueue()
        self.WIP.Record(float(numqueue))
        self.output_size()

    def Remove(self):
        # Remove the first entity from the queue and return the object
        # after updating the queue statistics
        if len(self.ThisQueue) > 0:
            remove = self.ThisQueue.pop(0)
            self.WIP.Record(float(self.NumQueue()))
            self.output_size()
            return remove


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
