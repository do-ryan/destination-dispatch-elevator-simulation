import pythonsim.SimClasses as SimClasses

from abc import abstractmethod
from pathlib import Path
import csv
import numpy as np


def CI_95(data):
    a = np.array(data)
    n = len(a)
    m = np.mean(a)
    sd = np.std(a, ddof=1)
    hw = 1.96 * sd / np.sqrt(n)
    return m, [m - hw, m + hw]


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

    def BatchedQuantile(self, b: int, q: float):
        """
        Args:
            - b: batch size
            - q: quantile
        """

        batched_observations = np.array_split(np.random.permutation(self.Observations),
                                             indices_or_sections=b,
                                             axis=0)
        return np.array([np.quantile(batch, q) for batch in batched_observations])


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
