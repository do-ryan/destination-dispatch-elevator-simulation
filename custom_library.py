import pythonsim.SimClasses as SimClasses

from abc import abstractmethod


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
