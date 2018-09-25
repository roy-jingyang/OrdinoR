# -*- coding: utf-8 -*-

'''
This module contains the definition of the abstract class
'BaseExecutionModeMiner', as well as the definition of a class which is a naive
implementation of the baseline execution model mining approach.
'''

from abc import ABC, abstractmethod

class BaseExecutionModeMiner(ABC):
    '''This abstract class implements the definition of the data structure for
    case/activity/time types as python dicts, which contain the mappings from
    case/activity/time types to their corresponding identifiers.
    '''

    _ctypes = dict()
    is_ctypes_verified = False
    _atypes = dict()
    is_atypes_verified = False
    _ttypes = dict()
    is_ttypes_verified = False

    def __init__(self, el):
        self._build_ctypes(el)
        self._build_atypes(el)
        self._build_ttypes(el)
        self.verify()

    def _build_ctypes(self, el):
        '''Mine the case types.

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is discovered.

        Returns
        -------
        is_valid : a boolean flag indicating the status of the operation.
        '''
        pass

    def _build_atypes(self, el):
        '''Mine the activity types.

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is discovered.

        Returns
        -------
        is_valid : a boolean flag indicating the status of the operation.
        '''
        pass

    def _build_ttypes(self, el):
        '''Mine the time types.

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is discovered.

        Returns
        -------
        is_valid : a boolean flag indicating the status of the operation.
        '''
        pass

    @abstractmethod
    def convert_event_log(self, el):
        '''Create a new log given the original log after the execution
        modes have been discovered and verified. The collections of case/
        activity/time identifiers in the original event log will be mapped
        onto the corresponding execution modes.

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is discovered.

        Returns
        -------
        rl: DataFrame
            The converted log in pandas DataFrame form.
        '''
        pass

    def verify(self):
        self.is_ctypes_verified = (
                len(self._ctypes) == 0 or self.is_ctypes_verified)
        self.is_atypes_verified = (
                len(self._atypes) == 0 or self.is_atypes_verified)
        self.is_ttypes_verified = (
                len(self._ttypes) == 0 or self.is_ttypes_verified)

        if (self.is_ctypes_verified and self.is_atypes_verified and
                self.is_ttypes_verified):
            print('-' * 80)
            print('Count of Types in the current {}:'.format(
                self.__class__.__name__))
            print('Number of C Types:\t\t{}'.format(len(self._ctypes)))
            print('Number of A Types:\t\t{}'.format(len(self._atypes)))
            print('Number of T Types:\t\t{}'.format(len(self._ttypes)))
            print('-' * 80)
        else:
            exit('[Error] Failed to verify the collected execution modes')

    
