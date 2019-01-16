# -*- coding: utf-8 -*-

'''
This module contains the definition of the abstract class
'BaseMiner', which is the base class for any approach for discovering 
execution modes.

An execution mode miner class inherited from the BaseMiner should
learn and store the mappings C -> CT, A -> AT, T -> TT, and should enable the
conversion from a source event log to a derived resource log.
'''

from abc import ABC, abstractmethod

class BaseMiner(ABC):
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

    @abstractmethod
    def _build_ctypes(self, el, **kwargs):
        '''Mine the case types.
        Each type should be stored as a key-value pair where the key is of
        string type and the value is a set of strings.

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
    def _build_atypes(self, el, **kwargs):
        '''Mine the activity types.
        Each type should be stored as a key-value pair where the key is of
        string type and the value is a set of strings.

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
    def _build_ttypes(self, el, **kwargs):
        '''Mine the time types.
        Each type should be stored as a key-value pair where the key is of
        string type and the value is a set of strings.

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
    def derive_resource_log(self, el):
        '''Derive a 'resource log' given the original log AFTER the execution
        modes have been discovered and verified. The collections of case/
        activity/time identifiers in the original event log will be mapped
        onto the corresponding execution modes.

        Each 'resource event' in the derived resource log is corresponded with
        an event in the source event log exactly (even if the resource log is a
        multiset).

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is discovered.

        Returns
        -------
        rl: DataFrame
            The derived resource log in pandas DataFrame form.
        '''
        pass

    def verify(self):
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
    
