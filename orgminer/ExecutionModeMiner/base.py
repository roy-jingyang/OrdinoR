# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

class BaseMiner(ABC):
    """This abstract class acts as an interface and should be used as a
    base class for implementing any approach for learning execution 
    modes.

    Parameters
    ----------
    el : DataFrame
        An event log from which the execution modes are learned.

    Attributes
    ----------
    _ctypes : 

    is_ctypes_verified : bool

    _atypes : 

    is_atypes_verified : bool

    _ttypes : 

    is_ttypes_verified : bool

    Methods
    -------
    derive_resource_log(el)

    verify_partition(whole_set, partitioning)

    verify():

    get_type_by_value(value)

    get_values_by_type(type_name)
    

    Notes
    -----

    get_type_by_value(value)

    get_values_by_type(type_name)
    An execution mode miner class should inherit from the BaseMiner and 
    enables learning and storing the mappings:

    - from case ids to Case Types,

    .. math:: \\mathcal{C} \\rightarrow \\mathcal{CT}

    - from activity labels to Activity Types,

    .. math:: \\mathcal{A} \\rightarrow \\mathcal{AT}

    - from timestamps to Time Types,

    .. math:: \\mathcal{T} \\rightarrow \\mathcal{TT}

    These mappings should be built using python `dicts`.

    Also, it should enable the conversion from a source event log to a 
    derived resource log.
    """
    _ctypes = None
    is_ctypes_verified = False

    _atypes = None
    is_atypes_verified = False

    _ttypes = None
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

        Note that, such 'resource event's are required to contain resource
        information, i.e. events with no resource information in the source 
        event log will be implicitly discarded.

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

    def verify_partition(self, whole_set, partitioning):
        '''Verify if the given partitioning (as a dict) is indeed a
        partitioning of values of the given set.

        Parameters
        ----------
        whole_set : set
        partitioning : dict

        Returns
        -------
        : a boolean flag 
        '''
        # since it is given as a dict, clusters are naturally mutual exclusive
        is_disjoint = True
        is_union = set(partitioning.keys()) == whole_set
        return is_disjoint and is_union

    def verify(self):
        '''Verify if the built execution modes are valid and print prompting
        information.

        Parameters
        ----------

        Returns
        -------
        '''
        if (self.is_ctypes_verified and self.is_atypes_verified and
                self.is_ttypes_verified):
            print('-' * 80)
            print('Count of Types in the current {}:'.format(
                self.__class__.__name__))
            print('Number of C Types:\t\t{}'.format(
                len(set(self._ctypes.values())) 
                if self._ctypes is not None else 'n/a (1)'))
            print('Number of A Types:\t\t{}'.format(
                len(set(self._atypes.values()))
                if self._atypes is not None else 'n/a (1)'))
            print('Number of T Types:\t\t{}'.format(
                len(set(self._ttypes.values()))
                if self._ttypes is not None else 'n/a (1)'))
            print('-' * 80)
        else:
            print('C Types:\t{}'.format('VERIFIED' if self.is_ctypes_verified
                else 'INVALID'))
            print('A Types:\t{}'.format('VERIFIED' if self.is_atypes_verified
                else 'INVALID'))
            print('T Types:\t{}'.format('VERIFIED' if self.is_ttypes_verified
                else 'INVALID'))
            exit('[Error] Failed to verify the collected execution modes: ')

    # from original activity labels/case ids/timestamps to types
    # TODO:
    def get_type_by_value(self, value):
        '''Query the built type given a value (of either an activity label, a
        case id, or a timestamp).

        Parameters
        ----------
        value : str
            The given value to be queried.

        Returns
        -------
        type_name : str
            The corresponding type name.
        '''
        pass

    # from types to the originals
    def get_values_by_type(self, type_name):
        '''Query the original values (of activity labels, case ids, or 
        timestamps) given a type name.

        Parameters
        ----------
        type_name : str
            The given type name to be queried.

        Returns
        -------
        values : list of str
            The corresponding values.
        '''
        if type_name.startswith('CT'):
            return list(k for k, v in self._ctypes.items()
                if v == type_name)

        if type_name.startswith('AT'):
            return list(k for k, v in self._atypes.items()
                if v == type_name)

        if type_name.startswith('TT'):
            return list(k for k, v in self._ttypes.items()
                if v == type_name)

        exit('[Error] Failed to parse the queried type.')

    
