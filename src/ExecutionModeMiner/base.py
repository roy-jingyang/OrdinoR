# -*- coding: utf-8 -*-

'''
This module contains the definition of the class 'ExecutionModeMap', as well
as the implementation of a naive execution mode mining method.
'''

class ExecutionModeMap:
    '''
    The class implements the definition of case/activity/time types, which
    contains the following mappings from the universes of case identifiers,
    activity identifiers and time identifiers: C -> CT, A -> AT and T -> TT.
    '''
    def __init__(self):
        self.ctypes = dict()
        self.atypes = dict()
        self.ttypes = dict()
        self.is_verified = False

    def get(self, exec_mode):
        '''Query for an execution mode from the ExecutionModeMap

        Parameters
        ----------
        exec_mode : a 3-tuple of (ct, at, tt)

        Returns
        -------
        a 3-tuple containing the original case/activity/time identifiers.
        '''
        return tuple((
            self.ctypes[exec_mode[0]] if exec_mode[0] is not None else None,
            self.atypes[exec_mode[1]] if exec_mode[1] is not None else None,
            self.ttypes[exec_mode[2]] if exec_mode[2] is not None else None))

    def set_c_types(self, par_c, index=None):
        '''Set up the case types.

        Parameters
        ----------
        par_c : a list of disjoint sets of case identifiers
        index : a list of indices corresponding the par_c, defaults to None,
        i.e. CT will be automatically indexed.

        Returns
        -------
        is_set : a boolean flag indicating the status of the operation.
        '''
        # check if disjoint
        if len(self.ctypes) == 0 and len(set.intersection(*par_c)) == 0:
            if index is None:
                for i, coll in enumerate(par_c):
                    self.ctypes['CT.{}'.format(i)] = coll.copy()
            else:
                for i in range(len(index)):
                    self.ctypes[index[i]] = par_c[i].copy()
            return True
        else:
            return False
        

    def set_a_types(self, par_a, index=None):
        '''Set up the activity types.

        Parameters
        ----------
        par_a : a list of disjoint sets of activity identifiers
        index : a list of indices corresponding the par_a, defaults to None,
        i.e. AT will be automatically indexed.

        Returns
        -------
        '''
        if len(self.atypes) == 0 and len(set.intersection(*par_a)) == 0:
            if index is None:
                for i, coll in enumerate(par_a):
                    self.atypes['AT.{}'.format(i)] = coll.copy()
            else:
                for i in range(len(index)):
                    self.atypes[index[i]] = par_a[i].copy()
            return True
        else:
            return False

    def set_t_types(self, par_t, index=None):
        '''Set up the time types.

        Parameters
        ----------
        par_t : a list of disjoint sets of time identifiers
        index : a list of indices corresponding the par_t, defaults to None,
        i.e. TT will be automatically indexed.

        Returns
        -------
        '''
        if len(self.ttypes) == 0 and len(set.intersection(*par_t)) == 0:
            if index is None:
                for i, coll in enumerate(par_t):
                    self.ttypes['TT.{}'.format(i)] = coll.copy()
            else:
                for i in range(len(index)):
                    self.ttype[index[i]] = par_t[i].copy()
            return True
        else:
            return False

    def verify(self, el):
        '''Check the validity of the current built case/activity/time types,
        against the input event log,
        i.e. the defined types of a dimension should be a partitioning of the
        corresponding universe of identifiers

        This method MUST be invoked whenever a ExecutionModeMap object is
        created and set (filled with actual data).

        Parameters
        ----------
        el : DataFrame
            The event log in pandas DataFrame form, from which the current
            execution mode is learned.

        Returns
        -------
        '''

        # check if the built types align with the formal definition
        # i.e. partitioning the corresponding universe
        # Note: the disjoint constraint is fulfilled in the set up methods
        c_valid = (
                len(self.ctypes) == 0 # if is not considered
                or 
                (set.union(*self.ctypes.values()) ==
                    el.groupby('case_id').groups.keys()))
        a_valid = (
                len(self.atypes) == 0 # if is not considered
                or
                (set.union(*self.atypes.values()) ==
                    el.groupby('activity').groups.keys()))
        t_valid = (
                len(self.ttypes) == 0 # if is not considered
                or
                (set.union(*self.ttypes.values()) ==
                    el.groupby('timestamp').groups.keys()))

        self.is_verified = (c_valid and a_valid and t_valid)
        return


def naive_miner(el):
    '''
    The baseline approach for discovering execution modes, i.e.
    AT = {{a} | a in A}, CT = {C} (not considered), TT = {T} (not considered).

    Params:
        el: DataFrame
            The imported event log.
    Returns:
        exec_mode_map: an object of ExecutionModeMap
    '''

    exec_mode_map = ExecutionModeMap()
    # Case Types: N/A
    # Activity Types
    exec_mode_map.set_a_types(
            [{x} for x in el.groupby('activity').groups.keys()])
    # Time Types: N/A
    
    # Verify the map
    exec_mode_map.verify(el)
    
    if exec_mode_map.is_verified:
        return exec_mode_map
    else:
        exit('[Error] Invalid result in discovering execution modes')
    
