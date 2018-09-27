# -*- coding: utf-8 -*-

'''
This module contains the definition of the class OrganizationalModel, of which
an object would be the discovery result of organizational model mining methods
included in the sibling modules.

This module also contains the implementation of the default mining method
by Song & van der Aalst (ref. Song & van der Aalst, DSS 2008), as a simple
approach.
'''

from collections import defaultdict

class OrganizationalModel:
    '''This class provides mainly the definition of the underly data structure:
        * Resource Group ("rg")
        * Membership ("mem"): Resource Group -> PowerSet(Resource)
        * Capability ("cap"): Resource Group -> PowerSet(Execution Mode)

    In the implementation, a integer id will be assigned to each resource
    group, which will then act as an "external key".
    
    Methods related to the query (retrieval?) on an organizaitonal model are
    also defined.
    '''

    # TODO: Resource Group should be a mapping from RG id to their descriptions
    # For now the description part is not be used since we don't know how to
    # annotate a resource group.
    _rg = dict()
    _mem = dict()
    _cap = dict()
    
    # An extra python dict recording the belonging of each resource, i.e. a
    # reverse mapping of Membership ("mem") - the individual resource POV.
    _rmem = defaultdict(lambda: set())
    # An extra python dict recording the qualified groups for each execution
    # mode, i.e. a reverse mapping of Capability ("cap").
    _rcap = defaultdict(lambda: set())

    def __init__(self):
        self._rg_id = -1
    
    def add_group(self, og, exec_modes):
        '''Add a new group into the organizational model.

        Parameters
        ----------
        og: iterator
            The ids of resources to be added as a resource group.
        exec_modes: iterator
            The execution modes corresponding to the og.

        Returns
        -------
        '''
        self._rg_id += 1
        self._rg[self._rg_id] = dict()

        self._mem[self._rg_id] = set()
        # two-way dict here
        for r in og:
            self._mem[self._rg_id].add(r)
            self._rmem[r].add(self._rg_id)

        # another two-way dict here
        self._cap[self._rg_id] = set()
        for cap in exec_modes:
            self._cap[self._rg_id].add(cap)
            self._rcap[cap].add(self._rg_id)
        
        return

    def size(self):
        '''Query the size (number of organizational groups) of the model.

        Parameters
        ----------

        Returns
        -------
        int
            The number of the groups.
        '''
        return len(self._rg)

    def resources(self):
        '''Simply return all the resources involved in the model.

        Parameters
        ----------

        Returns
        -------
        set
            The set of all resources.
        '''
        return set(self._rmem.keys())

    def find_group(self, r):
        '''Query the membership (i.e. belonging to which groups) of a resource
        given its identifier.

        Parameters
        ----------
        r: 
            The identifier of a resource given.

        Returns
        -------
        list of sets
            The groups to which the queried resource belong.
        '''
        return [self._mem[rg_id] for rg_id in self._rmem[r]]

    def find_all_groups(self):
        '''Simply return all the discovered groups.

        Parameters
        ----------

        Returns
        -------
        list of sets
            The groups to which the queried resource belong.
        '''
        return [g for g in self._mem.values()]
    
    def get_candidates(self, exec_mode):
        '''Query the capability (i.e. execution modes allowed by the model) 
        of a resource given its identifier.

        Parameters
        ----------
        exec_mode: set of 3-tuples
            The identifier of a resource given.

        Returns
        -------
        list of sets
            The groups which is capable of this execution mode.
        '''
        return [self._mem[rg_id] for rg_id in self._rcap[exec_mode]]

    # IO related methods
    # TODO 
    def to_file_csv(self, f):
        '''Export and write the current organizational model to a csv file.

        Data exchange format in the csv file (each row):
        Resource Group id, [Resource x; ...], [CTx|ATx|TTx; ...]

        Parameters
        ----------
        f: file object
            The destination csv file to be written.

        Returns
        -------
        '''
        from csv import writer
        writer = writer(f)

        rows = list()
        for rg_id in sorted(self._rg.keys()):
            str_rg_id = str(rg_id)
            str_members = ';'.join(sorted(str(r) for r in self._mem[rg_id]))
            str_exec_modes = ';'.join(sorted(
                    '|'.join(str(t) for t in mode)
                    for mode in self._cap[rg_id]))

            rows.append([str_rg_id, str_members, str_exec_modes])

        writer.writerows(rows)

    @classmethod
    def from_file_csv(cls, f):
        '''Read from a csv file and return an organizational model.

        Data exchange format in the csv file (each row):
        Resource Group id, [Resource x; ...], [CTx|ATx|TTx; ...]

        Parameters
        ----------
        f: file object
            The sourcd csv file to be read from.

        Returns
        -------
        '''
        from csv import reader
        om_obj = cls()
        for row in reader(f):
            group = row[1].split(';')
            # keep order: we assume the imported model is generated from
            # applying the method 'to_file_csv'
            exec_modes = list()
            for str_mode in row[2].split(';'):
                exec_modes.append(tuple(str_mode.split('|')))
            om_obj.add_group(group, exec_modes)
        return om_obj


# Note: this method does not require explicitly discovered resource profiles to
# be used.
# To stick to the original design of the method, each activity name in
# the source event log should be mapped to a single Activity Type in the
# resource log.
def default_mining(rl):
    '''
    The default mining method.

    Params:
        rl: DataFrame
            The resource log.

    Returns:
        om: OrganizationalModel object
            The discovered organizational model.
        ogs: list of sets
            A list of organizational groups.
    '''

    print('Applying Default Mining:')
    ogs = list()
    for atype, events in rl.groupby('activity_type'):
        ogs.append(set(events['resource']))
    print('{} organizational groups discovered.'.format(len(ogs)))
    return ogs

