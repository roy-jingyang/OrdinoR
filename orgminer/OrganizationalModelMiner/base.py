# -*- coding: utf-8 -*-

'''
This module contains the definition of the class OrganizationalModel, of which
an object would be the discovery result of organizational model mining methods
included in the sibling modules.

This module also contains the implementation of the default mining method
by Song & van der Aalst (ref. Song & van der Aalst, DSS 2008), as a simple
approach.
'''

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

    def __init__(self):
        self._rg_id = -1
        # TODO: Resource Group should be a mapping from RG id to their descriptions
        # For now the description part is not be used since we don't know how to
        # annotate a resource group.
        self._rg = dict()
        self._mem = dict()
        self._cap = dict()
        
        from collections import defaultdict
        # An extra python dict recording the belonging of each resource, i.e. a
        # reverse mapping of Membership ("mem") - the individual resource POV.
        self._rmem = defaultdict(set)
        # An extra python dict recording the qualified groups for each execution
        # mode, i.e. a reverse mapping of Capability ("cap").
        self._rcap = defaultdict(set)
        
    def add_group(self, og, exec_modes):
        '''Add a new group into the organizational model.

        Parameters
        ----------
        og: iterator
            The ids of resources to be added as a resource group.
        exec_modes: iterator (list or dict of lists)
            The execution modes corresponding to the group.

        Returns
        -------
        '''

        if type(exec_modes) == list:
            # no refinement applied
            self._rg_id += 1
            self._rg[self._rg_id] = '' # TODO: further description of a group

            self._mem[self._rg_id] = set()
            # two-way dict here
            for r in og:
                self._mem[self._rg_id].add(r)
                self._rmem[r].add(self._rg_id)

            # another two-way dict here
            self._cap[self._rg_id] = list()
            for m in exec_modes:
                self._cap[self._rg_id].append(m)
                self._rcap[m].add(self._rg_id)

        elif type(exec_modes) == list:
            # refinement applied
            for subog, subm in exec_modes.items():
                if subog not in self._mem.values():
                    self._rg_id += 1
                    self._rg[self._rg_id] = ''

                    self._mem[self._rg_id] = set()
                    for r in subog:
                        self._mem[self._rg_id].add(r)
                        self._rmem[r].add(self._rg_id)

                    self._cap[self._rg_id] = list()
                    for m in subm:
                        self._cap[self._rg_id].append(m)
                        self._rcap[m].add(self._rg_id)

                else:
                    pass

        else:
            exit('[Error] Invalid execution modes')
            
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
        frozenset
            The set of all resources.
        '''
        return frozenset(self._rmem.keys())

    def find_group_ids(self, r):
        '''Query the id of groups that contain a resource given its identifier.

        Parameters
        ----------
        r: 
            The identifier of a resource given.

        Returns
        -------
        list of ints
            The id of the groups to which the queried resource belong.
        '''
        return list(self._rmem[r])

    def find_groups(self, r):
        '''Query the groups that contain a resource given its identifier.

        Parameters
        ----------
        r: 
            The identifier of a resource given.

        Returns
        -------
        list of frozensets
            The groups to which the queried resource belong.
        '''
        return [frozenset(self._mem[rg_id]) for rg_id in self._rmem[r]]

    def find_execution_modes(self, r):
        '''Query the allowed execution modes of a resource given its identifier.

        Parameters
        ----------
        r: 
            The identifier of a resource given.

        Returns
        -------
        list of 3-tuples
            The allowed execution modes specific to the given resource.
        '''
        return list(set.union(
            *[self._cap[rg_id] for rg_id in self._rmem[r]]))
    
    def find_candidate_groups(self, exec_mode):
        '''Query the capable groups (i.e. groups that can perform the execution
        mode according to the model) given an execution mode.

        Parameters
        ----------
        exec_mode: 3-tuple
            The execution mode given.

        Returns
        -------
        list of frozensets
            The groups of resources which are capable of this execution mode.
        '''
        return [frozenset(self._mem[rg_id]) for rg_id in self._rcap[exec_mode]]

    def find_all_groups(self):
        '''Simply return all the groups.

        Parameters
        ----------

        Returns
        -------
        list of 2-tuples: (int, frozenset)
            The ids and member resources of the groups.
        '''
        return [(rg_id, frozenset(self._mem[rg_id])) 
            for rg_id in self._rg.keys()]

    def find_group_execution_modes(self, rg_id):
        '''Query the capable execution modes given a group identified by its
        id.

        Parameters
        ----------
        rg_id: int
            The given group id.

        Returns
        -------
        list of 3-tuples
            The allowed execution modes specific to the given group.
        '''
        return self._cap[rg_id]
    
    def find_all_execution_modes(self):
        '''Simply return all the execution modes related to the groups.

        Parameters
        ----------

        Returns
        -------
        list of lists of 3-tuples
            The execution modes related.
        '''
        return [em for em in self._cap.values()]
    
    # IO related methods
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
            if len(self._cap) > 0:
                str_exec_modes = ';'.join(sorted(
                        '|'.join(str(t) for t in mode)
                        for mode in self._cap[rg_id]))
            else:
                str_exec_modes = ''

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
        ogs: list of frozensets
            A list of organizational groups.
    '''
    print('Applying Default Mining:')
    ogs = list()
    for atype, events in rl.groupby('activity_type'):
        ogs.append(frozenset(events['resource']))
    #print('{} organizational groups discovered.'.format(len(ogs)))
    return ogs
