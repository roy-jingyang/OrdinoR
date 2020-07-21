# -*- coding: utf-8 -*-

"""This module contains the definition of class `OrganizationalModel`.

This module also contains the implementation of the default mining 
method by Song and van der Aalst [1]_, as a naive approach for
organizational model discovery from event logs.

References
----------
.. [1] Song, M., & van der Aalst, W. M. P. (2008). Towards comprehensive 
   support for organizational mining. *Decision Support Systems*, 46(1), 
   300-317. `<https://doi.org/10.1016/j.dss.2008.07.002>`_
"""
class OrganizationalModel:
    """This class defines an organizational model.

    An organizational model includes the following data structures as
    fundamentals:

        - Resource Group ID (`_rg_id`), resource group ids,
        - Resource Group (`_rg`), a mapping from resource group ids to
          group descriptions,
        - Membership (`_mem`), a mapping from resource group ids to
          member resource ids,
        - Capability (`_cap`), a mapping from resource group ids to
          capable execution modes.

    Attributes
    ----------
    group_number : int
        The number of resource groups in the model.
    resources : frozenset
        All the resources involved in the model.

    Methods
    -------
    add_group(og, exec_modes)
        Add a resource group into the organizational model.
    find_group_ids(r)
        Query the id of groups which contain a resource given its 
        identifier.
    find_groups(r)
        Query the groups which contain a resource given its identifier.
    find_execution_modes(r)
        Query the allowed execution modes of a resource given its 
        identifier.
    find_candidate_groups(exec_mode)
        Query the capable groups (i.e., groups that can perform the 
        execution mode according to the model) given an execution mode.
    find_all_groups()
        Return all resource groups involved in the model.
    find_group_members(rg_id)
        Query the group members given a group identified by its id.
    find_group_execution_modes(rg_id)
        Query the capable execution modes given a group identified by 
        its id.
    find_all_execution_modes()
        Return all execution modes involved in the model.
    to_file_csv(f)
        Export the organizational model to an external CSV file (with no 
        header line).
    from_file_csv(f)
        Import an organizational model from an external CSV file.

    Notes
    -----
    Methods related to the query on an organizational model are also 
    defined.
    """

    def __init__(self):
        self._rg_id = -1
        self._rg = dict()
        self._mem = dict()
        self._cap = dict()

        from collections import defaultdict
        self._rmem = defaultdict(set)
        self._rcap = defaultdict(set)
        

    def add_group(self, og, exec_modes):
        """Add a resource group into the organizational model.

        Parameters
        ----------
        og : iterator
            Ids of resources to be added as a resource group.
        exec_modes : iterator, list or dict of lists
            Execution modes corresponding to the group to be added.

        Returns
        -------

        Raises
        ------
        TypeError
            If the parameter type for `exec_modes` is unexpected.
        """
        if type(exec_modes) is list:
            # no refinement applied
            self._rg_id += 1
            self._rg[self._rg_id] = ''

            self._mem[self._rg_id] = set()
            for r in og:
                self._mem[self._rg_id].add(r)
                self._rmem[r].add(self._rg_id)

            self._cap[self._rg_id] = list()
            for m in exec_modes:
                self._cap[self._rg_id].append(m)
                self._rcap[m].add(self._rg_id)
        elif type(exec_modes) is dict:
            # refinement applied
            # TODO: consider deprecating
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
            raise TypeError('Invalid type for parameter `{}`: {}'.format(
                'exec_modes', type(exec_modes)))
            

    @property
    def group_number(self):
        """Return the number of resource groups of the model.

        Parameters
        ----------

        Returns
        -------
        int
            Number of resource groups.
        """
        return len(self._rg)


    @property
    def resources(self):
        """Return all the resources involved in the model.

        Parameters
        ----------

        Returns
        -------
        frozenset
            Set of all resources.
        """
        return frozenset(self._rmem.keys())


    def find_group_ids(self, r):
        """Query the id of groups which contain a resource given its 
        identifier.

        Parameters
        ----------
        r : str
            The identifier of a given resource.

        Returns
        -------
        list of ints
            Id of the groups to which the queried resource belongs.
        """
        return list(self._rmem[r])


    def find_groups(self, r):
        """Query the groups which contain a resource given its 
        identifier.

        Parameters
        ----------
        r : str
            The identifier of a given resource.

        Returns
        -------
        list of frozensets
            Groups to which the queried resource belongs.
        """
        return [frozenset(self._mem[rg_id]) 
            for rg_id in self._rmem[r]]


    def find_execution_modes(self, r):
        """Query the allowed execution modes of a resource given its 
        identifier.

        Parameters
        ----------
        r : str
            The identifier of a given resource.

        Returns
        -------
        list of 3-tuples
            Allowed execution modes specific to the queried resource.
        """
        resource_cap = set()
        for rg_id in self._rmem[r]:
            resource_cap.update(set(self._cap[rg_id]))
        return list(resource_cap)
    

    def find_candidate_groups(self, exec_mode):
        """Query the capable groups (i.e., groups that can perform the 
        execution mode according to the model) given an execution mode.

        Parameters
        ----------
        exec_mode : 3-tuple
            The execution mode given.

        Returns
        -------
        list of frozensets
            Groups of resources which are capable of this execution mode.
        """
        return [frozenset(self._mem[rg_id]) 
            for rg_id in self._rcap[exec_mode]]


    def find_all_groups(self):
        """Return all resource groups involved in the model.

        Parameters
        ----------

        Returns
        -------
        list of 2-tuples
            Ids and member resources of the groups.
        """
        return [(rg_id, frozenset(self._mem[rg_id])) 
            for rg_id in self._rg.keys()]


    def find_group_members(self, rg_id):
        """Query the capable execution modes given a group identified by 
        its id.

        Parameters
        ----------
        rg_id : int
            The given group id.

        Returns
        -------
        frozenset
            Set of member resources.
        """
        return frozenset(self._mem[rg_id])


    def find_group_execution_modes(self, rg_id):
        """Query the capable execution modes given a group identified by 
        its id.

        Parameters
        ----------
        rg_id : int
            The given group id.

        Returns
        -------
        list of 3-tuples
            Allowed execution modes specific to the given group.
        """
        return self._cap[rg_id]
    

    def find_all_execution_modes(self):
        """Return all execution modes involved in the model.

        Parameters
        ----------

        Returns
        -------
        list of 3-tuples
            Execution modes.
        """
        all_modes = list()
        for modes in self._cap.values():
            all_modes.extend(modes)
        return all_modes
    

    def to_file_csv(self, f):
        """Export the organizational model to an external CSV file (with
        no header line).

        Parameters
        ----------
        f : File object
            A destination CSV file to be written.

        Returns
        -------

        Notes
        -----
        Data exchange format in the CSV file (by each line):

        * No header line; 
        * 3 columns: Group ID, Member resource IDs, Execution modes
        * Values of member resource IDs and execution modes are separated
          by semicolons ``;`` within the columns;
        * A vertical bar ``|`` is used as the separator for values of the
          three types within an execution mode.
        
        An example is given as following:

        ======== =========================== ========================
        GroupID1 Resource ID 1;Resource ID 2 CT1|AT2|TT1;CT.0|AT2|TT3
        -------- --------------------------- ------------------------
        GroupID2 Resource ID 3;Resource ID 4 CT1|AT3|TT1;CT.0|AT2|TT4
        ======== =========================== ========================

        See Also
        --------
        OrganizationalModel.from_file_csv
        """
        from csv import writer
        writer = writer(f)

        rows = list()
        for rg_id in sorted(self._rg.keys()):
            str_rg_id = str(rg_id)
            str_members = ';'.join(sorted(str(r) for r in self._mem[rg_id]))
            if len(self._cap) > 0:
                str_exec_modes = ';'.join(sorted('|'.join(str(t) for t in mode)
                    for mode in self._cap[rg_id]))
            else:
                str_exec_modes = ''

            rows.append([str_rg_id, str_members, str_exec_modes])

        writer.writerows(rows)


    @classmethod
    def from_file_csv(cls, f):
        """Import an organizational model from an external CSV file.

        Parameters
        ----------
        f: File object
            A sourced CSV file to be read.

        Returns
        -------
        OrganizationalModel
            The imported organizational model.

        Notes
        -----
        Data exchange format in the CSV file (by each line):

        * No header line; 
        * 3 columns: Group ID, Member resource IDs, Execution modes
        * Values of member resource IDs and execution modes are separated
          by semicolons ``;`` within the columns;
        * A vertical bar ``|`` is used as the separator for values of the
          three types within an execution mode.
        
        An example is given as follows:

        ======== =========================== ========================
        GroupID1 Resource ID 1;Resource ID 2 CT1|AT2|TT1;CT.0|AT2|TT3
        -------- --------------------------- ------------------------
        GroupID2 Resource ID 3;Resource ID 4 CT1|AT3|TT1;CT.0|AT2|TT4
        ======== =========================== ========================

        See Also
        --------
        OrganizationalModel.to_file_csv
        """
        from csv import reader
        om_obj = cls()
        for row in reader(f):
            if not row:
                continue

            group = row[1].split(';')
            exec_modes = list()
            for str_mode in row[2].split(';'):
                exec_modes.append(tuple(str_mode.split('|')))
            om_obj.add_group(group, exec_modes)
        return om_obj


def default_mining(rl):
    """The default mining method.

    Parameters
    ----------
    rl : DataFrame
        A resource log.

    Returns
    -------
    ogs : list of frozensets
        A list of resource groups.

    Notes
    -----
    This method does not require using resource profiles.
    """
    print('Applying Default Mining:')
    ogs = list()
    for activity_type, events in rl.groupby('activity_type'):
        ogs.append(frozenset(events['resource']))
    return ogs

