"""
Base definition of organizational models
"""

import csv

import ordinor.exceptions as exc

class OrganizationalModel:
    """
    This class defines an organizational model.

    An organizational model includes the following data structures as
    fundamentals:

        - Resource Group ID (`_rg_id`), resource group ids,
        - Resource Group (`_rg`), a mapping from resource group ids to
          group descriptions,
        - Membership (`_mem`), a mapping from resource group ids to
          member resource ids,
        - Capability (`_cap`), a mapping from resource group ids to
          capable execution contexts.

    Attributes
    ----------
    group_number : int
        The number of resource groups in the model.
    resources : frozenset
        All the resources involved in the model.

    Methods
    -------
    add_group(og, exe_ctxs)
        Add a resource group into the organizational model.
    find_group_ids(r)
        Query the id of groups which contain a resource given its 
        identifier.
    find_groups(r)
        Query the groups which contain a resource given its identifier.
    find_execution_contexts(r)
        Query the allowed execution contexts of a resource given its 
        identifier.
    find_candidate_groups(exe_context)
        Query the capable groups (i.e., groups that can perform the 
        execution context according to the model) given an execution
        context.
    find_all_groups()
        Return all resource groups involved in the model.
    find_group_members(rg_id)
        Query the group members given a group identified by its id.
    find_group_execution_contexts(rg_id)
        Query the capable execution contexts given a group identified by 
        its id.
    find_all_execution_contexts()
        Return all execution contexts involved in the model.
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
        

    def add_group(self, og, exe_ctxs):
        """
        Add a resource group into the organizational model.

        Parameters
        ----------
        og : iterator
            Ids of resources to be added as a resource group.
        exe_ctxs : iterator, list or dict of lists
            Execution contexts corresponding to the group to be added.

        Returns
        -------
        """
        if type(exe_ctxs) is list:
            # no refinement applied
            self._rg_id += 1
            self._rg[self._rg_id] = ''

            self._mem[self._rg_id] = set()
            for r in og:
                self._mem[self._rg_id].add(r)
                self._rmem[r].add(self._rg_id)

            self._cap[self._rg_id] = list()
            for m in exe_ctxs:
                self._cap[self._rg_id].append(m)
                self._rcap[m].add(self._rg_id)
        else:
            raise exc.InvalidParameterError(
                param='exe_ctxs',
                reason='Expected a list'
            )
            

    @property
    def group_number(self):
        """
        Return the number of resource groups of the model.

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
        """
        Return all the resources involved in the model.

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
        """
        Query the groups which contain a resource given its 
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


    def find_execution_contexts(self, r):
        """
        Query the allowed execution contexts of a resource given its 
        identifier.

        Parameters
        ----------
        r : str
            The identifier of a given resource.

        Returns
        -------
        list of 3-tuples
            Allowed execution contexts specific to the queried resource.
        """
        resource_cap = set()
        for rg_id in self._rmem[r]:
            resource_cap.update(set(self._cap[rg_id]))
        return list(resource_cap)
    

    def find_candidate_groups(self, exe_ctx):
        """
        Query the capable groups (i.e., groups that can perform the 
        execution context according to the model) given an execution
        context.

        Parameters
        ----------
        exe_ctx : 3-tuple
            The execution context given.

        Returns
        -------
        list of frozensets
            Groups of resources which are capable of this execution context.
        """
        return [frozenset(self._mem[rg_id]) 
            for rg_id in self._rcap[exe_ctx]]


    def find_all_groups(self):
        """
        Return all resource groups involved in the model.

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
        """
        Query the capable execution contexts given a group identified by 
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


    def find_group_execution_contexts(self, rg_id):
        """
        Query the capable execution contexts given a group identified by 
        its id.

        Parameters
        ----------
        rg_id : int
            The given group id.

        Returns
        -------
        list of 3-tuples
            Allowed execution contexts specific to the given group.
        """
        return self._cap[rg_id]
    

    def find_all_execution_contexts(self):
        """
        Return all execution contexts involved in the model.

        Parameters
        ----------

        Returns
        -------
        list of 3-tuples
            Execution contexts.
        """
        all_ctxs = list()
        for contexts in self._cap.values():
            all_ctxs.extend(contexts)
        return all_ctxs
    

    def to_file_csv(self, f):
        """
        Export the organizational model to an external CSV file (with no
        header line).

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
        * 3 columns: Group ID, Member resource IDs, Execution contexts
        * Values of member resource IDs and execution contexts are separated
          by semicolons ``;`` within the columns;
        * A vertical bar ``|`` is used as the separator for values of the
          three types within an execution context.
        
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
        writer = csv.writer(f)

        rows = list()
        for rg_id in sorted(self._rg.keys()):
            str_rg_id = str(rg_id)
            str_members = ';'.join(sorted(str(r) for r in self._mem[rg_id]))
            if len(self._cap) > 0:
                str_exe_ctxs = ';'.join(
                    sorted('|'.join(str(t) for t in context)
                    for context in self._cap[rg_id])
                )
            else:
                str_exe_ctxs = ''

            rows.append([str_rg_id, str_members, str_exe_ctxs])

        writer.writerows(rows)


    @classmethod
    def from_file_csv(cls, f):
        """
        Import an organizational model from an external CSV file.

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
        * 3 columns: Group ID, Member resource IDs, Execution contexts
        * Values of member resource IDs and execution contexts are separated
          by semicolons ``;`` within the columns;
        * A vertical bar ``|`` is used as the separator for values of the
          three types within an execution context.
        
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
        om_obj = cls()
        for row in csv.reader(f):
            if not row:
                continue

            group = row[1].split(';')
            exe_ctxs = list()
            for str_ctxs in row[2].split(';'):
                exe_ctxs.append(tuple(str_ctxs.split('|')))
            om_obj.add_group(group, exe_ctxs)
        return om_obj
