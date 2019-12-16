# -*- coding: utf-8 -*-

"""This module contains methods for importing event log data from files 
in specific formats.

Event log formats currently supported:

    - Disco-exported CSV format (https://fluxicon.com/disco/)
    - (TODO) eXtensible Event Stream (XES) (http://xes-standard.org/)
    - (TODO) MXML (ProM 5)

All methods in this module should return the successfully imported event 
log as a pandas DataFrame.

Notes
-----
The original event log data should at least provide the information of 
the activity ids. It is expected that the resource ids and timestamps 
are presented as well (not mandatory though).
Other event data attributes can be appended according to different 
purposes of working projects and event data available.

See Also
--------
pandas.DataFrame : The primary pandas data structure.
"""
def _describe_event_log(el):
    """Output prompting information for a successfully imported event
    log.

    Parameters
    ----------
    el : DataFrame
        An event log imported.

    Returns
    -------
    """
    print('-' * 80)
    print('Number of events:\t\t{}'.format(len(el)))
    print('Number of cases:\t\t{}'.format(len(el.groupby('case_id'))))
    #print('Event log attributes:\n\t') # TODO
    print('-' * 80)


def read_disco_csv(f, mapping=None, header=True):
    """Import an event log from a file in CSV (Column-Separated Values)
    format, exported from Disco.

    There are four expected default event attributes, including:

        - case_id
        - activity
        - resource
        - timestamp

    Parameters
    ----------
    f : File object
        File object of the event log being imported.
    mapping : dict, optional
        A python dictionary denoting the mapping from CSV column index
        to event log attributes.
    header : bool, optional
        A boolean flag indicating whether the input event log file
        contains a header line.

    Returns
    -------
    el : DataFrame
        An event log.
    """
    from csv import reader

    ld = list()
    is_header_line = True
    line_count = 0

    for row in reader(f):
        line_count += 1
        if is_header_line:
            is_header_line = False
            pass
        else:
            # the default mapping consistent with Disco
            e = {
                'case_id': row[0],
                'activity': row[1],
                'resource': row[2],
                'timestamp': row[3]
            }
            # append additional attributes
            if mapping is not None:
                for attr, col_num in mapping.items():
                    e[attr] = row[col_num]
            ld.append(e)

    from pandas import DataFrame
    el = DataFrame(ld)

    print('Imported successfully. {} lines scanned.'.format(line_count))

    _describe_event_log(el)
    return el


def read_xes(f):
    """Import an event log from a file in IEEE XES (eXtensible Event 
    Stream) format.

    There are four expected default event attributes, including:

        - case_id
        - activity
        - resource
        - timestamp

    Parameters
    ----------
    f : File object
        File object of the event log being imported.

    Returns
    -------
    el : DataFrame
        An event log.
    """
    raise NotImplementedError

