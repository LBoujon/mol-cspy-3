"""
Module to support the creation and analysis of Disconnectivity Graphs from
`cspy-threshold` jobs.

Classes
-------

- `DisconnectivityGraph`: To create and analyse disconnectivity graphs.
- `DrawFilters`: To filter structures from the disconnectivity graph plots
created using `DisconnectivityGraph.draw()`.

Functions
---------

- `get_en_id`: Generate an energy ID from a minimized ID
- `get_min_id`: Generate an minimized ID from a energy ID and minimization step
- `get_info_from_id`: Get trial, minimization and MC step information from ID

Named Tuples
------------

- `Job`: To store the contents of a valid leaf structure

Usage
-----
First create an instance of the DisconnectivityGraph class with your database

```python
    disconn = DisconnectivityGraph(ds = ...)
```

Now you can query the graph to extract information using the class methods
available:

```python
    # Lid value connecting two structures
    lid = disconn.barrier_between_ids("ID1", "ID2")

    # All structures from a leaf to the root node
    lid_structs = disconn.structures_in_branch("ID")
```

Or you can save the disconnectivity graph to a file:

```python
    disconn.draw("plot_name")
```

It is also possible to create a DisconnectivityGraph instance from
a pickled graph instead of a database. Just beware that some class methods
will not be available as they need to read data from the database.
"""

import sys
import argparse
import json
import logging
import pickle
import time
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.figure import Figure as pltFigure

from cspy.clustering.trialstates import Min_Trial, Range, Trial
from cspy.db import CspDataStore
from cspy.util.logging_config import FORMATS, DATEFMT

from typing import Union, List, Dict, Tuple, Iterator, Callable, Set, Any
from typing_extensions import Literal
from pathlib import Path


LOG = logging.getLogger(__name__)
Job = namedtuple("Job", "uni_id trial_number density energy state min_id is_ini")


def get_en_id(min_id: str) -> str:
    """
    Generate an energy ID from a minimized ID.

    Arguments
    ---------
    min_id : str
        The id of the structure

    Returns
    -------
    new_id : str
        The new ID formatted as the input ID
    
    Raises
    ------
    NotImplementedError
        It is raised when the min_id is not formatted as expected
    """
    contents = min_id.split("-")
    if len(contents) == 8: # Handle the case where the crystal ID is from a 
                           # reoptimization job i.e. it has the -OPT-N suffix
        name, task, sg, tn, tmp, ms, opt, _ = min_id.split("-")
        return "-".join([name, task, sg, tn, tmp, ms, opt, "0"])
    elif len(contents) == 6: # Case for the usual crystal ID
        name, task, sg, tn, _, ms = min_id.split("-")
        return "-".join([name, task, sg, tn, "0", ms])
    else:
        raise NotImplementedError("Incompalible id for disconnectivity graph")


def get_min_id(en_id: str, max_min_step: int) -> str:
    """
    Generate a minimized ID from an ID.

    Arguments
    ---------
    en_id : str
        The id of the structure

    max_min_step : int
        The minimization step to write to the ID

    Returns
    -------
    new_id : str
        The new ID formatted as the input ID

    Raises
    ------
    NotImplementedError
        It is raised when the en_id is not formatted as expected
    """
    contents = en_id.split("-")
    if len(contents) == 8: # Handle the case where the crystal ID is from a 
                           # reoptimization job i.e. it has the -OPT-N suffix
        name, task, sg, tn, tmp, ms, opt, _ = en_id.split("-")
        return "-".join([name, task, sg, tn, tmp, ms, opt, str(max_min_step)])
    elif len(contents) == 6: # Case for the usual crystal ID
        name, task, sg, tn, _, ms = en_id.split("-")
        return "-".join([name, task, sg, tn, str(max_min_step), ms])
    else:
        raise NotImplementedError("Incompalible id for disconnectivity graph")


def get_info_from_id(id: str) -> Tuple[int, int, int]:
    """
    Get the trial number, minimization step and mc step from an ID.

    Arguments
    ---------
    id : str
        The id of the structure

    Returns
    -------
    (tn, mins, ms) : Tuple[int, int, int]
        The trial number, minimization step and mc step
    """
    contents = id.split("-")
    if len(contents) == 8: # Handle the case where the crystal ID is from a 
                           # reoptimization job i.e. it has the -OPT-N suffix
        _name, _task, _sg, tn, _tmp, ms, _opt, mins = id.split("-")
    elif len(contents) == 6: # Case for the usual crystal ID
        _name, _task, _sg, tn, mins, ms = id.split("-")
    else:
        raise NotImplementedError("Incompalible id for disconnectivity graph")

    return (int(tn), int(mins), int(ms))

class DrawFilters():
    """
    Filters to select what to draw from a DisconnectivityGraph
    object.

    - `initial_structures`: Draw only the starting structures of the trajectories.
    - `only_ids`: Show only the structure nodes that are in a list of IDs.
    - `lower_e`: Remove all leaf structures that are above an energy threshold.
    - `lowest_n`: Show the N structures with the lowest energy.

    This is just a collection of some usefult filters, but you can create
    your own (read below to see how).

    Usage
    -----
    To filter structures in the disconnectivity graph plot
    you must pass a function with the following inputs and output to
    the `draw()` method:

    ```python
        def f(graph: graph: nx.Graph, node_id: str) -> bool:
            ...
    ```

    When True is returned, the node with node_id will be erased from 
    the figure.

    So, in order to plot only the initial trajectory structures
    we can do the following:

    ```python
        disconn = DisconnectivityGraph(...)
        disconn.draw("save_file_name", DrawFilters.initial_structures())

        # Which is the same as
        disconn.draw_initial("save_file_name")
    ```

    **CAUTION**: Do not remove any nodes of kind 'thre' from the graph or the connections will
    be broken and no plot will be able to be created.
    """
    
    @staticmethod
    def initial_structures() -> Callable[[nx.Graph, str], bool]:
        """
        Filter all structures except the trajectory starts.

        Returns
        -------
        func : Callable[[nx.Graph, str], bool]
            The filter function
        """
        def func(graph: nx.Graph, node_id: str) -> bool:
            node_data = graph.nodes[node_id]
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                return False
            if node_data["kind"] == "min" and node_data["mc_step"] == 0:
                return False
            if node_data["kind"] == "ts":
                neighs = graph.neighbors(node_id)
                for neigh in neighs:
                    data = graph.nodes[neigh]
                    if data["kind"] == "min" and data["mc_step"] == 0:
                        return False
            return True
        return func
    
    @staticmethod
    def only_ids(ids: List[str]) -> Callable[[nx.Graph, str], bool]:
        """
        Filter all structures except the ones whose IDs match
        the argument.

        Arguments
        ---------
        ids : List[str]
            List of structure IDs to keep

        Returns
        -------
        func : Callable[[nx.Graph, str], bool]
            The filter function
        """
        def func(graph: nx.Graph, node_id: str) -> bool:
            node_data = graph.nodes[node_id]
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                return False
            if node_data["kind"] == "min" and node_id in ids:
                return False
            if node_data["kind"] == "ts":
                neighs = graph.neighbors(node_id)
                for neigh in neighs:
                    data = graph.nodes[neigh]
                    if data["kind"] == "min" and neigh in ids:
                        return False
            return True
        return func
    
    @staticmethod
    def lower_e(energy_lim: float) -> Callable[[nx.Graph, str], bool]:
        """
        Filter all structures except the ones whose energy is below
        `energy_lim`.

        Arguments
        ---------
        energy_lim : float
            The energy threshold

        Returns
        -------
        func : Callable[[nx.Graph, str], bool]
            The filter function
        """
        def func(graph: nx.Graph, node_id: str) -> bool:
            node_data = graph.nodes[node_id]
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                return False
            if node_data["kind"] == "min" and node_data["energy"] <= energy_lim:
                return False
            if node_data["kind"] == "ts":
                neighs = graph.neighbors(node_id)
                for neigh in neighs:
                    data = graph.nodes[neigh]
                    if data["kind"] == "min" and data["energy"] <= energy_lim:
                        return False
            return True
        return func
    
    @staticmethod
    def lowest_n(ds: CspDataStore, num: int) -> Callable[[nx.Graph, str], bool]:
        """
        Filter all structures except the `num` with lowest energy.

        Arguments
        ---------
        ds : CspDataStore
            The threshold database

        num : int
            The number of structures to keep

        Returns
        -------
        func : Callable[[nx.Graph, str], bool]
            The filter function
        """
        ids = ds.query("select distinct c.id from equivalent_to t join crystal c on t.unique_id = c.id order by c.energy").fetchall()
        if len(ids) == 0:
            LOG.warning(f"Database {ds.filename} contains no unique structures. Fetching data from all structures")
            ids = ds.query("select distinct(id) from crystal c order by c.energy").fetchall()
        ids = [i[0] for i in ids[:num]]
        def func(graph: nx.Graph, node_id: str) -> bool:
            node_data = graph.nodes[node_id]
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                return False
            if node_data["kind"] == "min" and node_data["uni_id"] in ids:
                return False
            if node_data["kind"] == "ts":
                neighs = graph.neighbors(node_id)
                for neigh in neighs:
                    data = graph.nodes[neigh]
                    if data["kind"] == "min" and data["uni_id"] in ids:
                        return False
            return True
        return func
    
    @staticmethod
    def limit_state(state: int) -> Callable[[nx.Graph, str], bool]:
        """
        Filter all structures except the ones connecting at lids
        below `state`.

        Arguments
        ---------
        state : int
            The state of the lid limit

        Returns
        -------
        func : Callable[[nx.Graph, str], bool]
            The filter function
        """
        def func(graph: nx.Graph, node_id: str) -> bool:
            node_data = graph.nodes[node_id]
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                return False
            if node_data["kind"] == "min":
                ts_id = next(graph.neighbors(node_id))
                node_id = ts_id
                node_data = graph.nodes[node_id]
            if node_data["kind"] == "ts":
                neighs = graph.neighbors(node_id)
                for neigh in neighs:
                    data = graph.nodes[neigh]
                    if data["kind"] == "thre" and data["state"] <= state:
                        return False
            return True
        return func


class DisconnectivityGraph:
    """
    Class to create and process disconnectivity graph from threshold algorithm data.

    Attributes
    ----------

    graph : networkx.Graph
        The disconnectivity graph object

    name : str
        The name of the object, used for naming some files

    min_trial : Min_Trial
        Used to calculate the lid states and energies of the disconnectivity graph 
    
    num_nodes : int
        The number of structure nodes in the graph (kind='min')

    ds : CspDataStore
        The datastore object of the traget threshold database

    yticks : List[float]
        The energies of the lids in increasing order

    basin_roots : List[str]
        A list of the IDs of the root nodes

    Notes
    -----
    The graph has nodes and edges of different kinds holding different data. To access the node data
    you can call `self.graph.nodes(data=True)` to iterate over a tuple of `(node_id, node_data)`, node_data 
    is a dictionary and node_id is a string.

    Node types (available at `node_data["kind"]`):
    1. `thre`. Has the data keys: kind, energy, state, path, covered, num_nodes, num_initials.  
        They are lid nodes that have one or more structures found.
    2. `fake_thre`. Has the same data keys as `thre`: kind, energy, state, path, covered, num_nodes, num_initials.
        These nodes are created to account for lids where no new structures were found.
    3. `ts`. Has the data keys: kind, energy.
        It is used to connect a `min` node with a `thre` node, used to draw the branches in
        the disconnectivity graph plot.
    1. `min`. Has the data keys: kind, energy, density, thre_energy, num, mc_step, uni_id.
        Represents a unique structure. The `node_id` of these kind of nodes is the crystal ID from 
        the database that appears in the lowest and earliest MC step from all trajectories.
    
    Edge types:
    1. `thre`. Has the data keys: kind.
        Describes the connection between two `thre` or `fake_thre` nodes.
    2. `quench`. Has the data keys: kind, density.
        Connects a `ts` node with a `min` node.
    3. `perturb`. Has the data keys: kind, density.
        Connection between a `ts` node and a `thre` node.

    Beware! When merging multiple trajectories that start from the same initial structure
    but do not have the same energy lid increases an extra fake_thre node will be created. This
    node will not have any structures connected to it, so `num_nodes` and `num_initials` will be 0.
    """

    def __init__(self, ds: Union[CspDataStore, Path], name: str = "disconnectivity_graph", save_graph=False, *args, **kwargs) -> None:
        """
        Create an instance of DisconnectivityGraph.

        The `construct_graph()` method is called if `ds` is a CspDataStore
        object. Otherwise, the program will assume that `ds` is a pickled
        object and will try to extract the data from it.
        
        Arguments
        ---------
        ds : Union[CspDataStore, Path]
            The threshold database or a pickled object path

        name : str, optional
            The name to use for plots and others. Default is "disconnectivity_graph"
        
        save_graph : bool, optional
            Save the networkx.Graph object to a file with pickle. Default is False 
        """
        self.name: str = name
        if isinstance(ds, CspDataStore):
            self.ds: Union[CspDataStore, None] = ds
            self.construct_graph(**kwargs)
        else:
            self.ds: Union[CspDataStore, None] = None
            self.update_data_from_pickle(ds)
        
        if save_graph:
            self.save_data_to_pickle(Path(f"{name}.pickle"))


    def update_data_from_pickle(self, pickle_path: Path) -> None:
        """
        Update the object data from a pickle file.

        The attributes read from file are: graph, min_trial, 
        num_nodes, yticks, basin_roots.

        Arguments
        ---------
        pickle_path : Path
            The path of the pickle file

        Raises
        ------
        FileNotFoundError
            It is raised when pickle_path is not a file

        Any
            Raises any exception that occurs while reading the data
            from a pickle file
        """
        if pickle_path.is_file():
            LOG.info(f"Trying to read data from pickle file: {pickle_path}")
            try:
                (
                    self.graph,
                    self.min_trial,
                    self.num_nodes,
                    self.yticks,
                    self.basin_roots,
                ) = pickle.load(open(pickle_path, "rb"))
            except Exception as e:
                LOG.info(f"Could not read disconnectivity graph data from pickled object due to: {e}")
                raise e
        else:
            LOG.error(f"Pickle file {pickle_path} does not exist")
            raise FileNotFoundError(f"File {pickle_path} not found")
        

    def save_data_to_pickle(self, pickle_path: Path) -> None:
        """
        Save the disconnectivity graph data to a pickle file.

        The attributes saved to file are: graph, min_trial, 
        num_nodes, yticks, basin_roots.

        Arguments
        ---------
        pickle_path : Path
            The path of the pickle file
        """
        LOG.info("Disconnectivity graph data (graph, min_trial, "
        f"num_nodes, yticks, basin_roots) will be saved to {pickle_path}")
        pickle.dump(
            (
                self.graph,
                self.min_trial,
                self.num_nodes,
                self.yticks,
                self.basin_roots,
            ),
            open(pickle_path, "wb"),
        )

    
    def export_graph(self, file_path: Path, format: Literal["gml", "graphml"] = "gml") -> None:
        """
        Export the disconnectivity graph to a file.

        Arguments
        ---------
        file_path : Path
            The path of the file to save the graph

        format : "gml" or "graphml", optional
            The format of the file. Default is "gml"

        Raises
        ------
        ValueError
            If format is not one of the accepted types
        """
        if format == "gml":
            nx.write_gml(self.graph, file_path)
        elif format == "graphml":
            nx.write_graphml(self.graph, file_path)
        else:
            raise ValueError("Format must be one of: ['gml', 'graphml']")


    def construct_graph(self, **kwargs) -> None:
        """
        Construct the disconnectivity graph into self.graph
        by reading the data from the database (self.ds). It overwrites
        any previous graphs stored in the object.

        Arguments
        ---------
        **kwargs : Dict[str, Any]
            These are passed to the _read_data() and _conn() methods

        Raises
        ------
        ValueError
            If the instance of DisconnectivityGraph was not created 
            with a linked CspDataStore database
        """
        if not isinstance(self.ds, CspDataStore):
            LOG.error("To construct graph you must provide a valid CspDataStore object")
            raise ValueError(f"Invalid CspDataStore object: {self.ds}")

        trial_data, trial_connections, trials, min_trial, max_state = DisconnectivityGraph._read_data(
            self.ds, **kwargs
        )

        self.trials = trials
        self.min_trial: Min_Trial = min_trial

        (
            graph,
            num_nodes,
            yticks,
            basin_roots,
        ) = DisconnectivityGraph._conn(
            trial_data, trial_connections, min_trial, max_state
        )

        self.graph: nx.Graph = graph
        self.num_nodes: int = num_nodes
        self.yticks: List[float] = yticks
        self.basin_roots: List[str] = basin_roots


    def cluster(self, update_equiv=False) -> Dict[str, List[str]]:
        """
        Cluster structures from the disconnectivity graph.
    
        It clusters all the structures in one root to the lowest energy
        structure of that root.

        > Warning! This method adds data to the CspDataStore object (self.ds) if 
        > update_equiv is set to `True`.
        
        Arguments
        ---------    
        update_equiv : bool, optional 
            Update the equivalent_to table in database to basin minima.
            Default is `False`.

        Returns
        -------
        equivalent_table : Dict[str, List[str]]
            The keys are the IDs of the lowest energy structures of the root, 
            the values are the list of IDs of the rest of structures in the 
            root.

        Raises
        ------
        ValueError
            If the instance of DisconnectivityGraph was not created 
            with a linked CspDataStore database
        """
        if not isinstance(self.ds, CspDataStore):
            LOG.error("You must provide a valid CspDataStore object")
            raise ValueError(f"Invalid CspDataStore object: {self.ds}")
        
        equivalent_table: Dict[str, List[str]] = {}
        LOG.info("clustering from disconnectivity graph")
        LOG.info("basin root nodes: {}".format(self.basin_roots))

        for root in self.basin_roots:
            min_ids = self.ids_in_branch(root)
            equivalent_table[min_ids[0][1]] = [x[1] for x in min_ids[1:]]

        if update_equiv:
            self.ds.add_equivalent_structures(equivalent_table)
            self.ds.commit()

        return equivalent_table


    def ids_in_branch(self, root: str) -> List[Tuple[str, str, float]]:
        """
        Get the IDs and energies of the structures under `root`.

        The resulting list is sorted in order of increasing energy.

        Arguments
        ---------
        root : str
            The node ID of the root

        Returns
        -------
        min_ids : List[Tuple[str, str, float]]
            First position in the tuple is the node ID of the structure, second
            is the unique ID (the one in equivalent_to table in the database), and third
            is the energy
        """
        neighbors: Iterator[str] = self.graph.neighbors(root)
        root_data: Dict[str, Union[str, float]] = self.graph.nodes[root]

        min_ids: List[Tuple[str, float]] = []
        for neighbor in neighbors:
            neighbor_data: Dict[str, Union[str, float]] = self.graph.nodes[neighbor]
            if (
                neighbor_data["kind"] == "thre" or neighbor_data["kind"] == "fake_thre"
            ) and neighbor_data["state"] < root_data["state"]:
                min_ids += self.ids_in_branch(neighbor)
            elif neighbor_data["kind"] == "ts":
                ts_neighbors = self.graph.neighbors(neighbor)
                for ts_neighbor in ts_neighbors:
                    ts_neighbor_data = self.graph.nodes[ts_neighbor]
                    if ts_neighbor_data["kind"] == "min":
                        min_ids.append(
                            (
                                ts_neighbor,
                                ts_neighbor_data["uni_id"],
                                ts_neighbor_data["energy"],
                            )
                        )

        return sorted(min_ids, key=lambda x: x[2])


    def save_basin_minima(self) -> None:
        """
        Dump the leaf structures of all basins in the disconnectivity
        graph into files. Each file is formatted as basin_{basin_number}_minima.txt.

        The file contents are as follows:
        ```
            unique_id,structure_energy
            ...
        ```
        """
        for i, root in enumerate(self.basin_roots):
            min_ids = self.ids_in_branch(root)
            with open("basin_{}_minima.txt".format(i), "w") as f:
                f.write("unique_id,structure_energy\n")
                for min_id in min_ids:
                    f.write("{},{}\n".format(min_id[1], min_id[2]))


    def initial_minima_ids(self, return_ids: Literal["unique", "minima"] = "minima") -> Tuple[List[str], List[str]]:
        """
        Get the unique or minima IDs of the trajectory starts and all other
        minima.

        - unique IDs: are the ones that can be found in the equivalent_to
        table of the database.
        - minima IDs: are the ones shown in the disconnectivity graph labels, and
        are the node IDs.

        Arguments
        ---------
        return_ids : "unqiue" or "minima", optional
            Which type of ID to return. Default is "minima"

        Returns
        -------
        (ini_list, min_list) : (List[str], List[str])
            The ids of the trajectory starts and the ids of the other
            minima structures

        Raises
        ------
        ValueError
            If return_ids is not of the accepted types
        """
        if return_ids not in ["unique", "minima"]:
            raise ValueError("return_ids must be one of: {}".format(["unique", "minima"]))

        min_list: List[str] = []
        ini_list: List[str] = []

        for node, node_data in self.graph.nodes.data():
            if node_data["kind"] == "min":
                if node_data["mc_step"] == 0:
                    if return_ids == "minima":
                        ini_list.append(node)
                    else:
                        ini_list.append(node_data["uni_id"])
                    continue
                
                if return_ids == "minima":
                    min_list.append(node)
                else:
                    min_list.append(node_data["uni_id"])

        return ini_list, min_list
    

    def _min_id_from_ts(self, ts_id: str) -> str:
        """
        Obtain the ID of the leaf structure that is connected to 
        the ts node.

        Arguments
        ---------
        ts_id : str
            ID of the ts node

        Returns
        -------
        neighbour : str
            ID of the leaf structure

        Raises
        ------
        ValueError
            If the ID is not in the graph or it is not a ts node 

        RuntimeError
            If the ts node does not have a min node connected, which
            should never happen (if it does, graph has not been created
            as expected)
        """
        if ts_id not in self.graph:
            raise ValueError("ID not in graph")

        ts: Dict[str, Union[str, float, int]] = self.graph.nodes[ts_id]
        if ts["kind"] != "ts":
            raise ValueError("Node is not ts kind")
        
        neighbours: Iterator[str] = self.graph.neighbors(ts_id)
        for neighbour in neighbours:
            neigh_data: Dict[str, Union[str, float, int]] = self.graph.nodes[neighbour]
            if neigh_data["kind"] == "min":
                return neighbour
        
        raise RuntimeError("ts nodes are always expected to have a min node connected") 
    

    def structures_in_lid(self, lid_id: str) -> List[str]:
        """
        Obtain the IDs of the leaf structures that are connected
        to a lid.
        
        Arguments
        ---------
        lid_id : str
            ID of the lid node to count structures from

        Returns
        -------
        structures : List[str]
            A list of the structure IDs

        Raises
        ------
        ValueError
            If the ID is not in the graph or it is not a lid node 

        Examples
        --------

        ```
        disconn.structures_in_lid("thre-0-9")
            ['perylene-SG-1-0-3-7772',
            'perylene-SG-1-1-3-8210',
            'perylene-SG-1-1-3-8258',
            'perylene-SG-1-1-3-8635']
        ```
        """
        if lid_id not in self.graph:
            raise ValueError("ID not in graph")
        
        start_struct: Dict[str, Union[str, float, int]] = self.graph.nodes[lid_id]
        if start_struct["kind"] == "fake_thre":
            return []
        elif start_struct["kind"] != "thre":
            raise ValueError("Node is not a lid, it has kind: {}".format(start_struct["kind"]))
        
        neighbours: Iterator[str] = self.graph.neighbors(lid_id)
        structures: List[str] = []
        for neighbour in neighbours:
            neigh_data: Dict[str, Union[str, float, int]] = self.graph.nodes[neighbour] 
            if neigh_data["kind"] == "ts":
                structures.append(self._min_id_from_ts(neighbour))
        
        return structures
    

    def structures_in_branch(self, structure_id: str) -> Dict[int, Tuple[str, List[str]]]:
        """
        Obtain the structures per lid in a branch from a leaf structure to the root.

        Arguments
        ---------
        structure_id : str
            The ID of the structure node from which to start

        Returns
        -------
        lid_structures : Dict[int, Tuple[str, List[str]]]
            The keys of the dictionary are the lids and the values are a tuple
            with the node ID of the lid and a list of structure IDs

        Raises
        ------
        ValueError
            If the ID is not in the graph or it is not a leaf node

        RuntimeError
            If connectivity of graph is not as expected
        """
        if structure_id not in self.graph:
            raise ValueError("ID not in graph")
        
        start_struct: Dict[str, Union[str, float, int]] = self.graph.nodes[structure_id]
        if start_struct["kind"] != "min":
            raise ValueError("Node is not a leaf structure, it has kind: {}".format(start_struct["kind"]))
        
        neighbours: Iterator[str] = list(self.graph.neighbors(structure_id))
        if len(neighbours) != 1:
            raise RuntimeError("Leaf node must only have one connected node of kind ts")
        _ts_node = self.graph.nodes[neighbours[0]]
        neighbours: Iterator[str] = list(self.graph.neighbors(neighbours[0]))
        if len(neighbours) != 2:
            raise RuntimeError("ts node must only have two connected nodes of kind thre and min")
        lid_node: Union[None, str] = None
        for neighbour in neighbours:
            neigh_data: Dict[str, Union[str, float, int]] = self.graph.nodes[neighbour] 
            if neigh_data["kind"] == "thre":
                lid_node = neighbour
                break
        
        lid_structures: Dict[int, Tuple[str, List[str]]] = {}
        while lid_node is not None:
            lid_data: Dict[str, Union[str, float, int]] = self.graph.nodes[lid_node]
            state = lid_data["state"]
            lid_structures[state] = (lid_node, self.structures_in_lid(lid_node))
            neighbours: Iterator[str] = self.graph.neighbors(lid_node)
            lid_node = None
            for neighbour in neighbours:
                neigh_data: Dict[str, Union[str, float, int]] = self.graph.nodes[neighbour]
                if neigh_data["kind"] in ["thre", "fake_thre"] and neigh_data["state"] > state:
                    lid_node = neighbour
                    break

        return lid_structures


    def barrier_between_ids(self, id1: str, id2: str) -> Union[int, None]:
        """
        Get the lid value connecting two leaf ids. 
        
        If no connection exists, a value of None is returned.

        Arguments
        ---------
        id1 : str
            Id of the first structure
        
        id2 : str
            Id of the second structure

        Returns
        -------
        connecting_lid : union[int, None]
            The lid value where the two structure are connected

        Raises
        ------
        ValueError
            If the ID is not in the graph or it is not a leaf node

        RuntimeError
            If connectivity of graph is not as expected

        Notes
        -----
        Once you get the lid value, you can get the energy by using the 
        get_energy_from_trial() method of self.min_trial:
        
        ```python
            disconn = DisconnectivityGraph(...)
            lid = disconn.barrier_between_ids("ID1", "ID2")
            lid_energy = disconn.min_trial.get_energy_from_trial(lid)
        ```
        """
        lids_id1 = self.structures_in_branch(id1)
        lids_id2 = self.structures_in_branch(id2)

        for lid, val in sorted(lids_id1.items()):
            if lid in lids_id2.keys() and lids_id2[lid][0] == val[0]:
                return lid
            
        return None
    

    def lid_ids_from_level(self, level: int) -> List[str]:
        """
        Get the IDs of the lid nodes at a set threshold level.

        If there is more than one lid node at a certain level it means
        that all trajectories have not joined up to that level.

        Arguments
        ---------
        level : int
            The threshold level

        Returns
        -------
        lid_ids : List[str]
            IDs of the lids at level

        Raises
        ------
        ValueError
            If level is smaller or equal than 0 or it is not integer
        """
        if level <= 0 or type(level) is float:
            raise ValueError("Level must be positive integer")

        lid_ids = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data["kind"] != "thre" and node_data["kind"] != "fake_thre":
                continue
            
            if node_data["state"] == level:
                lid_ids.append(node_id)
        
        return lid_ids
    

    def equivalent_structs(self, structure_id: str) -> List[str]:
        """
        Get a list of structures that are equivalent to the input structure.

        This returns the contents of the `equivalent_to` table in the database. 

        Arguments
        ---------
        structure_id : str
            The ID of the structure node

        Returns
        -------
        matches : List[str]
            A list of IDs of structures that are equivalent to structure_id

        Raises
        ------
        ValueError
            If the instance of DisconnectivityGraph was not created 
            with a linked CspDataStore database
        """
        if not isinstance(self.ds, CspDataStore):
            LOG.error("You must provide a valid CspDataStore object")
            raise ValueError(f"Invalid CspDataStore object: {self.ds}")
        
        if structure_id not in self.graph:
            raise ValueError("ID not in graph")
        
        search_struct: Dict[str, Union[str, float, int]] = self.graph.nodes[structure_id]
        if search_struct["kind"] != "min":
            raise ValueError("Node is not a leaf structure, it has kind: {}".format(search_struct["kind"]))
        
        matches = self.ds.equivalent_structures(search_struct["uni_id"])

        return matches
    

    def equivalent_structs_in_lid(self, structure_id: str, lid_val: int) -> List[str]:
        """
        Get a list of structures that are equivalent to the input structure and
        are found at a certain lid height.

        Arguments
        ---------
        structure_id : str
            The ID of the structure node

        lid_val : int
            The lid value

        Returns
        -------
        lid_matches : List[str]
            A list of the IDs of structures that match
        """
        all_matches = self.equivalent_structs(structure_id)
        lid_matches = []
        for match_id in all_matches:
            tn, _, ms = get_info_from_id(match_id)
            energy = self.trials[tn].get_current_energy(ms)
            lid = self.min_trial.get_state_from_energy(energy)
            if lid == lid_val:
                lid_matches.append(match_id)

        return lid_matches
    

    def disconnectivity_matrix(self) -> np.ndarray:
        """
        Create a matrix of the energy barriers between the lids.

        Returns
        -------
        matrix : np.ndarray
            The matrix of the energy barriers. It is a square matrix
            of length equal to the number of initial structures in the
            disconnectivity graph
        """
        ini_ids, _ = self.initial_minima_ids()
        sorted_ini_list = sorted(ini_ids, key=lambda x: int(x.split("-")[3]))
        matrix = np.zeros((len(sorted_ini_list), len(sorted_ini_list)))
        for i, ini1 in enumerate(sorted_ini_list):
            for j, ini2 in enumerate(sorted_ini_list[i+1:], start=i+1):
                level = self.barrier_between_ids(ini1, ini2)
                if level is None:
                    barrier_1_2 = None
                    barrier_2_1 = None
                else:
                    lid_energy = self.min_trial.get_energy_from_state(level)
                    barrier_1_2 = lid_energy - self.graph.nodes[ini1]["energy"]
                    barrier_2_1 = lid_energy - self.graph.nodes[ini2]["energy"]
                matrix[i][j] = barrier_1_2
                matrix[j][i] = barrier_2_1

        return matrix


    def draw_disconn_matrix(self, out_name: str) -> None:
        """
        Draw the disconnectivity matrix and save it to a file.

        Arguments
        ---------
        out_name : str
            The name of the file to save the plot
        """
        save_file = Path(f"{out_name}.png")
        self.plot_disconn_matrix()
        plt.savefig(fname=save_file, dpi=300)

    
    def plot_disconn_matrix(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the disconnectivity matrix. Returns the figure and axes
        objects so further modifications to the plot can be carried
        out.

        Returns
        -------
        fig, ax : Tuple[plt.Figure, plt.Axes]
            The figure and axes objects

        Notes
        -----
        The matrix is created with the `disconnectivity_matrix()` method.
        """
        matrix = self.disconnectivity_matrix()
        ini_ids, _ = self.initial_minima_ids()
        sorted_ini_list = sorted(ini_ids, key=lambda x: int(x.split("-")[3]))

        fig, ax = plt.subplots()
        im = ax.imshow(matrix)
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Energy", rotation=-90, va="bottom")
        ax.spines[:].set_visible(False)
        ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=4)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_xticks(range(len(sorted_ini_list)), labels=sorted_ini_list,
              rotation=45, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(sorted_ini_list)), labels=sorted_ini_list)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i == j:
                    continue
                text = ax.text(j, i, round(matrix[i, j], 2),
                            ha="center", va="center", color="w")

        ax.set_title("Energy barriers")
        fig.tight_layout()

        return fig, ax
    

    def _draw_core_task1(self, **kwargs) -> nx.Graph:
        """
        Task number 1 in plotting disconnectivity graphs: remove
        any filtered nodes that should not be plotted.

        Requires
        --------

        filter_func: Filter the nodes in the graph to remove

        drawing_graph: The Networkx graph object
        """
        LOG.debug("Starting drawing task 1: Filter nodes")
        if kwargs["filter_func"] is not None:
            to_remove = []
            for node in kwargs["drawing_graph"].nodes():
                if kwargs["filter_func"](kwargs["drawing_graph"], node):
                    to_remove.append(node)
            
            kwargs["drawing_graph"].remove_nodes_from(to_remove)
            LOG.info(f"Removed {len(to_remove)} nodes from graph to be drawn")
            # update num_nodes and num_initials after removing 
            # filtered nodes
            for (n, d) in kwargs["drawing_graph"].nodes(data=True):
                if d["kind"] != "thre" and d["kind"] != "fake_thre":
                    continue

                d["num_nodes"] = 0
                d["num_initials"] = 0

            LOG.info("Recalculating the number of min and ini structures in each lid node")
            for (n, d) in kwargs["drawing_graph"].nodes(data=True):
                if d["kind"] != "ts":
                    continue

                min_data = None
                lid_data = None
                lid_id = ""
                for neigh in kwargs["drawing_graph"].neighbors(n):
                    data = kwargs["drawing_graph"].nodes[neigh]
                    if data["kind"] == "min":
                        min_data = data
                    else:
                        lid_data = data
                        lid_id = neigh
                
                is_ini = min_data["mc_step"] == 0
                lower_count = 1
                lower_count_ini = 0
                if is_ini:
                    lower_count_ini = 1
                while lid_data is not None:
                    lid_next = None
                    lid_id_next = None
                    for n2 in kwargs["drawing_graph"].neighbors(lid_id):
                        d2 = kwargs["drawing_graph"].nodes[n2]
                        if d2["kind"] != "thre" and d2["kind"] != "fake_thre":
                            continue
                        if d2["state"] > lid_data["state"]:
                            lid_id_next = n2
                            lid_next = d2
                            break
                    
                    LOG.debug(f"Adding to lid id {lid_id} {lower_count} to num_nodes and {lower_count_ini} to num_initials")
                    lid_data["num_nodes"] += lower_count
                    lid_data["num_initials"] += lower_count_ini
                    lid_data = lid_next
                    lid_id = lid_id_next

            # add only previous lid counts to highest lid in graph
            highest_id = None
            highest_lid = -1
            for (n, d) in kwargs["drawing_graph"].nodes(data=True):
                if d["kind"] != "thre" and d["kind"] != "fake_thre":
                    continue
                if d["state"] > highest_lid:
                    highest_lid = d["state"]
                    highest_id = n
            count_num_nodes = 0
            count_num_initials = 0
            for (n, d) in kwargs["drawing_graph"].nodes(data=True):
                if d["kind"] != "thre" and d["kind"] != "fake_thre":
                    continue
                if d["state"] == highest_lid - 1:
                    count_num_nodes += d["num_nodes"]
                    count_num_initials += d["num_initials"]
            data = kwargs["drawing_graph"].nodes[highest_id]
            data["num_nodes"] = count_num_nodes
            data["num_initials"] = count_num_initials
            LOG.debug(f"Recalculated number of nodes in highest lid node is {count_num_nodes} total nodes and {count_num_initials} trajectory starts")

            to_remove = []
            for (n, d) in kwargs["drawing_graph"].nodes(data=True):
                if d["kind"] != "thre" and d["kind"] != "fake_thre":
                    continue

                if d["num_nodes"] == 0:
                    to_remove.append(n)

            LOG.debug(f"Removing lid nodes that have no connections below them: {to_remove}")
            kwargs["drawing_graph"].remove_nodes_from(to_remove)
            LOG.info(f"Removed {len(to_remove)} lid nodes that have 0 min nodes connected")
        else:
            LOG.debug("No nodes to filter")

        return kwargs["drawing_graph"]


    def _draw_core_task2(self, **kwargs) -> Tuple[Dict[str, List], List, List, List, List, int, List, Union[float, None]]:
        """
        Task number 2 in plotting disconnectivity graphs: calculate
        some values from the nx.Graph object.

        Requires
        --------

        drawing_graph: Networkx graph

        relative_energy: Bool
        """
        LOG.debug("Starting drawing task 2: Calculate values from graph object")
       
        min_energy_node = min(kwargs["drawing_graph"].nodes(data=True), key=lambda x: x[1]["energy"])[1]
        min_energy = min_energy_node["energy"]
        if kwargs["relative_energy"]:
            for _, node_data in kwargs["drawing_graph"].nodes(data=True):
                node_data['energy'] = node_data['energy'] - min_energy
                
        thre = defaultdict(list)
        thre_list = []
        ts_list = []
        min_list = []
        ini_list = []
        num_nodes = 0
        energies = []
        for node, node_data in kwargs["drawing_graph"].nodes(data=True):
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                thre_list.append(node)
                state = node_data["state"]
                thre[state].append((node, node_data))
            elif node_data["kind"] == "ts":
                ts_list.append(node)
            elif node_data["mc_step"] == 0:
                ini_list.append(node)
                num_nodes += 1
                energies.append(node_data["energy"])
            else:
                min_list.append(node)
                num_nodes += 1
                energies.append(node_data["energy"])

        LOG.debug(f"Read data from {num_nodes} nodes")
        LOG.debug(f"There are {len(thre_list)} lid (thre and fake_thre) nodes")
        LOG.debug(f"There are {len(ts_list)} ts nodes")
        LOG.debug(f"There are {len(ini_list)} trajectory start nodes")
        LOG.debug(f"There are {len(min_list)} other minima nodes")

        return thre, thre_list, ts_list, min_list, ini_list, num_nodes, energies, min_energy        


    def _draw_core_task3(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Normalize, List[float]]:
        """
        Task number 3 in plotting disconnectivity graphs: assign
        the colours of the nodes and edges in the plot.

        Requires
        --------

        cluster_file: The cluster file to colour nodes

        drawing_graph: The Networkx graph

        cluster_by_unique_id: Bool

        plot_density: Bool

        colour_ini_basins: Bool

        ini_list: List of ini node IDs
        """
        LOG.debug("Starting drawing task 3: Assign colours to nodes and edges")
        min_color_lists = {}
        color_normalize = None
        ini_color_lists = None
        edge_lists = None
        densities = []
        if kwargs["cluster_file"] is not None:
            LOG.debug(f"Assigning colours to edges from cluster file: {kwargs['cluster_file']}")
            basins = {}
            ids = defaultdict(int)
            with open(kwargs["cluster_file"], "r") as f:
                for line in f:
                    content = line.split()
                    id = content[0]
                    ids[id] += 1
                    cluster = int(
                        content[3]
                    )  ### hard coded color category (as int from 0)
                    basins[id] = cluster

            min_color_lists = defaultdict(list)
            ini_color_lists = defaultdict(list)
            edge_lists = defaultdict(list)
            for node, node_data in kwargs["drawing_graph"].nodes.data():
                if node_data["kind"] != "min":
                    continue
                if kwargs["cluster_by_unique_id"]:
                    check_node = node_data["uni_id"]
                else:
                    check_node = node
                if check_node not in basins:
                    LOG.warning("{} not in cluster file".format(check_node))
                    basin = -1
                else:
                    basin = basins[check_node]
                ts_id = kwargs["drawing_graph"].neighbors(node)
                for edge in kwargs["drawing_graph"].edges(ts_id):
                    edge_lists[basin].append(edge)
                    if node_data["mc_step"] == 0:
                        ini_color_lists[basin].append(check_node)
                    else:
                        min_color_lists[basin].append(check_node)
                # edge_lists[basin].append(G.edges(node))
        elif kwargs["plot_density"]:
            LOG.debug("Assigning colours to edges from the density of the crystal structures")
            edge_lists = defaultdict(list)
            for node_1, node_2, edge_data in kwargs["drawing_graph"].edges.data():
                if edge_data["kind"] == "thre":
                    edge_lists[-1].append((node_1, node_2))
                else:
                    densities.append(edge_data["density"])
                    edge_lists["d"].append(((node_1, node_2), edge_data["density"]))
            color_normalize = Normalize(vmin=min(densities), vmax=max(densities))
        elif kwargs["colour_ini_basins"]:
            LOG.debug("Assigning colours to edges from the order of the unique basin they correspond to")
            edge_lists = defaultdict(list)
            sorted_ini_list = sorted(kwargs["ini_list"], key=lambda x: int(x.split("-")[3]))
            for i, ini_id in enumerate(sorted_ini_list):
                ts_id = list(kwargs["drawing_graph"].neighbors(ini_id))[0]
                edge_lists[i].append((ini_id, ts_id, {}))
                for neigh in kwargs["drawing_graph"].neighbors(ts_id):
                    if neigh != ini_id:
                        thre_id = neigh
                        break
                edge_lists[i].append((ts_id, thre_id, {}))
                num_initials = 1
                while num_initials == 1 and isinstance(thre_id, str):
                    thre_data = kwargs["drawing_graph"].nodes[thre_id]
                    # get num_initials of thre node
                    num_initials = thre_data["num_initials"]
                    if num_initials == 1:
                        # iterate over neighs and colour them
                        # set thre_id to next lid id
                        curr_id = thre_id
                        thre_id = None
                        for neigh in kwargs["drawing_graph"].neighbors(curr_id):
                            neigh_data = kwargs["drawing_graph"].nodes[neigh]
                            if neigh_data["kind"] == "thre" or neigh_data["kind"] == "fake_thre":
                                if neigh_data["state"] > thre_data["state"]:
                                    # LOG.info(f"{curr_id} with {thre_data["state"]} and {neigh} with ")
                                    thre_id = neigh 
                                    edge_lists[i].append((curr_id, thre_id, {}))
                            if neigh_data["kind"] == "ts":
                                edge_lists[i].append((curr_id, neigh, {}))
                                for neigh2 in kwargs["drawing_graph"].neighbors(neigh):
                                    if neigh2 != curr_id:
                                        edge_lists[i].append((neigh, neigh2, {}))
                                        break
        else:
            LOG.debug("Assigning the default colours to the edges")
            edge_lists = {-1: kwargs["drawing_graph"].edges()}
            min_color_lists = {
                -1: [
                    x
                    for x, y in kwargs["drawing_graph"].nodes.data()
                    if y["kind"] == "min" and y["mc_step"] > 0
                ]
            }
            ini_color_lists = {
                -1: [
                    x
                    for x, y in kwargs["drawing_graph"].nodes.data()
                    if y["kind"] == "min" and y["mc_step"] == 0
                ]
            }

        return min_color_lists, ini_color_lists, edge_lists, color_normalize, densities


    def _draw_core_task4(self, **kwargs) -> Dict[str, Tuple[float, float]]:
        """
        Task number 4 in plotting disconnectivity graphs: assign 
        an x coordinate to each leaf node so that there are no
        overlapping edges on the plot.

        Requires
        --------

        num_nodes: Number of structures in the graph (ini+min)

        thre: Lid nodes dictionary

        drawing_graph: Networkx graph object

        ini_list: List of ini node IDs
        """
        LOG.debug("Starting drawing task 4: Assign x coordinates to structure nodes")
        pos = {}
        r = Range(kwargs["num_nodes"])
        for state in sorted(kwargs["thre"], reverse=True):
            LOG.debug(f"Checking lid at height {state}")
            for thre_id, thre_data in sorted(kwargs["thre"][state], key=lambda x: x[1]["path"]):
                path = thre_data["path"]
                start, end = r.get_range(path, thre_data["num_nodes"])
                LOG.debug(f"Lid node {thre_id} has range: [{start}, {end}]")
                pos_thre = (start + end) / 2
                pos[thre_id] = [pos_thre, thre_data["energy"]]
                if kwargs["drawing_graph"].nodes[thre_id]["kind"] == "fake_thre":
                    continue
                num_tmp = 0
                for i, ts_id in enumerate(kwargs["drawing_graph"].neighbors(thre_id)):
                    if (
                        kwargs["drawing_graph"].nodes[ts_id]["kind"] == "thre"
                        or kwargs["drawing_graph"].nodes[ts_id]["kind"] == "fake_thre"
                    ):
                        continue
                    num_tmp += 1

                    if thre_data["num_nodes"] == 1:
                        pos_ts_min = pos_thre
                    elif (i % 2 == 0) or (end <= (pos_thre + 1)):
                        pos_ts_min = start
                        start += 1
                    elif (i % 2 == 1) or (start >= pos_thre):
                        end -= 1
                        pos_ts_min = end
                    else:
                        LOG.error(
                            "Error: wrong position around thre, "
                            "thre %s, start %s, end %s",
                            pos_thre,
                            start,
                            end,
                        )

                    for min_id in kwargs["drawing_graph"].neighbors(ts_id):
                        if kwargs["drawing_graph"].nodes[min_id]["kind"] == "min":
                            break
                    
                    LOG.debug(f"Position of ts ({ts_id}) and min ({min_id}) nodes is {pos_ts_min}")
                    pos[ts_id] = [pos_ts_min, kwargs["drawing_graph"].nodes[ts_id]["energy"]]
                    pos[min_id] = [pos_ts_min, kwargs["drawing_graph"].nodes[min_id]["energy"]]
                r.update_range(path, start, end)
                r.get_overlap(path, thre_data["covered"])
            r.update_ranges()
            r.update_overlap()

        for i in sorted(kwargs["ini_list"], key=lambda x:pos[x][0]):
            LOG.info(f"{i} {pos[i]}")

        return pos


    def _draw_core_task5(self, **kwargs) -> Tuple[pltFigure, plt.Axes]:
        """
        Task number 5 in plotting disconnectivity graphs: create
        the actual plot.

        Requires
        --------

        plot_ini_labels: Bool

        plot_lid_heights: Bool

        ini_list: List of ini node IDs

        thre_list: List of lid node IDs

        drawing_graph: Networkx graph

        relative_energy: Bool

        min_energy: Energy of the global minimum

        energies: list of energies of the nodes

        show_nodes: Bool

        pos: (x,y) positions dictionary by node IDs

        ts_list: lis of transition state node IDs

        min_list: lis of min node IDs

        edge_lists: list of edge colours

        color_normalize: Normalize for colour in plot

        densities: list of densities

        plot_all_labels: Bool

        auto_ylim: Bool

        y_bottom: float

        y_top: float
        """
        LOG.debug("Starting drawing task 5: Plotting the data into a matplotlib figure")
        labels = {}

        if kwargs["plot_ini_labels"]:
            for min_id in kwargs["ini_list"]:
                trial_number = min_id.split('-')[3]
                labels[min_id] = trial_number 

        if kwargs["plot_lid_heights"]:
            for thre_id in kwargs["thre_list"]:
                energy = kwargs["drawing_graph"].nodes[thre_id]["energy"]
                labels[thre_id] = round(energy, 3)

        # calculate top ylim to include from the top lid
        # that has any structures
        for s in range(len(self.yticks), 0, -1):
            lid_ids = self.lid_ids_from_level(s)
            if len(lid_ids) > 1:
                break
            try:
                n_neighs = len(list(kwargs["drawing_graph"].neighbors(lid_ids[0])))
            except nx.exception.NetworkXError:
                continue
            if n_neighs > 2:
                break
        max_y = self.min_trial.get_energy_from_state(s+1)
        if kwargs["relative_energy"]:
            max_y = max_y - kwargs["min_energy"]

        # calculate bottom ylim
        min_y = min(kwargs["energies"]) - self.min_trial.increase

        plt.cla()
        plt.clf()
        fig, ax = plt.subplots()
        if kwargs["relative_energy"]:
            ax.set_ylabel("Relative energy (kJ/mol)", fontsize=30)
        else:
            ax.set_ylabel("Energy (kJ/mol)", fontsize=30)
        fig.set_size_inches([24, 18])
        if isinstance(self.ds, CspDataStore):
            title_text = f"Disconnectivity graph of {self.ds.filename}"
        else:
            title_text = "Disconnectivity graph"
        ax.set_title(title_text, fontsize=30, pad=15)

        if kwargs["show_nodes"]:
            nx.draw_networkx_nodes(kwargs["drawing_graph"], pos=kwargs["pos"], nodelist=kwargs["thre_list"], node_size=0)
            nx.draw_networkx_nodes(kwargs["drawing_graph"], pos=kwargs["pos"], nodelist=kwargs["ts_list"], node_size=0)
            nx.draw_networkx_nodes(kwargs["drawing_graph"], pos=kwargs["pos"], nodelist=kwargs["min_list"], node_size=30)
            nx.draw_networkx_nodes(kwargs["drawing_graph"], pos=kwargs["pos"], nodelist=kwargs["ini_list"], node_color='orange', node_size=60, node_shape="d")
        nx.draw_networkx_edges(kwargs["drawing_graph"], pos=kwargs["pos"], edge_color='k', width=kwargs["edge_width"])

        color_list = cm.Dark2.colors[:-1]
        for edge_i, edge_list in kwargs["edge_lists"].items():
            if edge_i == -1:
                LOG.debug(f"{len(edge_list)} edges being drawn have no user assigned colour")
                nx.draw_networkx_edges(kwargs["drawing_graph"], pos=kwargs["pos"], edgelist=edge_list, edge_color='k', width=kwargs["edge_width"]+0.1)
            elif edge_i == 'd':
                LOG.debug(f"{len(edge_list)} edges are being drawn using the 'd' colour scheme")
                for edge, density in edge_list:
                    norm_density = kwargs["color_normalize"](density)
                    nx.draw_networkx_edges(
                        kwargs["drawing_graph"], pos=kwargs["pos"], edgelist=[edge], 
                        edge_color=cm.coolwarm(norm_density), width=kwargs["edge_width"]+0.1
                    )

                scalarmappaple = cm.ScalarMappable(norm=kwargs["color_normalize"], cmap=cm.coolwarm)
                scalarmappaple.set_array(kwargs["densities"])
                cb = plt.colorbar(scalarmappaple, shrink=0.75, pad=0.02, ax=plt.gca())
                cb.ax.set_ylabel("Density (g c$m^{-3}$)", fontsize=30, rotation=270)
                cb.ax.yaxis.labelpad = 35
                cb.ax.tick_params(labelsize=22)
            else:
                LOG.debug(f"{len(edge_list)} edges are being drawn using the {edge_i % len(color_list)} position of the color_list object")
                nx.draw_networkx_edges(
                    kwargs["drawing_graph"], pos=kwargs["pos"], edgelist=edge_list, edge_color=color_list[edge_i % len(color_list)],
                    width=kwargs["edge_width"]+0.1, # edge_cmap=cm.gnuplot
                )
        ax.tick_params(left=True, labelleft=True, labelsize=30)
        if kwargs["plot_lid_heights"]:
            LOG.debug("Adding horizontal lines at lids")
            for i, ytick in enumerate(self.yticks):
                plt.axhline(ytick, linestyle="--", color="black", alpha=0.3)
        ax.set_xmargin(-0.02)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if labels or kwargs["plot_all_labels"]:
            if labels:
                if kwargs["plot_ini_labels"] and kwargs["plot_lid_heights"]:
                    lbls_txt = "ini_labels and lid_labels"
                elif kwargs["plot_ini_labels"]:
                    lbls_txt = "ini_labels"
                else:
                    lbls_txt = "lid_labels"
                LOG.debug(f"Adding the labels of {lbls_txt} to plot")
                labels_text: Dict = nx.draw_networkx_labels(kwargs["drawing_graph"], pos=kwargs["pos"], labels=labels, font_size=15, font_color='grey')
            elif kwargs["plot_all_labels"]:
                LOG.debug("Adding all labels to plot")
                labels = {}
                for node, node_data in kwargs["drawing_graph"].nodes(data=True):
                    if node_data["kind"] in ["min", "thre", "fake_thre"]:
                        labels[node] = node
                labels_text: Dict[str] = nx.draw_networkx_labels(kwargs["drawing_graph"], pos=kwargs["pos"], labels=labels, font_size=15, font_color='grey', horizontalalignment="right")

            for k, t in labels_text.items():
                if not k.startswith("thre-"): # do not rotate lid labels
                    t.set_rotation('vertical')
        
        if kwargs["auto_ylim"]:
            LOG.debug(f"Calculated min and max y values are: ({min_y}, {max_y})")
            ax.set_ylim(min_y, max_y)
        else:
            if kwargs["y_bottom"] is not None:
                ax.set_ylim(bottom=kwargs["y_bottom"])
            if kwargs["y_top"] is not None:
                ax.set_ylim(top=kwargs["y_top"])

        return fig, ax


    def _draw_core(self, filter_func: Union[Callable[[nx.Graph, str], bool], None], **kwargs) -> Tuple[pltFigure, plt.Axes]:
        """
        Core logic for plotting the disconnectivity graph.

        Arguments
        ---------
        filter_func : Callable[[nx.Graph, str], bool]
            Function to filter the nodes to plot

        **kwargs : Dict[str, Any]
            Additional arguments to pass to the plotting function

        Returns
        -------
        fig, ax : Tuple[plt.Figure, plt.Axes]
            The figure and axes objects of the plot

        Notes
        -----
        The available kwargs are:
        - `cluster_file`, str | None: The file with the cluster information. None
        - `plot_density`, bool: Plot the density of the structures. False
        - `plot_ini_labels`, bool: Plot the labels of initial structures. False
        - `plot_lid_heights`, bool: Plot the lid heights. False
        - `plot_all_labels`, bool: Plot all labels. False
        - `y_bottom`, float | None: The lower limit of the plot. None
        - `y_top`, float | None: The upper limit of the plot. None
        - `auto_ylim`, bool: Use the `y_top` and `y_bottom` values. False
        - `show_nodes`, bool: Show the nodes. False
        - `relative_energy`, bool: Plot the relative energy. False
        - `colour_ini_basins`, bool: Colour the initial basins. False
        - `cluster_by_unique_id`, bool: Cluster by unique id instead of node id. False
        - `edge_width`, float: The width of the graph edges in the plot. 1.8
        """
        drawing_settings = {
            "cluster_file": None,
            "plot_density": False,
            "plot_ini_labels": False,
            "plot_lid_heights": False,
            "plot_all_labels": False,
            "y_bottom": None,
            "y_top": None,
            "show_nodes": False,
            "relative_energy": False,
            "colour_ini_basins": False,
            "auto_ylim": True,
            "cluster_by_unique_id": False,
            "edge_width": 1.8
        }
        drawing_settings.update(kwargs)
        LOG.debug(f"The drawing settings are: {drawing_settings}")
        drawing_data: Dict[str, Any] = {
            "filter_func": filter_func
        }

        # clone the graph to avoid unexpectedly modifying
        # the objects graph, deepcopy gives an error so we
        # use the copy method, which is a shallow copy and therefore
        # any list or dict in the nodes will be shared between the two
        drawing_data["drawing_graph"] = self.graph.copy()

        drawing_data["drawing_graph"] = self._draw_core_task1(**drawing_data)

        (drawing_data["thre"],
         drawing_data["thre_list"],
         drawing_data["ts_list"],
         drawing_data["min_list"],
         drawing_data["ini_list"],
         drawing_data["num_nodes"],
         drawing_data["energies"],
         drawing_data["min_energy"]) = self._draw_core_task2(**drawing_settings, **drawing_data)

        (drawing_data["min_color_lists"], 
         drawing_data["ini_color_lists"], 
         drawing_data["edge_lists"], 
         drawing_data["color_normalize"], 
         drawing_data["densities"]) = self._draw_core_task3(**drawing_settings, **drawing_data)
        
        edge_nums = list(drawing_data["edge_lists"].keys())
        LOG.info(f"Number of structures in the graph {drawing_data['num_nodes']}")
        LOG.info("Number of edge lists %s", len(edge_nums))

        # Set x axis for each vertice to plot the graph
        drawing_data["pos"] = self._draw_core_task4(**drawing_data)

        drawing_data["fig"], drawing_data["ax"] = self._draw_core_task5(**drawing_settings, **drawing_data)

        return drawing_data["fig"], drawing_data["ax"]


    def create_plot(self, filter_func: Union[Callable[[nx.Graph, str], bool], None] = None, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the disconnectivity graph and return the matplotlib figure and axes objects.

        This can be useful when you want to further edit the plot.

        Arguments
        ---------
        filter_func : Union[Callable[[nx.Graph, str], bool], None]
            An optional function that filters out nodes to remove from the plot. Default is None

        **kwargs
            Arguments passed to _draw_core() method 
        
        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            The matplotlib figure and axes

        Notes
        -----
        The see available kwargs see the _draw_core() method docs.
        """
        LOG.info("Creating disconnectivity graph plot")
        return self._draw_core(filter_func, **kwargs)


    def draw(self, out_name: str, filter_func: Union[Callable[[nx.Graph, str], bool], None] = None, **kwargs) -> None:
        """
        Draw disconnectivity graph to an image file.

        Arguments
        ---------
        out_name : str
            The name of the output file (without the file extension)

        filter_func : Union[Callable[[nx.Graph, str], bool], None]
            An optional function that filters out nodes to remove from the plot. Default is None

        **kwargs
            Arguments passed to _draw_core() method 

        Notes
        -----
        The see available kwargs see the _draw_core() method docs.
        """
        save_file = Path(f"{out_name}.png")
        LOG.info(f"Plotting disconnectivity graph to: {save_file}")
        self._draw_core(filter_func, **kwargs)
        plt.savefig(save_file, bbox_inches='tight', dpi=300)


    def draw_initial(self, out_name: str, **kwargs) -> None:
        """
        Draw disconnectivity graph of only the initial structures.
        
        Arguments
        ---------
        out_name : str
            The name of the output file (without the file extension)

        **kwargs
            Arguments passed to _draw_core() method 
        
        Notes
        -----
        The see available kwargs see the _draw_core() method docs.
        """
        save_file = Path(f"{out_name}.png")
        LOG.info(f"Plotting branch disconnectivity graph to: {save_file}")
        self._draw_core(DrawFilters.initial_structures(), **kwargs)
        plt.savefig(save_file, bbox_inches='tight', dpi=300)


    @classmethod
    def _read_data(cls, ds: CspDataStore, **kwargs):
        """
        Read crystals from database file.

        Arguments
        ---------
        ds : CspDataStore
            The database from which to get the data

        **kwargs : Dict[str, Any]
            Additional keyword arguments

        Returns
        -------
        trial_data, trial_connections, trials, min_trial, max_state : Tuple[Dict[int, Dict[int, List[Job]]], Dict[int, Dict[int, Set[int]], Dict[int, int]]
            The data for the trials, the connections between trials, the trials themselves, the Min_Trial object and the maximum state
        """
        en_min = kwargs.get("en_min", None)  # set the minimum energy included
        en_interval = kwargs.get(
            "en_interval", None
        )  # set energy interval between lids
        up_limit = kwargs.get("up_limit", None)  # set max energy included
        up_lid = kwargs.get("up_lid", None)

        LOG.info(f"Loading data from {ds.filename}")
        trials: Dict[int, Trial] = {}
        en_mins = []
        en_interval_tmp = 1e10
        for trial_number, metadata in ds.query(
            "select trial_number, metadata from trial"
        ).fetchall():
            meta = metadata
            while isinstance(meta, str): # in case metadata is saved as a string of json
                meta = json.loads(meta)
                
            if meta["increase"] < en_interval_tmp:
                en_interval_tmp = meta["increase"]

            first_lid_energy = meta["trajectory"][0][1]
            ini_energy = first_lid_energy - meta["increase"]
            en_mins.append(ini_energy)
            trials[trial_number] = Trial(meta["trajectory"])

        # Generate the grid for connectivity graph
        if en_min is None:
            en_min = min(en_mins)
        if en_interval is None:
            LOG.info(f"Reading threshold data with lid increases of {en_interval_tmp} kJ/mol")
            en_interval = en_interval_tmp
        else:
            LOG.info(f"Using lid increases of {en_interval_tmp} kJ/mol")
        min_trial = Min_Trial(en_min, increase=en_interval)

        state_up = np.inf
        if up_limit is not None:
            state_up = min_trial.get_state_from_energy(up_limit)

        trial_data: Dict[int, Dict[int, List[Job]]] = defaultdict(lambda: defaultdict(list)) # trial: lid_num: [structs_in_lid]
        trial_connections: Dict[int, Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set)) # trial: lid_num: [connections with other trials]
        trial_starts: Dict[int, int] = defaultdict(lambda: np.inf) # trial: lid_num
        max_state = 0
        count_ini = 0
        for uni_id, trial_number, density, energy in ds.query(
            "select distinct(id), trial_number, density, energy from crystal "
            "join trial_structure using(id) "
            "join equivalent_to on crystal.id = equivalent_to.unique_id "
        ).fetchall():
            mc_step = int(uni_id.split("-")[5])
            thre_energy = trials[trial_number].get_current_energy(mc_step)
            state = min_trial.get_state_from_energy(thre_energy)

            # Remove all false minimization leads to uphill energy
            if energy > thre_energy:
                continue

            ini_id = None
            ini_trial = None
            if mc_step == 0:
                ini_id = uni_id
                ini_trial = trial_number
            ids = defaultdict(list)
            trial_app: Dict[int, Set[int]] = defaultdict(set)
            trial_numbers: Set[int] = set()

            valid = False
            if state <= state_up:
                valid = True
                if state > max_state:
                    max_state = state
                ids[trial_number].append((uni_id, state, mc_step))
                trial_app[state].add(trial_number)
                trial_numbers.add(trial_number)
            for content in ds.query(
                'select equivalent_id from equivalent_to where unique_id="{}"'.format(
                    uni_id
                )
            ).fetchall():
                dup_id = content[0]
                if dup_id is not None:
                    mc_step = int(dup_id.split("-")[5])
                    trial_number = int(dup_id.split("-")[3])
                    if mc_step == 0:
                        if ini_trial is None:
                            ini_trial = trial_number
                            ini_id = dup_id
                        elif trial_number < ini_trial:
                            ini_trial = trial_number
                            ini_id = dup_id
                    thre_energy = trials[trial_number].get_current_energy(mc_step)
                    state = min_trial.get_state_from_energy(thre_energy)
                    if state <= state_up:
                        valid = True
                        if state > max_state:
                            max_state = state
                        trial_app[state].add(trial_number)
                        trial_numbers.add(trial_number)
                        ids[trial_number].append((dup_id, state, mc_step))
            if not valid:
                continue
            # Get the earliest stage where the structure appears in each trajectory
            min_ids = []
            for trial_number, content in ids.items():
                min_id, state, mc_step = min(content, key=lambda x: (x[2], x[0]))
                min_id, state, mc_step = min(content, key=lambda x: (x[1], x[2]))
                if state < trial_starts[trial_number]:
                    trial_starts[trial_number] = state
                min_ids.append((trial_number, min_id, state, mc_step))

            # Set the earliest stage over all trials as the unique_id
            min_trial_num, min_uni_id, min_state, _ = min(min_ids, key=lambda x: (x[2], x[3]))
            min_trial_num, min_uni_id, min_state, _ = min(min_ids, key=lambda x: x[2])
            is_ini = False
            if ini_id is not None:
                min_uni_id = ini_id
                min_trial_num = ini_trial
                is_ini = True
                count_ini += 1
            trial_data[min_trial_num][min_state].append(
                Job(
                    uni_id=uni_id,
                    trial_number=min_trial_num,
                    density=density,
                    energy=energy,
                    state=min_state,
                    min_id=min_uni_id,
                    is_ini=is_ini,
                )
            )

            for trial_num in trial_numbers:
                conns: Set[int] = set()
                for state in sorted(trial_app.keys()):
                    b = False
                    for i in trial_app[state]:
                        if i != trial_num:
                            conns.add(i)
                        else:
                            b = True
                    if b:
                        break
                    
                for i in conns:
                    trial_connections[trial_num][state].add(i)

        for trial, trial_c in trial_connections.items():
            min_conn_state = min(trial_c)
            min_start = trial_starts[trial]
            assert min_conn_state >= min_start, f"Trial {trial} has a connection state {min_conn_state} lower than the start state {min_start}"

        LOG.debug(f"Number of initial structures: {count_ini}")

        return trial_data, trial_connections, trials, min_trial, max_state

    @classmethod
    def _conn(cls, trial_data: Dict[int, Dict[int, List[Job]]], trial_connections: Dict[int, Dict[int, Set[int]]], min_trial: Min_Trial, max_state: int):
        """
        Find connections between minima from data read with _read_data().
        Store the graph as networkx graph object.

        Arguments
        ---------
        trial_data : Dict[int, Dict[int, List[Job]]]
            The data for the trials

        trial_connections : Dict[int, Dict[int, Set[int]]]
            The connections between trials

        min_trial : Min_Trial
            The Min_Trial object, used to get the energy from the state

        max_state : int
            The maximum state

        Returns
        -------
        graph, num_total, yticks, last_covered : Tuple[nx.Graph, int, List[float], List[str]]
            The networkx graph object, the total number of nodes, the yticks and the last covered basins
        """
        LOG.info("Creating disconnectivity graph")

        trial_overlay: Dict[int, List[int]] = defaultdict(list) # state: [trial_numbers]
        for trial, trial_d in trial_data.items():
            min_state = min(trial_d.keys())
            for state in range(min_state, max_state+1):
                trial_overlay[state].append(trial)

        # Find lowest connections between trajectories, and make sure they are symmetric
        connections: Dict[int, Dict[int, int]] = defaultdict(dict) # trial: {conn_trial: state}
        for trial, trial_c in trial_connections.items():
            if len(trial_c.keys()) == 0:
                continue
            for state, conns in trial_c.items():
                for conn in conns:
                    if conn not in connections[trial]:
                        connections[trial][conn] = state
                    else:
                        if state < connections[trial][conn]:
                            connections[trial][conn] = state

                    if trial not in connections[conn]:
                        connections[conn][trial] = state
                    else:
                        if state < connections[conn][trial]:
                            connections[conn][trial] = state

        for trial, conns in connections.items():
            # get earliest connection
            min_conn = np.inf
            for _, conn in conns.items():
                if conn < min_conn:
                    min_conn = conn
            
            for state in range(min_conn, max_state+1):
                if trial not in trial_overlay[state]:
                    trial_overlay[state].append(trial)

        for trial, conns in connections.items():
            for trial_conn, conn in conns.items():
                assert connections[trial_conn][trial] == conn, f"Connection between {trial} and {trial_conn} at state {conn} is not symmetric"

        # organise the trials into their respective basins
        basins: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list)) # state: {basin: [trial_numbers]}
        trial_basins: Dict[int, Dict[int, int]] = defaultdict(dict) # state: {trial: basin}
        basin_id = 0
        mem: Dict[int, int] = {} # trial: basin
        for state, trials in sorted(trial_overlay.items(), key=lambda x: x[0]):
            # LOG.info(state)
            for trial in trials:
                if trial not in mem:
                    mem[trial] = basin_id
                    basin_id += 1
            
            changes = True
            while changes:
                changes = False
                for trial in trial_data:
                    for conn_trial, conn_state in connections[trial].items():
                        if conn_state <= state:
                            # find smallest basin
                            basin = mem[trial]
                            conn_basin = mem[conn_trial]
                            min_basin = min(basin, conn_basin)
                            if min_basin != conn_basin or min_basin != basin:
                                changes = True
                            mem[trial] = min_basin
                            mem[conn_trial] = min_basin

            for trial in mem:
                basins[state][mem[trial]].append(trial)
                trial_basins[state][trial] = mem[trial]

        for state, state_basins in sorted(basins.items(), key=lambda x: x[0]):
            in_state = []
            for basin, trials in state_basins.items():
                for trial in trials:
                    assert trial not in in_state, f"Trial {trial} is in more than one basin in state {state}"

        # create the graph from lowest lid to highest,
        # using the basins created in previous step
        graph = nx.Graph()
        ts_energy = min_trial.increase / 3.0
        num_total = 0
        num_ini_total = 0
        ini_node_ids = []
        for state, state_basins in sorted(basins.items(), key=lambda x: x[0]):
            all_covering = []
            for basin, trials in state_basins.items():
                thre_id = f"thre-{basin}-{state}"
                thre_energy = min_trial.get_energy_from_state(state)
                # calculate covered basins of previous state
                if state > 1:
                    covered = []
                    for trial in trials:
                        if trial in trial_basins[state-1]:
                            prev_basin = trial_basins[state-1][trial]
                            if prev_basin not in covered:
                                covered.append(prev_basin)
                    
                    if len(covered) == 0:
                        covered = [basin]
                else:
                    covered = [basin]
                
                for c in covered:
                    assert c not in all_covering, f"Basin {c} is covered by more than one thre node in state {state}"
                    all_covering.append(c)

                if len(covered) > 1:
                    kind = "thre"
                else:
                    kind = "fake_thre"
                graph.add_node(
                    thre_id,
                    kind = kind,
                    energy = thre_energy,
                    state = state,
                    path = basin,
                    covered = covered,
                    num_nodes = 0,
                    num_initials = 0
                )

                num_nodes = 0
                num_initials = 0

                if state > 1:
                    for basin in covered:
                        if basin in basins[state-1]:
                            prev_thre_id = f"thre-{basin}-{state-1}"
                            graph.add_edge(prev_thre_id, thre_id, kind="thre")
                            prev_thre_data = graph.nodes[prev_thre_id]
                            num_nodes += prev_thre_data["num_nodes"]
                            num_initials += prev_thre_data["num_initials"]

                new_num_nodes = 0
                new_num_initials = 0
                for trial in trials:
                    if state not in trial_data[trial]:
                        continue
                    trial_data_state = trial_data[trial][state]
                    for job in trial_data_state:
                        num_total += 1
                        mc_step = int(job.min_id.split("-")[5])
                        if job.is_ini:
                            new_num_initials += 1
                            num_ini_total += 1
                            mc_step = 0
                            ini_node_ids.append(job.min_id)
                        new_num_nodes += 1
                        graph.add_node(
                            get_en_id(job.min_id),
                            kind = "ts",
                            energy = thre_energy - ts_energy
                        )
                        graph.add_node(
                            job.min_id,
                            kind = "min",
                            energy = job.energy,
                            density = job.density,
                            thre_energy = thre_energy,
                            mc_step = mc_step,
                            uni_id = job.uni_id
                        )
                        graph.add_edge(get_en_id(job.min_id), job.min_id, kind="quench", density=job.density)
                        graph.add_edge(get_en_id(job.min_id), thre_id, kind="perturb", density=job.density)
                
                thre_data = graph.nodes[thre_id]
                if new_num_nodes > 0:
                    thre_data["kind"] = "thre"

        # count the number of ini structures and total number of structures in lid node
        # done separately just to make sure that lids do not connect to more than
        # one lid at a higher state, as well to check that min, ts, thre and fake_thre
        # nodes are connected as expected
        connected_lid_nodes = []
        for ini_id in ini_node_ids:
            running_count = 0
            ts_id = next(graph.neighbors(ini_id))
            thre_id = None
            for neigh in graph.neighbors(ts_id):
                neigh_data = graph.nodes[neigh]
                if neigh_data["kind"] == "thre":
                    thre_id = neigh
                    connected_lid_nodes.append(thre_id)
                    break
            while thre_id is not None:
                connected_lid_nodes.append(thre_id)
                thre_data = graph.nodes[thre_id]
                next_thre_id = None
                for neigh in graph.neighbors(thre_id):
                    neigh_data = graph.nodes[neigh]
                    if neigh_data["kind"] == "thre" or neigh_data["kind"] == "fake_thre":
                        if neigh_data["state"] > thre_data["state"]:
                            assert next_thre_id is None, "Thre node is connected to more than one upper basin which should not be possible: {}, {}".format(next_thre_id, neigh)
                            next_thre_id = neigh
                    elif neigh_data["kind"] == "ts" and thre_data["num_nodes"] == 0:
                        running_count += 1
                    assert neigh_data["kind"] == "ts" or neigh_data["kind"] == "fake_thre" or neigh_data["kind"] == "thre", "No other kind of node aside from ts, thre and fake_thre nodes can be connected to a thre or fake_thre node: {}".format(neigh_data["kind"])
                
                thre_data["num_nodes"] += running_count
                thre_data["num_initials"] += 1
                thre_id = next_thre_id

        # Sanitize the graph to remove any nodes that are not connected
        # to any other node, this is done to avoid issues with plotting
        # and to ensure that the graph is clean
        removed_count = 0
        for node, node_data in list(graph.nodes(data=True)):
            assert node_data["kind"] in ["min", "ts", "thre", "fake_thre"], f"Node {node} has an unexpected kind: {node_data['kind']}"
            if len(list(graph.neighbors(node))) == 0:
                LOG.debug(f"Removing node {node} from the graph as it is not connected to any other node")
                graph.remove_node(node)
                removed_count += 1
            if node_data["kind"] == "thre" or node_data["kind"] == "fake_thre":
                if node not in connected_lid_nodes:
                    # connect any stuff that is connected to it between each other so if any crystal structures are connected to
                    # it they are not left hanging.
                    neighs = list(graph.neighbors(node))
                    if len(neighs) > 1:
                        high_thre_node = None
                        for neigh in neighs:
                            neigh_data = graph.nodes[neigh]
                            if (neigh_data["kind"] == "thre" or neigh_data["kind"] == "fake_thre") and neigh_data["state"] > node_data["state"]:
                                assert high_thre_node is None, f"{node} is connected to more than one thre node at higher energy"
                                high_thre_node = neigh
                        for neigh in neighs:
                            if neigh == high_thre_node:
                                continue
                            graph.add_edge(high_thre_node, neigh, kind="thre")
                            
                    LOG.debug(f"Removing node {node} from the graph as it is not connected to any initial structure")
                    graph.remove_node(node)
                    removed_count += 1
        LOG.debug(f"Removed {removed_count} nodes from the graph that were not connected to any other node")

        # must add last thre node connecting everything for plotting reasons
        last_state = max(basins.keys())
        last_basins = basins[last_state]
        num_nodes = 0
        num_initials = 0
        last_covered = []
        for basin in last_basins.keys():
            thre_id = f"thre-{basin}-{last_state}"
            last_covered.append(thre_id)
            thre_data = graph.nodes[thre_id]
            num_nodes += thre_data["num_nodes"]
            num_initials += thre_data["num_initials"]
        last_thre_id = f"thre-0-{last_state+1}"
        last_thre_energy = min_trial.get_energy_from_state(last_state+1)
        graph.add_node(
            last_thre_id,
            kind = "thre",
            energy = last_thre_energy,
            state = last_state+1,
            path = 0,
            covered = list(last_basins.keys()),
            num_nodes = num_nodes,
            num_initials = num_initials
        )
        
        LOG.info(f"Total number of structures: {num_total}")
        LOG.info(f"Total number of ini structure: {num_ini_total}")

        return graph, num_total, min_trial.get_ticks(last_state+1), last_covered


def parse_args(args: Union[List[str], None], name: Union[str, None]) -> argparse.Namespace:
    if name is None:
        name = __file__
        
    parser = argparse.ArgumentParser(
        prog=name,
        description="Create a disconnectivity graph from a threshold algorithm database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=str, help="Input clustered database file or pickled data")
    cluster_group = parser.add_argument_group("Clustering", 
                                              "Options to colour the disconnectivity graph edges")
    cluster_group_exc = cluster_group.add_mutually_exclusive_group()
    cluster_group_exc.add_argument(
        "-c",
        "--cluster-file",
        type=str,
        default=None,
        help=(
            "Clustering file to be coloured on disconnectivity graph, currently "
            "hard coded as fourth row is label of cluster"
        ),
    )
    cluster_group_exc.add_argument(
        "--plot-density",
        action="store_true",
        default=False,
        help="Colour leaf edges by density of the crystal",
    )
    cluster_group_exc.add_argument(
        "--colour-ini-basins",
        action="store_true",
        default=False,
        help="Colour edges by basins of individual trajectories",
    )
    disconn_group = parser.add_argument_group("Disconnectivity",
                                              "Modify how the disconnectivity graph is generated")
    disconn_group.add_argument(
        "-e", "--en-min", type=float, default=None, help="""Minimum energy, only works when reading 
                               from a database not a pickled object""")
    disconn_group.add_argument("--up-limit", type=float, default=None, help="""Upper-lid energy limit, only works when reading 
                               from a database not a pickled object""")
    disconn_group.add_argument(
        "--interval",
        type=float,
        default=None,
        help="""Energy interval for disconnectivity graph, only works when reading 
                from a database not a pickled object""",
    )
    plot_group = parser.add_argument_group("Plotting", 
                                           "Control the plot settings and look")
    plot_group.add_argument(
        "-o", "--output", type=str, default="disconnectivity_graph", 
        help="Name for the output file"
    )
    plot_group.add_argument(
        "--plot-branch",
        action="store_true",
        default=False,
        help="Dump main branch disconnecitivity graph instead of the full graph",
    )
    plot_group.add_argument(
        "--plot-nodes",
        action="store_true",
        default=False,
        help="Show leaf nodes",
    )
    plot_group.add_argument(
        "--plot-ini-labels",
        action="store_true",
        default=False,
        help="Plot trial numbers of the initial structures",
    )
    plot_group.add_argument(
        "--plot-lid-heights",
        action="store_true",
        default=False,
        help="Plot lid heights",
    )
    plot_group.add_argument(
        "--plot-all-labels",
        action="store_true",
        default=False,
        help="Plot labels of all nodes on graph",
    )
    plot_group.add_argument(
        "--plot-by-unique-id",
        action="store_true",
        help="Use the unique ids of the structures instead"
        " of the node ids",
    )
    plot_group.add_argument(
        "--plot-by-relative-energy",
        action="store_true",
        help="Plot using relative energy form global energy minimum",
    )
    parser.add_argument(
        "--save-pickle",
        action="store_true",
        default=False,
        help="Save pickle file of disconnectivtiy graph object",
    )
    parser.add_argument(
        "-ll",
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    return parser.parse_args(args)


def main(args: Union[List[str], None] = None, name: Union[str, None] = None) -> None:
    args = parse_args(args, name)
    logging.basicConfig(
        format=FORMATS[args.log_level], datefmt=DATEFMT, level=args.log_level
    )

    file = Path(args.input)
    if not file.is_file():
        LOG.info(f"ERROR: {args.input} is not a valid file")
        sys.exit(1)
    
    LOG.info("Loading %s", args.input)
    tic = time.time()
    if args.input.endswith(".db"):
        ds = CspDataStore(args.input)
    else:
        ds = Path(args.input)
    disconn = DisconnectivityGraph(
        ds,
        name=args.output,
        save_graph=args.save_pickle,
        en_min=args.en_min,
        en_interval=args.interval,
        up_limit=args.up_limit,
    )
    if not args.input.endswith(".db"):
        LOG.info(f"Total number of structures: {disconn.num_nodes}")
        ini_ids, _ = disconn.initial_minima_ids()
        LOG.info(f"Total number of ini structure: {len(ini_ids)}")
    LOG.info("Analyze time %.3fs", time.time() - tic)
    
    if args.up_limit is not None:
        filter = DrawFilters.limit_state(disconn.min_trial.get_state_from_energy(args.up_limit))
    else:
        filter = None

    if args.plot_branch:
        disconn.draw_initial(
            f"{args.output}_branch",
            cluster_file=args.cluster_file,
            plot_density=args.plot_density,
            plot_ini_labels=args.plot_ini_labels,
            plot_lid_heights=args.plot_lid_heights,
            plot_all_labels=args.plot_all_labels,
            show_nodes=args.plot_nodes,
            colour_ini_basins=args.colour_ini_basins,
            cluster_by_unique_id=args.plot_by_unique_id,
            relative_energy=args.plot_by_relative_energy
        )
    else:
        disconn.draw(
            args.output,
            filter_func=filter,
            cluster_file=args.cluster_file,
            plot_density=args.plot_density,
            plot_ini_labels=args.plot_ini_labels,
            plot_lid_heights=args.plot_lid_heights,
            plot_all_labels=args.plot_all_labels,
            show_nodes=args.plot_nodes,
            colour_ini_basins=args.colour_ini_basins,
            cluster_by_unique_id=args.plot_by_unique_id,
            relative_energy=args.plot_by_relative_energy
        )

if __name__ == "__main__":
    main()
