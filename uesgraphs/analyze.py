import uesgraphs as ug


import os

from typing import List, Dict, Generator, Optional
import pyarrow.parquet as pq
import re
import pandas as pd

import logging

from datetime import datetime
import tempfile

import numpy as np
import sys
from pathlib import Path

from uesgraphs.data.mat_handler import mat_to_parquet

#### Global Variables ####

MASKS = None # Dictionary to store masks for column names

#### Functions 1: Logger ####
def set_up_logger(name,log_dir = None,level=int(logging.ERROR)):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        if log_dir == None:
            log_dir = tempfile.gettempdir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        print(f"Logfile findable here: {log_file}")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(filename)s:%(lineno)d] - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

#### Functions 2: Data Processing ####

def process_parquet_file(file_path: str, filter_list: List[str], 
                        chunk_size: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """
    Process a parquet file in chunks to reduce memory usage.
    
    Args:
        file_path: Path to the parquet file
        filter_list: List of column patterns to filter
        chunk_size: Number of rows to process at once
    """
    # Read parquet file metadata to get columns
    parquet_file = pq.ParquetFile(file_path)
    all_columns = parquet_file.schema.names
    
    # Pre-filter columns based on filter_list to reduce memory usage
    columns_to_read = []
    for pattern in filter_list:
        if pattern.endswith('$'):
            # Regex filter
            regex_pattern = pattern[:-1] + '$'
            columns_to_read.extend(
                col for col in all_columns 
                if re.match(regex_pattern, col)
            )
        else:
            # Simple string filter
            columns_to_read.extend(
                col for col in all_columns 
                if pattern in col
            )
    
    # Remove duplicates while preserving order
    columns_to_read = list(dict.fromkeys(columns_to_read))
    
    # Read and process the file in chunks
    for chunk in parquet_file.iter_batches(batch_size=chunk_size, columns=columns_to_read):
        yield chunk.to_pandas()

def check_input_file(file_path):
    if not file_path:
        raise ValueError("File path cannot be empty")

    base_path = os.path.splitext(file_path)[0]
    gzip_path = f"{base_path}.gzip"

    # Check for gzip first
    if os.path.exists(gzip_path):
        return gzip_path
    # Then check for .mat
    mat_path = f"{base_path}.mat"
    if os.path.exists(mat_path):
        try:
            print(f"Converting .mat file to parquet: {mat_path}")
            gzip_new = mat_to_parquet(save_as = base_path, fname = mat_path,with_unit=False)
            print(f"Converted .mat file to parquet: {gzip_new}")
            return gzip_new
        except:
            raise ValueError(f"Could not convert .mat file to parquet: {mat_path}") 
    # Finally check if file exists with any extension
    if not os.path.exists(file_path):
        raise ValueError(f"File does not exist: {file_path}")


def process_simulation_result(file_path: str, filter_list: List[str]) -> pd.DataFrame:
    """
    Process a single simulation result file and return the processed DataFrame.
    
    Args:
        file_path: Complete path to the simulation result file
        filter_list: List of column patterns to filter
        
    Returns:
        pd.DataFrame: Processed and filtered DataFrame from the simulation results
        
    Raises:
        FileNotFoundError: If the specified file does not exist
        ValueError: If the file path is empty or invalid
    """
    file_path = check_input_file(file_path=file_path)
    print(f"Processing: {file_path}")
    
    # Initialize an empty list for filtered chunks
    filtered_chunks = []
    
    # Process the file in chunks
    for chunk in process_parquet_file(file_path, filter_list):
        filtered_chunks.append(chunk)
    
    # Combine all chunks into a single DataFrame
    if not filtered_chunks:
        return pd.DataFrame()  # Return empty DataFrame if no data was processed
        
    result_df = pd.concat(filtered_chunks, axis=0)
    
    # Clear the chunks list to free memory
    filtered_chunks.clear()
    
    return result_df

#### Functions 3: Data Processing ####

def prepare_DataFrame(df, base_date=datetime(2024, 1,1),time_interval="15min",start_date=None,end_date=None):
    """
    Prepare a DataFrame with a datetime index using customizable parameters.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be processed
    base_date : datetime, optional
        The starting date for the index (default: 2024-01-01)
    time_interval : str, optional
        Frequency of the time intervals (e.g., '15min', '1h', '30min', default: '15min')
    start_date : datetime, optional
        If provided, slice the DataFrame from this date (inclusive)
    end_date : datetime, optional
        If provided, slice the DataFrame until this date (inclusive)
    
    Returns:
   
    DataFrame: A DataFrame containing the data from the parquet file for the specified time period.
    """
    try:
        # Create datetime index with specified frequency    
        datetime_index = pd.date_range(start=base_date, periods=len(df), freq=time_interval)
        
        # Set the index of the DataFrame to the datetime index
        df.index = datetime_index

        #Filter by data_range if specified
        if start_date is not None and end_date is not None:
            return df.loc[start_date:end_date]
        elif start_date is not None:
            return df.loc[start_date:]
        elif end_date is not None:
            return df.loc[:end_date]
        # Benenne den Index um
        df.index.name = 'DateTime'

        return df
    except ValueError as e:
        raise ValueError(f"Error create data range with frequency {time_interval} and base date {base_date}."
                         f"Original error: {e}") 

def get_mostfrequent_value(liste):
    #Find unique values and frequencies
    unique_values, counts = np.unique(liste, return_counts=True)
    #Find index of most frequent value and save that value
    max_index = np.argmax(counts)
    result = float(unique_values[max_index])
    return result

def get_peripheral_value(node, edge, heating_network, param, value_a, value_b):
    for node2 in edge:
        if node2 != node:
            value_assigned_node = heating_network.nodes[node2][param]
            if abs(value_a - value_assigned_node) < abs(value_b - value_assigned_node):
                result = value_b
            else:
                result = value_a
    return result

def get_node_values(heating_network, row,pipe_type=""):
    nodes = sorted(heating_network.nodes)
    assigned_nodes = []
    nodes_peripheral = []
    #Neighbor-algorihm: find the most frequent value at a node where at least 2 edges meet
    for node in nodes:         
        edges =  heating_network.edges(node) #Find adjacent edges
        #If we find only one edge we are regarding a peripheral node, which cant be solved by the neigbor algorithm
        if len(edges)> 1:
            pressures = []
            temperatures=[]

            for edge in edges:
                pipe_name= heating_network.edges[edge]['name']
                pressure_a = row[MASKS["p_a"].format(pipe_code=pipe_name, type=pipe_type)]
                pressure_b = row[MASKS["p_b"].format(pipe_code=pipe_name, type=pipe_type)]

                temperature_a = row[MASKS["T_a"].format(pipe_code=pipe_name, type=pipe_type)]
                temperature_b = row[MASKS["T_b"].format(pipe_code=pipe_name, type=pipe_type)]

                pressures.extend([pressure_a, pressure_b])
                temperatures.extend([temperature_a,temperature_b])
            pressure = get_mostfrequent_value(pressures)
            temperature = get_mostfrequent_value(temperatures)
            if pressure_a == pressure:
                heating_network.nodes[node]["press_name"] = MASKS["p_a"].format(pipe_code=pipe_name, type=pipe_type)
            elif pressure_b == pressure:
                heating_network.nodes[node]["press_name"] = MASKS["p_b"].format(pipe_code=pipe_name, type=pipe_type)
            heating_network.nodes[node]["press_flow"] = pressure

            if temperature_a == temperature:
                heating_network.nodes[node]["temp_name"] = MASKS["T_a"].format(pipe_code=pipe_name, type=pipe_type)
            elif temperature_b == temperature:
                heating_network.nodes[node]["temp_name"] = MASKS["T_b"].format(pipe_code=pipe_name, type=pipe_type)
            heating_network.nodes[node]["temperature_supply"] = temperature
            assigned_nodes.append(node)
        else:
            nodes_peripheral.append(node)
    #Peripheral-algorithm: Takes the not solved nodes from the neighbor-algorithm. Since we're still not sure
            # if the port_a or ports_b value is the right one, we compare the simulation results with the already assigned node
            # that was solved by the neighbor-algorithm.
    for node in nodes_peripheral:
        edge = list(heating_network.edges(node))[0]
        pipe_name = heating_network.edges[edge]["name"]
        pressure_a = row[MASKS["p_a"].format(pipe_code=pipe_name, type=pipe_type)]
        pressure_b = row[MASKS["p_b"].format(pipe_code=pipe_name, type=pipe_type)]

        temperature_a = row[MASKS["T_a"].format(pipe_code=pipe_name, type=pipe_type)]
        temperature_b = row[MASKS["T_b"].format(pipe_code=pipe_name, type=pipe_type)]
        
        pressure = get_peripheral_value(node, edge, heating_network, "press_flow", pressure_a, pressure_b)
        if pressure_a == pressure:
            heating_network.nodes[node]["press_name"] = MASKS["p_a"].format(pipe_code=pipe_name, type=pipe_type)
        elif pressure_b == pressure:
            heating_network.nodes[node]["press_name"] = MASKS["p_b"].format(pipe_code=pipe_name, type=pipe_type)
        heating_network.nodes[node]["press_flow"] = pressure

        temperature = get_peripheral_value(node, edge, heating_network, "temperature_supply", temperature_a, temperature_b)
        if temperature_a == temperature:
            heating_network.nodes[node]["temp_name"] = MASKS["T_a"].format(pipe_code=pipe_name, type=pipe_type)
        elif temperature_b == temperature:
            heating_network.nodes[node]["temp_name"] = MASKS["T_b"].format(pipe_code=pipe_name, type=pipe_type)
        heating_network.nodes[node]["temperature_supply"] = temperature
        
        assigned_nodes.append(node)
    
    ####Assert
    if len(assigned_nodes) == len(nodes):
        print("Assignment of pressure to nodes completed")
    else:
        print(f"Assignmnet failed -- severe {len(assigned_nodes)} und {len(nodes)}")
    return heating_network

def check_supply_type(graph):
    # Check if the graph is a supply or return graph
    if "supply_type" not in graph.graph:
        raise ValueError("The graph does not have a supply_type attribute")
    if graph.graph["supply_type"] not in ["supply", "return"]:
        raise ValueError("The graph supply_type attribute must be either 'supply' or 'return'")
    return graph.graph["supply_type"]

def get_MASKS(aixlib_version):
    """Returns the correct variable masks for different AixLib versions.
    
    The naming convention for variables in the simulation model depends on the AixLib version
    used to build it. The key difference is in how ports are referenced:
    - Version 2.1.0: Uses direct port access (e.g., port_b.p)
    - Earlier versions: Uses array indexing for ports_b (e.g., ports_b[1].p)
    
    Args:
        aixlib_version: Version string of AixLib used to build the model
        
    Returns:
        Dictionary mapping variable types to their full path in the simulation model
        
    """
    if aixlib_version == "2.1.0":
        masks = {"m_flow": "networkModel.pipe{pipe_code}{type}.port_a.m_flow",
         "p_a": "networkModel.pipe{pipe_code}{type}.port_a.p",
         "p_b": "networkModel.pipe{pipe_code}{type}.port_b.p",
         "T_a": "networkModel.pipe{pipe_code}{type}.sta_a.T",
         "T_b": "networkModel.pipe{pipe_code}{type}.sta_b.T",
         }
    else:
        masks = {"m_flow": "networkModel.pipe{pipe_code}{type}.port_a.m_flow",
         "p_a": "networkModel.pipe{pipe_code}{type}.port_a.p",
         "p_b": "networkModel.pipe{pipe_code}{type}.ports_b[1].p",
         "T_a": "networkModel.pipe{pipe_code}{type}.sta_a.T",
         "T_b": "networkModel.pipe{pipe_code}{type}.sta_b[1].T",
         }
    return masks


#### Functions 4: Data Assignment (main) ####

def assign_data_to_uesgraphs(graph,sim_data,start_date,end_date, aixlib_version ="2.1.0",time_interval="15min"):
    
    check_supply_type(graph) # Check if the graph is a supply or return graph

    supply_type_prefix = {"supply": "", "return": "R"}

    global MASKS
    MASKS = get_MASKS(aixlib_version)
    try:

        filter_list = []
        for edge in graph.edges:
            pipe_code = graph.edges[edge]["name"]
            for mask in MASKS:
                filter_list.append(MASKS[mask].format(pipe_code=pipe_code, 
                                                    type=supply_type_prefix[graph.graph["supply_type"]]))
        df = process_simulation_result(file_path=sim_data, filter_list=filter_list)
        df = prepare_DataFrame(df,start_date=start_date, end_date=end_date,time_interval=time_interval)
        
        graph = get_node_values(graph, df.iloc[0],pipe_type=supply_type_prefix[graph.graph["supply_type"]])
        
        for node in graph.nodes:
            graph.nodes[node]["press_flow"] = df[graph.nodes[node]["press_name"]]
            graph.nodes[node]["temperature_supply"] = df[graph.nodes[node]["temp_name"]]
        
        for edge in graph.edges:
            graph.edges[edge]["m_flow"] = df[MASKS["m_flow"].format(pipe_code=graph.edges[edge]["name"],
                                                                type=supply_type_prefix[graph.graph["supply_type"]])]
            graph.edges[edge]["press_drop"] = abs(graph.nodes[edge[0]]["press_flow"] - graph.nodes[edge[1]]["press_flow"])
            graph.edges[edge]["press_drop_length"] = graph.edges[edge]["press_drop"] / graph.edges[edge]["length"]
            
            graph.edges[edge]["temp_diff"] = abs(graph.nodes[edge[0]]["temperature_supply"] - graph.nodes[edge[1]]["temperature_supply"])
    except KeyError as e_key:
        if "ports_b[1]" in str(e_key):
            raise KeyError(f"Key: {e_key}  not found in data."
                  'Try using aixlib_version="2.1.0" when calling assign_data_to_uesgraphs'
                  "For more information see method get_MASKS(aixlib_version) in analyze.py"
            ) from e_key
        elif "port_b" in str(e_key):
            raise KeyError(f"Key: {e_key}  not found in data."
                           'Try using aixlib_version="2.0.0" when calling assign_data_to_uesgraphs'
                  " For more information see method get_MASKS(aixlib_version) in analyze.py"
            ) from e_key
        else:
            raise KeyError(f"Key: {e_key}  not found in data."
                  "Unknown Error. Check your data if it complies with mapping of get_MASKS(aixlib_version) in analyze.py"
            ) from e_key
    return graph


#### Functions 5: Data post-processing ####

def calculate_pump_power(graph):
    """
    Calculate the pump power based on the pressure drop and mass flow rate.
    
    This function calculates the pump power for each edge in the graph using the formula:
        Pump Power = Pressure Drop * Mass Flow Rate
    The results are stored in the edge attributes of the graph.
    
    Returns:
        None
    """
    #if graph uesgraphs:
    source_node, source_edge = find_source_node(graph)
    calculate_accumulated_pressure_drop([graph], source_node, "press_drop")


def find_source_node(heat_net):
    nodes = sorted(heat_net.nodes)
    for node in nodes:
        name = heat_net.nodes[node]["name"]
        if isinstance(name, str):
            if name.lower() == "supply1":
                edge =  heat_net.edges(node) #Find adjacent edges
                if len(edge) == 1:
                    return node, list(edge)[0]
                else:
                    logger.error(f"Source node {node} has more than one edge. Check graph")
    return "no source (name = supply1) node found. Please make sure your json file has a node with the name supply1 and only one edge connected to it."

def calculate_accumulated_pressure_drop(heat_nets, source_node, key):
    for heat_net in heat_nets:
        #Calculate the accumulated pressure drops from source node to all other nodes
        pressure = nx.single_source_dijkstra_path_length(heat_net, source_node, weight=key)
        for nodes in heat_net.nodes():
            heat_net.nodes[nodes]["acc_press_drop"] = pressure[nodes]
    return heat_nets

#### Functions 6: Documentation ####
    
EDGE_ATTRIBUTES = {
    # Static attributes
    "diameter": {"unit": None, "description": "Pipe inner diameter", "is_timeseries": False},
    "length": {"unit": None, "description": "Pipe length", "is_timeseries": False},
    "pipeID": {"unit": None, "description": "Pipe identifier", "is_timeseries": False},
    "name": {"unit": None, "description": "Pipe name", "is_timeseries": False},
    "node_0": {"unit": None, "description": "Start node identifier", "is_timeseries": False},
    "node_1": {"unit": None, "description": "End node identifier", "is_timeseries": False},
    "dIns": {"unit": "m", "description": "Insulation diameter", "is_timeseries": False},
    "kIns": {"unit": "W/(m·K)", "description": "Insulation thermal conductivity", "is_timeseries": False},
    "m_flow_nom": {"unit": "kg/s", "description": "Nominal mass flow rate", "is_timeseries": False},
    "fac": {"unit": None, "description": "Flow factor", "is_timeseries": False},
    
    # Time series attributes
    "m_flow": {"unit": None, "description": "Mass flow rate", "is_timeseries": True},
    "press_drop": {"unit": None, "description": "Pressure drop", "is_timeseries": True},
    "press_drop_length": {"unit": None, "description": "Pressure drop per length", "is_timeseries": True},
    "temp_diff": {"unit": None, "description": "Temperature difference", "is_timeseries": True},
    #"asd": {"unit": None, "description": "asd", "is_timeseries": True}
}

NODE_ATTRIBUTES = {
    # Static attributes
    "node_type": {"unit": None, "description": "Type of the network node", "is_timeseries": False},
    "network_id": {"unit": None, "description": "Network identifier", "is_timeseries": False},
    "position": {"unit": None, "description": "Geographical position coordinates", "is_timeseries": False},
    "name": {"unit": None, "description": "Node name", "is_timeseries": False},
    "press_name": {"unit": None, "description": "Reference name for pressure data", "is_timeseries": False},
    "temp_name": {"unit": None, "description": "Reference name for temperature data", "is_timeseries": False},
    
    # Time series attributes
    "press_flow": {"unit": "Pa", "description": "Pressure flow time series", "is_timeseries": True},
    "temperature_supply": {"unit": "K", "description": "Supply temperature time series", "is_timeseries": True}
}

def analyze_node_types(graph):
    """
    Analyze and categorize different node types in the graph with their attributes.
    
    This function identifies different node types and their specific attributes,
    which is helpful for understanding the structure of the graph.
    
    Parameters:
        graph: NetworkX graph with simulation data
        
    Returns:
        dict: Dictionary with node types as keys and their specific attributes as values
    """
    node_types = {}
    
    # Iterate through all nodes
    for _, data in graph.nodes(data=True):
        node_type = data.get('node_type', 'unknown')
        
        # Initialize node type if not seen before
        if node_type not in node_types:
            node_types[node_type] = {
                'count': 0,
                'attributes': set(),
                'timeseries_attributes': set()
            }
        
        # Count this node
        node_types[node_type]['count'] += 1
        
        # Add all attributes
        for key, value in data.items():
            node_types[node_type]['attributes'].add(key)
            
            # Try to detect time series data
            if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                try:
                    if len(value) > 1:
                        node_types[node_type]['timeseries_attributes'].add(key)
                except TypeError:
                    # Not a sequence with length
                    pass
    
    return node_types

def generate_graph_data_report(graph, output_path=None, node_types=["heating","building"]):
    """
    Generate a comprehensive report on the graph data, attributes, and validation results.
    Includes both edge and node analysis.
    
    Parameters:
        graph: UESGraph with simulation data
        output_path: Path to save the report (if None, will print to console)
        node_types: List of node types to include in the report (default: ["heating"])
    """

    
  
    # Build report content
    report_lines = []
    report_lines.append("# Graph Data Report")
    report_lines.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Graph summary section
    report_lines.append(f"\n## Graph Summary")
      
    
    # Handle edge counting - may need to adapt based on your class implementation
    try:
        # First try without parameters
        num_edges = len(list(graph.edges()))
        report_lines.append(f"- Number of edges: {num_edges}")
    except Exception as e:
        report_lines.append(f"- Error counting edges: {str(e)}")
    
    if hasattr(graph, "graph") and isinstance(graph.graph, dict) and "supply_type" in graph.graph:
        report_lines.append(f"- Supply type: {graph.graph['supply_type']}")
    
    # Analyze node types
    node_type_analysis = analyze_node_types(graph)
    report_lines.append(f"- Node types identified: {len(node_type_analysis)}")
    for nt, info in node_type_analysis.items():
        report_lines.append(f"  - {nt}: {info['count']} nodes")
    
    #------------------------#
    # EDGE ATTRIBUTES ANALYSIS
    #------------------------#
    report_lines.append(f"\n## EDGE ATTRIBUTES ANALYSIS")
    
    # Get actual edge attributes
    actual_edge_attrs = set()
    for u, v, data in graph.edges(data=True):
        actual_edge_attrs.update(data.keys())
    
    # Get expected attributes from schema
    expected_edge_attrs = set(EDGE_ATTRIBUTES.keys())
    
    # Calculate missing and unexpected attributes
    missing_edge_attrs = expected_edge_attrs - actual_edge_attrs
    unexpected_edge_attrs = actual_edge_attrs - expected_edge_attrs
    
    # Edge attribute summary section
    report_lines.append(f"\n### Edge Attribute Summary")
    report_lines.append(f"- Total attributes in schema: {len(expected_edge_attrs)}")
    report_lines.append(f"- Total attributes in graph: {len(actual_edge_attrs)}")
    report_lines.append(f"- Attributes in both: {len(actual_edge_attrs & expected_edge_attrs)}")
    report_lines.append(f"- Undocumented attributes: {len(unexpected_edge_attrs)}")
    report_lines.append(f"- Missing attributes: {len(missing_edge_attrs)}")
    
    # Documented edge attributes section
    report_lines.append(f"\n### Documented Edge Attributes")
    report_lines.append(f"Attributes that are known and expected by this script. You can access for example with:")
    report_lines.append(f"for edge in graph.edges: \n \t `graph.edges[edge]['Attribute']`")
    report_lines.append(f"| Attribute | Unit | Type | Description |")
    report_lines.append(f"|-----------|------|------|-------------|")
    
    for attr in sorted(actual_edge_attrs & expected_edge_attrs):
        info = EDGE_ATTRIBUTES[attr]
        data_type = "Time series" if info["is_timeseries"] else "Static value"
        unit = info["unit"] if info["unit"] else "-"
        report_lines.append(f"| {attr} | {unit} | {data_type} | {info['description']} |")
    
    # Undocumented edge attributes section
    if unexpected_edge_attrs:
        report_lines.append(f"\n### Undocumented Edge Attributes")
        report_lines.append(f"| Attribute | Count | Example Value |")
        report_lines.append(f"|-----------|-------|---------------|")
        
        # Get example values and counts
        attr_examples = {}
        attr_counts = {attr: 0 for attr in unexpected_edge_attrs}
        
        for u, v, data in graph.edges(data=True):
            for attr in unexpected_edge_attrs:
                if attr in data:
                    attr_counts[attr] += 1
                    if attr not in attr_examples and data[attr] is not None:
                        # Safely convert to string and truncate long values
                        try:
                            example_value = str(data[attr])[:50]
                            if len(str(data[attr])) > 50:
                                example_value += "..."
                            attr_examples[attr] = example_value
                        except:
                            attr_examples[attr] = "[Complex data]"
        
        for attr in sorted(unexpected_edge_attrs):
            example = attr_examples.get(attr, "None")
            report_lines.append(f"| {attr} | {attr_counts[attr]} | {example} |")
    
    # Missing edge attributes section
    if missing_edge_attrs:
        report_lines.append(f"\n### Missing Edge Attributes")
        report_lines.append(f"| Attribute | Unit | Type | Description |")
        report_lines.append(f"|-----------|------|------|-------------|")
        for attr in sorted(missing_edge_attrs):
            info = EDGE_ATTRIBUTES[attr]
            data_type = "Time series" if info["is_timeseries"] else "Static value"
            unit = info["unit"] if info["unit"] else "-"
            report_lines.append(f"| {attr} | {unit} | {data_type} | {info['description']} |")
    
    #------------------------#
    # NODE ATTRIBUTES ANALYSIS
    #------------------------#
    report_lines.append(f"\n## NODE ATTRIBUTES ANALYSIS")
    
    # Get all nodes of the specified types
    analyzed_nodes = []
    for _, data in graph.nodes(data=True):
        node_type_value = data.get('node_type', 'unknown')
        if node_type_value in node_type_analysis.keys():  # Include if in specified types or if no types specified
            analyzed_nodes.append(data)
    
    if not analyzed_nodes:
        report_lines.append(f"\nNo nodes of specified types {node_types} found.")
    else:
        # Get actual node attributes from nodes of specified types
        actual_node_attrs = set()
        for data in analyzed_nodes:
            actual_node_attrs.update(data.keys())
        
        # Get expected attributes from schema
        expected_node_attrs = set(NODE_ATTRIBUTES.keys())
        
        # Calculate missing and unexpected attributes
        missing_node_attrs = expected_node_attrs - actual_node_attrs
        unexpected_node_attrs = actual_node_attrs - expected_node_attrs
        
        # Node attribute summary section
        report_lines.append(f"\n### Node Attribute Summary")
        report_lines.append(f"- Total attributes in schema: {len(expected_node_attrs)}")
        report_lines.append(f"- Total attributes in graph nodes: {len(actual_node_attrs)}")
        report_lines.append(f"- Attributes in both: {len(actual_node_attrs & expected_node_attrs)}")
        report_lines.append(f"- Undocumented attributes: {len(unexpected_node_attrs)}")
        report_lines.append(f"- Missing attributes: {len(missing_node_attrs)}")
        
        # Documented node attributes section
        report_lines.append(f"\n### Documented Node Attributes")
        report_lines.append(f"Attributes that are known and expected by this script.")
        report_lines.append(f"| Attribute | Unit | Type | Description |")
        report_lines.append(f"|-----------|------|------|-------------|")
        
        for attr in sorted(actual_node_attrs & expected_node_attrs):
            info = NODE_ATTRIBUTES[attr]
            data_type = "Time series" if info["is_timeseries"] else "Static value"
            unit = info["unit"] if info["unit"] else "-"
            report_lines.append(f"| {attr} | {unit} | {data_type} | {info['description']} |")
        
        # Undocumented node attributes section
        if unexpected_node_attrs:
            report_lines.append(f"\n### Undocumented Node Attributes")
            report_lines.append(f"| Attribute | Count | Example Value |")
            report_lines.append(f"|-----------|-------|---------------|")
            
            # Get example values and counts
            attr_examples = {}
            attr_counts = {attr: 0 for attr in unexpected_node_attrs}
            
            for data in analyzed_nodes:
                for attr in unexpected_node_attrs:
                    if attr in data:
                        attr_counts[attr] += 1
                        if attr not in attr_examples and data[attr] is not None:
                            # Safely convert to string and truncate long values
                            try:
                                example_value = str(data[attr])[:50]
                                if len(str(data[attr])) > 50:
                                    example_value += "..."
                                attr_examples[attr] = example_value
                            except:
                                attr_examples[attr] = "[Complex data]"
            
            for attr in sorted(unexpected_node_attrs):
                example = attr_examples.get(attr, "None")
                report_lines.append(f"| {attr} | {attr_counts[attr]} | {example} |")
        
        # Missing node attributes section
        if missing_node_attrs:
            report_lines.append(f"\n### Missing Node Attributes")
            report_lines.append(f"| Attribute | Unit | Type | Description |")
            report_lines.append(f"|-----------|------|------|-------------|")
            for attr in sorted(missing_node_attrs):
                info = NODE_ATTRIBUTES[attr]
                data_type = "Time series" if info["is_timeseries"] else "Static value"
                unit = info["unit"] if info["unit"] else "-"
                report_lines.append(f"| {attr} | {unit} | {data_type} | {info['description']} |")
        
        # Node type specific attribute analysis
        report_lines.append(f"\n### Node Type Specific Attributes")
        report_lines.append(f"Analysis of attributes by node type.")
        
        for node_type, info in sorted(node_type_analysis.items()):
            if node_type in node_types or not node_types:  # Only include specified types
                report_lines.append(f"\n#### {node_type} ({info['count']} nodes)")
                
                static_attrs = info['attributes'] - info['timeseries_attributes']
                report_lines.append(f"- Static attributes: {len(static_attrs)}")
                report_lines.append(f"- Time series attributes: {len(info['timeseries_attributes'])}")
                
                # List static attributes
                if static_attrs:
                    report_lines.append(f"\n**Static Attributes:**")
                    report_lines.append(f"| Attribute | In Schema |")
                    report_lines.append(f"|-----------|-----------|")
                    for attr in sorted(static_attrs):
                        in_schema = "Yes" if attr in NODE_ATTRIBUTES else "No"
                        report_lines.append(f"| {attr} | {in_schema} |")
                
                # List time series attributes
                if info['timeseries_attributes']:
                    report_lines.append(f"\n**Time Series Attributes:**")
                    report_lines.append(f"| Attribute | In Schema |")
                    report_lines.append(f"|-----------|-----------|")
                    for attr in sorted(info['timeseries_attributes']):
                        in_schema = "Yes" if attr in NODE_ATTRIBUTES else "No"
                        report_lines.append(f"| {attr} | {in_schema} |")
    
    #------------------------#
    # DATA QUALITY VALIDATION
    #------------------------#
    report_lines.append(f"\n## Data Quality Summary")
    
    # Edge validation
    report_lines.append(f"\n### Edge Data Quality")
    report_lines.append("- Basic validation of edge attributes (checking for missing values and anomalies)")
    
    # Count edges with missing required attributes
    required_edge_attrs = ["diameter", "length"]  # Customize based on your requirements
    missing_required = 0
    edge_issues = {}
    
    for u, v, data in graph.edges(data=True):
        edge_id = f"{u}-{v}"
        issues = []
        
        for attr in required_edge_attrs:
            if attr not in data or data[attr] is None:
                issues.append(f"Missing required attribute: {attr}")
                
        if issues:
            edge_issues[edge_id] = issues
            missing_required += 1
    
    report_lines.append(f"- Edges missing required attributes: {missing_required} of {len(list(graph.edges()))}")
    
    # Node validation
    report_lines.append(f"\n### Node Data Quality")
    report_lines.append("- Basic validation of node attributes (checking for missing values and anomalies)")
    
    # Define required node attributes based on node type
    required_node_attrs = {
        "network_heating": ["position", "name"],
        "building": ["position", "name"],
        # Add other node types as needed
    }
    
    # Count nodes with issues
    nodes_with_issues = 0
    node_issues = {}
    
    for node, data in graph.nodes(data=True):
        node_type_value = data.get('node_type', 'unknown')
        
        # Skip if not in specified types
        if node_types and node_type_value not in node_types:
            continue
            
        issues = []
        
        # Check for required attributes based on node type
        if node_type_value in required_node_attrs:
            for attr in required_node_attrs[node_type_value]:
                if attr not in data or data[attr] is None:
                    issues.append(f"Missing required attribute: {attr}")
        
        # Check for time series data consistency
        for attr in data:
            if attr in NODE_ATTRIBUTES and NODE_ATTRIBUTES[attr]["is_timeseries"]:
                value = data[attr]
                # Check if this is a time series but empty or very short
                if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                    try:
                        if len(value) < 2:  # A time series should have multiple values
                            issues.append(f"Time series attribute '{attr}' has insufficient data points: {len(value)}")
                    except TypeError:
                        # Not a sequence with length
                        pass
        
        if issues:
            node_issues[node] = issues
            nodes_with_issues += 1
    
    analyzed_node_count = len(analyzed_nodes)
    report_lines.append(f"- Nodes with issues: {nodes_with_issues} of {analyzed_node_count} analyzed nodes")
    
    # Join report lines
    report_content = "\n".join(report_lines)
    
    # Output report
    if output_path:
        try:
                        
            # Explicitly check if output_path is a directory or file path
            output_path_obj = Path(output_path)
            
            # If output_path ends with a directory separator or exists as directory,
            # append a default filename
            if output_path.endswith(('/', '\\')) or (os.path.exists(output_path) and os.path.isdir(output_path)):
                filename = "report.md"
                output_file = output_path_obj / filename
            else:
                output_file = output_path_obj
            
            # Create parent directory if it doesn't exist
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Check write permissions explicitly
            if not os.access(output_file.parent, os.W_OK):
                print(f"WARNING: No write permission for {output_file.parent}")
                # Fall back to user's home directory
                backup_file = Path.home() / "report.md"
                print(f"Falling back to: {backup_file}")
                output_file = backup_file
            
            # Write report to file
            with open(output_file, 'w') as f:
                f.write(report_content)
                
            print(f"Report successfully saved to: {output_file}")
            
        except Exception as e:
            print(f"ERROR: Could not write report: {type(e).__name__}: {str(e)}")
            print(f"Using alternative approach to display report")
            print("=" * 80)
            print(report_content)
            print("=" * 80)
    else:
        print(report_content)

 
def generate_graph_data_html_report(graph, output_path=None, node_types=None):
    """
    Generate a comprehensive HTML report on the graph data, attributes, and validation results.
    
    Parameters:
        graph: UESGraph with simulation data
        output_path: Path to save the HTML report (if None, will print to console)
        node_types: List of node types to include in the report (default: ["heating"])
    """
    from datetime import datetime
    import os
    from html import escape
    
    # Use default node types if none specified
    if node_types is None:
        node_types = ["heating"]
    
    # Function to create an HTML table from data
    def create_html_table(headers, rows):
        html = '<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">\n'
        
        # Add headers
        html += '  <thead>\n    <tr>\n'
        for header in headers:
            html += f'      <th>{escape(header)}</th>\n'
        html += '    </tr>\n  </thead>\n'
        
        # Add rows
        html += '  <tbody>\n'
        for row in rows:
            html += '    <tr>\n'
            for cell in row:
                html += f'      <td>{escape(str(cell))}</td>\n'
            html += '    </tr>\n'
        html += '  </tbody>\n</table>'
        
        return html
    
    # CSS styles for the report
    css_styles = """
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        h2 {
            color: #3498db;
            margin-top: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 5px;
        }
        h3 {
            color: #2980b9;
            margin-top: 25px;
        }
        h4 {
            color: #27ae60;
            margin-top: 20px;
        }
        p {
            margin: 15px 0;
        }
        ul {
            margin: 15px 0;
            padding-left: 30px;
        }
        table {
            width: 100%;
            margin: 20px 0;
            border-collapse: collapse;
        }
        th {
            background-color: #f2f2f2;
            text-align: left;
        }
        td, th {
            padding: 10px;
            border: 1px solid #ddd;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .code {
            font-family: monospace;
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.4;
        }
        .alert {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .info {
            background-color: #d1ecf1;
            color: #0c5460;
            padding: 10px;
            border-radius: 5px;
            margin: 15px 0;
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
    </style>
    """
    
    # Analyze node types
    def analyze_node_types(graph):
        node_types_dict = {}
        
        # Iterate through all nodes
        for _, data in graph.nodes(data=True):
            node_type = data.get('node_type', 'unknown')
            
            # Initialize node type if not seen before
            if node_type not in node_types_dict:
                node_types_dict[node_type] = {
                    'count': 0,
                    'attributes': set(),
                    'timeseries_attributes': set()
                }
            
            # Count this node
            node_types_dict[node_type]['count'] += 1
            
            # Add all attributes
            for key, value in data.items():
                node_types_dict[node_type]['attributes'].add(key)
                
                # Try to detect time series data
                if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                    try:
                        if len(value) > 1:
                            node_types_dict[node_type]['timeseries_attributes'].add(key)
                    except TypeError:
                        # Not a sequence with length
                        pass
        
        return node_types_dict
    
    # Start building HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Graph Data Report</title>
    {css_styles}
</head>
<body>
    <h1>Graph Data Report</h1>
    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Graph Summary</h2>
    <ul>
"""
    
    # Custom handling for UESGraph node counting
    for node_type in node_types:
        try:
            num_nodes = graph.number_of_nodes(node_type=node_type)
            html_content += f"        <li>Number of {escape(node_type)} nodes: {num_nodes}</li>\n"
        except Exception as e:
            html_content += f"        <li>Error counting {escape(node_type)} nodes: {escape(str(e))}</li>\n"
    
    # Handle edge counting
    try:
        num_edges = len(list(graph.edges()))
        html_content += f"        <li>Number of edges: {num_edges}</li>\n"
    except Exception as e:
        html_content += f"        <li>Error counting edges: {escape(str(e))}</li>\n"
    
    if hasattr(graph, "graph") and isinstance(graph.graph, dict) and "supply_type" in graph.graph:
        html_content += f"        <li>Supply type: {escape(graph.graph['supply_type'])}</li>\n"
    
    # Analyze node types
    node_type_analysis = analyze_node_types(graph)
    html_content += f"        <li>Node types identified: {len(node_type_analysis)}\n"
    html_content += "            <ul>\n"
    for nt, info in sorted(node_type_analysis.items()):
        html_content += f"                <li>{escape(nt)}: {info['count']} nodes</li>\n"
    html_content += "            </ul>\n        </li>\n"
    
    html_content += "    </ul>\n"
    
    #------------------------#
    # EDGE ATTRIBUTES ANALYSIS
    #------------------------#
    html_content += """
    <div class="section">
        <h2>EDGE ATTRIBUTES ANALYSIS</h2>
"""
    
    # Get actual edge attributes
    actual_edge_attrs = set()
    for u, v, data in graph.edges(data=True):
        actual_edge_attrs.update(data.keys())
    
    # Get expected attributes from schema
    expected_edge_attrs = set(EDGE_ATTRIBUTES.keys())
    
    # Calculate missing and unexpected attributes
    missing_edge_attrs = expected_edge_attrs - actual_edge_attrs
    unexpected_edge_attrs = actual_edge_attrs - expected_edge_attrs
    
    # Edge attribute summary section
    html_content += """
        <h3>Edge Attribute Summary</h3>
        <ul>
"""
    html_content += f"            <li>Total attributes in schema: {len(expected_edge_attrs)}</li>\n"
    html_content += f"            <li>Total attributes in graph: {len(actual_edge_attrs)}</li>\n"
    html_content += f"            <li>Attributes in both: {len(actual_edge_attrs & expected_edge_attrs)}</li>\n"
    html_content += f"            <li>Undocumented attributes: {len(unexpected_edge_attrs)}</li>\n"
    html_content += f"            <li>Missing attributes: {len(missing_edge_attrs)}</li>\n"
    html_content += "        </ul>\n"
    
    # Documented edge attributes section
    html_content += """
        <h3>Documented Edge Attributes</h3>
        <p>Attributes that are known and expected by this script.</p>
        
        <div class="code">
for edge in graph.edges:
    graph.edges[edge]['Attribute']
        </div>
"""
    
    # Create table of documented attributes
    if actual_edge_attrs & expected_edge_attrs:
        headers = ["Attribute", "Unit", "Type", "Description"]
        rows = []
        
        for attr in sorted(actual_edge_attrs & expected_edge_attrs):
            info = EDGE_ATTRIBUTES[attr]
            data_type = "Time series" if info["is_timeseries"] else "Static value"
            unit = info["unit"] if info["unit"] else "-"
            rows.append([attr, unit, data_type, info["description"]])
        
        html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
    else:
        html_content += '        <p class="alert">No documented attributes found.</p>\n'
    
    # Undocumented edge attributes section
    if unexpected_edge_attrs:
        html_content += """
        <h3>Undocumented Edge Attributes</h3>
"""
        
        # Get example values and counts
        attr_examples = {}
        attr_counts = {attr: 0 for attr in unexpected_edge_attrs}
        
        for u, v, data in graph.edges(data=True):
            for attr in unexpected_edge_attrs:
                if attr in data:
                    attr_counts[attr] += 1
                    if attr not in attr_examples and data[attr] is not None:
                        # Safely convert to string and truncate long values
                        try:
                            example_value = str(data[attr])[:50]
                            if len(str(data[attr])) > 50:
                                example_value += "..."
                            attr_examples[attr] = example_value
                        except:
                            attr_examples[attr] = "[Complex data]"
        
        headers = ["Attribute", "Count", "Example Value"]
        rows = []
        
        for attr in sorted(unexpected_edge_attrs):
            example = attr_examples.get(attr, "None")
            rows.append([attr, attr_counts[attr], example])
        
        html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
    
    # Missing edge attributes section
    if missing_edge_attrs:
        html_content += """
        <h3>Missing Edge Attributes</h3>
"""
        
        headers = ["Attribute", "Unit", "Type", "Description"]
        rows = []
        
        for attr in sorted(missing_edge_attrs):
            info = EDGE_ATTRIBUTES[attr]
            data_type = "Time series" if info["is_timeseries"] else "Static value"
            unit = info["unit"] if info["unit"] else "-"
            rows.append([attr, unit, data_type, info["description"]])
        
        html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
    
    html_content += "    </div>\n"  # End of edge attributes section
    
    #------------------------#
    # NODE ATTRIBUTES ANALYSIS
    #------------------------#
    html_content += """
    <div class="section">
        <h2>NODE ATTRIBUTES ANALYSIS</h2>
"""
    
    # Get all nodes of the specified types
    analyzed_nodes = []
    for _, data in graph.nodes(data=True):
        node_type_value = data.get('node_type', 'unknown')
        if node_type_value in node_types or not node_types:  # Include if in specified types or if no types specified
            analyzed_nodes.append(data)
    
    if not analyzed_nodes:
        html_content += f'        <p class="alert">No nodes of specified types {", ".join(node_types)} found.</p>\n'
    else:
        # Get actual node attributes from nodes of specified types
        actual_node_attrs = set()
        for data in analyzed_nodes:
            actual_node_attrs.update(data.keys())
        
        # Get expected attributes from schema
        expected_node_attrs = set(NODE_ATTRIBUTES.keys())
        
        # Calculate missing and unexpected attributes
        missing_node_attrs = expected_node_attrs - actual_node_attrs
        unexpected_node_attrs = actual_node_attrs - expected_node_attrs
        
        # Node attribute summary section
        html_content += """
        <h3>Node Attribute Summary</h3>
        <ul>
"""
        html_content += f"            <li>Total attributes in schema: {len(expected_node_attrs)}</li>\n"
        html_content += f"            <li>Total attributes in graph nodes: {len(actual_node_attrs)}</li>\n"
        html_content += f"            <li>Attributes in both: {len(actual_node_attrs & expected_node_attrs)}</li>\n"
        html_content += f"            <li>Undocumented attributes: {len(unexpected_node_attrs)}</li>\n"
        html_content += f"            <li>Missing attributes: {len(missing_node_attrs)}</li>\n"
        html_content += "        </ul>\n"
        
        # Documented node attributes section
        html_content += """
        <h3>Documented Node Attributes</h3>
        <p>Attributes that are known and expected by this script.</p>
        
        <div class="code">
for node in graph.nodes:
    graph.nodes[node]['Attribute']
        </div>
"""
        
        # Create table of documented attributes
        if actual_node_attrs & expected_node_attrs:
            headers = ["Attribute", "Unit", "Type", "Description"]
            rows = []
            
            for attr in sorted(actual_node_attrs & expected_node_attrs):
                info = NODE_ATTRIBUTES[attr]
                data_type = "Time series" if info["is_timeseries"] else "Static value"
                unit = info["unit"] if info["unit"] else "-"
                rows.append([attr, unit, data_type, info["description"]])
            
            html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
        else:
            html_content += '        <p class="alert">No documented attributes found.</p>\n'
        
        # Undocumented node attributes section
        if unexpected_node_attrs:
            html_content += """
        <h3>Undocumented Node Attributes</h3>
"""
            
            # Get example values and counts
            attr_examples = {}
            attr_counts = {attr: 0 for attr in unexpected_node_attrs}
            
            for data in analyzed_nodes:
                for attr in unexpected_node_attrs:
                    if attr in data:
                        attr_counts[attr] += 1
                        if attr not in attr_examples and data[attr] is not None:
                            # Safely convert to string and truncate long values
                            try:
                                example_value = str(data[attr])[:50]
                                if len(str(data[attr])) > 50:
                                    example_value += "..."
                                attr_examples[attr] = example_value
                            except:
                                attr_examples[attr] = "[Complex data]"
            
            headers = ["Attribute", "Count", "Example Value"]
            rows = []
            
            for attr in sorted(unexpected_node_attrs):
                example = attr_examples.get(attr, "None")
                rows.append([attr, attr_counts[attr], example])
            
            html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
        
        # Missing node attributes section
        if missing_node_attrs:
            html_content += """
        <h3>Missing Node Attributes</h3>
"""
            
            headers = ["Attribute", "Unit", "Type", "Description"]
            rows = []
            
            for attr in sorted(missing_node_attrs):
                info = NODE_ATTRIBUTES[attr]
                data_type = "Time series" if info["is_timeseries"] else "Static value"
                unit = info["unit"] if info["unit"] else "-"
                rows.append([attr, unit, data_type, info["description"]])
            
            html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
        
        # Node type specific attribute analysis
        html_content += """
        <h3>Node Type Specific Attributes</h3>
        <p>Analysis of attributes by node type.</p>
"""
        
        for node_type, info in sorted(node_type_analysis.items()):
            if node_type in node_types or not node_types:  # Only include specified types
                html_content += f'        <h4>{escape(node_type)} ({info["count"]} nodes)</h4>\n'
                html_content += "        <ul>\n"
                
                static_attrs = info['attributes'] - info['timeseries_attributes']
                html_content += f"            <li>Static attributes: {len(static_attrs)}</li>\n"
                html_content += f"            <li>Time series attributes: {len(info['timeseries_attributes'])}</li>\n"
                html_content += "        </ul>\n"
                
                # List static attributes
                if static_attrs:
                    html_content += "        <h5>Static Attributes:</h5>\n"
                    
                    headers = ["Attribute", "In Schema"]
                    rows = []
                    
                    for attr in sorted(static_attrs):
                        in_schema = "Yes" if attr in NODE_ATTRIBUTES else "No"
                        rows.append([attr, in_schema])
                    
                    html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
                
                # List time series attributes
                if info['timeseries_attributes']:
                    html_content += "        <h5>Time Series Attributes:</h5>\n"
                    
                    headers = ["Attribute", "In Schema"]
                    rows = []
                    
                    for attr in sorted(info['timeseries_attributes']):
                        in_schema = "Yes" if attr in NODE_ATTRIBUTES else "No"
                        rows.append([attr, in_schema])
                    
                    html_content += "        " + create_html_table(headers, rows).replace("\n", "\n        ") + "\n"
    
    html_content += "    </div>\n"  # End of node attributes section
    
    #------------------------#
    # DATA QUALITY VALIDATION
    #------------------------#
    html_content += """
    <div class="section">
        <h2>Data Quality Summary</h2>
"""
    
    # Edge validation
    html_content += """
        <h3>Edge Data Quality</h3>
        <p>Basic validation of edge attributes (checking for missing values and anomalies)</p>
"""
    
    # Count edges with missing required attributes
    required_edge_attrs = ["diameter", "length"]  # Customize based on your requirements
    missing_required = 0
    edge_issues = {}
    
    for u, v, data in graph.edges(data=True):
        edge_id = f"{u}-{v}"
        issues = []
        
        for attr in required_edge_attrs:
            if attr not in data or data[attr] is None:
                issues.append(f"Missing required attribute: {attr}")
                
        if issues:
            edge_issues[edge_id] = issues
            missing_required += 1
    
    html_content += f'        <p>Edges missing required attributes: {missing_required} of {len(list(graph.edges()))}</p>\n'
    
    # Node validation
    html_content += """
        <h3>Node Data Quality</h3>
        <p>Basic validation of node attributes (checking for missing values and anomalies)</p>
"""
    
    # Define required node attributes based on node type
    required_node_attrs = {
        "network_heating": ["position", "name"],
        "building": ["position", "name"],
        # Add other node types as needed
    }
    
    # Count nodes with issues
    nodes_with_issues = 0
    node_issues = {}
    
    for node, data in graph.nodes(data=True):
        node_type_value = data.get('node_type', 'unknown')
        
        # Skip if not in specified types
        if node_types and node_type_value not in node_types:
            continue
            
        issues = []
        
        # Check for required attributes based on node type
        if node_type_value in required_node_attrs:
            for attr in required_node_attrs[node_type_value]:
                if attr not in data or data[attr] is None:
                    issues.append(f"Missing required attribute: {attr}")
        
        # Check for time series data consistency
        for attr in data:
            if attr in NODE_ATTRIBUTES and NODE_ATTRIBUTES[attr]["is_timeseries"]:
                value = data[attr]
                # Check if this is a time series but empty or very short
                if hasattr(value, '__len__') and not isinstance(value, (str, dict)):
                    try:
                        if len(value) < 2:  # A time series should have multiple values
                            issues.append(f"Time series attribute '{attr}' has insufficient data points: {len(value)}")
                    except TypeError:
                        # Not a sequence with length
                        pass
        
        if issues:
            node_issues[node] = issues
            nodes_with_issues += 1
    
    analyzed_node_count = len(analyzed_nodes)
    html_content += f'        <p>Nodes with issues: {nodes_with_issues} of {analyzed_node_count} analyzed nodes</p>\n'
    
    html_content += "    </div>\n"  # End of data quality section
    
    # Close HTML document
    html_content += """
</body>
</html>
"""
    
    # Output report
    if output_path:
        try:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Write the report
            with open(output_path, 'w') as f:
                f.write(html_content)
            print(f"HTML report generated and saved to: {output_path}")
        except PermissionError:
            print(f"ERROR: Permission denied when writing to {output_path}")
            print("Please check that you have write permissions for this location.")
        except Exception as e:
            print(f"ERROR: Could not write to {output_path}: {str(e)}")
    else:
        print(html_content)

def plot_network(graph):
    vis = ug.Visuals(graph)
    fig = vis.show_network(show_plot=show_plot)