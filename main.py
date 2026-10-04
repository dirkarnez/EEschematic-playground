#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import sys
# get_ipython().system('{sys.executable} -m pip install openai boto3 dotenv')
# get_ipython().system('jupyter nbconvert --to script notebook.ipynb')
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import openai
import requests
import re
import boto3
import json
import os
import csv
import time
import shutil
import pandas as pd
from scipy.fft import fft, fftfreq


# In[ ]:


# get_ipython().system('sudo apt install ngspice')
# get_ipython().system('sudo ngspice --version')


# In[ ]:


from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API URL and API key
api_base_url = os.getenv('API_URL')
api_key = os.getenv('API_KEY')

#print(f"API URL: {api_base_url}")
#print(f"API Key: {api_key}")


# In[ ]:


def make_chat_completion_request(prompt):
    # The base URL for your local server (note the trailing slash)
    #print("Generated API key:", api_key)
    
    # Step 2: Configure OpenAI client to use your custom endpoint
    openai.base_url = api_base_url  # Ensure trailing slash here
    openai.api_key = api_key

    # Optional: If you need to pass organization or project headers, you can do so.
    # For example:
    # openai.organization = "YOUR_ORG_ID"
    # openai.default_project = "YOUR_PROJECT_ID"

    # Step 3: Call the Chat Completions API with streaming enabled
    print("Making ChatCompletion request with streaming enabled...")
    try:
        response = openai.chat.completions.create(
            # model="anthropic.claude-3-haiku-20240307-v1:0",
            # model="gpt-4o-mini",
            model="gemini-3.5-flash-lite",
            # model="labiium-001",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            stream=True,
            max_tokens=2500,
        )
    except Exception as e:
        print("Error making API call:", e)
        return
    
    generated_content = " "
    print("Streaming response:")
    for chunk in response:
        # Each 'chunk' is a dict similar to what the OpenAI API returns.
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="")
            generated_content += content

    return generated_content


# In[ ]:


def make_chat_completion_request_function(prompt):
    # The base URL for your local server (note the trailing slash)
    #print("Generated API key:", api_key)
    
    # Step 2: Configure OpenAI client to use your custom endpoint
    openai.base_url = api_base_url  # Ensure trailing slash here
    openai.api_key = api_key

    # Optional: If you need to pass organization or project headers, you can do so.
    # For example:
    # openai.organization = "YOUR_ORG_ID"
    # openai.default_project = "YOUR_PROJECT_ID"

    # Step 3: Call the Chat Completions API with streaming enabled
    print("Making ChatCompletion request with function calling enabled...")
    functions = [
    {
            "name": "universal_circuit_tool",
            "description": (
                "A unified tool that enables simulation and analysis of analog and mixed-signal (AMS) circuits. "
                "It supports multiple functions including DC, AC, and transient simulations, as well as performance "
                "analysis such as gain, bandwidth, phase margin, power consumption, and more."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "simulation_type": {
                        "type": "string",
                        "enum": ["dc", "ac", "transient"],
                        "description": "Type of simulation to run: DC, AC, or transient."
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": [
                            "ac_gain", "output_swing", "offset", "ICMR", "tran_gain", 
                            "bandwidth", "unity_bandwidth", "phase_margin", "power", 
                            "thd_input_range", "cmrr_tran"
                        ],
                        "description": "Type of analysis to perform on the simulation results."
                    },
                    "simulation_tool": {
                        "type": "string",
                        "enum": ["run_ngspice"],
                        "description": "Name of the SPICE simulation tool to use (e.g., run_ngspice)."
                    }
                },
                "required": []
            }
        }
]

    try:
        response = openai.chat.completions.create(
            model="gemini-3.5-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            stream=False,
            tools = [
                {
                    "type": "function",
                    "function": functions[0]  # 包裹你原本定義的 tool
                }
            ]
        )
    except Exception as e:
        print("Error making API call:", e)
        return
    
    # Step 4: Print the complete JSON response.
    print("Non-streaming response:")
    print(response)

    return response


# ## User Input

# In[ ]:


tasks_generation_question = " This is a circuit netlist, optimize this circuit with a output swing above 1.7V, input offset smaller than 0.001V, input common mode range bigger than 1.6, ac gain and transient gain above 60dB, unity bandwidth above 10000000Hz, phase margin bigger than 50 degree, power smaller than 0.05W, cmrr bigger than 100dB and thd small than -26dB"
netlist = ''' 
.title Basic amp
.include '180nm_bulk.txt'
.param Vcm = 0.9
M3 d3 in2 midp vdd pmos W=1u L=180n
M4 d4 in1 midp vdd pmos W=1u L=180n
M6 midp bias1 vdd vdd pmos W=1u L=180n

M1 d1 in1 midn 0 nmos W=1u L=180n
M2 d2 in2 midn 0 nmos W=1u L=180n
M5 midn bias2 0 0 nmos W=1u L=180n

M7 d1 g7 vdd vdd pmos W=1u L=180n
M8 d2 g7 vdd vdd pmos W=1u L=180n
M9 g7 bias3 d1 vdd pmos W=1u L=180n
M10 d10 bias3 d2 vdd pmos W=1u L=180n

M15 g7 bias5 s15 0 nmos W=1u L=180n
M16 s15 bias6 g7 vdd pmos W=1u L=180n
M17 d10 bias5 s17 0 nmos W=1u L=180n
M18 s17 bias6 d10 vdd pmos W=1u L=180n

M11 s15 bias4 d4 0 nmos W=1u L=180n
M12 s17 bias4 d3 0 nmos W=1u L=180n
M13 d4 s15 0 0 nmos W=1u L=180n
M14 d3 s15 0 0 nmos W=1u L=180n

M19 out d10 vdd vdd pmos W=1u L=180n
M20 out s17 0 0 nmos W=1u L=180n

Cc1 out d10 5p
Cc2 out s17 5p
Cl out 0 10p 
Rl out 0 1k

vbias1 bias1 0 DC 0.9
vbias2 bias2 0 DC 0.9
vbias3 bias3 0 DC 0.9
vbias4 bias4 0 DC 0.9
vbias5 bias5 0 DC 0.9
vbias6 bias6 0 DC 0.9

vdd vdd 0 1.8

Vcm cm 0 DC {Vcm}
Eidp cm in1 diffin 0 1
Eidn cm in2 diffin 0 -1
Vid diffin 0 AC 1 SIN (0 1u 10k 0 0)
.op

.end

'''


# ## Task Decpmposition

# In[ ]:


## prompt for tasks generation based on user input
tasks_generation_template = f""" 

You are an expert at analogue circuit design. You are required to decomposit tasks from initial user input. Here are some examples:

# Example 1
question = '''
.title Basic Amplifier
Vdd vdd 0 1.8V
R1 vdd out 50kOhm
M1 out in 0 0 nm l=90nm w=1um
Vin in 0 DC 0.3V AC 1
.model nm nmos level=14 version=4.8.1   

.end

This is a circuit netlist, optimize this circuit with a gain above 20dB and bandwidth above 1Mag Hz.
'''

You can answer:
type_question: '''Analysis this netlist and tell me the type of the circuit ''' 
node_question: '''This is a Spice netlist. Tell me the input and output node name. And the input voltage source name.'''
sim_question: '''This is a Spice netlist for a circuit. Simulate it and give me the ac gain, transirnt gain and bandwidth.'''
sizing_question: '''Modify the parameter in the netlist. I want the gain at 20dB and bandwidth at 1Mag Hz''' 

Return a dictionary structured with a single key named "questions" with no premable or explanation, which maps to a list. Please do not output 'json' and any symbols such as ''' .
This list contains multiple dictionaries, each representing a specific question. Each dictionary within the list has a key named by the 
name of the question and the content of the key is the content of the question in question: content format. 
    
"""
tasks_generation_SYSTEM_PROMPT = tasks_generation_template


# In[ ]:


#task generation
tasks_generation_prompt = f'''system: {tasks_generation_SYSTEM_PROMPT},
"human": "Question: {tasks_generation_question},
Netlist: {netlist}"
'''
#print(tasks_generation_prompt)
tasks = make_chat_completion_request(tasks_generation_prompt)
print(tasks)


# In[ ]:


# set type_question, node_question, sim_question, sizing_question globally
def get_tasks(tasks):
    try:
        # Step 1: Remove triple backticks (if present)
        tasks = tasks.strip().strip('`')  # Remove leading/trailing backticks and whitespace
        tasks = tasks.replace("json", "").strip()
        # Step 2: Print the cleaned input for debugging
        print("Cleaned JSON string:")
        print(tasks)

        # Step 3: Parse the JSON string into a dictionary
        tasks_output = json.loads(tasks)

        # Step 4: Process the questions
        for question_dict in tasks_output["questions"]:
            for key, value in question_dict.items():
                #print(f"{key}: {value}")  # Print or process the key-value pairs
                # assign them to globals (not recommended unless necessary)
                globals()[key] = value

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        print("Invalid JSON string:", tasks)
    except KeyError as e:
        print(f"Missing key in JSON: {e}")

get_tasks(tasks)
#print(type_question)


# In[ ]:


target_value_question = tasks_generation_question
target_value_SYSTEM_PROMPT = ''' 
You are required to get the target value of some circuit performance from user input. 
Here is an example:
User input: 
    This is a circuit netlist, optimize this circuit with a gain above 30dB and bandidth above 10MHz.
Answer: 
    'gain_target': 30dB
    'bandwidth_target': 10000000Hz
If the input contents a range such as input_range is [1, 2]V, Then please output:
    'input_range_max_target':2V
    'input_range_min_target':1V
Please reserve the sign if target value have a + or -, such as:
Return a dictionary structured with a single key named "target_values" with no premable or explanation in JSON format, which maps to a list.  
This list contains multiple dictionaries, each representing a specific question. Each dictionary within the list has a key named by the 
name of the target performance and the content of the key is the value of the target performance in performance_target: value format. 
'''
target_value_prompt = f"""
system: {target_value_SYSTEM_PROMPT}
human: 
Question: {target_value_question}
"""
target_values = make_chat_completion_request(target_value_prompt)


# In[ ]:


type_identify_template = ''' 
You are an expert at analogue circuit design. You are required to identify the circuit type from user netlist input. 
'''
type_identify_prompt = f''' System: {type_identify_template}
Netlist: {netlist},
'''
print(type_identify_prompt)


# In[ ]:


type_identified = make_chat_completion_request(type_identify_prompt)
print(type_identified)


# In[ ]:


def nodes_extract(node):
    try:
        # Step 1: Clean the input string (remove backticks, "json", and extra whitespace)
        nodes = node.strip().strip('`')  # Remove leading/trailing backticks and whitespace
        nodes = nodes.replace("json", "").strip()  # Remove the word "json" and any extra whitespace

        # Step 2: Parse the JSON string into a dictionary
        node_name = json.loads(nodes)  # Use json.loads() for parsing strings

        # Step 3: Initialize lists to store input nodes, output nodes, and source names
        input_nodes = []
        output_nodes = []
        source_names = []

        # Step 4: Iterate through the "nodes" list
        for node in node_name["nodes"]:
            if "input_node" in node:
                input_nodes.append(node["input_node"])
            elif "output_node" in node:
                output_nodes.append(node["output_node"])
            elif "source_name" in node:
                source_names.append(node["source_name"])

        # Step 5: Return the extracted lists
        return input_nodes, output_nodes, source_names

    except json.JSONDecodeError as e:
        print("Error parsing JSON:", e)
        return [], [], []
    except KeyError as e:
        print(f"Missing key in JSON: {e}")
        return [], [], []

def extract_code(text):
    regex1 = r"'''(.+?)'''" 
    regex2 = r"```(.+?)```"

    matches1 = re.findall(regex1, text, re.DOTALL)
    matches2 = re.findall(regex2, text, re.DOTALL)

    extracted_code = "\n".join(matches1 + matches2)
    lines = extracted_code.split('\n')
    cleaned_lines = []

    for line in lines:
        if '*' in line:
            line = line.split('*')[0].strip()
        elif '#' in line:
            line = line.split('#')[0].strip()
        elif ';' in line:
            line = line.split(';')[0].strip()
        elif line.startswith('verilog'):
            line = '\n'
        if line:  
            cleaned_lines.append(line)
    
    cleaned_code = "\n".join(cleaned_lines)
    return cleaned_code

node_SYSTEM_PROMPT = '''
You aim to find the circuit output nodes, input nodes and input voltage source name from the given netlist. 
Here is an example output:
{"nodes": [{"input_node": "in"},
                {"output_node": "out"},        
                {"source_name": "Vin"}]}
Please do no add any description in the output, just the dictionary.
'''
node_prompt = f""" 
system: {node_SYSTEM_PROMPT},
human: Question: {node_question}
Netlist: {netlist}
"""


# In[ ]:


nodes = make_chat_completion_request(node_prompt)
print(nodes)


# In[ ]:


input_nodes, output_nodes, source_names = nodes_extract(nodes)
print("----------------------node extract-----------------------------")
print(f"input_nodes:{input_nodes}")
print(f"output_nodes:{output_nodes}")
print(f"source_names:{source_names}")


# In[ ]:


#fuctions for different simulation type
def dc_simulation(netlist, input_name, output_node):
     end_index = netlist.index('.end\n')

     #input_nodes_str = ' '.join(input_name)
     output_nodes_str = ' '.join(output_node)

     simulation_commands = f'''
    
    .control
      dc Vcm 0 1.8 0.001        
      wrdata output/output_dc.dat {output_nodes_str}  
    .endc
     '''
     new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
     print(f"dc netlist:{new_netlist}")
     return new_netlist

def ac_simulation(netlist, input_name, output_node):
     end_index = netlist.index('.end\n')

     output_nodes_str = ' '.join(output_node)
     simulation_commands = f'''
      .control
        ac dec 10 1 10G        
        wrdata output/output_ac.dat {output_nodes_str} 
      .endc
     '''
     new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
     return new_netlist

def trans_simulation(netlist, input_name, output_node):
    end_index = netlist.index('.end\n')
    output_nodes_str = ' '.join(output_node)
    simulation_commands = f'''
      .control
        tran 50n 500u
        wrdata output/output_tran.dat {output_nodes_str} I(vdd) in1
      .endc
     '''
    new_netlist = netlist[:end_index] + simulation_commands + netlist[end_index:]
    return new_netlist

def tran_inrange(netlist):
    import re
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist, flags=re.DOTALL)
    netlist_set = ""
    for line in modified_netlist.splitlines():
        if line.startswith("Vid"):
            # Append AC 1 to the Vcm line
            netlist_set += "Vid diffin 0 AC 1 SIN (0 10u 10k 0 0)\n"
        else:
            # Keep other lines unchanged
            netlist_set += line + "\n"

    end_index = netlist_set.index('.end\n')
    simulation_commands = f'''
    .control
      set_numthread = 8
      let start_vcm = 0
      let stop_vcm = 1.85
      let delta_vcm = 0.05
      let vcm_act = start_vcm
      rm output_tran_inrange.dat

      while vcm_act <= stop_vcm
        alter Vcm vcm_act
        tran 50n 500u 
        wrdata output/output_tran_inrange.dat out in1 in2 cm 
        set appendwrite
        let vcm_act = vcm_act + delta_vcm 
      end
    .endc
     '''
    new_netlist = netlist_set[:end_index] + simulation_commands + netlist_set[end_index:]
    return new_netlist


# In[ ]:


#fuctions for different results
from scipy.signal import find_peaks
Gain_init = None
Bw_init = None
UBw_init = None
Pm_init = None
CMRR_init = None
Power_init = None
Thd_init = None
OW_init = None
Offset_init = None
ICMR_init = None

def output_swing(filename):
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("Vcm"):
            line = 'Vin1 in1 0 DC {Vcm}'
        elif line.startswith("Eidn"):
            line = 'Vin2 in2 0 DC 0.9'
        elif line.startswith("Eidp"):
            line = '\n'
        elif line.startswith("Vid"):
            line = '\n'
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)
    simulation_commands = f'''
    .control
      dc Vin1 0 1.8 0.0001
      wrdata output/output_dc_ow.dat out in1
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_ow = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]
    print(netlist_ow)
    with open('output/netlist_ow.cir', 'w') as f:
        f.write(netlist_ow)
    run_ngspice(netlist_ow,'netlist_ow' )
    data_dc = np.genfromtxt(f'output/{filename}_ow.dat', skip_header=1)
    output = data_dc[0:,1]
    in1 = data_dc[0:,3]
    d_output_d_in1 = np.gradient(output, in1)

    # Replace zero or near-zero values with a small epsilon to avoid log10(0) error
    epsilon = 1e-10
    d_output_d_in1 = np.where(np.abs(d_output_d_in1) < epsilon, epsilon, d_output_d_in1)

    # Compute gain safely
    gain = 20 * np.log10(np.abs(d_output_d_in1))
    max_index = np.where(in1 == 0.9)
    if len(max_index[0]) > 0:
        # Use the first index (assuming you want the first match)
        index = max_index[0][0]

        # Extract the corresponding Idd value
        gain_mid = gain[index]
        print(f'gain_mid = {gain_mid}')

        #print(f'iquie (Idd when input = 0.9): {iquie}')
    else:
        gain_mid = None
        #print('No matching input value found.')

    #gain_max = np.max(gain)
    indices = np.where(gain > 0.8 * gain_mid)
    output_above_threshold = output[indices]
    print(output_above_threshold)
    if output_above_threshold.size > 0:
        vout_first = output_above_threshold[0]  # First value
        vout_last = output_above_threshold[-1]  # Last value
        ow = vout_first - vout_last # Difference

        print(f"First output: {vout_first}")
        print(f"Last output: {vout_last}")
        #print(f"Difference: {vout_diff}")
    else:
        ow = 0
        print("No values found where gain > 0.8 * gain_max")
    print(f'output swing: {ow}')
    return ow

def offset(filename):
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("M4") or line.startswith("M1"):
            line = re.sub(r'\bin1\b', 'out', line)  # Replace 'in1' with 'out'
        #if not (line.startswith("Rl") or line.startswith("Cl")):  # Skip lines starting with "Rl" or "Cl"
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)
    simulation_commands = f'''
    .control
      dc Vcm 0 1.8 0.001
      wrdata output/output_dc_offset.dat out
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_offset = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]
    #print(netlist_offset)
    with open('output/netlist_offset.cir', 'w') as f:
        f.write(netlist_offset)
    run_ngspice(netlist_offset,'netlist_offset' )
    data_dc = np.genfromtxt(f'output/{filename}_offset.dat', skip_header=1)
    
    # Extract input and output values from the data
    input = data_dc[19:-19, 0]   # Skip first and last points
    output = data_dc[19:-19, 1]
    #print(input)

    # Calculate the maximum voltage offset (difference between output and input)
    voff_max = np.max(np.abs(output - input))

    print(voff_max)
    
    return voff_max

def ICMR(filename):
    with open('output/netlist.cir', 'r') as f:
        netlist_content = f.read()

    # Remove control block
    modified_netlist = re.sub(r'\.control.*?\.endc', '', netlist_content, flags=re.DOTALL)

    # Update transistor connections
    updated_lines = []
    for line in modified_netlist.splitlines():
        if line.startswith("M4") or line.startswith("M1"):
            line = re.sub(r'\bin1\b', 'out', line)  # Replace 'in1' with 'out'
        #if not (line.startswith("Rl") or line.startswith("Cl")):  # Skip lines starting with "Rl" or "Cl"
        updated_lines.append(line)
    updated_netlist = '\n'.join(updated_lines)

    # Append simulation commands
    simulation_commands = '''
    .control
      dc Vcm 0 1.8 0.001
      wrdata output/output_dc_icmr.dat out I(vdd)
    .endc
    '''
    end_index = updated_netlist.index('.end')
    netlist_icmr = updated_netlist[:end_index] + simulation_commands + updated_netlist[end_index:]

    # Write modified netlist to a new file and run simulation
    with open('output/netlist_icmr.cir', 'w') as f:
        f.write(netlist_icmr)

    run_ngspice(netlist_icmr, 'netlist_icmr')

    # Read simulation data
    data_dc = np.genfromtxt(f'output/{filename}_icmr.dat', skip_header=1)

    # Extract relevant data
    input_vals = data_dc[19:-19, 0]   # Skip first and last points
    output_vals = data_dc[19:-19, 1]
    Idd = np.abs(data_dc[19:-19, 3])
    # Find indices where input_vals is equal to 0.9
    indices = np.where(input_vals == 0.9)

    # Check if any indices were found
    if len(indices[0]) > 0:
        # Use the first index (assuming you want the first match)
        index = indices[0][0]

        # Extract the corresponding Idd value
        iquie = Idd[index]

        print(f'iquie (Idd when input = 0.9): {iquie}')
    else:
        iquie = None
        print('No matching input value found.')
    
    voff = np.abs(output_vals - input_vals)  # Offset voltage
    voff_max = np.max(np.abs(output_vals - input_vals))
    # Define thresholds 20%
    threshold_idd = iquie - iquie * 0.1

    # Find index ranges 
    # 10% for output
    indices_idd = np.where(Idd > threshold_idd)[0]  
    indices_voff = np.where(voff < voff_max * 0.9)[0]  
    print(f'idd valid = {indices_idd}')
    print(f'voff valid = {indices_voff}')
    # Handle different cases
    if len(indices_idd) > 0 and len(indices_voff) > 0:
        ic_min_idd = input_vals[indices_idd[0]]
        ic_min_voff = input_vals[indices_voff[0]]
        ic_max_idd = input_vals[indices_idd[-1]]
        ic_max_voff = input_vals[indices_voff[-1]]
        ic_min = np.max([ic_min_idd, ic_min_voff])
        ic_max = np.min([ic_max_idd, ic_max_voff])
        icmr_out = ic_max - ic_min
    elif len(indices_idd) > 0:
        ic_max = 1.8
        ic_min = input_vals[indices_idd[0]]
        icmr_out = 1.8 - ic_min
    else:
        icmr_out = 0  # No valid range found
        
    print(icmr_out)
    print(f'ic_min = {ic_min}')
    print(f'ic_max = {ic_max}')

    return icmr_out

def tran_gain(file_name):
    data_tran = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_tran.shape[1]

    # for one output node
    if num_columns == 6:
        time = data_tran[:, 0]
        out = data_tran[:, 1]

    # Find the peaks (local maxima)
        peak= np.max(out)

    # Find the troughs (local minima) by inverting the signal
        trough = np.min(out)

    # Compute the gain using the difference between average peak and average trough
        tran_gain = 20 * np.log10(np.abs(peak - trough)/0.000002)
    else:
        raise ValueError("The input file must have 2 columns.")
    
    print(f"tran gain = {tran_gain}")
    
    return tran_gain

def ac_gain(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]

    # for one output node
    if num_columns == 3:
        frequency = data_ac[:, 0]
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:    
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        gain = 20 * np.log10(np.abs(v_d[0]))
    else:
        raise ValueError("The input file must have either 3 or 6 columns.")
    
    print(f"gain = {gain}")
    
    return gain

def bandwidth(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:, 0]
    
    # for one output node
    if num_columns == 3:    
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))

    half_power_point = gain - 3
    
    indices = np.where(output >= half_power_point)[0]
    
    if len(indices) > 0:
        f_l = frequency[indices[0]]
        f_h = frequency[indices[-1]]
        bandwidth = f_h - f_l
    else:
        f_l = f_h = bandwidth = 0

    print(f"bandwidth = {bandwidth}")
    
    return bandwidth

def unity_bandwidth(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:, 0]
    
    # for one output node
    if num_columns == 3:    
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))
    # for 2 output nodes
    elif num_columns == 6:
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
        output = 20 * np.log10(v_d)
        gain = 20 * np.log10(np.abs(v_d[0]))

    half_power_point = 0
    
    indices = np.where(output >= half_power_point)[0]
    
    if len(indices) > 0:
        f_l = frequency[indices[0]]
        f_h = frequency[indices[-1]]
        bandwidth = f_h - f_l
    else:
        f_l = f_h = bandwidth = 0

    print(f"unity bandwidth = {bandwidth}")
    
    return bandwidth

def phase_margin(file_name):
    data_ac = np.genfromtxt(f'output/{file_name}.dat', skip_header=1)
    num_columns = data_ac.shape[1]
    frequency = data_ac[:,0]
    # for one output node
    if num_columns == 3:
        v_d = data_ac[:, 1] + 1j * data_ac[:, 2]   
    # for 2 output nodes
    elif num_columns == 6:    
        v_d = data_ac[:, 4] + 1j * data_ac[:, 5]
    #gain
    gain_db = 20 * np.log10(np.abs(v_d))
    #phase
    phase = np.degrees(np.angle(v_d))

    #find the frequency where gain = 0dB
    gain_db_at_0_dB = np.abs(gain_db - 0)
    index_at_0_dB = np.argmin(gain_db_at_0_dB)
    frequency_at_0_dB = frequency[index_at_0_dB]
    phase_at_0_dB = phase[index_at_0_dB]

    initial_phase = phase[0]
    tolerance = 15
    if np.isclose(initial_phase, 180, atol=tolerance):
        return phase_at_0_dB
    elif np.isclose(initial_phase, 0, atol=tolerance):
        return 180 - np.abs(phase_at_0_dB)
    else:
        return 0

def calculate_static_current(simulation_data):
    static_currents = []
    threshold=5e-7
    # calculate the difference of two time points
    for i in range(len(simulation_data)):
        current_diff = np.abs(simulation_data[i] - simulation_data[i-1])        
        if current_diff <= threshold:
            static_currents.append(simulation_data[i])

    if static_currents:
        return np.mean(static_currents)
    else:
        return None

def power(filename, vdd=1.8):
    
    data_trans = np.genfromtxt(f'output/{filename}.dat')
    num_columns = data_trans.shape[1]
    if num_columns == 3:
        iout = data_trans[:, 3]  
        Ileak = calculate_static_current(iout)
        static_power = Ileak * vdd
    
    if num_columns == 6:
        iout = data_trans[:, 3]  
        Ileak = calculate_static_current(iout)
        static_power = np.abs(Ileak * vdd)

    print(f"power = {static_power}")

    return static_power
        
def cmrr_tran(netlist):
    #avd diff mode
    ac_backup_filename = f'output/output_ac_vd.dat'
    tran_backup_filename = f'output/output_tran_vd.dat'
    shutil.copy('output/output_ac.dat', ac_backup_filename)
    shutil.copy('output/output_tran.dat', tran_backup_filename)

    data_ac_vd = np.genfromtxt(f'output/output_ac_vd.dat')
    num_columns = data_ac_vd.shape[1]
    # for one output node
    if num_columns == 3:
        frequency = data_ac_vd[:, 0]
        vd = data_ac_vd[:, 1] + 1j * data_ac_vd[:, 2]
    
    # for 2 output nodes
    elif num_columns == 6:    
        vd = data_ac_vd[:, 4] + 1j * data_ac_vd[:, 5]
    
    data_tran_vd = np.genfromtxt(f'output/output_tran_vd.dat')
    time = data_tran_vd[:, 0]
    out = data_tran_vd[:, 1]
    altitude_vd = np.max(out) - np.min(out)

    #prepare netlist for common mode test
    netlist_cmrr = " "
    for line in netlist.splitlines():
        if line.startswith("Vcm"):
            # Append AC 1 to the Vcm line
            netlist_cmrr += "Vcm cm 0 DC 0.9 SIN(0 1u 10k 0 0) AC 1\n"
        elif line.startswith("Vid"):
            # Remove AC 1 from the Vid line
            netlist_cmrr += "Vid diffin 0 DC 0\n"
        else:
            # Keep other lines unchanged
            netlist_cmrr += line + "\n"

    with open('output/netlist_CMRR.cir', 'w') as f:
        f.write(netlist_cmrr)

    #run ngspice for common mode test
    run_ngspice(netlist_cmrr, 'netlist_CMRR')

    ac_backup_filename_vc = f'output/output_ac_vc.dat'
    tran_backup_filename_vc = f'output/output_tran_vc.dat'
    shutil.copy('output/output_ac.dat', ac_backup_filename_vc)
    shutil.copy('output/output_tran.dat', tran_backup_filename_vc)

    data_ac_vc = np.genfromtxt(f'output/output_ac_vc.dat')
    
    if num_columns == 3:
        frequency = data_ac_vc[:, 0]
        vc = data_ac_vc[:, 1] + 1j * data_ac_vc[:, 2]
    
    # for 2 output nodes
    elif num_columns == 6:    
        vc = data_ac_vc[:, 4] + 1j * data_ac_vc[:, 5]
    
    data_tran_vc = np.genfromtxt(f'output/output_tran_vc.dat')
    time = data_tran_vc[:, 0]
    out_vc = data_tran_vc[:, 1]
    altitude_vc = np.max(out_vc) - np.min(out_vc)
    
    cmrr_tran = 20 * np.log10(np.abs(altitude_vd/altitude_vc))
    print(f"cmrr_tran = {cmrr_tran}")

    cmrr_ac = 20 * np.log10(np.abs(vd[0]/vc[0]))
    print(f"cmrr_ac = {cmrr_ac}")
    
    return cmrr_tran

def thd_input_range(filename):
    thd_values = []
    valid_inputs = []
    threshold_thd = -24.7
    #read origin netlist
    with open('output/netlist.cir', 'r') as file:
      netlist0 = file.read()
    #replace the simulation setting 
    netlist_inrange = tran_inrange(netlist0)
    #print(netlist_inrange)
    run_ngspice(netlist_inrange, 'netlist_inrange')

    #data preperation
    data_tran = np.genfromtxt(f'output/{filename}_inrange.dat')
    time = data_tran[:,0]
    other_data = data_tran[:, 1:]  # Extract other columns
    iteration_indices = np.where(time == 0)[0]
    batch_numbers = np.zeros_like(time, dtype=int)
    # Assign batch numbers based on iteration resets
    for i, idx in enumerate(iteration_indices):
      batch_numbers[idx:] = i
    # Create a DataFrame with batch information
    columns = ['time', 'batch'] + [f'col_{i}' for i in range(1, other_data.shape[1] + 1)]
    data_with_batches = np.column_stack((time, batch_numbers, other_data))
    df = pd.DataFrame(data_with_batches, columns=columns)

    #fft
    #plt.figure(figsize=(12, 8))
    for batch, group in df.groupby('batch'):
      time = group['time'].reset_index(drop=True).to_numpy()  
      #print(time)
      output = group['col_1'].reset_index(drop=True).to_numpy()    
      output_nodc = output - np.mean(output)
      #print(output)
      # Calculate the sampling frequency (Fs)
      time_intervals = 5e-8
      fs = 1 / time_intervals  # Sampling frequency in Hz
    
      N = len(output)  # Length of the signal
      fft_values = fft(output_nodc)
      fft_magnitude = np.abs(fft_values[:N//2])  # Take magnitude of FFT values (only positive frequencies)
      fft_freqs = fftfreq(N, d=1/fs)[:N//2]  # Corresponding frequency values (only positive frequencies)

      # Identify the fundamental frequency (largest peak)
      fundamental_idx = np.argmax(fft_magnitude)
      fundamental_freq = fft_freqs[fundamental_idx]
      fundamental_amplitude = fft_magnitude[fundamental_idx]

      # Calculate harmonic amplitudes (sum magnitudes of multiples of the fundamental frequency)
      harmonics_amplitude = 0
      for i in range(2, N // fundamental_idx):  # Start from second harmonic
          idx = i * fundamental_idx
          if idx < len(fft_magnitude):  # Ensure the index is within bounds
              harmonics_amplitude = harmonics_amplitude + fft_magnitude[idx] ** 2

      harmonics_rms = np.sqrt(harmonics_amplitude)

      # Calculate Total Harmonic Distortion (THD)
      thd_db = 20 * np.log10(harmonics_rms / fundamental_amplitude)
      thd_values.append(thd_db)
    
      if thd_db < threshold_thd:
            valid_inputs.append(np.max(group['col_7']))

    thd = np.max(thd_values)
    print(thd)
    print(valid_inputs)

    if not valid_inputs:  # Check if valid_inputs is empty
        input_ranges = [(0, 0)]  # Return default range if no valid inputs
    
    else:
        input_ranges = []  # List to store the ranges
        start = valid_inputs[0]  # Start of the current range

        for i in range(1, len(valid_inputs)):
            if valid_inputs[i] - valid_inputs[i - 1] > 0.11:
            # If the difference exceeds the threshold, close the current range
                input_ranges.append((start, valid_inputs[i - 1]))
                start = valid_inputs[i]  # Start a new range

        # Add the last range
        input_ranges.append((start, valid_inputs[-1]))

    print(input_ranges)
    
    return thd, input_ranges

def is_range_covered(outer_range, sub_ranges):
    """
    Check if an outer range is fully covered by a list of subranges.

    Args:
        outer_range (tuple): The outer range as (start, end).
        sub_ranges (list of tuples): A list of subranges as (start, end).

    Returns:
        bool: True if the outer range is fully covered by the subranges, False otherwise.
    """
    # Sort subranges by their start values
    sub_ranges = sorted(sub_ranges)
    
    # Check if the outer range is fully covered
    current_position = outer_range[0]  # Start from the beginning of the outer range
    
    for sub_range in sub_ranges:
        # If there's a gap between current position and the start of the subrange, it's not covered
        if sub_range[0] > current_position:
            return False
        
        # Extend the current covered position if the subrange extends it
        if sub_range[1] > current_position:
            current_position = sub_range[1]
        
        # If the current position exceeds the end of the outer range, it's fully covered
        if current_position >= outer_range[1]:
            return True
    
    # If we exit the loop and haven't reached the end of the outer range, it's not covered
    return False

def filter_lines(input_file, output_file):
    
    filtered_lines = []
    with open(input_file, "r") as infile:
        for line in infile:
            stripped_line = line.lstrip()  # Remove leading whitespace
            words = line.split()
            if (stripped_line.startswith("device") and len(words) > 1 and words[1].startswith('m')) or stripped_line.startswith("vth ") or stripped_line.startswith("vgs "):
                filtered_lines.append(line.strip())  # Store non-empty lines without leading/trailing spaces
    with open(output_file, "w") as outfile:
        outfile.write("\n".join(filtered_lines))  # Write all lines at once   

def convert_to_csv(input_txt, output_csv):

    headers = []
    vth_rows = []
    vgs_rows = []

    with open(input_txt, "r") as infile:
        for line in infile:
            stripped_line = line.strip()

            # If the line starts with 'device', extract the device names (excluding the word 'device')
            if stripped_line.startswith("device"):
                devices = stripped_line.split()[1:]  # Skip the word 'device'
                headers.append(devices)
        # If the line starts with 'gm', extract the gm values
            elif stripped_line.startswith("vth"):
                vth_values = stripped_line.split()[1:]  # Skip the word 'gm'
                if vth_values:  # Ensure there are gm values to add
                    vth_rows.append(vth_values)

            elif stripped_line.startswith("vgs"):
                vgs_values = stripped_line.split()[1:]  # Skip the word 'gm'
                if vgs_values:  # Ensure there are gm values to add
                    vgs_rows.append(vgs_values)
            #rows = [item for sublist in rows for item in sublist]
        #print(headers)
        vth_rows = [float(item) for sublist in vth_rows for item in sublist]
        vgs_rows = [float(item) for sublist in vgs_rows for item in sublist]
        headers = [str(item) for sublist in headers for item in sublist]

    num_columns = len(headers)
    vth_rows_2d = [vth_rows[i:i+num_columns] for i in range(0, len(vth_rows), num_columns)]
    vgs_rows_2d = [vgs_rows[i:i+num_columns] for i in range(0, len(vgs_rows), num_columns)]
    with open(output_csv, "w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        # Write the header row (device names)
        csv_writer.writerow(headers)
        # Write the gm values (rows)
        csv_writer.writerows(vth_rows_2d)   
        csv_writer.writerows(vgs_rows_2d)      

def format_csv_to_key_value(input_csv, output_txt):
    try:
        with open(input_csv, "r") as infile:
            csv_reader = csv.reader(infile)
            headers, vth_values, vgs_values = list(csv_reader)[:3]  # Read first 3 rows

        filtered_lines = [
            f"vgs - vth value of {header}: {diff:.4f}"
            for header, vth, vgs in zip(headers, vth_values, vgs_values)
            if (diff := float(vgs) - float(vth)) < 0
        ]

        with open(output_txt, "w") as outfile:
            outfile.write("\n".join(filtered_lines) if filtered_lines else "No values found where vgs - vth < 0.")

        print("Filtered output written to:", output_txt)

    except Exception as e:
        print(f"An error occurred: {e}")

def read_txt_as_string(file_path):

    try:
        with open(file_path, "r") as file:
            content = file.read()
        return content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
def run_ngspice(circuit, filename):
    output_file = 'output/op.txt'
    with open(f'output/{filename}.cir', 'w') as f:
        f.write(circuit)

    try:
        result = subprocess.run(['ngspice', '-b', f'output/{filename}.cir'], capture_output=True, text=True)
        ngspice_output = result.stdout
        with open(output_file, "w") as f:
            f.write(ngspice_output)
    except Exception as e:
        ngspice_output = f"Error running NGspice: {str(e)}"

    print("NGspice output:")
    #print(ngspice_output)

def tool_calling(tool_chain):
    global Gain_init, Bw_init, Pm_init, Dc_Gain_init,Tran_Gain_init, CMRR_init, Power_init, InputRange_Init,Thd_init, OW_init, Offset_init, UBw_init, ICMR_init
    gain = None
    Dc_gain = None
    tr_gain = None
    ow = None
    Offset = None
    bw = None
    ubw = None
    pm = None
    cmrr = None
    pr = None
    ir = None
    thd = None
    icmr = None
    sim_netlist = netlist

    input_txt = "output/op.txt"   # Replace with your actual input file
    filtered_txt = "output/vgscheck.txt"
    output_csv = "output/vgscheck.csv"
    output_txt = "output/vgscheck_output.txt"
    
    for tool_call in tool_chain['tool_calls']:
        if tool_call['name'].lower() == "dc_simulation":
            sim_netlist = dc_simulation(sim_netlist, source_names, output_nodes)

        elif tool_call['name'].lower() == "ac_simulation":
            sim_netlist = ac_simulation(sim_netlist, source_names, output_nodes)
            print(f"ac_netlist:{sim_netlist}")

        elif tool_call['name'].lower() == "transient_simulation":
            sim_netlist = trans_simulation(sim_netlist, source_names, output_nodes)

        elif tool_call['name'].lower() == "run_ngspice":
            run_ngspice(sim_netlist, 'netlist')
            filter_lines(input_txt, filtered_txt)
            convert_to_csv(filtered_txt, output_csv)
            format_csv_to_key_value(output_csv, output_txt)
            vgscheck = read_txt_as_string(output_txt)
            #print(vgscheck)

        elif tool_call['name'].lower() == "ac_gain":   
            gain = ac_gain('output_ac')
            Gain_init = gain
            print(f"ac_gain result: {gain}")   

        elif tool_call['name'].lower() == "output_swing":   
            ow = output_swing('output_dc')
            OW_init = ow
            print(f"output swing result: {ow}") 
        
        elif tool_call['name'].lower() == "icmr":   
            icmr = ICMR('output_dc')
            ICMR_init = icmr
            print(f"input common mode voltage result: {icmr}")
        
        elif tool_call['name'].lower() == "offset":   
            Offset = offset('output_dc')
            Offset_init = Offset
            print(f"input offset result: {Offset}")

        elif tool_call['name'].lower() == "tran_gain":   
            tr_gain = tran_gain('output_tran')
            Tran_Gain_init = tr_gain
            print(f"tran_gain result: {tran_gain}")

        elif tool_call['name'].lower() == "bandwidth":
            bw = bandwidth('output_ac')
            Bw_init = bw
            print(f"bandwidth result: {bw}")

        elif tool_call['name'].lower() == "unity_bandwidth":
            ubw = unity_bandwidth('output_ac')
            UBw_init = ubw
            print(f"unity bandwidth result: {ubw}")

        elif tool_call['name'].lower() == "phase_margin":
            pm = phase_margin('output_ac')
            Pm_init = pm
            print(f"phase margin: {pm}")

        elif tool_call['name'].lower() == "cmrr_tran":
            cmrr = cmrr_tran(sim_netlist)
            CMRR_init = cmrr
            print(f"cmrr: {cmrr}")

        elif tool_call['name'].lower() == "power":
            pr = power('output_tran')
            Power_init = pr
            print(f"power: {pr}")
        
        elif tool_call['name'].lower() == "thd_input_range":
            thd, ir= thd_input_range('output_tran')
            Thd_init = thd
            InputRange_Init = ir
            print(f"thd is {thd}")

    sim_output = f"Transistors below vth: {vgscheck}," +  f"ac_gain is {gain}, " + f"tran_gain is {tr_gain}, " +  f"output swing is {ow}, " +  f"input offset is {Offset}, " +  f"input common mode voltage range is {icmr}, " + f"unity bandwidth is {ubw}, " + f"phase margin is {pm}, " + f"power is {pr}, " + f"cmrr is {cmrr}," + f"thd is {thd},"
    
    return sim_output, sim_netlist

def extract_number(value):
    # Convert the value to a string (in case it's an integer or None)
    value_str = str(value) if value is not None else "0"
    
    # Use regex to find all numeric values (including decimals)
    match = re.search(r"[-+]?\d*\.\d+|\d+", value_str)
    if match:
        return float(match.group(0))
    return None


# In[ ]:


simulation_function_explanation = """
    
You should decide which tools to use from user input and think about the question and return an array for functions. 
When user ask to simulate or give specfic circuit performance, please think about which types of simulation should be done and which function in simulation functions can do this according to user requirement. e.g. Target is ac gain and transient gain, then return the type sc_simulation and tran_simulation.
e.g. Target is output swing and offset, then return the type dc_simulation.
Then, the netlist should be simulated by simulation tools run_ngspice. The should be only called once.
Finaly, to analysis the output data file, some specfic functions should be done to calculate and give the final result. Choose the analysis functions to do this.

Example:
human: This is a Spice netlist for a circuit. Simulate it and give me the ac gain, unity gain bandwidth and offset.
Assistant:
Doing simuation_type: ac and dc simulation by function calling
Using tool named ngspice by function calling run_ngspice
I want to have ac gain, unity bandwidth, and input offset.

"""


# In[ ]:


sim_prompts_gen = f''' 
system: {simulation_function_explanation}
human: {sim_question}
'''
sim_prompts = make_chat_completion_request(sim_prompts_gen)


# In[ ]:


tool = make_chat_completion_request_function(sim_prompts)


# In[ ]:


def extract_tool_data(tool):
    """Extract tool call data from the tool object."""
    message = tool.choices[0].message
    tool_call = message.tool_calls[0]
    arguments = tool_call.function.arguments
    formatted_arguments = "[" + arguments.replace("}{", "},{") + "]"
    return json.loads(formatted_arguments)

def format_simulation_types(tool_data_list):
    """Format unique simulation types."""
    unique_sim_types = set()
    formatted_output = []
    for tool_data in tool_data_list:
        sim_type = tool_data["simulation_type"]
        if sim_type not in unique_sim_types:
            unique_sim_types.add(sim_type)
            formatted_output.append({"name": f"{sim_type}_simulation"})
    return formatted_output

def format_simulation_tools(tool_data_list):
    """Format unique simulation tools."""
    unique_sim_tools = set()
    formatted_output = []
    for tool_data in tool_data_list:
        sim_tool = tool_data["simulation_tool"]
        if sim_tool not in unique_sim_tools:
            unique_sim_tools.add(sim_tool)
            formatted_output.append({"name": sim_tool})
    return formatted_output

def format_analysis_types(tool_data_list):
    """Format analysis types."""
    formatted_output = []
    for tool_data in tool_data_list:
        if isinstance(tool_data["analysis_type"], str):
            tool_data["analysis_type"] = tool_data["analysis_type"].split(",")
        formatted_output.extend([{"name": analysis} for analysis in tool_data["analysis_type"]])
    return formatted_output

def combine_results(sim_types, sim_tools, analysis_types):
    """Combine simulation types, tools, and analysis types into one list."""
    return sim_types + sim_tools + analysis_types

# Example usage
#tool = ...  # Replace with your actual tool object

# Step 1: Extract tool data
tool_data_list = extract_tool_data(tool)

print(f"!!!!!!{tool_data_list}!!!")

# Step 2: Format simulation types, tools, and analysis types
formatted_sim_types = format_simulation_types(tool_data_list)
formatted_sim_tools = format_simulation_tools(tool_data_list)
formatted_analysis_types = format_analysis_types(tool_data_list)

# Step 3: Combine results
tool_chain_row = combine_results(formatted_sim_types, formatted_sim_tools, formatted_analysis_types)
tool_chain = {"tool_calls": tool_chain_row}
# Step 4: Print the combined output
print(tool_chain)
print(type(tool_chain))


# In[ ]:


print("----------------------function used-----------------------------")
print(tool_chain)
sim_output, sim_netlist = tool_calling(tool_chain)
print("-------------------------result---------------------------------")
print(sim_output)
print("-------------------------netlist---------------------------------")
print(sim_netlist)


# ## Optimization

# In[ ]:


analysing_system_prompt = f""" 
You are an expert at analogue circuit design. You are required to design CMOS amplifier. 
Valid input range is the input that can generate output without to much distortion.
Please analysis their difference, and conclude on the relationship between specific parameters and performance and tell me the conclusion. e.g. Increase W of current mirror can increase the gain.

Let's think step by step to analysis the relationship:
Step1: detect the parameter difference in the provided netlist and output which were changed.
Step2: detect the performance difference.
Step3: Analysis the performance with target values, are they passed or not, or close to the target. Please notice that all the value given to you is for the standerd unit, e.g bandwidth = 100 means 100Hz, power = 0.01 means 0.01W.
Step4: Suggest possible relationship between different parameters and performance. You need to consider all the performance given to you. 
Please consider each subcircuit (differential pair, current mirror, current source, load...) and each parameters (W, L, C...) seperately because they are always with different size and have different impact on different performance.

You may output in this format:

Assistant: 
According to pervious results, I find that increase(decrease, or maintain) ... of differential pair (current mirror, current source, C)can increase(decrease, or maintain) gain.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)bandwidth.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)phase margin.
increase(decrease, or maintain) ... of differential pair (current mirror, current source, C) can increase(decrease, or maintain)the valid input range.
"""


# In[ ]:


sizing_Question = f"Currently, {sim_output}. " + sizing_question
#print(sizing_SYSTEM_PROMPT)
print(sizing_Question)


# In[ ]:


optimising_system_prompt = f""" 
You are an expert at analogue circuit design. Please generate a detailed circuit optimisation guide based on the actual situation provided to you. You should analysis the performance and the difference between component parameters. You should think about the relationship between parameters and performance. 
Please consider W, L, W/L, C, R seperately due to their different influence on different performance. You should know that vgs for pmos is Vs > Vg, vgs for pmos is vs - vg , so if vgs - vth for a pmos is small than 0, then you should take measures to make vg smaller or vs higher.

Let's think step by step:
Step 1: Detect the vgs -vth given to you, the value for all the transistor should be above 0, if not, please detect which are the transistor, the transistor type(p or n) and the size fo these transistors and bias voltage connected, then make changes to bias them correctly.
Step 2: Detect which performances should be improved according to current result and the target performance. Please notice that all the value given to you is for the standerd unit, e.g bandwidth = 100 means 100Hz, power = 0.01 means 0.01W.
Step 3: The difference between size. 
Step 4: The different between performance of the results given to you. 
Step 5: According to the difference between size and formar performance, current performance and target performance, please suggest how to adjust the size to reach the target. You should think about the circuit type and the function of each part to reach the target. The performance have a 5% tolerance. Please make sure the W of pmos is at least 2 times bigger than that of nmos.

An Example for you to referance: 
According to my observation, gain, phase margin and cmrr didn't reach the target. So I will focus on improving this performance.
According to the former results, the increase of W of M1 can increase the overall gain from 45dB to 47dB and the decrease of it decrease the overall gain from 47dB to 30dB, so I suggest increase W to get a higher gain.
The increase of L from 220n to 320n cause a decrease in gain from 45dB to 20dB, so I suggest a further decrease in L to get a higher gain...
"""


# In[ ]:


sizing_SYSTEM_PROMPT = f"""
You are an expert at analogue circuit design. You are required to size CMOS amplifier. 
Please size the devices based on the suggestions given to you and take the following constrains into consideration.
"""

sizing_output_template = f""" 
Design Constrains:
1. Do not change CL, RL, vdd and input cm voltage.
2. Please make sure the W of pmos is at least 2 times bigger than that of nmos.
3. Please always change L with W to maintain a proper ratio. 
4. The range for W is [180n, 500u], the range for L is [180n, 3u]. Please remain reasonable adjustment.
5. Please return the complete netlist, and please do not change the simulation settings.

Finally, change these value in the netlist for further simulation, only one netlist is enough. 
Return the netlist without comment between ''' '''

You may output in this format:

Assistant: 
According to the suggestion, I decide to change W of M.. from 1u. to 2u .

Based on the performance and suggestion, I modified the netlist.
Here's the modified netlist with the proposed changes:
'''
...
'''
"""


# In[ ]:


def optimization(tools, target_values, sim_netlist):
    max_iterations = 25
    tolerance = 0.05  # 5% tolerance
    iteration = 0
    converged = False

    gain_output = None
    tr_gain_output = None
    dc_gain_output = None
    bw_output = None
    ubw_output = None
    ow_output = None
    pm_output = None
    cmrr_output = None
    pr_output = None
    thd_output = None
    offset_output = None
    icmr_output = None

    input_txt = "output/op.txt"   # Replace with your actual input file
    filtered_txt = "output/vgscheck.txt"
    output_csv = "output/vgscheck.csv"
    output_txt = "output/vgscheck_output.txt"

    opti_output = None
    opti_netlist = sim_netlist
    previous_results = [f"{sim_output}, " + f",the netlist is {opti_netlist}"]

    gain_output_list = []
    dc_gain_output_list = []
    tr_gain_output_list = []
    bw_output_list = []
    ubw_output_list = []
    ow_output_list = []
    pm_output_list = []
    pr_output_list = []
    cmrr_output_list = []
    thd_output_list = []
    offset_output_list = []
    icmr_output_list = []
    
    target_values = target_values.strip().strip('`')  # Remove leading/trailing backticks and whitespace
    target_values = target_values.replace("json", "").strip()  # Remove the word "json" and any extra whitespace

    # Step 2: Parse the JSON string into a dictionary
    target_output = json.loads(target_values)
    for dict in target_output["target_values"]:
        for key, value in dict.items():
            globals()[key] = value
    
    gain_target = extract_number(globals().get('ac_gain_target', '0')) if 'ac_gain_target' in globals() else None
    bandwidth_target = extract_number(globals().get('bandwidth_target', '0')) if 'bandwidth_target' in globals() else None
    unity_bandwidth_target = extract_number(globals().get('unity_bandwidth_target', '0')) if 'unity_bandwidth_target' in globals() else None
    phase_margin_target = extract_number(globals().get('phase_margin_target', '0')) if 'phase_margin_target' in globals() else None
    tr_gain_target = extract_number(globals().get('transient_gain_target', '0')) if 'transient_gain_target' in globals() else None
    input_offset_target = extract_number(globals().get('input_offset_target', '0')) if 'input_offset_target' in globals() else None
    output_swing_target = extract_number(globals().get('output_swing_target', '0')) if 'output_swing_target' in globals() else None
    pr_target = extract_number(globals().get('power_target', '0')) if 'power_target' in globals() else None
    cmrr_target = extract_number(globals().get('cmrr_target', '0')) if 'cmrr_target' in globals() else None
    thd_target = extract_number(globals().get('thd_target', '0')) if 'thd_target' in globals() else None
    icmr_target = extract_number(globals().get('input_common_mode_range_target', '0')) if 'input_common_mode_range_target' in globals() else None


    gain_pass = True if gain_target not in globals() or gain_target is None else False
    tr_gain_pass = True if tr_gain_target not in globals() or tr_gain_target is None else False
    dc_gain_pass = True if tr_gain_target not in globals() or tr_gain_target is None else False
    ow_pass = True if output_swing_target not in globals() or output_swing_target is None else False
    bw_pass = True if bandwidth_target not in globals() or bandwidth_target is None else False
    ubw_pass = True if unity_bandwidth_target not in globals() or unity_bandwidth_target is None else False
    pm_pass = True if phase_margin_target not in globals() or phase_margin_target is None else False
    pr_pass = True if pr_target not in globals() or pr_target is None else False
    cmrr_pass = True if cmrr_target not in globals() or cmrr_target is None else False
    thd_pass = True if thd_target not in globals() or thd_target is None else False
    input_offset_pass = True if input_offset_target not in globals() or input_offset_target is None else False
    icmr_pass = True if icmr_target not in globals() or icmr_target is None else False

    sizing_Question = f"Currently, {sim_output}. " + sizing_question
    while iteration < max_iterations and not converged:
        time.sleep(10)
        print(f"----------------------iter = {iteration}-----------------------------")
        #print(f"previous_results:{previous_results}")
        #######################observation#########################
        analysising_prompt = f""" 
            system:{analysing_system_prompt}
            human: 
            Previous:{previous_results}
            Target performance:{sizing_question}
        """
        analysis = make_chat_completion_request(analysising_prompt)
        print(analysis)
        #analysis_content = analysis["text"]
        ##########################analysis#######################
        time.sleep(10)
        optimising_prompt = f""" 
            system:{optimising_system_prompt}
            human: 
            Observation:{analysis}
            Previous:{previous_results}
        """
        optimising = make_chat_completion_request(optimising_prompt)
        print(optimising)
        ###########################sizing/optimising######################
        time.sleep(10)
        sizing_prompt = f""" 
        system:{sizing_SYSTEM_PROMPT}
        human: 
        Question: {sizing_Question}
        Netlist: {opti_netlist}
        Please follow the instructions to update parameters in netlist:{optimising} 
        Please follow the design restrain : {sizing_output_template}
        """
        modified = make_chat_completion_request(sizing_prompt)
        print(modified)
        modified_output = modified
        print("----------------------Modified-----------------------------")
        print(modified_output)
        time.sleep(10)
        ###########################function calling######################
        print("------------------------result-----------------------------")
        for tool_call in tools['tool_calls']:
            opti_netlist = extract_code(modified_output)
            print(opti_netlist)

            if tool_call['name'].lower() == "run_ngspice":
                run_ngspice(opti_netlist, 'netlist')
                print("running ngspice")
                filter_lines(input_txt, filtered_txt)
                convert_to_csv(filtered_txt, output_csv)
                format_csv_to_key_value(output_csv, output_txt)
                vgscheck = read_txt_as_string(output_txt)
        
            elif tool_call['name'].lower() == "ac_gain":
                #run_ngspice(sim_netlist)
                gain_output = ac_gain('output_ac')
                #print(f"ac_gain result: {gain_output}") 

            elif tool_call['name'].lower() == "dc_gain":
                #run_ngspice(sim_netlist)
                dc_gain_output = dc_gain('output_dc')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "output_swing":
                #run_ngspice(sim_netlist)
                ow_output = output_swing('output_dc')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "offset":
                #run_ngspice(sim_netlist)
                offset_output = offset('output_dc')

            elif tool_call['name'].lower() == "icmr":
                #run_ngspice(sim_netlist)
                icmr_output = ICMR('output_dc')

            elif tool_call['name'].lower() == "tran_gain":
                #run_ngspice(sim_netlist)
                tr_gain_output = tran_gain('output_tran')
                #print(f"ac_gain result: {gain_output}")

            elif tool_call['name'].lower() == "bandwidth":
                #run_ngspice(sim_netlist)
                bw_output = bandwidth('output_ac')
                #print(f"bandwidth result: {bw_output}")

            elif tool_call['name'].lower() == "unity_bandwidth":
                #run_ngspice(sim_netlist)
                ubw_output = unity_bandwidth('output_ac')
                #print(f"bandwidth result: {bw_output}")

            elif tool_call['name'].lower() == "phase_margin":
                #run_ngspice(sim_netlist)
                pm_output = phase_margin('output_ac')
                #print(f"phase margin result: {pm_output}")

            elif tool_call['name'].lower() == "power":
                #run_ngspice(sim_netlist)
                pr_output = power('output_tran')
                #print(f"phase margin result: {pm_output}")

            elif tool_call['name'].lower() == "thd_input_range":
                #run_ngspice(sim_netlist)
                thd_output, ir_output = thd_input_range('output_tran')

            elif tool_call['name'].lower() == "cmrr_tran":
                #run_ngspice(sim_netlist)
                cmrr_output = cmrr_tran(opti_netlist)

            
        
        #print(vgscheck)
            
        opti_output = f"Transistors below vth: {vgscheck}," + f"ac_gain is {gain_output}, " + f"tran_gain is {tr_gain_output}, " + f"output_swing is {ow_output}, " + f"input offset is {offset_output}, " + f"input common mode voltage range is {icmr_output}, "  + f"unity bandwidth is {ubw_output}, " + f"phase margin is {pm_output}, " + f"power is {pr_output}, " + f"cmrr is {cmrr_output}," + f"thd is {thd_output}," 

        #save the output value in a list
        gain_output_list.append(gain_output)
        tr_gain_output_list.append(tr_gain_output)
        dc_gain_output_list.append(dc_gain_output)
        ow_output_list.append(ow_output)
        bw_output_list.append(bw_output)
        ubw_output_list.append(ubw_output)
        pm_output_list.append(pm_output)
        pr_output_list.append(pr_output)
        cmrr_output_list.append(cmrr_output)
        thd_output_list.append(thd_output)
        offset_output_list.append(offset_output)
        icmr_output_list.append(icmr_output)
                    
        print(opti_output)

        #comparison
        if gain_target is not None:
            if gain_output >= gain_target - gain_target * tolerance:
                gain_pass = True
            else:
                gain_pass = False

        if tr_gain_target is not None:
            if tr_gain_output >= tr_gain_target - tr_gain_target * tolerance:
                tr_gain_pass = True
            else:
                tr_gain_pass = False

        if output_swing_target is not None:
            if ow_output >= output_swing_target - output_swing_target * tolerance:
                ow_pass = True
            else:
                ow_pass = False

        if input_offset_target is not None:
            if offset_output <= input_offset_target - input_offset_target * tolerance:
                input_offset_pass = True
            else:
                input_offset_pass = False     

        if icmr_target is not None:
            if icmr_output >= icmr_target - icmr_target * tolerance:
                icmr_pass = True
            else:
                icmr_pass = False   
        
        if bandwidth_target is not None:
            if bw_output >= bandwidth_target - bandwidth_target * tolerance:
                bw_pass = True
            else:
                bw_pass = False

        if unity_bandwidth_target is not None:
            if ubw_output >= unity_bandwidth_target - unity_bandwidth_target * tolerance:
                ubw_pass = True
            else:
                ubw_pass = False

        if phase_margin_target is not None:
            if pm_output >= phase_margin_target - phase_margin_target * tolerance:
                pm_pass = True
            else:
                pm_pass = False

        if pr_target is not None:
            if pr_output <= pr_target + pr_target * tolerance:
                pr_pass = True
            else:
                pr_pass = False

        if cmrr_target is not None:
            if cmrr_output >= cmrr_target - cmrr_target * tolerance:
                cmrr_pass = True
            else:
                cmrr_pass = False

        if thd_target is not None:
            if thd_output <= thd_target + np.abs(thd_target) * tolerance:
                thd_pass = True
            else:
                thd_pass = False


        if gain_pass and ubw_pass and pm_pass and tr_gain_pass and pr_pass and cmrr_pass and dc_gain_pass and thd_pass and ow_pass and input_offset_pass and icmr_pass:
            converged = True

        sizing_Question = f"Currently,{opti_output}" + sizing_question
        
        iteration += 1
        previous_results.append(f"Currently, {opti_output}, the netlist is {opti_netlist}")
        if len(previous_results) > 3:
            previous_results.pop(0)  # Remove the oldest result

        print(f"gain_target:{gain_target}, tr_gain_target:{tr_gain_target},output_swing_target:{output_swing_target}, input_offset_target:{input_offset_target}, icmr_target:{icmr_target}, unity_bandwidth_target:{unity_bandwidth_target}, phase_margin_target:{phase_margin_target}, power_target:{pr_target}, cmrr_target:{cmrr_target}, thd_target:{thd_target}")
        print(f"gain_pass:{gain_pass},tr_gain_pass:{tr_gain_pass},output_swing_pass:{ow_pass},input_offset_pass:{input_offset_pass}, icmr_pass:{icmr_pass}, unity_bandwidth_pass:{ubw_pass}, phase_margin_pass:{pm_pass}, power_pass:{pr_pass}, cmrr_pass:{cmrr_pass} , thd_pass:{thd_pass}")
    ##########################################################################################################
    # save the value in file
    file_empty = not os.path.exists('output/g13.csv') or os.stat('output/g13.csv').st_size == 0
    with open('output/g13.csv', 'a', newline='') as csvfile:
        fieldnames = ['iteration', 'gain_output', 'tr_gain_output', 'output_swing_output', 'input_offset_output',  'icmr_output', 'ubw_output', 'pm_output', 'pr_output', 'cmrr_output', 'thd_output' ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if file_empty:
            writer.writeheader()
        writer.writerow({'iteration': 0, 'gain_output': Gain_init,'tr_gain_output': Tran_Gain_init, 'output_swing_output': OW_init, 'input_offset_output': Offset_init, 'icmr_output': ICMR_init, 'ubw_output': UBw_init, 'pm_output': Pm_init, 'pr_output':Power_init, 'cmrr_output': CMRR_init, 'thd_output': Thd_init})
        for i in range(len(gain_output_list)):
            writer.writerow({'iteration': i+1, 'gain_output': gain_output_list[i], 'tr_gain_output': tr_gain_output_list[i], 'output_swing_output': ow_output_list[i], 'input_offset_output': offset_output_list[i], 'icmr_output': icmr_output_list[i], 'ubw_output': ubw_output_list[i], 'pm_output': pm_output_list[i], 'pr_output': pr_output_list[i], 'cmrr_output': cmrr_output_list[i], 'thd_output': thd_output_list[i] })

    return {'converged': converged, 
            'iterations': iteration, 
            'gain_output': gain_output if 'gain_output' in locals() else None, 
            'tr_gain_output': tr_gain_output if 'tr_gain_output' in locals() else None,
            'output_swing_output': ow_output if 'ow_output' in locals() else None,
            'input_offset_output': offset_output if 'offset_output' in locals() else None,
            'ubw_output': ubw_output if 'ubw_output' in locals() else None,
            'pm_output':pm_output if 'pm_output' in locals() else None,
            'pr_output':pr_output if 'pr_output' in locals() else None,
            'cmrr_output':cmrr_output if 'cmrr_output' in locals() else None,
            'icmr_output':icmr_output if 'icmr_output' in locals() else None,
            'thd_output':thd_output if 'thd_output' in locals() else None}, opti_netlist


# In[ ]:


def run_multiple_optimizations(target_values, sim_netlist, num_runs=1):
    results = []  # List to store results of each run

    for i in range(num_runs):
        print(f"Starting optimization run {i + 1}")
        result, opti_netlist = optimization(tool_chain, target_values, sim_netlist)
        results.append(result)  # Append the result of each run to the results list
        print(f"Run {i + 1} result: {result}")
        print("----------------------\n")    
results = run_multiple_optimizations(target_values, sim_netlist)


# In[ ]:


# save the netlist
#print("Summary of all optimization runs:")
source_file = 'output/netlist.cir'
destination_file = f'output/netlist_g13/a2.cir'
shutil.copyfile(source_file, destination_file)
print(f"Netlist copy to {destination_file}")


# In[ ]:


df = pd.read_csv('output/g13.csv')
df['batch'] = (df['iteration'] == 0).cumsum()
df['bw_output_dB'] = 20 * np.log10(df['ubw_output'] + 1e-9)

# Filter the DataFrame for the first 5 batches
filtered_df = df[df['batch'] <= 5]

# Create a 4x2 grid of subplots
fig, axes = plt.subplots(5, 2, figsize=(10, 10))  # Adjust figure size for 4x2 layout
fig.subplots_adjust(hspace=0.1, wspace=0.3)  # Adjust spacing between subplots

# Generate a list of colors
colors = plt.cm.viridis(np.linspace(0, 0.9, filtered_df['batch'].max() + 1))
# Function to plot in a subplot
def plot_subplot(ax, x, y, xlabel, ylabel, ylim_min, ylim_max, fill_range=None, fill_label=None, log_scale=False):
    for batch, group in filtered_df.groupby('batch'):
        ax.plot(group[x], group[y], marker='o', color=colors[batch], label=f'Attempt {batch}', markersize=4, linewidth=1)
    ax.set_xlim(filtered_df[x].min() - 1, filtered_df[x].max() + 1)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    #ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(False)
    if fill_range:
        ax.fill_between(ax.get_xlim(), fill_range[0], fill_range[1], color='blue', alpha=0.1, label=fill_label)
    if log_scale:
        ax.set_yscale('log')
    #ax.legend(fontsize=10)

# Plot each metric in a subplot
#plot_subplot(axes[0, 0], 'iteration', 'gain_output', 'Iterations', 'Gain (dB)', 'Gain', 0, 120, fill_range=(65, 85), fill_label='Target Range')
plot_subplot(axes[1, 0], 'iteration', 'pm_output', 'Iterations', 'Phase Margin (°)', -5, 100, fill_range=(55, 100), fill_label='Target Range')
plot_subplot(axes[0, 1], 'iteration', 'bw_output_dB', 'Iterations', 'UGBW (dB)', 50, 20 * np.log10(1e7) + 20, fill_range=(20 * np.log10(1e7), 20 * np.log10(1e7) + 20), fill_label='Target Range')
plot_subplot(axes[0, 0], 'iteration', 'tr_gain_output', 'Iterations', 'Gain (dB)', -30, 90, fill_range=(65, 90), fill_label='Target Range')
plot_subplot(axes[1, 1], 'iteration', 'pr_output', 'Iterations', 'Power (W)', -0.0005, 0.015, fill_range=(-0.0005, 1e-2), fill_label='Target Range')
plot_subplot(axes[3, 0], 'iteration', 'cmrr_output', 'Iterations', 'CMRR (dB)', 40, 150, fill_range=(100 * 0.95, 150), fill_label='Target Range')
plot_subplot(axes[2, 0], 'iteration', 'output_swing_output', 'Iterations', 'Output Swing (V)', 0, 1.8, fill_range=(1.75 * 0.95, 1.8), fill_label='Target Range')
plot_subplot(axes[3, 1], 'iteration', 'thd_output', 'Iterations', 'THD (dB)', -60, -15, fill_range=(-60, -26), fill_label='Target Range')
plot_subplot(axes[2, 1], 'iteration', 'input_offset_output', 'Iterations', 'Offset (V)', -0.001, 0.015, fill_range=(-0.001, 0.001), fill_label='Target Range')
plot_subplot(axes[4, 0], 'iteration', 'icmr_output', 'Iterations', 'ICMR', 0, 1.8, fill_range=(1.6, 1.8), fill_label='Target Range')

labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
#for i, ax in enumerate(axes.flat):
    #ax.text(1, 0.15, labels[i], transform=ax.transAxes, fontsize=14, va='top', ha='right')

for ax in axes[:-1, :].flat:
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_xlabel('')  # Remove x-axis label

# Create a common legend
handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one subplot
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=6, fontsize=10)

# Save the figure
plt.savefig('railtorail_subplots_4x2.pdf', format='pdf', bbox_inches='tight')
plt.show()

