# ABM-Thesis
Improving activity schedules of an agent-based model with bus travel and grocery shopping using Python

Code for a study that aims to improve an agent-based model (ABM) which quantifys diurnal NO2 exposure based on daily activity schedules with the integration of bus travel and grocery shopping trips. The goal is a better understanding of the consequences of long-term exposure on the public health.

## How to use it
1. Download files
2. Check folder names in *streamline_osmnx_4_improved.py*
3. Run *streamline_osmnx_4_improved.py* -> output: activity schedules
4. Check folder names in *calc_exposure4_improved.py*
5. Run *calc_exposure4_improved.py* -> output: exposure data
6. Check folder names in *calc_exposure4_improved_plot.py*
7. Run *calc_exposure4_improved_plot.py* -> output: exemplary plot for two agents

## Input data files
The files are stored here: https://1drv.ms/f/s!AhrgTI5gv_meiCL65MDP39LPVifD?e=yHT8ai <br>

*input* is the folder for all input data files. <br>
*results1* is the standard folder to save the results. <br>
*r1_improved* inside *results1* is an exemplary folder name to save the results of an individul simulation run. <br>

## Python files
*streamline_osmnx_4_improved.py* is the programm to **simulate** the activity schedules of the agents. <br>
*calc_exposure4_improved.py* is the programm to **calculate** the exposure of the agents.  <br>
*calc_exposure4_improved_plot.py* is the programm to **plot** the exposure of the agents.  <br>
*model_functions4_improved.py* contains the **functions** used in the programs. <br>
