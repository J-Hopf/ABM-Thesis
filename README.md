# ABM-Thesis
Improving activity schedules of an agent-based model with bus travel and grocery shopping using Python

Code for a study that aims to improve an agent-based model (ABM) which quantifys diurnal NO2 exposure based on daily activity schedules with the integration of bus travel and grocery shopping trips. The goal is a better understanding of the impact of daily activities on long-term exposure. 

## How to use it
1. Download files and input data
2. Open programs with an IDE of your choice
3. Check folder names in *streamline_osmnx_4_improved.py*
4. Specify number of iterations
5. Run *streamline_osmnx_4_improved.py* -> output: activity schedules
6. Check folder names in *calc_exposure4_improved.py*
7. Run *calc_exposure4_improved.py* -> output: exposure data + two plots

To get more plots for specific agents: <br>
9. Check folder names in *calc_exposure4_improved_plot.py*
10. Specify which agents of which iteration to plot
11. Run *calc_exposure4_improved_plot.py* -> output: exemplary plot for two agents of an iteration

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

## Authors
Original model by [Meng Lu](https://github.com/mengluchu/agentmodel). <br>
Improved with flexible schedules to add bus travel and grocery shopping by [Jan Hopfer](https://github.com/J-Hopf).
