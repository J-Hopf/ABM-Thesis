# ABM-Thesis
Improving activity schedules of an agent-based model with bus travel and grocery shopping using Python

Code for a study that aims to improve an agent-based model (ABM) which quantifys diurnal NO2 exposure based on daily activity schedules with the integration of bus travel and grocery shopping trips. The goal is a better understanding of the impact of daily activities on long-term exposure. 

## Input data files
The files are stored here: https://1drv.ms/f/s!AhrgTI5gv_meiCL65MDP39LPVifD?e=yHT8ai <br>

*input* is the folder for all input data files. <br>
*results1* is the standard folder to save the results. <br>
*r1_improved* inside *results1* is an exemplary folder name to save the results of an individul simulation run. <br>

## How to use it
To calculate exposure of agents:
1. Download files and input data
2. Export input data into working directory
3. Create folder `/results1/r1_improved/` in working directory
4. Open programs with an IDE of your choice
5. Check folder names in *streamline_osmnx_4_improved.py* [ `filedir` ]
6. Install necessary libraries
7. Specify number of iterations [number of iterations = `rageend` - `rangestart` ] 
8. Run *streamline_osmnx_4_improved.py* -> output: activity schedules 
9. Check folder names in *calc_exposure4_improved.py*  [ `filedir` ]
10. Install necessary libraries
11. Run *calc_exposure4_improved.py* -> output: exposure data + exemplary plot of two agents

To plot exposure of specific agents:
1. Check folder names in *calc_exposure4_improved_plot.py*
2. Specify which agents of which iteration to plot
3. Run *calc_exposure4_improved_plot.py* -> output: plot for specified agents

## Python files
*streamline_osmnx_4_improved.py* is the program to **simulate** the activity schedules of the agents. <br>
*calc_exposure4_improved.py* is the program to **calculate** and **plot** the exposure of the agents.  <br>
*calc_exposure4_improved_plot.py* contains the plotting functions of *calc_exposure4_improved.py* to **plot** the exposure of specific agents.  <br>
*model_functions4_improved.py* contains the **functions** used in the programs. <br>

## Added Python functions
*model_functions4_improved.py* <br>
`input_bus_stops` Imports bus stops from a CSV file into a usable DataFrame format. <br>
`input_shops` Imports grocery shops from a CSV file into a usable DataFrame format. <br>
`get_busstops` Gets closest bus stops to the home and destination location. <br>
`travelmean_bus` Chooses mode of travel with bus travel. <br>
`distance_eucl` Calculates Euclidean distance between two points. <br>
`testpoint` Quick validity test for output point in Utrecht. <br>
`get_shop` Returns one of the k nearest shop locations to the input points. <br>
`get_bufferradius` Returns buffer values. <br>
`checkroutebuffer` Calculates the shop locations inside a buffer. <br>
`check_in_extend` Checks if the point locations are inside the concentration raster. <br>
 
## Adapted Python functions
*streamline_osmnx_4_improved.py* <br>
`generate_activity_all` Chooses and generates activity schedules. <br>

*calc_exposure4_improved.py* and *calc_exposure4_improved_plot.py* <br>
`getcon2` Samples the exposure values from the exposure rasters. <br>
`cal_exp2` Calculates the exposure of the agents. <br>
`plotact2` Plots the line graphs of the agents diurnal exposure. <br>

*model_functions4_improved.py* <br>
`schedule_general_shop_h` Schedule function for a grocery shopping trip from home. <br>
`schedule_general_shop_r` Schedule function for grocery shopping on the way home. <br>
`schedule_general_bus` Schedule function for bus travel. <br>
`schedule_general_bus_shop_h` Schedule function for bus travel and a grocery shopping trip from home. <br>
`getroute_2` Calculates route. <br>
`getroute_shop` Calculates route to shop locations. <br>

## Authors
Original model by [Meng Lu](https://github.com/mengluchu/agentmodel). <br>
Improved with flexible schedules to add bus travel and grocery shopping by [Jan Hopfer](https://github.com/J-Hopf).
