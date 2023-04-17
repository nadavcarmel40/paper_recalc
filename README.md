# paper_recalc
this repository includes all the code and data for recreation of the figures from the paper titled "Hybrid Logical-Physical Qubit Interaction for Quantum Metrology".

the "data_2021" folder contains all data for redrawing the graphs of the paper. 

to redraw a graph, just go to the relevant .py file in the main folder:
figure_2_general_exploration.py
figure_3_a_kitaev.py
figure_3_b_IPEA.py

and inside set:
generate_data = False
plot_data_new = False
plot_data_from_2021 = True

to generate data with other parameters, just change the constants in the "constants to control this file" section in each of the .py files.
the parameters should be self-explanatory, where the only exception is the parameter "e" which should be "0" for ground state where we measure Rz, or "1" for the excited state.
each file contains a constant termed "name" or "foldername", which corresponds to the name of the folder in which the new data is saved.
