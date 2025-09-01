### The proper code is in `AUT_algo/Final_aut_90.py` file and `MES_algo/sim92s_v21_clean.py` file
### In MES directory equation is solved using finite element method
### In AUT directory problem is solved by cellular automaton method

------------------

Below are some information used while working at the project - not important
### legenda do logfile

1. Iteracja
2. Time
3. Unixowy czas w sekundach
4. total_production_nodes
5. average_release
6. inner_iteration - 573 linia działa pętla na tym
7. total_nt_mass ?
8. zone1_nt_m - SYNTHESIS ZONE
9. zone2_nt_m - INNER ZONE
10. zone3_nt_m - RELEASE ZONE
11. prod_nt_mass
12. time_diff -

#### Artykuły użyte:
 - https://www.youtube.com/watch?v=X7vBbelRXn0
 - https://kompendium.plgrid.pl
 - https://nealhughes.net/cython1/
 - https://numpy.org/doc/stable/reference/arrays.nditer.html
 - https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html#cython.parallel.prange
 - https://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html#id4
 - https://stackoverflow.com/questions/2846653/how-do-i-use-threading-in-python

#### Komendy do profilowania

'''bash
python -m cProfile -o my_program.prof ./kolbka_synaptyczna/sim92s_v21.py
snakeviz .\my_program.prof
'''
