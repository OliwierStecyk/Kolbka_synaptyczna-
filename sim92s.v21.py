from termcolor import colored, cprint

# SIM92s ======
# better mesh - version III.2025 ====

import time
epoch = time.time()
from datetime import datetime

print("\nIMP", datetime.now())
from scipy.linalg import norm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import bicgstab as bicgs
# Introduction
import matplotlib as mpl
from matplotlib import cm
from numba import jit

# from mpl_toolkits.mplot3d import Axes3D
# import mpl_toolkits.mplot3d as mp3d
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
# Plot description and layout
from matplotlib import rcParams
import os
PLOTPATH = './plots/'   # SaporPuppis -- tutaj trzeba zrobić sobie folder
if not os.path.exists(PLOTPATH):
    os.makedirs(PLOTPATH)
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 16


# function of initial density of neurotransmitter
czasv = time.time()
#@jit(nopython=True, fastmath=True)
def init_densities(xxx, yyy, zzz, a_dens, b_dens, c_dens):
    rr1 = xxx**2 + yyy**2 + zzz**2
    vector_f = a_dens * np.exp((2.0 * np.random.rand(len(xxx)) - 1.0) * c_dens + b_dens * rr1)
    return vector_f
czasv2 = time.time()

# function of incoming impulses
def f_impulse(t):
    fraction = t - np.floor(t)  #  HERE It IS ASSUMED THAT MAJOR STIMULATION CYCLE TIME IS 1 SECOND
    if fraction < point_of_change:  #  assumed [0.0000s;0.5000s)  !!! OR [0;0.1)
        if freq_high * fraction >= np.floor(freq_high * (fraction + offset_high)):
            if freq_high * fraction < np.floor(freq_high * (fraction + offset_high + tau)):
                ff = 1.0
            else:
                ff = 0.0
        else:
            ff = 0.0
    else:                           #  assumed [0.5000s;1.0000s)  !!! OR [0.1;0;2)
        if freq_low * fraction >= np.floor(freq_low * (fraction + offset_low)):
            if freq_low * fraction < np.floor(freq_low * (fraction + offset_low + tau)):
                ff = 1.0
            else:
                ff = 0.0
        else:
            ff = 0.0
    return ff


# tpx, tpy and tpz are tables with coordinates of 4 points
# returned values are ABCD - coefficients of four shape functions
def coefficients(tpx, tpy, tpz):
    matrix = np.zeros([4, 4])
    v1 = np.zeros(4)
    v2 = np.zeros(4)
    v3 = np.zeros(4)
    v4 = np.zeros(4)
    v1[0] = 1
    v2[1] = 1
    v3[2] = 1
    v4[3] = 1
    for i in range(4):
        matrix[i, 0] = tpx[i]
        matrix[i, 1] = tpy[i]
        matrix[i, 2] = tpz[i]
        matrix[i, 3] = 1
    sol1 = np.linalg.solve(matrix, v1)
    sol2 = np.linalg.solve(matrix, v2)
    sol3 = np.linalg.solve(matrix, v3)
    sol4 = np.linalg.solve(matrix, v4)
    return [sol1, sol2, sol3, sol4]


#  Function of production of neurotransmitter -
def production_V(volume, alfa_i, alfa_mean):
    production = 0.25 * dt * volume * synthesis_rate * (synthesis_threshold - 0.2 * alfa_i - 0.8 * alfa_mean )
    return production


#  P R O G R A M   S T A R T   ====================================================================
#  Begin - set parameters
print("\nBEG", datetime.now(), time.time())
diffusion           =   6.0
permeability        =  80.0
permeability       *=   0.2
synthesis_rate      =  50.0
synthesis_threshold = 370.0
pool_range          =  20                  # pool numbers from -10 to +9 (pool_range-10)
pool_center         =  10
print("\n DIFF ", diffusion, " PERM ", permeability, " SYNTH_R ", synthesis_rate, " SYNTH_T ", synthesis_threshold)
dt = 0.00001
stimulation_cycle = 1.0  #  Cycle high freq /low freq repeated each 1.0 s
tau = 0.0004             #  Duration of stimulation impuls
point_of_change = 0.5  #  Frequency of stimulation changes !!! 0.3 + 0.7 (??) s
freq_high   = 200.0  #  in Hz i.e. 20 impulses in  first 0.5 s each second
offset_high = 0.5 * (stimulation_cycle / freq_high)  #  Stimulation begins in mid-cycle
freq_low    = 100.0  #  in Hz i.e. 10 impulses in second 0.5 s each second
offset_low  = 0.5 * (stimulation_cycle / freq_low)  #  Stimulation begins in mid-cycle
n_print = 10000    #  Print message every n_print'th tetrahedron
n_iter  = 10000    #  Start numbering iterations from n_iter
n_plot  =   200    #  Plot density every n_plot'th iteration
n_timec =     1    #  Calculate NT mass(es) every n_timec iteration
n_timep =   200    #  Plot NT mass(es) vs time every n_timep'th iteration
maxit   = 10000    #  max number of iterations for bicgs
toler   = 1.0E-12  # bicgs tolerance
file_prefix   = './'         #  Unix - ARES


# PROBABILITIES
prob_p = 1.8  #  prob. of release
prob_q = 0.1  #  prob. of return


#  Open log file ------------------
logfile = open(file_prefix + 'log90.txt', 'w')
logfile.write("# Next lines: output from sim90s\n")
logfile.close()


# detecting small tetrahedra and scaling bouton
vol_limit = 1.0E-06
scale_factor = ((3.0*1.1/(4.0*np.pi))**(1.0/3.0)) / 10.0  #  desired radius vs recorded radius
# scale_factor = approx. 0.06403755
vol_scale = scale_factor ** 3
x_scale   = scale_factor
xn_scale  = scale_factor


#  Scaling scatter plot     color
vs_scale = 1.0


# Initialize values of coefficients of density function
a_dens = 7500.00   #  Max of Gauss curve - FORMER VALUE !!!!!!!!
b_dens =   -0.28   #  sigma2 of Gauss curve
c_dens =    0.00   #  coefficient of distortion of Gauss curve


#  Secretion: read number of points (nodes) and node flags from node file
try:
    file  = open(file_prefix + 'param90.1.node', 'r')
except:
    print("File param90.1.node not found")
    exit(1)

filen = open(file_prefix + 'param90_2_node.txt', 'w')

line = file.readline()
tab = line.split()
nodes_number = int(tab[0])
print("\nNumber of nodes from node file", nodes_number, "\n")
points_number = nodes_number
nodes_flag = np.zeros(nodes_number,dtype=np.int32)
for line in file:
    #print(line) #SaporPuppis nie printuje tego
    tab = line.split()
    if tab[0] != '#':
        nodes_n = int(tab[0])
        nodes_x = float(tab[1]) * xn_scale
        nodes_y = float(tab[2]) * xn_scale
        nodes_z = float(tab[3]) * xn_scale
        nodes_flag[int(tab[0])-1] = int(tab[4])
        filen.write("{:10.0f} {:20.8f} {:20.8f} {:20.8f} {:10.0f} \n".
                    format(nodes_n, nodes_x, nodes_y, nodes_z, nodes_flag[int(tab[0])-1]))
file.close()
filen.close()


####################  INITIALIZE PROD FLAGS ######################


# Shifted numbers of iterations used to label output files
i1_range  =  10001  #  Do i1_range iterations, incl. last
i1_offset = n_iter  #  numbering them from (including) i1_offset


#  CALCULATE IMPULSES IN TIME INTERVALS
impulse_x = np.arange(i1_range) * dt
impulse_y = np.vectorize(f_impulse)(impulse_x)
#  PLOT IMPULSES VS TIME
fig = plt.figure()
plt.xlabel('time [ s ]')
plt.ylabel(' impulse (1) or lack of impulse (0)')
plt.scatter(impulse_x, impulse_y, s=5)
plt.savefig( file_prefix + PLOTPATH + 'impulse90.png')
plt.close()


#  STARTING TIME POINT ======
t_zero = 0.0  # starting time


# Initialize matrices A, G and f
matrix_A  = lil_matrix((points_number, points_number))  #  SPARSE
matrix_G  = lil_matrix((points_number, points_number))  #  SPARSE
matrix_A1 = lil_matrix((points_number, points_number))  #  SPARSE
vector_f = np.zeros(points_number)


# Initialize values of graph vectors and set some plot parameters
xxx = np.zeros(points_number)
yyy = np.zeros(points_number)
zzz = np.zeros(points_number)
a_g = 0.0
b_g = 1.0
sss = 20.0 / 100.0  #  was 20.0 / 50.0
v_scatt = 20.0
xfliml = -0.8      #  Minimum value of x in a scatter plot of neurotransmitter density
xflimh = +0.8      #  Maximum value of x in a scatter plot of neurotransmitter density
yfliml = -0.8      #  Minimum value of y in a scatter plot of neurotransmitter density
yflimh = +0.8      #  Maximum value of y in a scatter plot of neurotransmitter density
zfliml = -0.8      #  Minimum value of z in a scatter plot of neurotransmitter density
zflimh = +0.8      #  Maximum value of z in a scatter plot of neurotransmitter density
vfliml = +0.20E3   #  Minimum value of v in a scatter plot of neurotransmitter density
vflimh = +0.45E3   #  Maximum value of v in a scatter plot of neurotransmitter density
srliml = +0.0      #  Minimum value of scaled r in a scatter plot of density vs radius
srlimh = +0.7      #  Maximum value of scaled r in a scatter plot of density vs radius
syliml = +0.00E3   #  Minimum value of scaled y in a scatter plot of density vs radius
sylimh = +0.45E3   #  Maximum value of scaled y in a scatter plot of density vs radius
txliml = +0.0000   #  Starting time for all time plots
txlimh = +0.2000   #  Ending time for these time plots
tyliml = +0.300E3  #  Min value of total nt mass
tylimh = +0.401E3  #  Max value of total nt mass


# Initialize values of
# table of coefficients for calculation of volume of neurotransmitter
# each node has its coefficient equal to 0.25*no.of tetrahedra inside
# the bouton, or, alternatively, each tetrahedron has a coefficient
# equal to the volume of neurotrnsmitter inside it
# matrices A,G and A1 (A1 - secretion)

try:
    file = open(file_prefix + 'ele90.1b.txt', 'r')
except:
    print("File ele90.1b.txt not found")
    exit(1)

nt_vol = np.array([])  # SaporPuppis    nt_vol =[]
work_xyz = np.array([]) # SaporPuppis    work_xyz =[]
vol_pool_n = np.zeros(pool_range)   # Number of tets in flag value class
vol_pool_v = np.zeros(pool_range)   # volumes (20 just for sure to grab all flags)
total_volume = 0.0
i = 0

for line in file:
    i += 1
    tab = line.split()
    #element_number = int(tab[0])  #SaporPuppis i tak nikt tego nie używa Rather not used later !! (??)  SaporPuppis
    point_numbers = np.array([int(tab[1]) - 1, int(tab[2]) - 1, int(tab[3]) - 1, int(tab[4]) - 1])
    pn = point_numbers
    pn = np.append( pn, float(tab[17]) * vol_scale)
    #  In position 22 there is a code: -3 synthesis -2 inside -1 secretion
    pn =  np.append(int(tab[27]))
    nt_vol = np.append(nt_vol, pn)

    tpx = np.zeros(4)
    tpy = np.zeros(4)
    tpz = np.zeros(4)
    node_flag = np.zeros(4)

    tpx[0] = np.float64(tab[5])
    tpy[0] = np.float64(tab[6])
    tpz[0] = np.float64(tab[7])

    tpx[1] = np.float64(tab[8])
    tpy[1] = np.float64(tab[9])
    tpz[1] = np.float64(tab[10])

    tpx[2] = np.float64(tab[11])
    tpy[2] = np.float64(tab[12])
    tpz[2] = np.float64(tab[13])

    tpx[3] = np.float64(tab[14])
    tpy[3] = np.float64(tab[15])
    tpz[3] = np.float64(tab[16])

    for work_i in range(17):
        work_xyz = np.append(work_xyz, float(tab[work_i]))

    #  scaling coordinates
    tpx = tpx * x_scale
    tpy = tpy * x_scale
    tpz = tpz * x_scale

    # =============================================================================
    # From now on the values are rescaled
    # =============================================================================

    # Adjust graph vectors
    for i_xyz in range(4):
        xxx[point_numbers[i_xyz]] = tpx[i_xyz]  # czasem indeks point_numbers[i_xyz] powtarza się więc nadpisujemy wartość, nie wiadomo czy zamierzone działanie
        yyy[point_numbers[i_xyz]] = tpy[i_xyz]  # -||-
        zzz[point_numbers[i_xyz]] = tpz[i_xyz]  # -||-

    #  scaling tab[17] to detect small volumes
    vol = float(tab[17]) * vol_scale
    total_volume += vol
    if (vol < vol_limit * vol_scale):
        print("In line", i, "volume is", vol)
    #  ADDITIONALLY !! >> VOL for pools
    vol_pool_n[pool_center+int(tab[22])] += 1
    vol_pool_v[pool_center+int(tab[22])] += vol

    coefficients_tab = coefficients(tpx, tpy, tpz)

    for point_1 in range(4):
        for point_2 in range(point_1, 4):
            A1 = coefficients_tab[point_1][0]
            B1 = coefficients_tab[point_1][1]
            C1 = coefficients_tab[point_1][2]
            D1 = coefficients_tab[point_1][3]

            A2 = coefficients_tab[point_2][0]
            B2 = coefficients_tab[point_2][1]
            C2 = coefficients_tab[point_2][2]
            D2 = coefficients_tab[point_2][3]

            contribution = diffusion * vol * (A1 * A2 + B1 * B2 + C1 * C2)

            global_point_1 = point_numbers[point_1]
            global_point_2 = point_numbers[point_2]

            if point_1 == point_2:
                matrix_A[global_point_1, global_point_2] += contribution
                matrix_G[global_point_1, global_point_2] += vol / 10.0
            else:
                matrix_A[global_point_1, global_point_2] += contribution
                matrix_A[global_point_2, global_point_1] += contribution

                matrix_G[global_point_1, global_point_2] += vol / 20.0
                matrix_G[global_point_2, global_point_1] += vol / 20.0

    if (i % n_print) == 0:
        print(" TET #", i, ", TIME", datetime.now())


print("\n Number of tetrahedra = ", i, " with volume ", total_volume)
print("\n Coords: ", min(xxx),max(xxx), min(yyy), max(yyy), min(zzz), max(zzz))

n_tet = i
print("  >> volumes: first", nt_vol[0], ", last", nt_vol[n_tet-1])
file.close()
print(" Volume pool sizes: ", end=" ")
for pooln in range(pool_range):
    if vol_pool_n[pooln] > 0.0:
        print(pooln, vol_pool_n[pooln], end=" ")
print(" ")


print("\n", "Write node file calculated and rescaled from ele file")
n2file = open(file_prefix + 'param90_e_node.txt', 'w')
for i in range(points_number):
    n2file.write("{:10.0f} {:20.8f} {:20.8f} {:20.8f} \n".format(i+1, xxx[i], yyy[i], zzz[i]))
n2file.close()


# Read secreting faces
file = open(file_prefix + 'param90.1.face', 'r')
line = file.readline()
tab = line.split()
faces_number = int(tab[0])
print("Number of faces", faces_number)
area_t = []
no_of_tra = 0
no_of_trs = 0
total_area = 0.0
total_area_2 = 0.0
total_area_s = 0.0

czas1 = time.time()
for line in file:
    tab = line.split()
    if tab[0] == '#':
        continue
    no_of_tra += 1
    node1 = int(tab[1])-1
    node2 = int(tab[2])-1
    node3 = int(tab[3])-1
    len_12 = np.sqrt(np.square(xxx[node1]-xxx[node2])+np.square(yyy[node1]-yyy[node2])+np.square(zzz[node1]-zzz[node2]))
    len_13 = np.sqrt(np.square(xxx[node1]-xxx[node3])+np.square(yyy[node1]-yyy[node3])+np.square(zzz[node1]-zzz[node3]))
    len_23 = np.sqrt(np.square(xxx[node3]-xxx[node2])+np.square(yyy[node3]-yyy[node2])+np.square(zzz[node3]-zzz[node2]))
    p = (len_12+len_13+len_23)/2
    area_tr = np.sqrt(p*(p-len_12)*(p-len_13)*(p-len_23))
    if int(tab[4]) == -2:
        total_area_2 += area_tr
    total_area += area_tr
    if int(tab[4]) == -1:  #  The triangle belongs to the secretion zone
        no_of_trs += 1
        prob = 0.0
        area_t.append([area_tr, node1, node2, node3, prob])
        total_area_s += area_tr
        diagonal_contribution = permeability * area_tr / 6.0
        off_diagonal_contribution = permeability * area_tr / 12.0
        matrix_A1[node1,node1] += diagonal_contribution
        matrix_A1[node2,node2] += diagonal_contribution
        matrix_A1[node3,node3] += diagonal_contribution
        matrix_A1[node1,node3] += off_diagonal_contribution
        matrix_A1[node3,node1] += off_diagonal_contribution
        matrix_A1[node2,node3] += off_diagonal_contribution
        matrix_A1[node3,node2] += off_diagonal_contribution
        matrix_A1[node2,node1] += off_diagonal_contribution
        matrix_A1[node1,node2] += off_diagonal_contribution
print("There are", no_of_tra, "triangles,", no_of_trs, "secreting, with areas:")
print(area_t[0], area_t[1],area_t[2],'...',area_t[no_of_trs-3],area_t[no_of_trs-2],area_t[no_of_trs-1])
print("Total secretion area is", total_area_s, " total area2 is", total_area_2, " total area is", total_area)
file.close()
cprint("Time for reading faces  "+ str(time.time()-czas1)+ "s", "green", "on_light_magenta")

# Initial (and permanent!) radius
rrr = np.sqrt(xxx**2 + yyy**2 + zzz**2)
# "CONTINUOUS" initial density - Gauss curve

czas2=time.time()
vector_f = init_densities(xxx, yyy, zzz, a_dens, b_dens, c_dens)
cprint("Time for calculating vector_f "+ str(time.time()-czas2)+ "s", "green", "on_light_magenta")
cprint(" Whole time:  "+str(time.time()-epoch)+"s", "green", "on_light_magenta")
cprint("Compiling function init_densities "+ str(czasv2-czasv)+ "s", "green", "on_light_magenta")
# Calculate total amount of neurotransmitter
total_nt_mass = 0.0
for number_of_tet in range(n_tet):
    for number_of_node in range(4):
        total_nt_mass += 0.25 * vector_f[nt_vol[number_of_tet][number_of_node]] * nt_vol[number_of_tet][4]

print("   ====> TOTAL NT MASS =", total_nt_mass)


# PLOT DETAILED AND RADIAL DISTRIBUTION OF NT AND ITS TOTAL VOLUME CHANGE IN TIME
# Set time and volume for total volume plot
iter_t = []
iter_v = []


# ITERATION No. 0
i1 = 0
t = t_zero + i1 * dt
print("Time =", t)
iter_t.append(t)
ii = i1 + i1_offset
vvv = a_g + b_g * vector_f


# DETAILED
fig = plt.figure()
colmap = cm.ScalarMappable(cmap=cm.hot)
colmap.set_array(vvv)
colmap.set_clim(vfliml, vflimh)


cmap_hot = plt.get_cmap("hot")
# colmap = cm.ScalarMappable(cmap=cm.hot)
ax = fig.add_subplot(111, projection='3d', alpha=1.0)
ax.grid(True)
# ax.set_xlabel('x [ micrometer ]')
# ax.set_ylabel('y [ micrometer ]')
# ax.set_zlabel('z [ micrometer ]')
ax.scatter(xxx, yyy, zzz, 'z', s=sss*vvv, c=vvv, cmap=cmap_hot, vmin=vfliml, vmax=vflimh, lw=0) #SaporPuppis ax.scatter(xxx, yyy, zzz, vvv, s=sss*vvv, c=vvv, cmap=cmap_hot, vmin=vfliml, vmax=vflimh, lw=0)
plt.xlim(xfliml,xflimh)
plt.ylim(yfliml,yflimh)
plt.tight_layout()
ax.set_zlim(zfliml,zflimh)
cb = fig.colorbar(colmap, ax=ax)    #SaporPuppis cb = fig.colorbar(colmap)
plt.savefig(file_prefix + PLOTPATH + 'grph90nr' + str(ii) + '.png', dpi=300)
plt.close()


# RADIAL
fig = plt.figure()
cmap_hot = plt.get_cmap("hot")
plt.grid(True)
# plt.xlabel('r [ micrometer ]')
# plt.ylabel(' density (number of vesicles per cubic micrometer)')
plt.scatter(rrr, vector_f, cmap=cmap_hot, c=zzz, vmin=zfliml, vmax=zflimh, s=v_scatt, lw=0)
plt.xlim(srliml, srlimh)
plt.ylim(syliml, sylimh)
plt.savefig(file_prefix + PLOTPATH+ 'scatt90nr' + str(ii) + '.png')
plt.close()


# TOTAL
iter_v.append(total_nt_mass)
fig=plt.figure()
plt.xlabel('time [s]')
plt.ylabel('number of vesicles')
plt.scatter(iter_t, iter_v, c='k', s = 5.0)
plt.xlim(txliml, txlimh)
plt.ylim(tyliml, tylimh)
plt.savefig(file_prefix + PLOTPATH+ 'total90nr' + str(ii) + '.png')
plt.close()


print("START", datetime.now())
matrix_left = 2 * matrix_G + dt * (matrix_A + matrix_A1 * f_impulse(t))
matrix_right = 2 * matrix_G - dt * (matrix_A + matrix_A1 * f_impulse(t - dt))
print("MAT_LEFT (AND RIGHT)", matrix_left[0, 0], matrix_left[points_number - 1, points_number - 1])
print("LEFT AND RIGHT", datetime.now())


synthesis_vector = np.zeros(points_number)
synth_flag       = np.zeros(points_number, dtype=int)


previous_release = 0.0

logfile = open(file_prefix + 'log90.txt', 'a')
startbigpentla = time.time()
for i1 in range(i1_range):
    inlooptime = time.time()
    logfile.write("{:8.0f} {:7.4f} {:20.8f}".format(i1, t, time.time()))
    time_diff = 0.0 - time.time()
    ii = i1 + 1 + i1_offset
    matrix_left = 2 * matrix_G + dt * (matrix_A + matrix_A1 * f_impulse(t))
    matrix_right = 2 * matrix_G - dt * (matrix_A + matrix_A1 * f_impulse(t - dt))
    time_diff += time.time()


    #  SYNTHESIS (production) - calculate, print and plot
    vector_f_times_right = matrix_right.dot(vector_f)
    #  CALCULATE F WITHOUT PRODUCTION
    time_diff = 0.0 - time.time()
    sol_wo_p = bicgs(matrix_left, vector_f_times_right,x0=vector_f,atol=toler,maxiter=maxit) #SaporPuppis sol_wo_p = bicgs(matrix_left, vector_f_times_right,x0=vector_f,tol=toler,maxiter=maxit)
    time_diff += time.time()
    vector_f_wo_p = sol_wo_p[0]
    for ind in range(points_number):
        synthesis_vector[ind] = 0.0
        synth_flag[ind] = 0
    total_production_nodes = 0
    # print(" DETECT ROWS !! ")
    for row in nt_vol:
        # print(row, end=" " )
        if row[5] == -2:  # # # CORRECTED IN 90 VERSION, IS IT OK???
            if vector_f[row[0]] < synthesis_threshold and vector_f[row[1]] < synthesis_threshold and \
                            vector_f[row[2]] < synthesis_threshold and vector_f[row[3]] < synthesis_threshold:
                mean_f = 0.25 * (vector_f[row[0]]+vector_f[row[1]]+vector_f[row[2]]+vector_f[row[3]])
                synthesis_vector[row[0]] += production_V(row[4], vector_f[row[0]], mean_f)
                synthesis_vector[row[1]] += production_V(row[4], vector_f[row[1]], mean_f)
                synthesis_vector[row[2]] += production_V(row[4], vector_f[row[2]], mean_f)
                synthesis_vector[row[3]] += production_V(row[4], vector_f[row[3]], mean_f)
                for i_row in range(4):
                    synth_flag[row[i_row]] += 1

    # print(" ")
    total_production_nodes = sum([1 for i_prod in range(points_number) if synth_flag[i_prod] > 0])
    vector_f_times_right_copy = vector_f_times_right[:]
    total_synth = 2.0 * synthesis_vector
    print(">> IT", ii,"Inn IT 0 synth norm", norm(total_synth),
          "Pnodes =", total_production_nodes, end=" ")
    vector_f_times_right += total_synth

    logfile.write("{:8.0f}".format(total_production_nodes))
    previous_step_synthesis = synthesis_vector[:]
    inner_iteration = 1
    solution = bicgs(matrix_left, vector_f_times_right, x0=vector_f, atol=toler, maxiter=maxit)  #SaporPuppis solution = bicgs(matrix_left, vector_f_times_right, x0=vector_f, tol=toler, maxiter=maxit)
    
    vector_f_new = solution[0]


    while inner_iteration < 101:
        jumps_from_below_to_above = 0
        jumps_from_above_to_below = 0
        for row in nt_vol:  # petla po czworoscianach
            if vector_f[row[0]] < synthesis_threshold and vector_f[row[1]] < synthesis_threshold and \
                   vector_f[row[2]] < synthesis_threshold and vector_f[row[3]] < synthesis_threshold:
                if vector_f_new[row[0]] > synthesis_threshold or vector_f_new[row[1]] > synthesis_threshold or \
                   vector_f_new[row[2]] > synthesis_threshold or vector_f_new[row[3]] > synthesis_threshold:
                        jumps_from_below_to_above += 1
            if vector_f[row[0]] > synthesis_threshold or vector_f[row[1]] > synthesis_threshold or \
                   vector_f[row[2]] > synthesis_threshold or vector_f[row[3]] > synthesis_threshold:
                if vector_f_new[row[0]] < synthesis_threshold and vector_f_new[row[1]] < synthesis_threshold and \
                   vector_f_new[row[2]] < synthesis_threshold and vector_f_new[row[3]] < synthesis_threshold:
                        jumps_from_above_to_below += 1
        # print("ABOVE -> BELOW: ",jumps_from_above_to_below, "  BELOW->ABOVE: ",jumps_from_below_to_above)
        vector_f = vector_f_new[:]


        #  PRODUCTION
        production_flag = 0
        for ind in range(points_number):
             synthesis_vector[ind] = 0.0
        for row in nt_vol:
               if row[5] == -2:   # # # Corrected in 90 version, is it OK????
                     if vector_f[row[0]] < synthesis_threshold and vector_f[row[1]] < synthesis_threshold and \
                            vector_f[row[2]] < synthesis_threshold and vector_f[row[3]] < synthesis_threshold:
                         mean_f = 0.25 * (vector_f[row[0]]+vector_f[row[1]]+vector_f[row[2]]+vector_f[row[3]])
                         synthesis_vector[row[0]] += production_V(row[4], vector_f[row[0]], mean_f)
                         synthesis_vector[row[1]] += production_V(row[4], vector_f[row[1]], mean_f)
                         synthesis_vector[row[2]] += production_V(row[4], vector_f[row[2]], mean_f)
                         synthesis_vector[row[3]] += production_V(row[4], vector_f[row[3]], mean_f)
                         production_flag = 1

        total_synth = synthesis_vector + previous_step_synthesis
        # print("  inner IT", inner_iteration+1, "synth n", norm(total_synth), end=" ")
        vector_f_times_right = vector_f_times_right_copy + total_synth

        solution = bicgs(matrix_left, vector_f_times_right, x0=vector_f, atol=toler, maxiter=maxit) #SaporPuppis solution = bicgs(matrix_left, vector_f_times_right, x0=vector_f, tol=toler, maxiter=maxit)
        vector_f_new = solution[0]

        # PRINT AND PLOT SYNTHESIS (PRODUCTION)
        if production_flag == 1:
            if np.array_equal(synthesis_vector, np.zeros(points_number)):
                print(" OOOOOOOOOOOOOOOOOOO ITER ", ii, " synth_v = 0 !! ")
            else:
                # print(" SYNTHESIS = ", synthesis_vector, "PREVIOUS_SYNTHESIS = ", previous_step_synthesis)
                #  PLOT THREE-DIMENSIONAL, not in ALL iterations !!!
                if ii%n_plot == 0:
                    fig = plt.figure()
                    colmap.set_array(synthesis_vector)
                    print(" mmm ", max(synthesis_vector), " mmm ")
                    colmap.set_clim(0, 0.002)
                    cmap_YoB = plt.get_cmap("YlOrBr")
                    ax.grid(True)   ###  >>>  added 11.XI.2024 to make similar to gr plots
                    ax = fig.add_subplot(111, projection='3d', alpha=1.0)

                    ax.scatter(xxx, yyy, zzz, synthesis_vector, s=v_scatt, c=synthesis_vector,
                               cmap=cmap_YoB, vmin=0.0, vmax=3E-3, lw=0)
                    plt.tight_layout()
                    ax.set_zlim(zfliml,zflimh)
                    cb = fig.colorbar(colmap)
                    plt.savefig(file_prefix + PLOTPATH+ 'gsss90nr' + str(ii) + '.png')
                    plt.close()
                    #  PLOT TWO-DIMENSIONAL
                    # fig = plt.figure()
                    # plt.xlabel('x [um]')
                    # plt.ylabel('z [um]')
                    # plt.scatter(iter_t, iter_v, c=synthesis_vector, cmap=cm,vmin=0.0,vmax=0.01)
                    # plt.xlim(xfliml, xflimh)
                    # plt.ylim(zfliml, zflimh)
                    # plt.savefig(file_prefix + 'gssx90nr' + str(ii) + '.png')
                    # plt.close()


        inner_iteration += 1
        if jumps_from_below_to_above == 0 and jumps_from_above_to_below == 0:
            break

    print("In_it=",inner_iteration,", jmps",jumps_from_below_to_above,jumps_from_above_to_below,
          ", s_norm =", norm(total_synth), end=" ")
    vector_f = vector_f_new[:]
    if i1>2 and i1<i1_range-2:
        if impulse_y[i1-2] == 1 or impulse_y[i1+2] == 1 or ii%n_plot == 0:
            # PLOT      #  ^ ... WAS  if ii%n_plot==0:
            vvv = a_g + b_g * vector_f
            fig = plt.figure()
            # colmap = cm.ScalarMappable(cmap=cm.hot)
            colmap.set_array(vvv)
            colmap.set_clim(vfliml, vflimh)
            cb = fig.colorbar(colmap, ax=ax) #SaporPuppis cb = fig.colorbar(colmap)
            cmap_hot = plt.get_cmap("hot")
            ax = fig.add_subplot(111, projection='3d', alpha=1.0)
            ax.grid(True)
            # ax.set_xlabel('x [ micrometer ]')
            # ax.set_ylabel('y [ micrometer ]')
            # ax.set_zlabel('z [ micrometer ]')
            ax.scatter(xxx, yyy, zzz, 'z', s=sss*vvv, c=vvv, cmap=cmap_hot,
                       vmin=vfliml, vmax=vflimh, lw=0)    #SaporPuppis ax.scatter(xxx, yyy, zzz, vvv, s=sss*vvv, c=vvv, cmap=cmap_hot, vmin=vfliml, vmax=vflimh, lw=0) 
            plt.xlim(xfliml, xflimh)
            plt.ylim(yfliml, yflimh)
            plt.tight_layout()
            ax.set_zlim(zfliml, zflimh)
            cb = fig.colorbar(colmap,ax=ax)   #SaporPuppis cb = fig.colorbar(colmap)
            plt.savefig(file_prefix + PLOTPATH+ 'grph90nr' + str(ii) + '.png', dpi=300)
            plt.close()


            fig = plt.figure()
            cm = plt.get_cmap("hot")
            plt.grid(True)
            # plt.xlabel('r [ micrometer ]')
            # plt.ylabel(' density (number of vesicles per cubic micrometer)')
            plt.scatter(rrr, vector_f, cmap=cm, c=zzz, vmin=zfliml, vmax=zflimh, s=v_scatt, lw=0)
            plt.xlim(srliml, srlimh)
            plt.ylim(syliml, sylimh)
            plt.savefig(file_prefix + PLOTPATH+ 'scatt90nr' + str(ii) + '.png')
            plt.close()


    # Calculate release
    release = 0.0
    releasing_nodes = 0
    for triangle_no in range(len(area_t)):
        tr = area_t[triangle_no]
        release_contribution = tr[0] * (vector_f[tr[1]] + vector_f[tr[2]] + vector_f[tr[3]])
        release_contribution *= (f_impulse(t) * dt * permeability / 3.0)
        prob = np.random.rand()
        if prob < prob_p:
            release += release_contribution
            releasing_nodes += 1
    average_release = (release + previous_release) / 2.0
    print("==> TOTAL RELEASE =", average_release, end=" ")
    logfile.write("{:13.6f} {:8.0f}".format(average_release,inner_iteration))
    previous_release = release


    # Calculate total amount of neurotransmitter, also in pools
    if ii%n_timec==0:
        # Calculate
        total_nt_mass = 0.0
        zone1_nt_m = 0.0  #  SYNTHESIS ZONE
        zone2_nt_m = 0.0  #  INNER ZONE
        zone3_nt_m = 0.0  #  RELEASE ZONE
        for number_of_tet in range(n_tet):
            for number_of_node in range(4):
                nt_mass = 0.25 * vector_f[nt_vol[number_of_tet][number_of_node]] * nt_vol[number_of_tet][4]
                total_nt_mass += nt_mass
                if nt_vol[number_of_tet][5] == -3:
                    zone1_nt_m += nt_mass
                if nt_vol[number_of_tet][5] == -2:
                    zone2_nt_m += nt_mass
                if nt_vol[number_of_tet][5] == -1:
                    zone3_nt_m += nt_mass
        print("   ====> TOTAL (AND DETAILED...) NT MASS =", total_nt_mass, zone1_nt_m,
              zone2_nt_m, zone3_nt_m, end=" ")
        logfile.write("{:20.8f} {:20.8f} {:20.8f} {:20.8f}".format(total_nt_mass,zone1_nt_m,zone2_nt_m,zone3_nt_m))


    # Calculate synthesised amount of neurotransmitter !!!!!!!!!!!!!!!!!!!
    if ii%n_timec==0:
        # Calculate
        prod_nt_mass = 0.0
        delta_f = vector_f - vector_f_wo_p
        for n_of_tet in range(n_tet):
            for n_of_node in range(4):
                prod_nt_mass += 0.25 * delta_f[nt_vol[n_of_tet][n_of_node]] * nt_vol[n_of_tet][4]
        print("==(!!!)=> PRODUCED NT MASS =", prod_nt_mass, end=" ")
        logfile.write("{:20.8f}".format(prod_nt_mass))


    # Plot total NT mass vs time every n_timep'th iteration (but CALCULATE every iteration)
    iter_t.append(t)
    iter_v.append(total_nt_mass)
    if ii%n_timep==0:
        fig=plt.figure()
        plt.xlabel('time [s]')
        plt.ylabel('number of vesicles')
        plt.grid(True)
        plt.scatter(iter_t, iter_v, c='k', s = 5.0)
        plt.xlim(txliml, txlimh)
        plt.ylim(tyliml, tylimh)
        plt.savefig(file_prefix + PLOTPATH+ 'total90nr' + str(ii) + '.png')
        plt.close()


    print(" TIME = ", time_diff, " maxit ", solution[1], " minf ", min(vector_f), datetime.now())
    logfile.write("{:15.6f}\n".format(time_diff))

    t += dt
    cprint("Iteration time: "+str((time.time()-inlooptime)+" s", "black", "on_light_magenta")
    cprint("Mean iteration time: "+str((time.time()-startbigpentla)/(i1+1))+" s", "black", "on_light_magenta")
    # return to loop start, iterate over

cprint("Whole time for loop: "+str(time.time()-startbigpentla)+" s", "black", "on_light_magenta")
# END PLOT


# DETAILED
fig = plt.figure()
cm = plt.get_cmap("hot")
ax = fig.add_subplot(111, projection='3d', alpha=1.0)
ax.grid(True)
# ax.set_xlabel('x [ micrometer ]')
# ax.set_ylabel('y [ micrometer ]')
# ax.set_zlabel('z [ micrometer ]')
ax.scatter(xxx, yyy, zzz, 'z', s=sss*vvv, c=vvv, cmap=cm, vmin=vfliml, vmax=vflimh, lw=0) #SaporPuppis ax.scatter(xxx, yyy, zzz, vvv, s=sss*vvv, c=vvv, cmap=cm, vmin=vfliml, vmax=vflimh, lw=0)
plt.xlim(xfliml,xflimh)
plt.ylim(yfliml,yflimh)
plt.tight_layout()
ax.set_zlim(zfliml,zflimh)
cb = fig.colorbar(colmap,ax=ax) #SaporPuppis cb = fig.colorbar(colmap)
plt.savefig(file_prefix + PLOTPATH+ 'grph90nr' + str(ii) + '.png', dpi=300)
plt.close()


# RADIAL
fig = plt.figure()
cm = plt.get_cmap("hot")
plt.grid(True)
# plt.xlabel('r [ micrometer ]')
# plt.ylabel(' density (number of vesicles per cubic micrometer)')
plt.scatter(rrr, vector_f, cmap=cm, c=zzz, vmin=zfliml, vmax=zflimh, s=v_scatt, lw=0)
plt.xlim(srliml, srlimh)
plt.ylim(syliml, sylimh)
plt.savefig(file_prefix + PLOTPATH+ 'scatt90nr' + str(ii) + '.png')
plt.close()


# TOTAL
iter_t.append(t)
iter_v.append(total_nt_mass)
fig=plt.figure()
plt.grid(True)
plt.xlabel('time [s]')
plt.ylabel('number of vesicles')
plt.scatter(iter_t, iter_v, c='k', s = 5.0)
plt.xlim(txliml, txlimh)
plt.ylim(tyliml, tylimh)
plt.savefig(file_prefix + PLOTPATH+ 'total90nr' + str(ii) + '.png')
plt.close()


print("END", datetime.now())