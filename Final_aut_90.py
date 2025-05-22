print("\n Final Model, cellular automaton, version <90>, November 2023 ")
print(" All files reside in one directory C: Pythondata Graphics90s ")


from datetime import datetime
import matplotlib as mpl
# mpl.use('Agg')
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import cm
import statistics
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16
# plt.rcParams["figure.figsize"] = [12.00, 8.00]
print("\n S T A R T > > > ", datetime.now(), "\n")


def init_dens(x):
    rr = x[0] * x[0] + x[1] * x[1] + x[2] * x[2]
    dd = a_dens * math.exp((2.0 * np.random.rand() - 1.0) * c_dens + b_dens * rr)
    return dd


def S_ABC(v_list):
    def ll(xx1, yy1, zz1, xx2, yy2, zz2):
        return ((xx2 - xx1) ** 2 + (yy2 - yy1) ** 2 +  (zz2 - zz1) ** 2) ** 0.5
    d_AB = ll(xn[v_list[0]],yn[v_list[0]],zn[v_list[0]],xn[v_list[1]],yn[v_list[1]],zn[v_list[1]])
    d_AC = ll(xn[v_list[0]],yn[v_list[0]],zn[v_list[0]],xn[v_list[2]],yn[v_list[2]],zn[v_list[2]])
    d_BC = ll(xn[v_list[1]],yn[v_list[1]],zn[v_list[1]],xn[v_list[2]],yn[v_list[2]],zn[v_list[2]])
    p_ABC = 0.5 * (d_AB + d_AC + d_BC)
    return (p_ABC * (p_ABC - d_AB) * (p_ABC - d_AC) * (p_ABC - d_BC)) ** 0.5


def l_AB(xl1, yl1, zl1, xl2, yl2, zl2):
    return ((xl2 - xl1) ** 2 + (yl2 - yl1) ** 2 + (zl2 - zl1) ** 2) ** 0.5


diffusion      =     3.0  #  was 3.0, then 6.0, them 13.0
permeability   =    50.0  #  was 4.16, was ..., was 25000, was 30000, 31000, 20000  ...
print(" \n\n Increase permeability, probably to 30000? \n\n ")
synthesis_rate =     5.0
max_iter       =    2001
dt             = 0.00001


# arrays
t_time  = np.zeros(max_iter, dtype = np.float64)
t_volro = np.zeros(max_iter, dtype = np.float64)
for i in range(max_iter):
    t_time[i] = float(i) * dt
print(" Calculated time points \n",t_time)


max_zs  =   10.00
ro_     =   30.00   # synthesis threshold
a_dens  =  518.54   # Max of Gauss curve
b_dens  =   -1.40   # sigma2 of Gauss curve
c_dens  =    0.00   # coefficient of distortion of Gauss curve
Wil_Vol =    1.10   # Volume taken from WilhVolume calculated from Wilhelm dissertation
Cmp_Vol =  971.756  #
x_scale =  (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0) # COEFF OF SCLNG PT COORDS
# x_scale *= ((Wil_Vol / Cmp_Vol) ** (1.0 / 3.0)) ### mor prcis? but diff fr FE
x_scale *= ((Wil_Vol / (10**3)) ** (1.0 / 3.0))
print("\n\n * * * * SCALING X COORDINATES BY THE FACTOR OF ", x_scale, "\n\n")
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
# a_dens  = a_dens * 400.0 / 83492.185585
# Second adjustment >> make it as close to FE results as possible
# a_dens  = a_dens * 399.928383 / 400.017311
# VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
read_dir_name = 'C:\\PythonData\\Graphics90s\\'
fin5_dir_name = 'C:\\PythonData\\Graphics90s\\'
#----------------------------
n_print = 2500   #  For common verts, print line every n_print'th tetrahedron
n_plot  =   10   #  In main loop, draw pictures every n_plot'th iteration
# ---------------------------
# xyz_lim = approx(0.62)
xfliml, xflimh, yfliml, yflimh, zfliml, zflimh = -0.75, 0.75, -0.75, 0.75, -0.75, 0.75
vfliml, vflimh = 2.8E+02, 5.2E+02
# sss = 20.0 / 50.03
sss = 0.1


print(" \n\n P A R A M E T E R S  diff perm syn_r syn_t \n\n ",
      diffusion, permeability, synthesis_rate, ro_)


#  Read node file is read to get the copies of x, y, z coordinates
#  important: rescale read variables
file  = open(read_dir_name + "param90.1.node", "r")
line  = file.readline()
tab   = line.split()
n_nod = int(tab[0])
xn = np.zeros(n_nod, dtype=np.float32)
yn = np.zeros(n_nod, dtype=np.float32)
zn = np.zeros(n_nod, dtype=np.float32)
Ln = np.zeros(n_nod, dtype=np.int32)
i_node = 0
for line in file:
    tab = line.split()
    if tab[0] != "#":
        xn[i_node], yn[i_node], zn[i_node] = \
            float(tab[1]) * x_scale, float(tab[2]) * x_scale, float(tab[3]) * x_scale
        Ln[i_node] = int(tab[4])
        i_node += 1
file.close()
print(" \n Nodes file: read # of nodes = ", n_nod, ", calculated # of nodes = ", i_node, "\n")
print(" Ranges of x, y, z : ", min(xn), max(xn), min(yn), max(yn), min(zn), max(zn), "\n\n")
print(" Ranges of x, y, z internal [-1] : ",
      min([xn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1]),
      max([xn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1]),
      min([yn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1]),
      max([yn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1]),
      min([zn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1]),
      max([zn[i] for i in range(n_nod) if Ln[i] == 1 or Ln[i] == -1])  )
print(" Ranges of x, y, z internal [-2] : ",
    min([xn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2]),
    max([xn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2]),
    min([yn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2]),
    max([yn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2]),
    min([zn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2]),
    max([zn[i] for i in range(n_nod) if Ln[i] == 2 or Ln[i] == -2])  )
print(" Ranges of x, y, z internal [-3] : ",
      min([xn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3]),
      max([xn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3]),
      min([yn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3]),
      max([yn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3]),
      min([zn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3]),
      max([zn[i] for i in range(n_nod) if Ln[i] == 3 or Ln[i] == -3])  )
print(" Ranges of x, y, z internal [1] : ",
      min([xn[i] for i in range(n_nod) if Ln[i] == 1]),
      max([xn[i] for i in range(n_nod) if Ln[i] == 1]),
      min([yn[i] for i in range(n_nod) if Ln[i] == 1]),
      max([yn[i] for i in range(n_nod) if Ln[i] == 1]),
      min([zn[i] for i in range(n_nod) if Ln[i] == 1]),
      max([zn[i] for i in range(n_nod) if Ln[i] == 1]))
print(" Ranges of x, y, z internal [2] : ",
      min([xn[i] for i in range(n_nod) if Ln[i] == 2]),
      max([xn[i] for i in range(n_nod) if Ln[i] == 2]),
      min([yn[i] for i in range(n_nod) if Ln[i] == 2]),
      max([yn[i] for i in range(n_nod) if Ln[i] == 2]),
      min([zn[i] for i in range(n_nod) if Ln[i] == 2]),
      max([zn[i] for i in range(n_nod) if Ln[i] == 2]))
print(" Ranges of x, y, z internal [3] : ",
      min([xn[i] for i in range(n_nod) if Ln[i] == 3]),
      max([xn[i] for i in range(n_nod) if Ln[i] == 3]),
      min([yn[i] for i in range(n_nod) if Ln[i] == 3]),
      max([yn[i] for i in range(n_nod) if Ln[i] == 3]),
      min([zn[i] for i in range(n_nod) if Ln[i] == 3]),
      max([zn[i] for i in range(n_nod) if Ln[i] == 3]))
print(" Ranges of x, y, z internal [ALL] : ",
      min(xn), max(xn), min(yn), max(yn), min(zn), max(zn), "\n\n")


#  Read OLD tet file, in fact only the number of tets
#  Basing on the number of tets, create the tables:
file  = open(read_dir_name + "param90.1.ele", "r")
line  = file.readline()
tab   = line.split()
n_tet = int(tab[0])
file.close()
nt      = np.zeros(n_tet,     dtype=np.int32)   #  number of ith tetrahedron (from 1)
tn      = np.zeros((n_tet,4), dtype=np.int32)   #  numbers of vertices of ith tet. (from 1)
xt      = np.zeros((n_tet,4), dtype=np.float64) #  x coords of vertices of ith tet.
yt      = np.zeros((n_tet,4), dtype=np.float64) #  y coords of vertices of ith tet.
zt      = np.zeros((n_tet,4), dtype=np.float64) #  z coords of vertices of ith tet.
V_tet   = np.zeros(n_tet,     dtype=np.float64) #  volume of ith tetrahedron
fn      = np.zeros((n_tet,4), dtype=np.int32)   #  flags of vertices of ith tet.
tf      = np.zeros(n_tet,     dtype=np.int32)   #  flag of ith tetrahedron (-2 synth, -1 rest)
n_nei   = np.zeros(n_tet,     dtype=np.int32)   #  number of neighbrs for ith tetrahedron  # NEW!!
id_nei  = np.zeros((n_tet,4), dtype=np.int32)   #  ids of n_nei neighbors of a tetrahedron # NEW!!
S_neigh = np.zeros((n_tet,4), dtype=np.float64) #  areas of n_nei neighbors of a tetr.     # NEW!!
r_neigh = np.zeros((n_tet,4), dtype=np.float64) #  distances to n_nei neighbors of a tetr. # NEW!!
x_int   = np.zeros(n_tet,     dtype=np.float64) #  x coord of the middle point if ith tet. # NEW!!
y_int   = np.zeros(n_tet,     dtype=np.float64) #  y coord of the middle point if ith tet. # NEW!!
z_int   = np.zeros(n_tet,     dtype=np.float64) #  z coord of the middle point if ith tet. # NEW!!


# read tetrahedra - input file is ele90.1b.txt, it must be converted
# before converting, coordinates of nodes have to be rescaled
# additional data - neighboring tet numbers, "doorface" areas, distance to neighbors
# output arrays are: n_nei, id_nei, S_neigh, r_neigh
## First, read the input columns, calculate all - except neigh distances
file = open(read_dir_name + "ele90.1b.txt", "r")
i_tet = 0
freq_n_nei = [0, 0, 0, 0, 0]
for line in file:
    tab = line.split()  #  REMEMBER, lines are numbered from 0 !!
    nt[i_tet] = int(tab[0])
    for i in range(4):
        tn[i_tet, i] = int(tab[1 + i])
        xt[i_tet, i] = float(tab[5 + 3 * i]) * x_scale
        yt[i_tet, i] = float(tab[6 + 3 * i]) * x_scale
        zt[i_tet, i] = float(tab[7 + 3 * i]) * x_scale
    V_tet[i_tet] = float(tab[17]) * (x_scale ** 3)
    for i in range(4):
        fn[i_tet, i] = int(tab[18 + i])
    n_nei[i_tet] = int(tab[22])  #  no of neighbors of tet i_tet
    freq_n_nei[n_nei[i_tet]] += 1
    for i in range(4):
        id_nei[i_tet, i] = int(tab[23 + i])  #  ids of those tets
    tf[i_tet] = int(tab[27])
    x_int[i_tet] = statistics.mean(xt[i_tet,:])
    y_int[i_tet] = statistics.mean(yt[i_tet,:])
    z_int[i_tet] = statistics.mean(zt[i_tet,:])
    i_tet += 1


print(" \n Number of tets read:", i_tet, ", declared:", n_tet, "\n")
print(" n_neigh  freq", freq_n_nei, ", total n of neighbrs =", sum(freq_n_nei))
print(" Stats(tn) min", [min(tn[:,i]) for i in range(4)], ", max", [max(tn[:,i]) for i in range(4)])
print(" Stats(xt) min", [min(xt[:,i]) for i in range(4)], ", max", [max(xt[:,i]) for i in range(4)])
print(" Stats(yt) min", [min(yt[:,i]) for i in range(4)], ", max", [max(yt[:,i]) for i in range(4)])
print(" Stats(zt) min", [min(zt[:,i]) for i in range(4)], ", max", [max(zt[:,i]) for i in range(4)])
print(" Stats(x_int) min", min(x_int),  ", max", max(x_int))
print(" Stats(y_int) min", min(y_int),  ", max", max(y_int))
print(" Stats(z_int) min", min(z_int),  ", max", max(z_int))
print(" Stats(V_tet) min", min(V_tet),  ", max", max(V_tet),  ", sum", sum(V_tet))
print(", mean", statistics.mean(V_tet), ", std", statistics.stdev(V_tet, statistics.mean(V_tet)))


print(" \n\n !!! COMMON VERTS !!! ")
COMMON_VERTS = []
n_of_len_3 = 0
for i_tet in range(n_tet):
    if i_tet % n_print == 0:
        print(" i_tet: NNN nei -", i_tet, n_nei[i_tet])
    for i_nei in range(n_nei[i_tet]):   #   i_nei means 0th, 1st, 2nd and 3rd neighbour
        id_tnei = id_nei[i_tet][i_nei]  #   id_nei neighbours, id_tnei can be 0
        common_verts = list((set(tn[i_tet,:])).intersection(set(tn[id_tnei,:])))
        if len(common_verts) == 3:
            S_temp = S_ABC([common_verts[0] - 1, common_verts[1] - 1, common_verts[2] - 1])
            l_temp = l_AB(x_int[i_tet], y_int[i_tet], z_int[i_tet],
                          x_int[id_tnei], y_int[id_tnei], z_int[id_tnei])
            S_neigh[i_tet, i_nei] = S_temp
            r_neigh[i_tet, i_nei] = l_temp
            if S_temp < 2.00E-4:
                print(" >> SMALL S *!*!*!*!* >>> ", i_tet, i_nei, id_tnei, S_temp, l_temp)
            if l_temp > 5.50E-2:
                print(" >> LARGE L *!*!*!*!* >>> ", i_tet, i_nei, id_tnei, S_temp, l_temp)
            COMMON_VERTS.append([i_tet, i_nei, id_tnei, S_neigh, r_neigh])
            n_of_len_3 += 1
file.close()
print(" No of common faces = ", len(COMMON_VERTS), n_of_len_3)


print("\n S neigh MIN >>> ", [min(S_neigh[:,i]) for i in range(4)])
print(  " S neigh MAX >>> ", [max(S_neigh[:,i]) for i in range(4)])
print("\n r neigh min >>> ", [min(r_neigh[:,i]) for i in range(4)])
print(  " r neigh max >>> ", [max(r_neigh[:,i]) for i in range(4)], "\n")


for i in range(4):
    print(" &: ", min([S_neigh[i_tet, i] for i_tet in range(n_tet)]),
          max([S_neigh[i_tet, i] for i_tet in range(n_tet)]),
          sum([S_neigh[i_tet, i] for i_tet in range(n_tet)]),
          statistics.mean([S_neigh[i_tet, i] for i_tet in range(n_tet)]),
          statistics.stdev([S_neigh[i_tet, i] for i_tet in range(n_tet)]))
print(" \n a Counted number of elems = ", i_tet, " compared to read from ele file ", n_tet, end=" ")
print(", number of internal faces = ", n_of_len_3)


print("\n S_neigh and r_neigh - statistics min max mean ")
S_neigh_list = []
r_neigh_list = []
for i_tet in range(n_tet):
    if (i_tet == 0) or (i_tet == 15329):
        print( " S NEIGH LIST I_TET ID_NEI[I_TET] => ", i_tet, n_nei[i_tet], id_nei[i_tet])
    for i_nei in range(n_nei[i_tet]):
        id_tnei = id_nei[i_tet][i_nei]
        if (i_tet == 0) or (i_tet == 15329):
            print(" N Nnnn I nnn ", n_nei[i_tet], i_nei, id_tnei)
        if S_neigh[i_tet, i_nei] > 0.0:
            # print( " SSSS ", i_tet, i_nei, id_tnei, S_neigh[i_tet, i_nei])
            S_neigh_list.append([i_tet, id_tnei, S_neigh[i_tet, i_nei]])
            r_neigh_list.append([i_tet, id_tnei, r_neigh[i_tet, i_nei]])
len_S_n_list = len(S_neigh_list)
# print("\n\n", [S_neigh_list[i][2] for i in range(len_S_n_list)])
print("\n\n S_neigh_list stats: ", len(S_neigh_list),
      min([S_neigh_list[i][2] for i in range(len_S_n_list)]),
      max([S_neigh_list[i][2] for i in range(len_S_n_list)]),
      sum([S_neigh_list[i][2] for i in range(len_S_n_list)]), end=" ")
print(statistics.mean([S_neigh_list[i][2] for i in range(len_S_n_list)]),
      statistics.stdev([S_neigh_list[i][2] for i in range(len_S_n_list)]))
print("\n r_neigh_list stats: ", len(r_neigh_list),
      min([r_neigh_list[i][2] for i in range(len_S_n_list)]),
      max([r_neigh_list[i][2] for i in range(len_S_n_list)]),
      sum([r_neigh_list[i][2] for i in range(len_S_n_list)]), end=" ")
print(statistics.mean([r_neigh_list[i][2] for i in range(len_S_n_list)]),
      statistics.stdev([r_neigh_list[i][2] for i in range(len_S_n_list)]), "\n")


# faces - triangles - read face file - ATTENTION!  This set includes secreting faces !!
file = open(read_dir_name + 'param90.1.face', 'r')
line = file.readline()
tab  = line.split()
nnf  = int(tab[0])
print("Number of faces =", nnf, "\n")
n_abc   = np.zeros((nnf,3), dtype=int)
S_abc   = np.zeros(nnf, dtype=float)
df_label = np.zeros(nnf, dtype=int)
i_face = 0
for line in file:
    tab = line.split()
    if tab[0] != "#":
        nf1, nf2, nf3 =  int(tab[1]) - 1, int(tab[2]) - 1, int(tab[3]) - 1
        n_abc[i_face, :] = nf1, nf2, nf3
        x1 = xn[nf1]; y1 = yn[nf1]; z1 = zn[nf1]; x2 = xn[nf2]; y2 = yn[nf2]; z2 = zn[nf2]
        x3 = xn[nf3]; y3 = yn[nf3]; z3 = zn[nf3]
        df_label[i_face] = int(tab[4])
        r12 = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
        r13 = ((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2) ** 0.5
        r23 = ((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2) ** 0.5
        pp = 0.5 * (r12 + r13 + r23)
        S_abc[i_face] = (pp * (pp - r12) * (pp - r13) * (pp - r23)) ** 0.5
        if i_face < 10:
            print(i_face, int(tab[1]), int(tab[2]), int(tab[3]))
            print(x1, y1, z1, x2, y2, z2, x3, y3, z3, S_abc[i_face])
        i_face += 1
file.close()
print(" SUM n(ABC)  -3 = ", sum([1 for i in range(nnf) if df_label[i] == -3]))
print(" SUM n(ABC)  -2 = ", sum([1 for i in range(nnf) if df_label[i] == -2]))
print(" SUM n(ABC)  -1 = ", sum([1 for i in range(nnf) if df_label[i] == -1]))
print(" SUM S(ABC)  -3 = ", sum([S_abc[i] for i in range(nnf) if df_label[i] == -3]))
print(" SUM S(ABC)  -2 = ", sum([S_abc[i] for i in range(nnf) if df_label[i] == -2]))
print(" SUM S(ABC)  -1 = ", sum([S_abc[i] for i in range(nnf) if df_label[i] == -1]))
print(" SUM S(ABC) all = ", sum(S_abc))


# Now, merge faces to tetrahedrons =================================
# Purpose - faces for exo (external)
print("\n FACESSSS ", datetime.now())
S_tet = lil_matrix((n_tet, 2))     #  diffusion
#  S_tet[i,0] - secreting area; S_tet[i,1] - no. of secreting faces (in a tet)
for i_tet in range(n_tet):
    set_of_tet_nodes = {tn[i_tet, 0] - 1, tn[i_tet, 1] - 1, tn[i_tet, 2] - 1, tn[i_tet, 3] - 1}
    tet_flag = False  #  If False then no secreting faces spotted, otherwise True
    for ii_face in range(nnf):  #  both sets of node numbers counted from 0
        if df_label[ii_face] == -1:
            set_of_face_nodes = {n_abc[ii_face, 0], n_abc[ii_face, 1], n_abc[ii_face, 2]}
            common_verts = list((set_of_tet_nodes).intersection(set_of_face_nodes))
            if (len(common_verts) == 3):
                S_tet_face = S_ABC(common_verts)
                if not tet_flag:
                    S_tet[i_tet, 0]  = S_tet_face; S_tet[i_tet, 1]  = 1
                else:
                    S_tet[i_tet, 0] += S_tet_face; S_tet[i_tet, 1] += 1
                    tet_flag = True
    if i_tet % 2000 == 0:
        print(" TET # ", i_tet, datetime.now())

print(" No. of secreting tr's = ", sum([S_tet[i_tet, 1] for i_tet in range(n_tet)]))
print(" Total secreting area  = ", sum([S_tet[i_tet, 0] for i_tet in range(n_tet)]))


# Number of tetrahedrons is n_tet, rrr is needed for 2d plot (!!)
rrr = np.zeros(n_tet, dtype=np.float)
for i in range(n_tet):
    rrr[i] = (x_int[i] ** 2.0 + y_int[i] ** 2.0 + z_int[i] ** 2.0) ** 0.5


# Initial NT density in tets - assumed as NT in the middle of a tetrahedron
# Draw a test graph NT_dens as function of x_int, y_int, z_int
NT_dens = np.zeros(n_tet, dtype=np.float)
for i in range(n_tet):
    NT_dens[i] = init_dens([x_int[i], y_int[i], z_int[i]])
fig = plt.figure()
cmap_hot = plt.get_cmap("hot")
plt.grid(True)
plt.scatter(rrr, NT_dens, cmap=cmap_hot, c=z_int, vmin=-1.6, vmax=+1.6, s=30.0, lw=0)
plt.xlim(-0.0, 1.6)
plt.ylim(0.0, 40.0)
plt.savefig(fin5_dir_name + 'tsct90nr' + "10000" + '.png')
plt.close()


# print(" WRITE INIT DENSITIES IN NODES TO FILE ")
node_NT = [init_dens([xn[i], yn[i], zn[i]]) for i in range(n_nod)]
dens_N = open(fin5_dir_name + "dens_N_PC.txt", "w")
for i in range(n_nod):
    dens_N.write(str(node_NT[i]) + "\n")
dens_N.close()


print(" \n\n Statistics of NT amount - calculate NT mass array as vol_x_ro ")
print(" NT density in tetrahedra: MIN=", min(NT_dens), " MAX=", max(NT_dens))
vol_x_ro = np.zeros(n_tet, dtype=np.float)
NT_mass = 0.0
Vol_total = 0.0
for i in range(n_tet):
    Vol_total += V_tet[i]
    vol_x_ro[i] = NT_dens[i] * V_tet[i]
    NT_mass += vol_x_ro[i]
print(" Tetrahedron flag(tf) range: ", min(tf), max(tf))
print(" NT mass and bouton volume = ", NT_mass, Vol_total)
print(" Label -3 : ", sum([V_tet[i] for i in range(n_tet) if tf[i] == -3]))
print(" Label -2 : ", sum([V_tet[i] for i in range(n_tet) if tf[i] == -2]))
print(" Label -1 : ", sum([V_tet[i] for i in range(n_tet) if tf[i] == -1]))
print("\n VR ", max(vol_x_ro), " sum(VOL_RO) ", sum(vol_x_ro), " VR \n")


fig = plt.figure()
colmap = cm.ScalarMappable(cmap=cm.hot)
colmap.set_array(NT_dens)
colmap.set_clim(vfliml, vflimh)
cm = plt.get_cmap("hot")
ax = fig.add_subplot(111, projection='3d', alpha=1.0)
ax.grid(True)
ax.scatter(x_int, y_int, z_int, NT_dens, s=3.0, c=NT_dens, cmap=cm, )
plt.xlim(xfliml,xflimh)
plt.ylim(yfliml,yflimh)
plt.tight_layout()
ax.set_zlim(zfliml,zflimh)
cb = fig.colorbar(colmap)
plt.show()
plt.savefig(fin5_dir_name + 'grph_init.png', dpi=300)
plt.close()


print( " \n\n CHECK IF SUM OF FACE AREAS IS ZERO")
delta1_m_i = np.zeros(n_tet)
delta2_m_i = np.zeros(n_tet)
delta3_m_i = np.zeros(n_tet)
# print(" Check S_neigh and r_neigh ")
# matrix_Snei = lil_matrix((n_tet, n_tet))
# matrix_rnei = lil_matrix((n_tet, n_tet))
diff = 0
for itet in range(n_tet):
    if itet % 2000 == 0:
        print(" ITET mod ", itet, datetime.now())
    for i_nei in range(n_nei[itet]):
        id_tnei = id_nei[itet][i_nei]
        if id_tnei > itet:
            diff += S_neigh[itet, i_nei]
        else:
            diff -= S_neigh[itet, i_nei]
print(" Diff = ", diff, "\n\n")


for i in range(0, max_iter):  # 1 diffsn - n_nei: table [n_tet] of no. of tet neighbrs
    #             - id_nei: table [n_tet x 4] of (in fact) n_nei[i_tet] neighs ids
    #             - S_nei:  table [n_tet x 4] of (in fact) n_nei[i_tet] neighs faces
    #             - r_nei:  table [n_tet x 4] of (in fact) n_nei[i_tet] nghs distances
    for i_tet in range(n_tet):
        delta1_m_i[i_tet] = 0.0
        for i_nei in range(n_nei[i_tet]):
            jj = id_nei[i_tet, i_nei]
            delta1_m_i[i_tet] += \
                ((1.0/4.0) * diffusion * S_neigh[i_tet, i_nei] / r_neigh[i_tet, i_nei]) * \
                dt * (NT_dens[jj] - NT_dens[i_tet])


    DIFF_max = max(delta1_m_i)
    # print("Max diff",
    #       [(i, delta1_m_i[i]) for i in range(n_tet)
    #        if delta1_m_i[i] >= 0.99 * DIFF_max], end=" ")
    # print(" NT amnt 0 = ", sum(vol_x_ro),   " NT max 0 = ", max(vol_x_ro), end=" ")
    # print(" DIFF amnt = ", sum(delta1_m_i), " DIFF max = ", DIFF_max,      end=" ")
    # 2 synthesis
    for i_tet in range(n_tet):
        if (NT_dens[i_tet] < ro_) and (tf[i_tet] == -2):  #  label -2 means SynZone here
            delta2_m_i[i_tet] = \
                synthesis_rate * (ro_ - NT_dens[i_tet]) * V_tet[i_tet] * dt
        else:
            delta2_m_i[i_tet] = 0.0
    print(" SYNTH amnt = ", sum(delta2_m_i), end=" ")
    for i_tet in range(n_tet):
        vol_x_ro[i_tet] += delta1_m_i[i_tet] + delta2_m_i[i_tet]
        if vol_x_ro[i_tet] < 0.0:
            vol_x_ro[i_tet] = 0.0
        NT_dens[i_tet] = vol_x_ro[i_tet] / V_tet[i_tet]
    t_volro[i] = sum(vol_x_ro)
    # print(" NT amnt 1 = ", sum(vol_x_ro), " NT max 1 = ", max(vol_x_ro),  end=" ")
    # 3 release
    exocytosis = ((i >  210) and (i <  252)) or ((i >  711) and (i <  753)) or \
                 ((i > 1211) and (i < 1255)) or ((i > 1711) and (i < 1753)) or \
                 ((i > 2000) and ((i % 500) > 210) and ((i % 500) < 256))
    if exocytosis:
        for ii in range(n_tet):
            if S_tet[ii, 1] > 0:
                delta3_m_i[ii] = -permeability * NT_dens[ii] * 3.0 * S_tet[ii, 0] * dt
            else:
                delta3_m_i[ii] = 0.0
            vol_x_ro[ii] += delta3_m_i[ii] * V_tet[ii]
            NT_dens[ii] = vol_x_ro[ii] / V_tet[ii]
        print(" EXO: ", sum([delta3_m_i[ii] * V_tet[ii] for ii in range(n_tet) if S_tet[ii, 1] > 0]), end=" ")
    else:
        print(" EXO: ", 0.0, end=" ")
    # 4 control
    NT_mass = 0.0
    for ii in range(n_tet):
        NT_mass += NT_dens[ii] * V_tet[ii]
    print(" NT AMOUNT=", NT_mass, " AMIN=", min(vol_x_ro), " AMAX=", max(vol_x_ro),
          " RHOMIN=", min(NT_dens), " RHOMAX=", max(NT_dens),
          ", TIME=", datetime.now(), end=" ")
    # print(" << I =", i, ">> AMIN=", min(NT_dens), " AMAX=", max(NT_dens))
    print(" << I =", i, ">>")
    # 5 draw 3d and 2d plots of density
    if (i % n_plot == 0) or exocytosis:
        fig = plt.figure()
        cm = plt.get_cmap("hot")
        ax = fig.add_subplot(111, projection='3d', alpha=1.0)
        ax.grid(True)
        ax.scatter(x_int, y_int, z_int, NT_dens, c=NT_dens, s=sss*NT_dens, cmap=cm,
                   vmin=vfliml, vmax=vflimh, lw=0)
        plt.xlim(xfliml, xflimh)
        plt.ylim(yfliml, yflimh)
        plt.tight_layout()
        ax.set_zlim(zfliml, zflimh)
        cb = fig.colorbar(colmap)
        plt.savefig(fin5_dir_name + '/tur_' + str(10000 + i) + '.png', dpi=300)
        plt.close()
        ###################################################################################
        fig = plt.figure()
        cmap_hot = plt.get_cmap("hot")
        plt.grid(True)
        plt.scatter(rrr, NT_dens, cmap=cmap_hot, c=z_int, vmin=-1.6, vmax=+1.6, s=30.0, lw=0)
        plt.xlim(-0.00, 0.65)
        plt.ylim(280.0, 520.0)
        plt.savefig(fin5_dir_name + 'tsct90nr' + str(10000 + i) + '.png')
        plt.close()


print( " \n And now, ... the plot of NT_AMOUNT vs iteration \n ")
print(" limits ", min(t_time), max(t_time), min(t_volro), max(t_volro))


fig = plt.figure()
cmap_hot = plt.get_cmap("hot")
plt.grid(True)
plt.scatter(t_time, t_volro, c='r')
plt.xlim(0.0, 0.02)
plt.ylim(280.0, 520.0)
plt.show()
plt.close()
