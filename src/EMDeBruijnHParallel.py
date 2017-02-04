# This script will use EMDeBruijnHueristic.py to compute it (in parallel) over a large set of file names
import numpy as np
import multiprocessing as mp
import ctypes, os, sys, time, getopt
import EMDeBruijnHeuristic as EMD
import h5py
import scipy.io as sio


try:
    opts, args = getopt.getopt(sys.argv[1:],"f:d:o:h",["CountsFileNames=", "DistanceFile=","OutputFile="])
except getopt.GetoptError:
    print 'Unknown option, call using: python EMDeBruijnHParallel.py -f <CountsFileNames> -d <DistanceFileName> -o <OutputFile>'
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'call using: python EMDeBruijnHParallel.py -f <CountsFileNames> -d <DistanceFileName> -o <OutputFile>'
        sys.exit(2)
    elif opt in ("-f", "--CountsFileNames"):
        file_names_file = arg
    elif opt in ("-d", "--DistanceFile"):
        distmatfile = arg
    elif opt in ("-o", "--OutputFile"):
        output_file = arg


# Make the wrapper to do the EMDeBruijn calculation
def EMD_wrapper(tuple):
    (i, j) = tuple
    val = EMD.EMDeBruijnHeuristic(shared_dist, shared_counts[i, :], shared_counts[j, :])
    return val

# Get the distance matrix
extension = os.path.basename(distmatfile).split('.')[-1]
if extension == 'hdf5' or extension == 'h5':
    f = h5py.File(distmatfile, 'r')
    Distance = f['D'][:]
elif extension == 'mat':
    contents = sio.loadmat(distmatfile)
    Distance = contents['D']
else:
    print("Error: only HDF5 files or Matlab MAT files allowed")
    sys.exit(2)

# Get all the file names
fid = open(file_names_file, 'r')
file_names = list()
for file in fid:
    file_names.append(file.strip())
fid.close()

# Read a single one in to get the data size
file = file_names[0]
counts = np.genfromtxt(file)
num_kmers = len(counts)

# Import all the counts into one big shared array
shared_counts_base = mp.Array(ctypes.c_double, len(file_names)*num_kmers)
shared_counts = np.ctypeslib.as_array(shared_counts_base.get_obj())
shared_counts = shared_counts.reshape(len(file_names), num_kmers)

i = 0
for file in file_names:
    counts = np.genfromtxt(file)
    counts /= np.sum(counts)
    shared_counts[i, :] = counts[:]
    i += 1

# Put the distance matrix in a shared array
shared_dist_base = mp.Array(ctypes.c_double, num_kmers*num_kmers)
shared_dist = np.ctypeslib.as_array(shared_dist_base.get_obj())
shared_dist = shared_dist.reshape(num_kmers, num_kmers)
shared_dist[:, :] = Distance[:, :]

tuples =list()
for i in range(len(file_names)):
    for j in range(len(file_names)):
        if i > j:
            tuples.append((i, j))

pool = mp.Pool(processes=mp.cpu_count())
res = pool.map(EMD_wrapper, tuples)
pool.close()

res_dist = np.zeros((len(file_names), len(file_names)))
for it in range(len(tuples)):
    (i, j) = tuples[it]
    val = res[it]
    res_dist[i, j] = val
    res_dist[j, i] = val

np.savetxt(output_file, res_dist)
