
#This script if based on Jason McClelland's heuristic algorithm for computing the EMD. It needs and input distance D and probability vectors A and B (all as .mat files)
import numpy as np
import scipy.io as sio
import h5py
import os, sys, copy

# Need a better way to deal with the distance matrices....hdf5?

def EMDeBruijnHeuristic(Distance, inMassA, inMassB):
    MassA = copy.copy(inMassA)
    MassB = copy.copy(inMassB)
    maxd = np.max(Distance)
    MassA /= np.sum(MassA)
    MassB /= np.sum(MassB)
    EMD = 0  # Initialize the EMD computation.
    d = 0  # Initialize distance to move mass at 0.
    InitialMove = [min([MassA[i], MassB[i]]) for i in range(0, len(MassA))]
    Flow = np.zeros([len(MassA), len(MassA)], dtype=np.float64)  #Later, can make this a sparse matrix for computational efficiency
    MassA -= InitialMove  # Removes mass that was moved initially.
    MassB -= InitialMove
    while sum(MassA) > 1e-10 and d <= maxd:  # While we still have mass to move,
        d += 1  # increment distance to move
        IndicesSortedMassSourceA = np.flipud(np.argsort(MassA))  # sort the sources, big to small.
        for SourceA in IndicesSortedMassSourceA:  # Now, for each source of mass in A
            if MassA[SourceA] == 0:  # Have we gotten through all the sources with mass left? If so, break.
                break
            dNeighborsSourceA = np.argwhere(Distance[SourceA, :] == d)# Find the n-mers in B which are distance d from our source.
            dNeighborsSourceA = dNeighborsSourceA.flatten()
            IndicesSortedMassB = np.flipud(np.argsort(MassB[dNeighborsSourceA]))  # We order the sinks, we're going to fill from the top down.
            for SinkB in IndicesSortedMassB:  # Iterating over the indices of the sinks.
                if MassB[dNeighborsSourceA[SinkB]] == 0:
                    break
                if MassA[SourceA] - MassB[dNeighborsSourceA[SinkB]] >= 0:  # check to see if our source can fulfil this sink.
                    Flow[SourceA, dNeighborsSourceA[SinkB]] = MassB[dNeighborsSourceA[SinkB]]  # If so, note the flow.
                    EMD += d*MassB[dNeighborsSourceA[SinkB]]  # update the EMD calc,
                    MassA[SourceA] = MassA[SourceA] - MassB[dNeighborsSourceA[SinkB]]  # then remove mass from A
                    MassB[dNeighborsSourceA[SinkB]] = 0  # and remove mass from B.
                else: # otherwise, we've run out of mass from SourceA.
                    Flow[SourceA, dNeighborsSourceA[SinkB]] = MassA[SourceA]  # If so, note the flow to this last sink,
                    EMD += d*MassA[SourceA]  # update the EMD calc,
                    MassB[dNeighborsSourceA[SinkB]] = MassB[dNeighborsSourceA[SinkB]]-MassA[SourceA]  # remove mass from B,
                    MassA[SourceA] = 0  # then remove mass from A
                    break  # and end the loop, with no more mass to distribute from this source.
    return EMD


def EMDeBruijnHeuristicFromFiles(distmatfile, sampleAfile, sampleBfile):
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
    MassA = np.genfromtxt(sampleAfile)
    MassB = np.genfromtxt(sampleBfile)
    return EMDeBruijnHeuristic(Distance, MassA, MassB)





# Test
#distmatfile = '../Test/D6Symm.mat'
#sampleAfile = '../Test/SRR3545955-6mers.txt'
#sampleBfile = '../Test/SRR3546358-6mers.txt'
#print(EMDeBruijnHeuristicFromFiles(distmatfile, sampleAfile, sampleBfile))
