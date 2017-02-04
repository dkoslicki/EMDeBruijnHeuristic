
#This script if based on Jason McClelland's heuristic algorithm for computing the EMD. It needs and input distance D and probability vectors A and B (all as .mat files)
import numpy
import scipy.io as sio

contents = sio.loadmat('/nfs1/Koslicki_Lab/koslickd/CAMI/EvaluationMetrics/EMD/PhylogeneticTrees/D.mat')
Distance = contents['D'] # Using the distance matrix formed in the other script.

contents = sio.loadmat('/nfs1/Koslicki_Lab/koslickd/CAMI/EvaluationMetrics/EMD/PhylogeneticTrees/A.mat')
A = contents['A']
contents = sio.loadmat('/nfs1/Koslicki_Lab/koslickd/CAMI/EvaluationMetrics/EMD/PhylogeneticTrees/B.mat')
B = contents['B']

MassA = A[0]  # Pick off the masses for the support of A and B (We'll be editting these, so we leave A, B alone)
MassB = B[0]

EMD = 0  # Initialize the EMD computation.

d = 0  # Initialize distance to move mass at 0.

InitialMove = [min([MassA[i], MassB[i]]) for i in range(0, len(MassA))]
Flow = numpy.zeros([len(MassA), len(MassA)], dtype=numpy.float64)  #Later, can make this a sparse matrix for computational efficiency

MassA -= InitialMove # Removes mass that was moved initially.
MassB -= InitialMove

while sum(MassA) > 1e-10:  # While we still have mass to move,
    d += 1 # increment distance to move
    IndicesSortedMassSourceA = numpy.flipud(numpy.argsort(MassA))  # sort the sources, big to small.
    for SourceA in IndicesSortedMassSourceA:  # Now, for each source of mass in A
        if MassA[SourceA] == 0:  # Have we gotten through all the sources with mass left? If so, break.
            break

        dNeighborsSourceA = numpy.argwhere(Distance[SourceA, :] == d)# Find the n-mers in B which are distance d from our source.
        dNeighborsSourceA = dNeighborsSourceA.flatten()
        IndicesSortedMassB = numpy.flipud(numpy.argsort(MassB[dNeighborsSourceA]))  # We order the sinks, we're going to fill from the top down.
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
print EMD



