for each power of activating controlled U there is a folder.
in each folder:
num_points = 50
T2_list = np.linspace(Tgate*1, Tgate*100, num_points, endpoint=True)
num_angles = 50
phi_list = np.linspace(0,1,num_angles,endpoint=False)

idealState_angle<angle> - ideal state before measurement for Kitaev circuit with K=I and for the angle <angle> of controlled rotation
idealStateK_angle<angle> - ideal state before measurement for Kitaev circuit with K=S and for the angle <angle> of controlled rotation

traditionalState_T2<T2>_phi<angle> - state of traditional algorithm before measurement for Kitaev circuit with K=I and for T2 of <T2> and angle phi <angle>
traditionalStateK_T2<T2>_phi<angle> - state of traditional algorithm before measurement for Kitaev circuit with K=S and for T2 of <T2> and angle phi <angle>

logicalState_T2<T2>_phi<angle> - state of logical (6 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=I and for T2 of <T2> and angle phi <angle>
logicalStateK_T2<T2>_phi<angle> - state of logical (6 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=S and for T2 of <T2> and angle phi <angle>

logicalState1EC_T2<T2>_phi<angle> - state of logical (6 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=I and for T2 of <T2> and angle phi <angle>, with only one syndrome measurement at the end
logicalState1ECK_T2<T2>_phi<angle> - state of logical (6 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=S and for T2 of <T2> and angle phi <angle>, with only one syndrome measurement at the end

flaggedState_T2<T2>_phi<angle> - state of logical (8 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=I and for T2 of <T2> and angle phi <angle>, with flags for fault-tolerant rotation
flaggedStateK_T2<T2>_phi<angle> - state of logical (8 qubits) algorithm (4x4 matrix) before measurement for Kitaev circuit with K=S and for T2 of <T2> and angle phi <angle>, with flags for fault-tolerant rotation



traditional_T2_<T2>_angle_fidelities - list of fidelity(ideal,traditional) for all angles for given T2=<T2> and K=I
traditionalK_T2_<T2>_angle_fidelities - list of fidelity(ideal,traditional) for all angles for given T2=<T2> and K=S
traditional_T2_<T2>_lost_information - list of [1-trace(traditional state)] for all angles for given T2=<T2> and K=I
traditionalK_T2_<T2>_lost_information - list of [1-trace(traditional state)] for all angles for given T2=<T2> and K=S

logical_T2_<T2>_angle_fidelities - list of fidelity(ideal,logical) for all angles for given T2=<T2> and K=I
logicalK_T2_<T2>_angle_fidelities - list of fidelity(ideal,logical) for all angles for given T2=<T2> and K=S
logical_T2_<T2>_lost_information - list of [1-trace(logical state)] for all angles for given T2=<T2> and K=I
logicalK_T2_<T2>_lost_information - list of [1-trace(logical state)] for all angles for given T2=<T2> and K=S

logical1EC_T2_<T2>_angle_fidelities - list of fidelity(ideal,logical1EC) for all angles for given T2=<T2> and K=I, with only one syndrome measurement at the end
logical1ECK_T2_<T2>_angle_fidelities - list of fidelity(ideal,logical1EC) for all angles for given T2=<T2> and K=S, with only one syndrome measurement at the end
logical1EC_T2_<T2>_lost_information - list of [1-trace(logical1EC state)] for all angles for given T2=<T2> and K=I, with only one syndrome measurement at the end
logical1ECK_T2_<T2>_lost_information - list of [1-trace(logical1EC state)] for all angles for given T2=<T2> and K=S, with only one syndrome measurement at the end

flagged_T2_<T2>_angle_fidelities - list of fidelity(ideal,flagged) for all angles for given T2=<T2> and K=I, and fault tolerant rotation with flags
flaggedK_T2_<T2>_angle_fidelities - list of fidelity(ideal,flagged) for all angles for given T2=<T2> and K=S, and fault tolerant rotation with flags
flagged_T2_<T2>_lost_information - list of [1-trace(flagged state)] for all angles for given T2=<T2> and K=I, and fault tolerant rotation with flags
flaggedK_T2_<T2>_lost_information - list of [1-trace(flagged state)] for all angles for given T2=<T2> and K=S, and fault tolerant rotation with flags



traditional_lost_information_average
traditional_fidelity_average - list of average (over angles) fidelity between ideal and traditional for different T2s, and K=I
traditionalK_lost_information_average
traditionalK_fidelity_average - list of average (over angles) fidelity between ideal and traditional for different T2s, and K=S

logical_lost_information_average
logical_fidelity_average - list of average (over angles) fidelity between ideal and logical for different T2s, and K=I
logicalK_lost_information_average
logicalK_fidelity_average - list of average (over angles) fidelity between ideal and logical for different T2s, and K=S

logical1EC_lost_information_average
logical1EC_fidelity_average - list of average (over angles) fidelity between ideal and logical with one LPS at the end for different T2s, and K=I
logical1ECK_lost_information_average
logical1ECK_fidelity_average - list of average (over angles) fidelity between ideal and logical with one LPS at the end for different T2s, and K=S

flagged_lost_information_average
flagged_fidelity_average - list of average (over angles) fidelity between ideal and logical with flags for different T2s, and K=I
flaggedK_lost_information_average
flaggedK_fidelity_average - list of average (over angles) fidelity between ideal and logical with flags for different T2s, and K=S






