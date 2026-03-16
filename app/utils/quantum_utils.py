import numpy as np
import torch
try:
    import pennylane as qml
    from pennylane.qnn import TorchLayer
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False

def get_quantum_device(n_qubits, device_type="default.qubit"):
    """
    Initialize and return a PennyLane device.
    """
    if not PENNYLANE_AVAILABLE:
        return None
    return qml.device(device_type, wires=n_qubits)

def vqc_circuit(inputs, weights, n_qubits):
    """
    Variational Quantum Circuit definition.
    Extracted from CNN_WITH_VQC.ipynb.
    """
    qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

def qstate(x, n_qubits, device):
    """
    Compute quantum state from input angles.
    Extracted from CNN_SEP_FINETUNING_QSVM.ipynb.
    """
    @qml.qnode(device, interface="autograd")
    def _circuit(inputs):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])  # entangler
        return qml.state()
    
    return _circuit(x)

def compute_states(X_angles, n_qubits, device):
    """
    Compute quantum states for a batch of angle vectors.
    """
    return np.vstack([np.array(qstate(x, n_qubits, device)) for x in X_angles])

def kernel_from_states(A, B):
    """
    Compute the quantum kernel matrix from two sets of quantum states.
    Extracted from CNN_SEP_FINETUNING_QSVM.ipynb.
    """
    inner = np.dot(A.conj(), B.T)
    return np.abs(inner)**2

def scale_to_angles(X, mins=None, ranges=None):
    """
    Scale features to angles in the range [-pi/2, pi/2].
    """
    if mins is None:
        mins = X.min(axis=0)
    if ranges is None:
        maxs = X.max(axis=0)
        ranges = np.where(maxs - mins == 0, 1, maxs - mins)
    
    Xn = (X - mins) / ranges
    return (Xn - 0.5) * np.pi
