from jax._src.typing import Array
import iqpopt as iqp
import iqpopt.gen_qml as genq
from iqpopt.utils import local_gates
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)

def loss_estimate_iqp2(params1: jnp.ndarray, params2: jnp.ndarray, iqp_circuit1: iqp.IqpSimulator, iqp_circuit2: iqp.IqpSimulator,
                        all_ops1: jnp.ndarray, all_ops2: jnp.ndarray, n_samples: int, key: Array, init_coefs1: list = None,
                        init_coefs2: list = None, indep_estimates: bool = False, sqrt_loss: bool = False, return_expvals: bool = False,
                        max_batch_ops: int = None, max_batch_samples: int = None) ->  float | list:
    """Estimates the MMD Loss of an IQP circuit with respect to another IQP circuit distribution.

    Args:
        params1 (jnp.ndarray): The parameters of the IQP circuit 1 gates.
        params2 (jnp.ndarray): The parameters of the IQP circuit 2  gates.
        iqp_circuit1 (IqpSimulator): The IQP circuit 1 given by the class IqpSimulator.
        iqp_circuit2 (IqpSimulator): The IQP circuit 2 given by the class IqpSimulator.
        all_ops1 (jnp.ndarray): Matrix with the all the operators as rows (0s and 1s). Used to estimate the IQP 1 part of the loss.
        all_ops2 (jnp.ndarray): Matrix with the all the operators as rows (0s and 1s). Used to estimate the IQP 2 part of the loss.
        n_samples (int): Number of samples used to estimate the loss.
        key (Array): Jax key to control the randomness of the process.
        init_coefs1 (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates for the IQP circuit 1.
        init_coefs2 (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates for the IQP circuit 2.
        indep_estimates (bool, optional): Whether to use independent estimates of the ops in a batch (takes longer). Defaults to False.
        sqrt_loss (bool, optional): Whether to use the square root of the MMD^2 loss. Note estiamtes will no longer be unbiased. Defaults to False.
        return_expvals (bool, optional): Whether to return the expectation values of the IQP circuit or return the loss. Defaults to False.
        max_batch_ops (int, optional): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int, optional): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
         float | list: The value of the loss or list of expectation values for each operator.
    """
    tr_iqp_samples1 = iqp_circuit1.op_expval(params1, all_ops1, n_samples, key, init_coefs1, indep_estimates, return_samples=True,
                                           max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)
    tr_iqp_samples2 = iqp_circuit2.op_expval(params2, all_ops2, n_samples, key, init_coefs2, indep_estimates, return_samples=True,
                                           max_batch_ops=max_batch_ops, max_batch_samples=max_batch_samples)
    
    correction1 = jnp.mean(tr_iqp_samples1**2, axis=-1)/n_samples
    correction2 = jnp.mean(tr_iqp_samples2**2, axis=-1)/n_samples
    
    tr_iqp1 = jnp.mean(tr_iqp_samples1, axis=-1)
    tr_iqp2 = jnp.mean(tr_iqp_samples2, axis=-1)

    # add correction to make the first term unbiased
    res = (tr_iqp1*tr_iqp1-correction1)*n_samples/(n_samples-1) - \
          2*tr_iqp1*tr_iqp2 + \
          (tr_iqp2*tr_iqp2-correction2)*n_samples/(n_samples-1)

    res = jnp.mean(res) if not return_expvals else res
    res = jnp.sqrt(jnp.abs(res)) if sqrt_loss else res

    return res


def mmd_loss_iqp2(params1: jnp.ndarray, params2: jnp.ndarray, iqp_circuit1: iqp.IqpSimulator, iqp_circuit2: iqp.IqpSimulator,
                 sigma: float | list, n_ops: int, n_samples: int, key: Array, init_coefs1: list = None, init_coefs2: list = None,
                 wires1: list = None, wires2: list = None, indep_estimates: bool = False, jit: bool = True, sqrt_loss: bool = False,
                 return_expvals: bool = False, max_batch_ops: int = None, max_batch_samples: int = None) -> float | list:
    """Returns an estimate of the (squared) MMD Loss of an IQP circuit with respect to another IQP circuit distribution.
     The function uses a randomized method whose precision can be increased by using larger values of n_samples and/or
     n_ops.

    Args:
        params1 (jnp.ndarray): The parameters of the IQP circuit 1 gates.
        params2 (jnp.ndarray): The parameters of the IQP circuit 2  gates.
        iqp_circuit1 (IqpSimulator): The IQP circuit 1 given by the class IqpSimulator.
        iqp_circuit2 (IqpSimulator): The IQP circuit 2 given by the class IqpSimulator.
        sigma (float or list): The bandwidth of the kernel. If several are given as a list the average loss over each value will
            be returned.
        n_ops (int): Number of operators used to estimate the loss.
        n_samples (int): Number of samples used to estimate the loss.
        key (jax.random.PRNGKey): Jax PRNG key used to seed random functions.
        init_coefs1 (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates for the IQP circuit 1.
        init_coefs2 (list[float], optional): List or array of length len(init_gates) that specifies the fixed parameter
                values of init_gates for the IQP circuit 2.
        wires1 (list, optional): List of qubit positions that specifies the qubits whose measurement statistics are
            used to estimate the MMD loss. The remaining qubits will be traced out. Defaults to None, meaning all
            qubits are used. This is for the IQP circuit 1.
        wires2 (list, optional): List of qubit positions that specifies the qubits whose measurement statistics are
            used to estimate the MMD loss. The remaining qubits will be traced out. Defaults to None, meaning all
            qubits are used. This is for the IQP circuit 2.
        indep_estimates (bool): Whether to use independent estimates when estimating expvals of ops (takes longer).
        jit (bool): Whether to jit the loss (works only for circuits with sparse=False). Defaults to True.
        sqrt_loss (bool): Whether to use the square root of the MMD^2 loss. Note estimates will no longer be unbiased.
            Defaults to False.
        return_expvals (bool): If True, the expectation values of the IQP circuit used to estimate the loss are
            returned. Defaults to False.
        max_batch_ops (int): Maximum number of operators in a batch of op_expval. Defaults to None.
        max_batch_samples (int): Maximum number of samples in a batch of op_expval. Defaults to None.

    Returns:
        float | list: The value of the loss or
                      the list of expectation values for each operator or
                      the list of the losses for each sigma or
                      the list of the list of expectation values for each operator for each sigma.
    """
    
    
    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")
    
    if wires1 is None:
        wires1 = list(range(iqp_circuit1.n_qubits))
    
    if wires2 is None:
        wires2 = list(range(iqp_circuit2.n_qubits))
        
    if len(wires1) != len(wires2):
        raise ValueError("The number of wires 1 and 2 must be equal so the number of working qubits are the same.")
    
    if max(wires1) >= iqp_circuit1.n_qubits:
        raise ValueError("The wires1 qubit numbers can't be higher than the number of qubits in the circuit1.")
    
    if max(wires2) >= iqp_circuit2.n_qubits:
        raise ValueError("The wires2 qubit numbers can't be higher than the number of qubits in the circuit2.")
    
    sigmas = [sigma] if isinstance(sigma, (int, float)) else sigma
    init_coefs1 = jnp.array(init_coefs1) if init_coefs1 is not None else None
    init_coefs2 = jnp.array(init_coefs2) if init_coefs2 is not None else None
        
    losses = []
    for sigma in sigmas:
        p_MMD = (1-jnp.exp(-1/2/sigma**2))/2
        
        key, subkey = jax.random.split(key, 2)
        visible_ops1 = jnp.array(jax.random.binomial(
            subkey, 1, p_MMD, shape=(n_ops, len(wires1))), dtype='float64')
        
        key, subkey = jax.random.split(key, 2)
        visible_ops2 = jnp.array(jax.random.binomial(
            subkey, 1, p_MMD, shape=(n_ops, len(wires2))), dtype='float64')

        all_ops1 = []
        i = 0
        for q in range(iqp_circuit1.n_qubits):
            if q in wires1:
                all_ops1.append(visible_ops1[:, i])
                i += 1
            else:
                all_ops1.append(jnp.zeros(n_ops))
        all_ops1 = jnp.array(all_ops1, dtype='float64').T

        all_ops2 = []
        i = 0
        for q in range(iqp_circuit2.n_qubits):
            if q in wires2:
                all_ops2.append(visible_ops2[:, i])
                i += 1
            else:
                all_ops2.append(jnp.zeros(n_ops))
        all_ops2 = jnp.array(all_ops2, dtype='float64').T
        
        
        if jit and not iqp_circuit1.sparse and not iqp_circuit2.sparse:
            loss = jax.jit(loss_estimate_iqp2, static_argnames=[
                        "iqp_circuit1", "iqp_circuit2", "n_samples", "indep_estimates", "sqrt_loss", "return_expvals",
                        "max_batch_ops", "max_batch_samples"])
        else:
            loss = loss_estimate_iqp2

        losses.append(loss(params1, params2, iqp_circuit1, iqp_circuit2, all_ops1, all_ops2, n_samples, key, init_coefs1,
                           init_coefs2, indep_estimates, sqrt_loss, return_expvals=return_expvals, max_batch_ops=max_batch_ops,
                           max_batch_samples=max_batch_samples))

    if return_expvals:
        return losses
    else:
        return sum(losses)/len(losses)
    
