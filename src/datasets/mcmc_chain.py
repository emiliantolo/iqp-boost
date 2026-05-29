import jax
import jax.numpy as jnp
from functools import partial
import numpy as np

@partial(jax.jit, static_argnames=['n_qubits', 'burn_in', 'thinning', 'sweeps_per_sample', 'n_samples_per_chain'])
def _jax_mcmc_trajectory(chain_key, J, beta, n_qubits, burn_in, thinning, sweeps_per_sample, n_samples_per_chain):
    """Runs a single MCMC chain, burns it in, and collects multiple thinned samples."""
    k_init, k_burn, k_sample = jax.random.split(chain_key, 3)
    
    # Initialize random spins on device: {-1.0, 1.0}
    spins = jax.random.choice(k_init, jnp.array([-1.0, 1.0]), shape=(n_qubits,))
    
    def single_sweep(s, key):
        k1, k2 = jax.random.split(key)
        indices = jax.random.permutation(k1, n_qubits)
        rands = jax.random.uniform(k2, (n_qubits,))
        
        def update_spin(s_inner, x):
            idx, rand_val = x
            local_field = jnp.dot(J[idx], s_inner)
            p_plus = 1.0 / (1.0 + jnp.exp(-2.0 * beta * local_field))
            new_spin = jnp.where(rand_val < p_plus, 1.0, -1.0)
            return s_inner.at[idx].set(new_spin), None
            
        s_out, _ = jax.lax.scan(update_spin, s, (indices, rands))
        return s_out, None

    # 1. Burn-in Phase (Executed once per chain)
    burn_keys = jax.random.split(k_burn, burn_in)
    spins, _ = jax.lax.scan(single_sweep, spins, burn_keys)
    
    # 2. Sampling Phase (Collect thinned states along the trajectory)
    total_sweeps = n_samples_per_chain * thinning * sweeps_per_sample
    sample_keys = jax.random.split(k_sample, total_sweeps).reshape(n_samples_per_chain, thinning * sweeps_per_sample, 2)
    
    def sample_step(s_carry, keys_chunk):
        # Step forward by 'thinning' sweeps
        s_next, _ = jax.lax.scan(single_sweep, s_carry, keys_chunk)
        # Convert to binary {0, 1} int8 directly on-device to minimize H2D transfer size
        bits = ((1.0 - s_next) / 2.0).astype(jnp.int8)
        return s_next, bits

    _, sample_trajectory = jax.lax.scan(sample_step, spins, sample_keys)
    return sample_trajectory


def run_mcmc_chains(J: np.ndarray, beta: float, n_samples: int, burn_in: int, thinning: int = 1, sweeps_per_sample: int = 1, n_chains: int = 32, seed: int = 0) -> np.ndarray:
    """Runs a fixed pool of parallel MCMC chains and harvests samples along their trajectories."""
    actual_chains = min(int(n_chains), int(n_samples))
    actual_chains = max(actual_chains, 1)
    samples_per_chain = int(np.ceil(n_samples / actual_chains))
    
    key = jax.random.PRNGKey(int(seed))
    chain_keys = jax.random.split(key, actual_chains)
    J_jnp = jnp.array(J, dtype=jnp.float32)
    
    # Parallelize across the fixed pool of chains
    vmapped_sampler = jax.vmap(
        lambda k: _jax_mcmc_trajectory(k, J_jnp, float(beta), J.shape[0], int(burn_in), int(thinning), int(sweeps_per_sample), samples_per_chain)
    )
    
    # Shape: (actual_chains, samples_per_chain, n_qubits)
    samples_jnp = vmapped_sampler(chain_keys)
    
    # Flatten the chain pool on the host safely and cleanly
    samples = np.asarray(samples_jnp.reshape(-1, J.shape[0]))
    return samples[:n_samples]