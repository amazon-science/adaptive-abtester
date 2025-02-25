import numpy as np
from tqdm import tqdm

def compute_step_sizes(t, T, old=False, alpha=0.49, beta=0):
    if old:
        eta_t = np.sqrt(1 / T)
        delta_t = 0.5 * ((t + 1) ** (-((5 * np.log(T)) ** -0.5)))
    else:
        eta_t = 1 / (alpha * (t + 1))
        h_t = beta + np.exp((np.log(t + 2)) ** 0.25)
        delta_t = 1 / h_t
    return eta_t, delta_t


def sample_treatment_and_outcome(Pt, y1_val, y0_val, rng=None):
    Zt = np.random.binomial(1, Pt) if rng is None else rng.binomial(1, Pt)
    Yt = Zt * y1_val + (1 - Zt) * y0_val
    return Zt, Yt


def update_probabilities(Pt, Gt, eta_t, delta_t):
    Pt_updated = np.clip(Pt - eta_t * Gt, delta_t, 1 - delta_t)
    return Pt_updated


def compute_gradient(Yt, Pt, Zt):
    return (Yt**2) * (-Zt / (Pt**3) + (1 - Zt) / ((1 - Pt) ** 3))


def gd(
    y1,
    y0,
    dim=1,
    rng=None,
    old=False,
    alpha=0.49,
    beta=0,
    subsample=1,
    tqdm=False,
):

    y1 = np.array(y1)[:, np.newaxis]
    y0 = np.array(y0)[:, np.newaxis]

    T = y1.shape[0]
    T_sub = round(T / subsample)
    Pt_history = np.zeros((T_sub, dim))
    Zt_history = np.zeros((T_sub, dim))
    Yt_history = np.zeros((T_sub, dim))

    Pt = 0.5 * np.ones(dim)
    Gt = np.zeros(dim)

    iterator = range(T)
    if tqdm:
        iterator = tqdm(iterator)

    for t in iterator:
        eta_t, delta_t = compute_step_sizes(t, T, old, alpha, beta)
        Pt = update_probabilities(Pt, Gt, eta_t, delta_t)
        assert (
            ~np.any(np.isnan(Pt)) and Pt.max() <= 1 and Pt.min() >= 0
        ), "something went wrong with the probability update"
        Zt, Yt = sample_treatment_and_outcome(Pt, y1[t], y0[t], rng)
        assert (
            ~np.any(np.isnan(Zt)) and Zt.max() <= 1 and Zt.min() >= 0
        ), "something went wrong with ... sampling?"
        assert ~np.any(np.isnan(Yt)), "something went wrong with ... sampling?"
        Gt = compute_gradient(Yt, Pt, Zt)
        assert ~np.any(np.isnan(Gt)), "something went wrong with gradient computation?"

        add_ind = round(t / subsample)
        if t % subsample == 0 and add_ind < T_sub:
            Pt_history[add_ind] = Pt
            Zt_history[add_ind] = Zt
            Yt_history[add_ind] = Yt

    HT_vals = Yt_history * (
        Zt_history / Pt_history - (1 - Zt_history) / (1 - Pt_history)
    )
    HT_vals_cumulative = np.cumsum(HT_vals, axis=0)
    est = (
        HT_vals_cumulative
        / np.arange(1, HT_vals_cumulative.shape[0] + 1)[:, np.newaxis]
    )

    return Pt_history, Zt_history, Yt_history


def gd_noclip(y1, y0, dim=1, rng=None, old=False, subsample=1):

    T = len(y1)
    T_sub = int(T / subsample)
    Pt_history = np.zeros((T_sub, dim))
    Zt_history = np.zeros((T_sub, dim))
    Yt_history = np.zeros((T_sub, dim))

    Pt = 0.5 * np.ones(dim)
    Gt = np.zeros(dim)

    for t in tqdm(range(len(y1))):
        eta_t = np.sqrt(1 / len(y1)) if old else 1 / (t + 1)
        Pt = np.clip(Pt - eta_t * Gt, 0, 1)

        Zt = np.random.binomial(1, Pt) if rng is None else rng.binomial(1, Pt)
        Yt = Zt * y1[t] + (1 - Zt) * y0[t]

        Gt = (Yt**2) * (-Zt / (Pt**3) + (1 - Zt) / ((1 - Pt) ** 3))

        add_ind = int(t / subsample)
        if t % subsample == 0 and add_ind < T_sub:
            Pt_history[add_ind], Zt_history[add_ind], Yt_history[add_ind] = Pt, Zt, Yt

    return Pt_history.T, Zt_history.T, Yt_history.T


def gd_amgate(
    y1,
    y0,
    sets,
    dim=1,
    rng=None,
    old=False,
    alpha=0.49,
    beta=0,
    subsample=1,
    tqdm=False,
):
    T = len(y1)
    T_sub = round(T / subsample)
    n_groups = sets.shape[0]
    Pt_history = np.zeros((T_sub, dim))
    Ptg_history = np.zeros((T_sub, n_groups, dim))
    group_weight_history = np.zeros((T_sub, n_groups, dim))
    Zt_history = np.zeros((T_sub, dim))
    Yt_history = np.zeros((T_sub, dim))

    Pt = 0.5 * np.ones(dim)
    Gt = np.zeros(dim)

    iterator = range(T)
    if tqdm:
        iterator = tqdm(iterator)

    assert (
        sets.ndim == 2 and sets.shape[1] == y1.shape[0]
    ), "got incompatible sets shape, expected gxT matrix and got {}".format(sets.shape)

    # Create single AMGATE instance with k parallel sequences
    amgate_instance = AMGATE(d=n_groups, k=dim)
    for t in iterator:
        # Retrieve groups and add rep dimension
        active_sets = np.tile(sets[None, :, t], reps=[dim, 1])
        # # Set outcomes for this round
        outcomes = np.array([[y0[t], y1[t]]] * dim)
        # Execute one step
        Pt, Zt, Yt, group_weights, Ptg = amgate_instance.step(active_sets, outcomes)

        add_ind = round(t / subsample)
        if t % subsample == 0 and add_ind < T_sub:
            Pt_history[add_ind] = Pt
            Zt_history[add_ind] = Zt
            Yt_history[add_ind] = Yt
            Ptg_history[add_ind] = Ptg.T
            group_weight_history[add_ind] = group_weights.T

    return Pt_history, Zt_history, Yt_history, Ptg_history, group_weight_history


def get_neyman_stats(y1, y0):
    
    S0, S1 = np.mean(y0**2, axis=0), np.mean(y1**2, axis=0)
    p_star = 1 / (1 + np.sqrt(S0 / S1))

    var_starT = get_neyman_prob(y1, y0)
    return p_star, var_starT


def get_neyman_prob(y1, y0):
    
    y1, y0 = np.array(y1), np.array(y0)

    S0 = np.cumsum(y0.reshape(-1) ** 2)
    S1 = np.cumsum(y1.reshape(-1) ** 2)
    return 1 / (1 + np.sqrt(S0 / S1))


def get_rescaled_neyman_variance(y1, y0):

    y1, y0 = np.array(y1), np.array(y0)

    T = y1.shape[0]
    S0 = np.cumsum(y0**2, axis=0) / np.arange(1, T + 1)[:, np.newaxis]
    S1 = np.cumsum(y1**2, axis=0) / np.arange(1, T + 1)[:, np.newaxis]
    rho = np.cumsum(y0 * y1, axis=0) / (
        np.arange(1, T + 1)[:, np.newaxis] * np.sqrt(S0 * S1)
    )
    return np.mean(2 * (1 + rho) * np.sqrt(S0 * S1), axis=1)


def HT_eval_iter(p_arr, y_arr, z_arr):
    
    HT_vals = y_arr * (z_arr / p_arr - (1 - z_arr) / (1 - p_arr))
    HT_vals_cumulative = np.cumsum(HT_vals, axis=0)
    est = (
        HT_vals_cumulative
        / np.arange(1, HT_vals_cumulative.shape[0] + 1)[:, np.newaxis]
    )
    return est


class AATE:
    def __init__(self, k):
        """Initialize CLIP OGD algorithm for k parallel sequences."""
        self.p = np.full(k, 0.5)  # Initial probabilities
        self.g = np.zeros(k)  # Previous gradients
        self.t = 0  # Time step
        self.k = k

    def get_probabilities(self):
        """Get current treatment probabilities."""
        return self.p

    def step(self, gradients):
        """Update algorithm state with new gradients."""
        eta, delta = compute_step_sizes(
            self.t, 10000, old=False
        )  # MB swapped the code here for the one we use for ClipOGDSC (yes, we call this before incrementing the time variable)
        self.t += 1
        # eta = 1 / (0.49 * self.t) #MB swapped the code here for the one we use for ClipOGDSC
        # delta = np.exp(-np.sqrt(np.log(max(self.t, 2))))
        self.g = gradients
        self.p = np.clip(self.p - eta * self.g, delta, 1 - delta)


class ASOLO:
    def __init__(self, d, k):
        self.L = np.zeros((k, d))  # k parallel cumulative loss vectors
        self.q = np.zeros(k)  # k parallel cumulative squared norms

    def step(self, loss):
        """Update algorithm state with new loss matrix."""
        self.L += loss
        self.q += np.sum(loss**2, axis=1)

    def get_weights(self):
        """Calculate current weights for all sequences."""
        safe_q = np.where(self.q > 0, np.sqrt(self.q), 1)
        weights = -self.L / safe_q[:, np.newaxis]
        return np.maximum(0, weights)


class ASE:
    def __init__(self, d, k):
        """Initialize Sleeping Experts algorithm for k parallel sequences."""
        self.asolo = ASOLO(d, k)
        self.d = d
        self.k = k

    def get_weights(self, active_sets):
        """Get current weights for active experts across all sequences."""
        unscaled_weights = self.asolo.get_weights()

        # In the initial state or when ASOLO weights are all zero,
        # use uniform weights over active groups
        if np.all(unscaled_weights <= 0):
            return active_sets / np.maximum(
                np.sum(active_sets, axis=1, keepdims=True), 1
            )

        # Apply active sets mask and normalize
        active_weights = unscaled_weights * active_sets
        sums = np.sum(active_weights, axis=1, keepdims=True)

        # Where we have valid weights, normalize them
        # Where we don't, use uniform weights over active groups
        weights = np.where(
            sums > 1e-10,
            active_weights / np.maximum(sums, 1e-10),
            active_sets / np.maximum(np.sum(active_sets, axis=1, keepdims=True), 1),
        )

        return weights

    def step(self, loss, active_sets, current_weights):
        """Update algorithm state with new loss matrices."""
        inner_products = np.sum(loss * current_weights, axis=1, keepdims=True)
        surrogate_loss = active_sets * (loss - inner_products)
        self.asolo.step(surrogate_loss)


class AMGATE:
    def __init__(self, d, k):
        """Initialize Multigroup Adaptive Design for k parallel sequences."""
        self.d = d
        self.k = k
        self.ase = ASE(d, k)
        self.aates = {g: AATE(k) for g in range(d)}

    def step(self, active_sets, outcomes, unif=None):
        """Execute one step of the algorithm for all sequences."""

        # Create random variable if not fed into step
        if unif is None:
            unif = np.random.uniform(size=self.k)

        # Get treatment probabilities for all groups and sequences
        treatment_probs = np.zeros((self.k, self.d))
        for g in range(self.d):
            g_probs = self.aates[g].get_probabilities()
            treatment_probs[:, g] = g_probs

        # Get group weights from ASE
        group_weights = self.ase.get_weights(active_sets)

        # Calculate aggregated treatment probabilities (only for active groups)
        pt_agg = np.sum(treatment_probs * group_weights * active_sets, axis=1)
        pt_agg = np.clip(pt_agg, 1e-6, 1 - 1e-6)

        # Sample treatment decisions as Bernoulli variables with pt_agg rates
        # Zt_agg = (unif <= pt_agg).astype(int)
        Zt_agg = np.random.binomial(1, pt_agg)  # MB

        # Get realized outcomes
        yt = outcomes[:, 1] * Zt_agg + outcomes[:, 0] * (1 - Zt_agg)

        # Prepare matrices for updates
        loss_matrix = np.zeros((self.k, self.d))

        # Calculate common terms for all sequences
        ip_agg = Zt_agg / pt_agg + (1 - Zt_agg) / (1 - pt_agg)
        scaled_outcome = yt**2 * ip_agg

        # Update AATE instances and prepare loss matrix
        for g in range(self.d):
            pt_g = np.clip(treatment_probs[:, g], 1e-6, 1 - 1e-6)

            # Calculate group-specific terms
            ip_group_loss = Zt_agg / pt_g + (1 - Zt_agg) / (1 - pt_g)
            ip_group_grad = -Zt_agg / pt_g**2 + (1 - Zt_agg) / (1 - pt_g) ** 2

            # Calculate gradients and losses
            grad = scaled_outcome * ip_group_grad
            loss_matrix[:, g] = scaled_outcome * ip_group_loss

            # Only update AATE for sequences where group g is active
            active_mask = active_sets[:, g].astype(bool)
            if np.any(active_mask):
                self.aates[g].step(np.where(active_mask, grad, 0))

        # Update ASE with surrogate loss
        self.ase.step(loss_matrix, active_sets, group_weights)

        return pt_agg, Zt_agg, yt, group_weights, treatment_probs
