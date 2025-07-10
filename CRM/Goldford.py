import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import entropy


def get_metabolism(num_resources, flag='rand', p=0.1):
    if flag in ['thermo', 'sparse']:
        w = np.arange(num_resources, 0, -1)
        d = np.flip(w) / num_resources
        D = np.array([np.concatenate([np.zeros(i), d[:num_resources - i]]) for i in range(num_resources)])
        D = 0.3 * D.T
        if flag == 'sparse':
            M = np.random.binomial(1, p, (num_resources, num_resources))
            D *= M
    elif flag == 'rand':
        D = np.random.rand(num_resources, num_resources) * p
    else:
        raise ValueError("Unknown flag type.")
    return D


def get_consumer_priors(T, r, specialist, specialistVar, num_resources):
    R = np.random.rand(num_resources)
    if r is None:
        r = np.random.randint(0, num_resources)
    f = np.clip(np.random.normal(specialist, specialistVar), 0, 1)
    other = [i for i in range(num_resources) if i != r]
    R[other] = (T - f*T) * R[other] / R[other].sum()
    R[r] = f * T
    return R


def drchrnd(a, n):
    r = np.random.gamma(np.tile(a, (n, 1)), 1.0)
    return r / r.sum(axis=1, keepdims=True)


def make_phylo_consumers(num_species, num_resources, num_groups, priors, Total, non_normal):
    if not priors:
        priors = [Total * np.random.rand(num_resources) for _ in range(num_groups)]
    C_list = [drchrnd(np.array(prior), num_species) for prior in priors]
    C = np.vstack(C_list)
    group_labels = np.concatenate([[i + 1] * num_species for i in range(num_groups)])
    idx = np.random.choice(C.shape[0], size=num_species, replace=False)
    C = C[idx]
    group_labels = group_labels[idx]
    if non_normal:
        u = np.random.normal(1, 0.01, size=(num_species, 1))
        C *= u
    return {'C': C, 'group': group_labels, 'k': np.argsort(group_labels)}, priors


def mcrm_params(resource_idx, num_species, C, D, qual='eye', W_mode='shared'):
    num_resources = D.shape[1]
    var_idx = {
        'species': list(range(num_species)),
        'resources': list(range(num_species, num_species + num_resources))
    }

    if qual == 'eye':
        base_W = np.eye(num_resources)
    else:
        w = np.arange(num_resources, 0, -1)
        base_W = np.diag(w)

    if W_mode == 'shared':
        W = base_W - np.diag(D.sum(axis=0))
    elif W_mode == 'species':
        # Species-specific qualities
        W = np.random.uniform(0.5, 1.5, size=(num_species, num_resources))
    else:
        raise ValueError("Unknown W_mode: choose 'shared' or 'species'")

    alpha = np.zeros(num_resources)
    alpha[resource_idx] = 1e6
    B = np.random.rand(num_resources)
    B /= B.sum()
    x0 = np.random.rand(num_species + num_resources)

    return {
        'num_species': num_species,
        'num_resources': num_resources,
        'varIdx': var_idx,
        'C': C,
        'D': D,
        'W': W,
        'B': B,
        'alpha': alpha,
        'death_rate': np.zeros(num_species),
        'mu': 1.0,
        'T': 1.0,
        'tau': np.ones(num_resources),
        'x0': x0,
        'timeStep': 10000,
        'W_mode': W_mode
    }


def population_dynamics(t, x, params):
    v = params['varIdx']
    C, D = params['C'], params['D']
    B, T, alpha, tau = params['B'], params['T'], params['alpha'], params['tau']
    death_rate = params['death_rate']
    mu = params['mu']
    W = params['W']
    N = x[v['species']]
    R = x[v['resources']]
    dx = np.zeros_like(x)

    # Species-specific or shared W
    if params.get('W_mode', 'shared') == 'species':
        growth = np.sum(C * W * R, axis=1) - T
    else:
        growth = C @ (W @ R) - T

    dx[v['species']] = N * mu * growth - death_rate * N
    consumption = (C * R).T @ N
    production = D @ consumption
    dx[v['resources']] = (alpha - R) / tau - consumption + production + B * (death_rate @ N)

    return dx


def run_mcrm(params, num_points=10000):
    x0 = params['x0']
    t_eval = np.linspace(0, params['timeStep'], num_points)
    method = params.get('solverMethod', 'LSODA')  # fallback in case old params dict
    sol = solve_ivp(lambda t, x: population_dynamics(t, x, params),
                    [0, params['timeStep']], x0, method=method, t_eval=t_eval)

    v = params['varIdx']
    return {
        'species': sol.y[v['species'], :].T,
        'resources': sol.y[v['resources'], :].T,
        'communityStruct': sol.y[v['species'], -1],
        'environmentalStruct': sol.y[v['resources'], -1],
        'time': sol.t
    }


def coarse_grain_community_structure(abundance_vector, group_vector, max_groups):
    return np.array([abundance_vector[group_vector == (g+1)].sum() for g in range(max_groups)])


def remove_extinct_species(params, simulation, min_abundance=1e-4):
    x = simulation['species'][-1].copy()
    k = x > min_abundance
    params['num_species'] = np.sum(k)
    params['varIdx']['species'] = list(range(params['num_species']))
    params['varIdx']['resources'] = list(range(params['num_species'], params['num_species'] + params['num_resources']))
    params['x0'] = np.concatenate([x[k], simulation['resources'][-1]])
    params['C'] = params['C'][k]
    params['death_rate'] = params['death_rate'][k]
    return params


def knockout_species(params, species_idx):
    new_params = params.copy()
    new_params['C'] = new_params['C'].copy()
    new_params['C'][species_idx, :] = 0
    return new_params


def alpha_diversity(abundances, threshold=1e-4):
    abundances = abundances[abundances > threshold]
    p = abundances / abundances.sum()
    richness = len(p)
    shannon = entropy(p)
    simpson = (p ** 2).sum()
    inv_simpson = 1 / simpson if simpson > 0 else 0
    evenness = shannon / np.log(richness) if richness > 1 else 0
    return {
        "Richness": richness,
        "Shannon": shannon,
        "Simpson": simpson,
        "InvSimpson": inv_simpson,
        "Evenness": evenness
    }