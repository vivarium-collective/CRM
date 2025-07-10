import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import entropy
from numpy.linalg import eigvals
from typing import Dict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


class CRMBenchmarkEngine:
    def __init__(self, community_model_instance, t_max=1000, n_points=1000, compute_extra=True):
        """
        :param community_model_instance: An instance of the CommunityModels class
        :param t_max: Simulation end time
        :param n_points: Number of time points
        :param compute_extra: Whether to compute Bray-Curtis, PCA, etc.
        """
        self.community_model = community_model_instance
        self.t_max = t_max
        self.n_points = n_points
        self.compute_extra = compute_extra
        self.results = []

    @staticmethod
    def alpha_diversity(abundances, threshold=1e-4):
        abundances = np.array(abundances)
        abundances = abundances[abundances > threshold]
        if abundances.sum() == 0:
            return {"Richness": 0, "Shannon": 0, "Simpson": 0, "InvSimpson": 0, "Evenness": 0}
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

    @staticmethod
    def feasibility_check(species, resources, threshold=1e-6):
        final_species = species[-1]
        final_resources = resources[-1]
        feasible_species = np.all(final_species >= threshold)
        feasible_resources = np.all(final_resources >= threshold)
        return feasible_species, feasible_resources

    @staticmethod
    def compute_jacobian(f, x_eq, eps=1e-6):
        n = len(x_eq)
        J = np.zeros((n, n))
        for i in range(n):
            x1 = np.copy(x_eq)
            x2 = np.copy(x_eq)
            x1[i] -= eps / 2
            x2[i] += eps / 2
            f1 = f(0, x1)
            f2 = f(0, x2)
            J[:, i] = (f2 - f1) / eps
        return J

    @staticmethod
    def stability_check(f, x_eq):
        try:
            J = CRMBenchmarkEngine.compute_jacobian(f, x_eq)
            eigenvalues = eigvals(J)
            is_stable = np.all(np.real(eigenvalues) < 0)
        except Exception:
            eigenvalues = np.array([])
            is_stable = False
        return is_stable, eigenvalues

    @staticmethod
    def additional_metrics(X, species_slice=None, threshold=1e-6):
        from scipy.spatial.distance import pdist
        from sklearn.decomposition import PCA

        if species_slice is not None:
            species_trajectories = X[:, species_slice]
        else:
            species_trajectories = X

        species_means = np.mean(species_trajectories, axis=0)
        species_vars = np.var(species_trajectories, axis=0)

        normed = species_trajectories / (species_trajectories.sum(axis=1, keepdims=True) + threshold)
        bray_curtis = pdist(normed, metric='braycurtis')
        bc_mean = np.mean(bray_curtis)
        bc_var = np.var(bray_curtis)

        try:
            pca = PCA(n_components=2)
            pca.fit(normed)
            var_explained = pca.explained_variance_ratio_
        except Exception:
            var_explained = [0.0, 0.0]

        return {
            "MeanAbundance": np.mean(species_means),
            "VarAbundance": np.mean(species_vars),
            "BrayCurtisMean": bc_mean,
            "BrayCurtisVar": bc_var,
            "PCA1": var_explained[0],
            "PCA2": var_explained[1]
        }

    def run(self):
        results_dict = self.community_model.run_all(time_step=self.t_max, num_points=self.n_points)

        for name, output in results_dict.items():
            X = output['X'].T
            t = output['t']

            if name == 'mcrm':
                species = X[:, :self.community_model.params_mcrm['num_species']]
                resources = X[:, self.community_model.params_mcrm['num_species'] :]
            elif name == 'gibbs':
                species = X[:, -self.community_model.params_gibbs['N0'].shape[0]:]
                resources = X[:, :self.community_model.params_gibbs['R0'].shape[0]]
            elif name == 'micrm':
                species = X[:, -self.community_model.params_micrm['N0'].shape[0]:]
                resources = X[:, :self.community_model.params_micrm['R0'].shape[0]]
            else:
                continue

            alpha = self.alpha_diversity(species[-1])
            feas_s, feas_r = self.feasibility_check(species, resources)

            # Define model-specific lambda for Jacobian estimation
            f_dyn = lambda t, x: self.community_model._mcrm_dynamics(t, x) if name == 'mcrm' \
                else self.community_model._gibbs_dynamics(t, x) if name == 'gibbs' \
                else self.community_model._micrm_dynamics(t, x)

            stable, eigs = self.stability_check(f_dyn, X[-1])

            result = {
                "Model": name,
                "FeasibleSpecies": feas_s,
                "FeasibleResources": feas_r,
                "Stable": stable,
                "MaxEigenvalue": np.max(np.real(eigs)) if eigs.size > 0 else np.nan,
                **alpha
            }

            if self.compute_extra:
                extras = self.additional_metrics(X, species_slice=slice(0, species.shape[1]))
                result.update(extras)

            self.results.append(result)

        return pd.DataFrame(self.results)
