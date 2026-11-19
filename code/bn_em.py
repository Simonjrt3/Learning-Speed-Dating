import numpy as np
import pandas as pd
from itertools import product
from scipy.special import logsumexp

def fit_missing_bn_em(
    data,
    latent_groups,
    decision_cols=("decision", "decision_o"),
    max_iter=100,
    tol=1e-5,
    alpha=1.0,
    seed=42,
    verbose=True,
):
    rng = np.random.default_rng(seed)

    latent_names = list(latent_groups)
    L = len(latent_names)

    # All possible joint assignments to the binary latent variables.
    Z = np.array(list(product([0, 1], repeat=L)), dtype=int)
    K = len(Z)

    # Determine which latent variables are parents of each observed node.
    nodes = list(dict.fromkeys(
        [v for group in latent_groups.values() for v in group]
        + list(decision_cols)
    ))

    parent_map = {}
    for node in nodes:
        if node in decision_cols:
            parent_map[node] = latent_names.copy()
        else:
            parent_map[node] = [
                latent for latent, features in latent_groups.items()
                if node in features
            ]

    # Convert data to arrays and infer state cardinalities.
    X = {}
    cards = {}

    for node in nodes:
        if node not in data.columns:
            raise ValueError(f"Missing column: {node}")

        x = data[node].to_numpy(dtype=float)
        observed = x[np.isfinite(x)]

        if len(observed) == 0:
            raise ValueError(
                f"{node} is entirely missing. Remove it from the model."
            )

        values = np.unique(observed)

        # States must be encoded as 0, 1, ..., r-1.
        if not np.array_equal(values, np.arange(len(values))):
            raise ValueError(
                f"{node} must be encoded as consecutive integers "
                f"starting at zero. Found {values}"
            )

        X[node] = x
        cards[node] = len(values)

    # For every observed node, map each latent configuration to
    # the corresponding parent configuration of that node.
    parent_indices = {}
    parent_config_counts = {}

    for node, parents in parent_map.items():
        positions = [latent_names.index(p) for p in parents]

        if positions:
            parent_states = Z[:, positions]
            indices = np.ravel_multi_index(
                parent_states.T,
                dims=(2,) * len(positions)
            )
            n_configs = 2 ** len(positions)
        else:
            indices = np.zeros(K, dtype=int)
            n_configs = 1

        parent_indices[node] = indices
        parent_config_counts[node] = n_configs

    # Initialize latent prior probabilities.
    latent_probs = np.full((L, 2), 0.5)

    # Initialize each observed-node CPT with random probabilities.
    theta = {}

    for node in nodes:
        r = cards[node]
        n_configs = parent_config_counts[node]

        theta[node] = rng.dirichlet(
            np.ones(r), size=n_configs
        )

    def expectation():
        """Compute posterior probabilities of all latent configurations."""

        n = len(data)

        # Log P(Z), assuming independent latent roots.
        log_prior = np.zeros(K)

        for j in range(L):
            log_prior += np.log(latent_probs[j, Z[:, j]])

        log_joint = np.tile(log_prior, (n, 1))

        # Multiply in likelihood contributions from observed nodes.
        # Missing values contribute no likelihood term.
        for node in nodes:
            x = X[node]
            mask = np.isfinite(x)

            if not mask.any():
                continue

            values = x[mask].astype(int)
            pidx = parent_indices[node]

            # Shape: (number of observed rows, K)
            likelihood = theta[node][pidx[:, None], values[None, :]].T

            log_joint[mask] += np.log(
                np.maximum(likelihood, 1e-300)
            )

        log_normalizer = logsumexp(
            log_joint, axis=1, keepdims=True
        )

        posterior = np.exp(log_joint - log_normalizer)
        log_likelihood = log_normalizer.sum()

        return posterior, float(log_likelihood)

    # EM iterations.
    previous_ll = -np.inf

    for iteration in range(max_iter):
        posterior, ll = expectation()

        # M-step: update latent prior probabilities.
        n = len(data)

        for j in range(L):
            for state in range(2):
                mass = posterior[:, Z[:, j] == state].sum()
                latent_probs[j, state] = (
                    mass + alpha
                ) / (n + 2 * alpha)

        # M-step: update observed-node CPTs.
        for node in nodes:
            x = X[node]
            observed_mask = np.isfinite(x)
            values = x.astype(float)

            r = cards[node]
            n_configs = parent_config_counts[node]
            pidx = parent_indices[node]

            counts = np.zeros((n_configs, r))

            for parent_config in range(n_configs):
                latent_columns = np.flatnonzero(
                    pidx == parent_config
                )

                # Posterior mass assigned to this parent configuration.
                mass = posterior[:, latent_columns].sum(axis=1)

                for state in range(r):
                    mask = observed_mask & (values == state)
                    counts[parent_config, state] = mass[mask].sum()

            # Dirichlet smoothing prevents zero probabilities.
            theta[node] = (
                counts + alpha
            ) / (
                counts.sum(axis=1, keepdims=True) + alpha * r
            )

        if verbose:
            print(
                f"Iteration {iteration + 1:3d}: "
                f"log-likelihood = {ll:.4f}"
            )

        if iteration > 0 and abs(ll - previous_ll) < tol:
            if verbose:
                print("Converged.")
            break

        previous_ll = ll

    return {
        "latent_names": latent_names,
        "latent_configs": Z,
        "latent_probs": latent_probs,
        "theta": theta,
        "parent_map": parent_map,
        "parent_indices": parent_indices,
        "cards": cards,
        "log_likelihood": ll,
    }

def encode_train_test(train_raw, test_raw, q=3):
    train = pd.DataFrame(index=train_raw.index)
    test = pd.DataFrame(index=test_raw.index)

    for col in train_raw.columns:
        if (pd.api.types.is_numeric_dtype(train_raw[col])
                and train_raw[col].nunique() > 10):

            _, bins = pd.qcut(
                train_raw[col].dropna(),
                q=q,
                retbins=True,
                duplicates="drop"
            )

            bins[0] = -np.inf
            bins[-1] = np.inf

            train[col] = pd.cut(
                train_raw[col], bins=bins,
                labels=False, include_lowest=True
            ).astype(float)

            test[col] = pd.cut(
                test_raw[col], bins=bins,
                labels=False, include_lowest=True
            ).astype(float)

        else:
            # Encode categorical variables, preserving NaNs
            categories = pd.Index(train_raw[col].dropna().unique())
            mapping = {value: i for i, value in enumerate(categories)}

            train[col] = train_raw[col].map(mapping).astype(float)
            test[col] = test_raw[col].map(mapping).astype(float)

    return train, test

def encode_train_test(train_raw, test_raw, q=3):
    train = pd.DataFrame(index=train_raw.index)
    test = pd.DataFrame(index=test_raw.index)

    for col in train_raw.columns:
        # Bin numerical variables with many distinct values
        if (pd.api.types.is_numeric_dtype(train_raw[col])
                and train_raw[col].nunique() > 10):

            _, bins = pd.qcut(
                train_raw[col].dropna(),
                q=q,
                retbins=True,
                duplicates="drop"
            )

            bins[0] = -np.inf
            bins[-1] = np.inf

            train[col] = pd.cut(
                train_raw[col], bins=bins,
                labels=False, include_lowest=True
            ).astype(float)

            test[col] = pd.cut(
                test_raw[col], bins=bins,
                labels=False, include_lowest=True
            ).astype(float)

        else:
            # Encode categorical variables, preserving NaNs
            categories = pd.Index(train_raw[col].dropna().unique())
            mapping = {value: i for i, value in enumerate(categories)}

            train[col] = train_raw[col].map(mapping).astype(float)
            test[col] = test_raw[col].map(mapping).astype(float)

    return train, test

def predict_match_probability(data, fitted):
    latent_names = fitted["latent_names"]
    Z = fitted["latent_configs"]
    latent_probs = fitted["latent_probs"]
    theta = fitted["theta"]
    parent_indices = fitted["parent_indices"]

    n = len(data)
    K = len(Z)
    L = len(latent_names)

    # Prior over joint latent configurations.
    log_prior = np.zeros(K)

    for j in range(L):
        log_prior += np.log(latent_probs[j, Z[:, j]])

    log_joint = np.tile(log_prior, (n, 1))

    # Condition on all available features.
    # Deliberately do not condition on either decision.
    for node in fitted["parent_map"]:
        if node in ("decision", "decision_o"):
            continue

        x = data[node].to_numpy(dtype=float)
        mask = np.isfinite(x)

        if not mask.any():
            continue

        values = x[mask].astype(int)
        pidx = parent_indices[node]

        likelihood = theta[node][pidx[:, None], values[None, :]].T

        log_joint[mask] += np.log(
            np.maximum(likelihood, 1e-300)
        )

    posterior = np.exp(
        log_joint - logsumexp(log_joint, axis=1, keepdims=True)
    )

    # Probability of each decision being 1 under each latent assignment.
    p_decision = theta["decision"][
        parent_indices["decision"], 1
    ]

    p_decision_o = theta["decision_o"][
        parent_indices["decision_o"], 1
    ]

    # Conditional independence of decisions given latent variables.
    p_match_given_z = p_decision * p_decision_o

    return posterior @ p_match_given_z
