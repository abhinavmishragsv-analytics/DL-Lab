import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend, we only save PNGs
import matplotlib.pyplot as plt
import os
import json

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ==============================================================================
# 1. DATA GENERATION  (creates a realistic, imperfectly-separable CSV dataset)
# ==============================================================================
def generate_credit_risk_dataset(m=2000, seed=42, path="credit_risk_synthetic.csv"):
    """
    Synthetic loan-default dataset. Built so the true probability of default
    depends on a mostly-linear combination of features PLUS one nonlinear
    interaction term (debt_to_income * credit_utilization). The interaction
    term is what gives a hidden-layer network a genuine edge over a plain
    linear (logistic-regression-style) model, which is the whole point of
    building an ANN rather than a single-neuron perceptron.
    """
    rng = np.random.default_rng(seed)

    annual_income = rng.normal(60000, 20000, m).clip(15000, None)
    debt_to_income = rng.normal(0.35, 0.12, m).clip(0.01, 0.9)
    credit_score = rng.normal(650, 80, m).clip(300, 850)
    credit_utilization = rng.normal(0.40, 0.20, m).clip(0.0, 1.0)
    late_payments_12m = rng.poisson(1.2, m)
    loan_amount = rng.normal(15000, 7000, m).clip(1000, None)
    employment_years = rng.exponential(5, m).clip(0, 40)

    # True (unknown-to-the-model) generating process for P(default)
    z = (
        -0.000030 * (annual_income - 60000)
        + 3.50 * (debt_to_income - 0.35)
        - 0.010 * (credit_score - 650)
        + 2.00 * (credit_utilization - 0.40)
        + 0.40 * late_payments_12m
        + 0.000040 * (loan_amount - 15000)
        - 0.05 * employment_years
        + 2.50 * (debt_to_income * credit_utilization)   # <- nonlinear interaction
        + rng.normal(0, 0.5, m)                            # irreducible noise
    )
    prob_default = 1.0 / (1.0 + np.exp(-z))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "annual_income": annual_income,
        "debt_to_income": debt_to_income,
        "credit_score": credit_score,
        "credit_utilization": credit_utilization,
        "late_payments_12m": late_payments_12m,
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "default": default,
    })
    df.to_csv(path, index=False)
    return path


# ==============================================================================
# 2. MANUAL PREPROCESSING UTILITIES (no sklearn)
# ==============================================================================
def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=42):
    """Shuffle-then-slice split. Returns 6 arrays: train/val/test for X and y."""
    m = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(m)

    n_test = int(round(m * test_size))
    n_val = int(round(m * val_size))

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def standardize(X_train, *other_sets):
    """
    Z-score standardization: x' = (x - mean) / std, using TRAIN statistics only
    (fit on train, applied to val/test) to avoid data leakage.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # guard against constant columns
    scaled = [(X_train - mean) / std]
    for X in other_sets:
        scaled.append((X - mean) / std)
    return (*scaled, mean, std)


def one_hot(y_int, num_classes):
    """Integer class labels -> one-hot matrix, implemented manually."""
    m = y_int.shape[0]
    Y = np.zeros((m, num_classes))
    Y[np.arange(m), y_int.astype(int)] = 1.0
    return Y


# ==============================================================================
# 3. ACTIVATION FUNCTIONS (manual, numerically stable)
# ==============================================================================
def _stable_sigmoid(Z):
    """
    Numerically-stable logistic sigmoid.
    For Z >= 0:  1 / (1 + e^-Z)             (no overflow risk)
    For Z <  0:  e^Z / (1 + e^Z)             (rewrite to avoid e^-Z blowing up)
    """
    out = np.empty_like(Z, dtype=np.float64)
    pos = Z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-Z[pos]))
    ez = np.exp(Z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def _softmax(Z):
    """Row-wise softmax with max-subtraction for numerical stability."""
    Z_shift = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z_shift)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def activate(name, Z):
    if name == "relu":
        return np.maximum(0.0, Z)
    if name == "sigmoid":
        return _stable_sigmoid(Z)
    if name == "tanh":
        return np.tanh(Z)
    if name == "softmax":
        return _softmax(Z)
    if name == "linear":
        return Z
    raise ValueError(f"Unknown activation: {name}")


def activate_derivative(name, Z):
    """dA/dZ for the given activation, evaluated at pre-activation Z."""
    if name == "relu":
        return (Z > 0).astype(np.float64)
    if name == "sigmoid":
        s = _stable_sigmoid(Z)
        return s * (1.0 - s)
    if name == "tanh":
        t = np.tanh(Z)
        return 1.0 - t ** 2
    if name == "linear":
        return np.ones_like(Z)
    if name == "softmax":
        # Softmax's Jacobian is not diagonal; in this codebase softmax is only
        # ever used at the output layer paired with categorical cross-entropy,
        # where dZ = A - Y directly (see ANN._output_delta). This branch exists
        # only so the function never silently returns something wrong.
        raise NotImplementedError("Softmax derivative is handled via the combined CCE+softmax shortcut.")
    raise ValueError(f"Unknown activation: {name}")


# ==============================================================================
# 4. LOSS FUNCTIONS (manual)
# ==============================================================================
EPS = 1e-9  # clipping epsilon for log-domain numerical stability

def binary_cross_entropy(A, Y):
    A_c = np.clip(A, EPS, 1 - EPS)
    return -np.mean(Y * np.log(A_c) + (1 - Y) * np.log(1 - A_c))


def categorical_cross_entropy(A, Y):
    A_c = np.clip(A, EPS, 1.0)
    return -np.mean(np.sum(Y * np.log(A_c), axis=1))


def mean_squared_error(A, Y):
    return np.mean((A - Y) ** 2)


# ==============================================================================
# 5. THE NETWORK
# ==============================================================================
class ANN:
    """
    Configurable fully-connected feed-forward network.

    layer_dims  : e.g. [n_input, n_hidden1, n_hidden2, ..., n_output]
    activations : one entry per LAYER (len == len(layer_dims) - 1), e.g.
                  ["relu", "relu", "sigmoid"]
    loss        : "bce" (binary cross-entropy, pair with sigmoid output),
                  "cce" (categorical cross-entropy, pair with softmax output),
                  "mse" (mean squared error, pair with linear output)
    """

    def __init__(self, layer_dims, activations, loss="bce", seed=42):
        assert len(activations) == len(layer_dims) - 1, \
            "Need exactly one activation per layer (excluding the input layer)."
        self.layer_dims = layer_dims
        self.activations = activations
        self.loss = loss
        self.num_layers = len(layer_dims) - 1
        self.seed = seed
        self.parameters = self._initialize_parameters()

    # ---- 5.1 Weight initialization -----------------------------------------
    def _initialize_parameters(self):
        """
        He initialization for ReLU layers (std = sqrt(2 / n_in)) -- keeps the
        variance of activations roughly constant across layers when using
        ReLU, which zeroes out ~half its inputs.

        Xavier/Glorot uniform initialization for sigmoid / tanh / softmax /
        linear layers (limit = sqrt(6 / (n_in + n_out))) -- keeps the
        variance of both the forward activations and the backward gradients
        balanced for activations that saturate.
        """
        rng = np.random.default_rng(self.seed)
        params = {}
        for l in range(1, self.num_layers + 1):
            n_in, n_out = self.layer_dims[l - 1], self.layer_dims[l]
            act = self.activations[l - 1]
            if act == "relu":
                std = np.sqrt(2.0 / n_in)
                W = rng.standard_normal((n_in, n_out)) * std
            else:
                limit = np.sqrt(6.0 / (n_in + n_out))
                W = rng.uniform(-limit, limit, size=(n_in, n_out))
            b = np.zeros((1, n_out))
            params[f"W{l}"] = W
            params[f"b{l}"] = b
        return params

    # ---- 5.2 Forward propagation -------------------------------------------
    def forward(self, X):
        """
        For each layer l = 1..L:
            Z[l] = A[l-1] . W[l] + b[l]        (batch-rows convention:
                                                 A[l-1] is (m, n_{l-1}),
                                                 W[l]   is (n_{l-1}, n_l))
            A[l] = f_l( Z[l] )
        Caches every A and Z (needed for backprop).
        """
        A = X
        caches = {"A": [A], "Z": [None]}  # Z[0] is an unused placeholder so
                                           # caches["Z"][l] lines up with layer l
        for l in range(1, self.num_layers + 1):
            W, b = self.parameters[f"W{l}"], self.parameters[f"b{l}"]
            Z = A @ W + b
            A = activate(self.activations[l - 1], Z)
            caches["Z"].append(Z)
            caches["A"].append(A)
        return A, caches

    # ---- 5.3 Cost ------------------------------------------------------------
    def compute_cost(self, AL, Y, lambda_reg=0.0):
        if self.loss == "bce":
            cost = binary_cross_entropy(AL, Y)
        elif self.loss == "cce":
            cost = categorical_cross_entropy(AL, Y)
        elif self.loss == "mse":
            cost = mean_squared_error(AL, Y)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

        if lambda_reg > 0.0:
            # L2 (ridge) weight-decay penalty: (lambda / 2m) * sum(W^2), summed
            # over every weight matrix. Biases are conventionally excluded.
            m = Y.shape[0]
            l2 = sum(np.sum(self.parameters[f"W{l}"] ** 2) for l in range(1, self.num_layers + 1))
            cost += (lambda_reg / (2 * m)) * l2
        return cost

    # ---- 5.4 Output-layer delta (loss+activation combined shortcut) --------
    def _output_delta(self, AL, Y, caches):
        """
        See the written derivation in the report, but in short:

        (a) sigmoid output + binary cross-entropy:
                dL/dZ_L = A_L - Y
        (b) softmax output + categorical cross-entropy:
                dL/dZ_L = A_L - Y
            Both (a) and (b) collapse to the SAME beautifully simple
            expression because the log in the loss exactly cancels the
            exp in the activation during the chain rule.
        (c) linear output + MSE:
                Cost = (1/m) * sum_i L_i ,  where per-example L_i = (1/n_out) * sum_k (a_ik - y_ik)^2
                => dL_i/dA_ik = (2/n_out) * (a_ik - y_ik) ;  dA/dZ = 1 (linear)
                => "raw" per-example delta: dZ_L = (2/n_out) * (A_L - Y)

            NOTE: this delta is deliberately NOT divided by m here. Every dW/db
            below applies a single shared (1/m) batch-average, exactly mirroring
            how (a) and (b) hand back the raw per-example (A_L - Y) with no 1/m
            baked in. Dividing by m twice (once here, once in dW/db) was an
            earlier bug caught by gradient_check() -- it silently shrank every
            regression gradient by a factor of the batch size m.
        """
        out_act = self.activations[-1]
        if (self.loss == "bce" and out_act == "sigmoid") or \
           (self.loss == "cce" and out_act == "softmax"):
            return AL - Y
        if self.loss == "mse":
            n_out = AL.shape[1]
            dA = (2.0 / n_out) * (AL - Y)
            dZ = dA * activate_derivative(out_act, caches["Z"][self.num_layers])
            return dZ
        raise NotImplementedError(
            "This codebase only wires up (sigmoid+bce), (softmax+cce), (linear+mse)."
        )

    # ---- 5.5 Backpropagation --------------------------------------------------
    def backward(self, AL, Y, caches, lambda_reg=0.0):
        """
        Standard backprop chain rule, layer by layer, from output to input:

            dW[l] = (1/m) * A[l-1]^T . dZ[l]  [+ (lambda/m) * W[l] if L2 reg used]
            db[l] = (1/m) * sum_over_batch( dZ[l] )
            dA[l-1] = dZ[l] . W[l]^T
            dZ[l-1] = dA[l-1] * f'_{l-1}( Z[l-1] )   (elementwise)

        i.e. the error signal dZ[l] is propagated backward through W[l]^T to
        become dA[l-1], then multiplied elementwise by the local derivative
        of layer (l-1)'s activation to become dZ[l-1] -- repeat until layer 1.

        L2 regularization adds (lambda/2m)*sum(W^2) to the cost, whose
        gradient w.r.t. W is simply (lambda/m)*W -- added on top of the
        data-driven gradient for every weight matrix (not the biases).
        """
        m = Y.shape[0]
        grads = {}
        L = self.num_layers
        dZ = self._output_delta(AL, Y, caches)

        for l in range(L, 0, -1):
            A_prev = caches["A"][l - 1]
            W = self.parameters[f"W{l}"]

            grads[f"dW{l}"] = (1.0 / m) * (A_prev.T @ dZ)
            if lambda_reg > 0.0:
                grads[f"dW{l}"] += (lambda_reg / m) * W
            grads[f"db{l}"] = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)

            if l > 1:
                dA_prev = dZ @ W.T
                dZ = dA_prev * activate_derivative(self.activations[l - 2], caches["Z"][l - 1])

        return grads

    # ---- 5.6 Parameter update (gradient descent) ----------------------------
    def update_parameters(self, grads, learning_rate):
        """W := W - eta * dL/dW   ;   b := b - eta * dL/db"""
        for l in range(1, self.num_layers + 1):
            self.parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
            self.parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    # ---- 5.7 Training loop (mini-batch gradient descent) --------------------
    def train(self, X, Y, X_val=None, Y_val=None, epochs=300, learning_rate=0.05,
               batch_size=None, lambda_reg=0.0, early_stopping_patience=None,
               verbose=True, print_every=20):
        """
        early_stopping_patience: if set (and X_val/Y_val given), training
        stops once val_loss has failed to improve for that many consecutive
        epochs, and the parameters are rolled back to whichever epoch had the
        best val_loss. This is what actually fixes overfitting seen in the
        raw loss curve -- it is a cheap, principled alternative to just
        picking a good-looking epoch by eye.
        """
        m = X.shape[0]
        batch_size = m if batch_size is None else batch_size
        rng = np.random.default_rng(self.seed)

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = np.inf
        best_params = None
        epochs_without_improvement = 0
        stopped_epoch = epochs - 1

        for epoch in range(epochs):
            perm = rng.permutation(m)
            X_shuf, Y_shuf = X[perm], Y[perm]

            epoch_loss_sum = 0.0
            n_batches = int(np.ceil(m / batch_size))
            for b in range(n_batches):
                s, e = b * batch_size, min((b + 1) * batch_size, m)
                X_batch, Y_batch = X_shuf[s:e], Y_shuf[s:e]

                AL, caches = self.forward(X_batch)
                loss = self.compute_cost(AL, Y_batch, lambda_reg)
                epoch_loss_sum += loss * (e - s)

                grads = self.backward(AL, Y_batch, caches, lambda_reg)
                self.update_parameters(grads, learning_rate)

            train_loss = epoch_loss_sum / m
            history["train_loss"].append(train_loss)

            if X_val is not None:
                AL_val, _ = self.forward(X_val)
                val_loss = self.compute_cost(AL_val, Y_val, lambda_reg)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_params = {k: v.copy() for k, v in self.parameters.items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                msg = f"Epoch {epoch:4d}/{epochs}  train_loss={train_loss:.5f}"
                if X_val is not None:
                    msg += f"  val_loss={history['val_loss'][-1]:.5f}"
                print(msg)

            if (early_stopping_patience is not None and X_val is not None
                    and epochs_without_improvement >= early_stopping_patience):
                stopped_epoch = epoch
                if verbose:
                    print(f"Early stopping at epoch {epoch} "
                          f"(no val_loss improvement for {early_stopping_patience} epochs). "
                          f"Restoring best weights from val_loss={best_val_loss:.5f}.")
                break

        if best_params is not None:
            self.parameters = best_params

        history["stopped_epoch"] = stopped_epoch
        history["best_val_loss"] = float(best_val_loss) if best_params is not None else None
        return history

    # ---- 5.8 Inference --------------------------------------------------------
    def predict_proba(self, X):
        AL, _ = self.forward(X)
        return AL

    def predict(self, X):
        AL = self.predict_proba(X)
        out_act = self.activations[-1]
        if out_act == "sigmoid":
            return (AL >= 0.5).astype(int)
        if out_act == "softmax":
            return np.argmax(AL, axis=1)
        return AL  # regression: raw continuous output


# ==============================================================================
# 6. EVALUATION METRICS (manual -- no sklearn.metrics)
# ==============================================================================
def confusion_matrix_binary(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))
    cm = np.array([[TN, FP], [FN, TP]])
    return cm, TP, TN, FP, FN


def classification_metrics_binary(y_true, y_pred):
    cm, TP, TN, FP, FN = confusion_matrix_binary(y_true, y_pred)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1_score": f1, "confusion_matrix": cm}


def confusion_matrix_multiclass(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics_multiclass(y_true, y_pred, num_classes):
    cm = confusion_matrix_multiclass(y_true, y_pred, num_classes)
    accuracy = np.trace(cm) / np.sum(cm)
    precisions, recalls, f1s = [], [], []
    for k in range(num_classes):
        TP = cm[k, k]
        FP = cm[:, k].sum() - TP
        FN = cm[k, :].sum() - TP
        p = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        r = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p); recalls.append(r); f1s.append(f)
    return {"accuracy": accuracy, "precision_macro": float(np.mean(precisions)),
            "recall_macro": float(np.mean(recalls)), "f1_macro": float(np.mean(f1s)),
            "confusion_matrix": cm}


def regression_metrics(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


# ==============================================================================
# 7. GRADIENT CHECKING  (verifies backprop is mathematically correct)
# ==============================================================================
def gradient_check(model, X, Y, num_checks=8, epsilon=1e-7, seed=0):
    """
    Finite-difference check:  dCost/dtheta  ~=  (Cost(theta+eps) - Cost(theta-eps)) / (2*eps)
    Compares this numerical estimate against the analytic gradient produced by
    our hand-written backward() for a handful of randomly sampled parameter
    entries. Relative error should be on the order of 1e-6 or smaller.
    """
    AL, caches = model.forward(X)
    analytic_grads = model.backward(AL, Y, caches)

    rng = np.random.default_rng(seed)
    param_names = list(model.parameters.keys())
    results = []
    for _ in range(num_checks):
        pname = param_names[rng.integers(0, len(param_names))]
        shape = model.parameters[pname].shape
        idx = tuple(rng.integers(0, s) for s in shape)

        original = model.parameters[pname][idx]

        model.parameters[pname][idx] = original + epsilon
        cost_plus = model.compute_cost(model.forward(X)[0], Y)

        model.parameters[pname][idx] = original - epsilon
        cost_minus = model.compute_cost(model.forward(X)[0], Y)

        model.parameters[pname][idx] = original  # restore

        numerical_grad = (cost_plus - cost_minus) / (2 * epsilon)
        analytic_grad = analytic_grads["d" + pname][idx]
        rel_error = abs(numerical_grad - analytic_grad) / (
            abs(numerical_grad) + abs(analytic_grad) + 1e-12)

        results.append({"param": pname, "index": idx,
                         "numerical": numerical_grad, "analytic": analytic_grad,
                         "rel_error": rel_error})
    return results


# ==============================================================================
# 8. VISUALIZATION
# ==============================================================================
def plot_loss_curve(history, save_path):
    plt.figure(figsize=(7, 4.5))
    plt.plot(history["train_loss"], label="Train loss", linewidth=2)
    if history.get("val_loss"):
        plt.plot(history["val_loss"], label="Validation loss", linewidth=2, linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss vs. Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title, fontsize=12, pad=12)
    plt.colorbar(fraction=0.046, pad=0.04)
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                      color="white" if cm[i, j] > thresh else "black", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ==============================================================================
# 9. MAIN DEMO -- Binary classification: Credit default risk
# ==============================================================================
def run_binary_classification_demo():
    print("=" * 78)
    print("DEMO 1/3: BINARY CLASSIFICATION -- Credit Default Risk")
    print("=" * 78)

    csv_path = generate_credit_risk_dataset(m=2000, seed=42,
                                             path=os.path.join(OUT_DIR, "credit_risk_synthetic.csv"))
    df = pd.read_csv(csv_path)

    feature_cols = ["annual_income", "debt_to_income", "credit_score",
                     "credit_utilization", "late_payments_12m", "loan_amount",
                     "employment_years"]
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["default"].to_numpy(dtype=np.float64).reshape(-1, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15, seed=42)
    X_train_s, X_val_s, X_test_s, mean, std = standardize(X_train, X_val, X_test)

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{y.mean():.1%} positive (default) rate")
    print(f"Split -> train:{X_train.shape[0]}  val:{X_val.shape[0]}  test:{X_test.shape[0]}")

    model = ANN(layer_dims=[X.shape[1], 16, 8, 1],
                activations=["relu", "relu", "sigmoid"],
                loss="bce", seed=42)

    # ---- sanity check the backprop math before trusting the training run ----
    gc = gradient_check(model, X_train_s[:5], y_train[:5], num_checks=8)
    max_err = max(r["rel_error"] for r in gc)
    print(f"\nGradient check on {len(gc)} random parameters -> max relative error: {max_err:.2e}")
    print("  (values below ~1e-5 confirm the hand-written backward() is correct)")

    print("\nTraining (with L2 regularization + early stopping on val_loss)...")
    history = model.train(X_train_s, y_train, X_val_s, y_val,
                           epochs=400, learning_rate=0.08, batch_size=64,
                           lambda_reg=0.15, early_stopping_patience=25,
                           verbose=True, print_every=20)

    plot_loss_curve(history, os.path.join(OUT_DIR, "binary_loss_curve.png"))

    y_pred_test = model.predict(X_test_s)
    metrics = classification_metrics_binary(y_test, y_pred_test)
    plot_confusion_matrix(metrics["confusion_matrix"], ["No Default", "Default"],
                           os.path.join(OUT_DIR, "binary_confusion_matrix.png"),
                           title="Credit Default -- Confusion Matrix (Test Set)")

    print("\nTest-set performance:")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1-score : {metrics['f1_score']:.4f}")
    print(f"  Confusion matrix [[TN,FP],[FN,TP]]:\n{metrics['confusion_matrix']}")

    result = {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1_score"],
        "gradient_check_max_rel_error": max_err,
    }
    return result


# ==============================================================================
# 10. SECONDARY DEMO -- Multi-class classification (proves softmax+CCE works)
# ==============================================================================
def run_multiclass_demo():
    print("\n" + "=" * 78)
    print("DEMO 2/3: MULTI-CLASS CLASSIFICATION -- 3-cluster synthetic dataset")
    print("=" * 78)

    rng = np.random.default_rng(7)
    m_per_class = 300
    centers = np.array([[0, 0], [4, 4], [4, -4]])
    X_list, y_list = [], []
    for k, c in enumerate(centers):
        pts = rng.normal(loc=c, scale=1.3, size=(m_per_class, 2))
        X_list.append(pts)
        y_list.append(np.full(m_per_class, k))
    X = np.vstack(X_list)
    y_int = np.concatenate(y_list)
    num_classes = 3

    X_train, y_train_int, X_val, y_val_int, X_test, y_test_int = train_val_test_split(
        X, y_int, val_size=0.15, test_size=0.15, seed=7)
    X_train_s, X_val_s, X_test_s, mean, std = standardize(X_train, X_val, X_test)

    Y_train = one_hot(y_train_int, num_classes)
    Y_val = one_hot(y_val_int, num_classes)

    model = ANN(layer_dims=[2, 10, num_classes],
                activations=["tanh", "softmax"],
                loss="cce", seed=7)

    gc = gradient_check(model, X_train_s[:5], Y_train[:5], num_checks=6)
    max_err = max(r["rel_error"] for r in gc)
    print(f"Gradient check max relative error: {max_err:.2e}")

    history = model.train(X_train_s, Y_train, X_val_s, Y_val,
                           epochs=150, learning_rate=0.3, batch_size=32,
                           verbose=True, print_every=30)

    plot_loss_curve(history, os.path.join(OUT_DIR, "multiclass_loss_curve.png"))

    y_pred_test = model.predict(X_test_s)
    metrics = classification_metrics_multiclass(y_test_int, y_pred_test, num_classes)
    plot_confusion_matrix(metrics["confusion_matrix"], ["C0", "C1", "C2"],
                           os.path.join(OUT_DIR, "multiclass_confusion_matrix.png"),
                           title="3-Class Softmax -- Confusion Matrix (Test Set)")

    print(f"Test accuracy: {metrics['accuracy']:.4f} | "
          f"macro-F1: {metrics['f1_macro']:.4f}")

    return {"test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["f1_macro"],
            "gradient_check_max_rel_error": max_err}


# ==============================================================================
# 11. TERTIARY DEMO -- Regression (proves linear-output + MSE works)
# ==============================================================================
def run_regression_demo():
    print("\n" + "=" * 78)
    print("DEMO 3/3: REGRESSION -- synthetic nonlinear function")
    print("=" * 78)

    rng = np.random.default_rng(3)
    m = 600
    X = rng.uniform(-3, 3, size=(m, 2))
    y = (0.5 * X[:, 0] ** 2 - 1.5 * X[:, 1] + np.sin(X[:, 0] * X[:, 1])
         + rng.normal(0, 0.2, m)).reshape(-1, 1)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X, y, val_size=0.15, test_size=0.15, seed=3)
    X_train_s, X_val_s, X_test_s, mean, std = standardize(X_train, X_val, X_test)

    model = ANN(layer_dims=[2, 16, 8, 1], activations=["tanh", "tanh", "linear"],
                loss="mse", seed=3)

    gc = gradient_check(model, X_train_s[:5], y_train[:5], num_checks=6)
    max_err = max(r["rel_error"] for r in gc)
    print(f"Gradient check max relative error: {max_err:.2e}")

    history = model.train(X_train_s, y_train, X_val_s, y_val,
                           epochs=300, learning_rate=0.02, batch_size=64,
                           early_stopping_patience=40,
                           verbose=True, print_every=50)

    plot_loss_curve(history, os.path.join(OUT_DIR, "regression_loss_curve.png"))

    y_pred_test = model.predict(X_test_s)
    metrics = regression_metrics(y_test, y_pred_test)
    print(f"Test MSE: {metrics['mse']:.4f}  RMSE: {metrics['rmse']:.4f}  "
          f"MAE: {metrics['mae']:.4f}  R2: {metrics['r2']:.4f}")

    return metrics


# ==============================================================================
if __name__ == "__main__":
    all_results = {}
    all_results["binary_classification"] = run_binary_classification_demo()
    all_results["multiclass_classification"] = run_multiclass_demo()
    all_results["regression"] = run_regression_demo()

    with open(os.path.join(OUT_DIR, "results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    print("\n" + "=" * 78)
    print("All demos complete. Plots + results_summary.json saved to ./outputs/")
    print("=" * 78)