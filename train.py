import os
import gc
import itertools
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Centralized Configuration
# ---------------------------------------------------------------------------
config = {
    # ── Paths ──────────────────────────────────────────────────────────────
    # Root folder that contains one sub-folder per class, e.g.:
    #   ECG_DATA/
    #     Normal/
    #     Myocardial Infarction/
    #     History of MI/
    #     Abnormal/
    "DATASET_PATH": r"ECG_DATA",                          # <-- CHANGE THIS
    "BASE_OUTPUT_DIR": r"results",                         # <-- CHANGE THIS

    # ── Reproducibility ────────────────────────────────────────────────────
    "RANDOM_STATE": 42,
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    # ── Base Model ─────────────────────────────────────────────────────────
    # Options: 'mobilenet_v3_small' | 'resnet50' | 'efficientnet_b0'
    #          'densenet121' | 'inception_v3'
    "BASE_MODEL_NAME": "mobilenet_v3_small",

    # ── K-Fold Cross-Validation ────────────────────────────────────────────
    "K_FOLDS": 4,

    # ── Federated Learning ─────────────────────────────────────────────────
    "NUM_ROUNDS": 100,
    "NUM_CLIENTS": 3,
    "CLIENT_EPOCHS": 3,

    # ── Training Hyper-parameters ──────────────────────────────────────────
    "BATCH_SIZE": 32,
    "NUM_CLASSES": 4,
    "INITIAL_LEARNING_RATE": 0.001,
    "L2_REG": 1e-5,

    # Inverse Squeeze-and-Excitation expansion ratio
    "SE_RATIO": 16,

    # Set > 0 only if you have multiple CPU cores and no Windows issues
    "NUM_WORKERS": 0,
}

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "mobilenet_v3_small": {
        "input_size": 224,
        "model_fn": models.mobilenet_v3_small,
        "weights_fn": models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    },
    "resnet50": {
        "input_size": 224,
        "model_fn": models.resnet50,
        "weights_fn": models.ResNet50_Weights.IMAGENET1K_V1,
    },
    "efficientnet_b0": {
        "input_size": 224,
        "model_fn": models.efficientnet_b0,
        "weights_fn": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    },
    "densenet121": {
        "input_size": 224,
        "model_fn": models.densenet121,
        "weights_fn": models.DenseNet121_Weights.IMAGENET1K_V1,
    },
    "inception_v3": {
        "input_size": 299,
        "model_fn": models.inception_v3,
        "weights_fn": models.Inception_V3_Weights.IMAGENET1K_V1,
    },
}


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
def setup_environment() -> None:
    """Seed everything and populate derived config keys."""
    np.random.seed(config["RANDOM_STATE"])
    torch.manual_seed(config["RANDOM_STATE"])
    if "cuda" in config["DEVICE"]:
        torch.cuda.manual_seed_all(config["RANDOM_STATE"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[INFO] Device : {config['DEVICE']}")

    if config["BASE_MODEL_NAME"] not in MODEL_CONFIG:
        raise ValueError(
            f"Unsupported BASE_MODEL_NAME: '{config['BASE_MODEL_NAME']}'. "
            f"Choose from: {list(MODEL_CONFIG.keys())}"
        )

    spec = MODEL_CONFIG[config["BASE_MODEL_NAME"]]
    config["IMG_SIZE"] = (spec["input_size"], spec["input_size"])
    config["BASE_MODEL_FN"] = spec["model_fn"]
    config["WEIGHTS_FN"] = spec["weights_fn"]

    print(
        f"[INFO] Base model : {config['BASE_MODEL_NAME']}  "
        f"| Input size : {config['IMG_SIZE']}"
    )


# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------
def get_image_paths_and_labels(dataset_path: str):
    """
    Walk *dataset_path* and return parallel arrays of file paths and integer
    labels, plus the ordered list of class names.
    """
    print("[INFO] Scanning dataset …")
    all_paths, all_labels = [], []

    class_names = sorted(
        d
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    if not class_names:
        raise FileNotFoundError(
            f"No class sub-directories found in '{dataset_path}'."
        )

    class_map = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        files = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        for fname in files:
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(class_map[class_name])

    print(
        f"[INFO] Found {len(all_paths)} images across "
        f"{len(class_names)} classes: {class_names}"
    )
    return np.array(all_paths), np.array(all_labels), class_names


class ECGDataset(Dataset):
    """PyTorch Dataset for ECG image classification."""

    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[idx], dtype=torch.long)


def create_client_dataloaders(train_dataset: Dataset, num_clients: int):
    """Split *train_dataset* evenly among *num_clients* and return DataLoaders."""
    n = len(train_dataset)
    indices = list(range(n))
    spc = n // num_clients  # samples per client

    print(f"[INFO] Distributing {n} training samples across {num_clients} clients …")
    loaders, sizes = [], []
    for i in range(num_clients):
        start = i * spc
        end = (i + 1) * spc if i < num_clients - 1 else n
        subset = Subset(train_dataset, indices[start:end])
        loaders.append(
            DataLoader(
                subset,
                batch_size=config["BATCH_SIZE"],
                shuffle=True,
                num_workers=config["NUM_WORKERS"],
                pin_memory=True,
            )
        )
        sizes.append(len(subset))
        print(f"   Client {i + 1}: {len(subset)} samples")

    return loaders, sizes


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class InverseSEBlock(nn.Module):
    """
    Inverse Squeeze-and-Excitation (ISE) block.

    Unlike the standard SE block (compress → expand), ISE first *expands*
    the feature vector into a higher-dimensional space and then *squeezes*
    it back, producing richer inter-feature attention weights.

    Reference: Section III-B of the companion paper.
    """

    def __init__(self, num_features: int, ratio: int):
        super().__init__()
        self.expand = nn.Linear(num_features, num_features * ratio, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Linear(num_features * ratio, num_features, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.squeeze(self.relu(self.expand(x))))
        return x * attn


class CustomECGModel(nn.Module):
    """
    Transfer-learning model with a frozen pre-trained backbone and a custom
    classification head that includes the ISE attention block.

    Head structure:
        Linear(num_ftrs → 256) → ReLU → ISE(256) → Dropout(0.3) → Linear(256 → num_classes)
    """

    def __init__(self, num_classes: int):
        super().__init__()

        weights = config["WEIGHTS_FN"]
        self.base_model = config["BASE_MODEL_FN"](weights=weights)

        # Freeze backbone
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Replace the built-in classifier with Identity so we can attach
        # our own head and still use the backbone as a feature extractor.
        name = config["BASE_MODEL_NAME"]
        if "mobilenet" in name:
            num_ftrs = self.base_model.classifier[0].in_features
            self.base_model.classifier = nn.Identity()
        elif "efficientnet" in name:
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif "densenet" in name:
            num_ftrs = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif "resnet" in name or "inception" in name:
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise NotImplementedError(
                f"Feature-extraction logic not implemented for '{name}'."
            )

        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            InverseSEBlock(256, config["SE_RATIO"]),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inception v3 returns (logits, aux_logits) during training
        if "inception" in config["BASE_MODEL_NAME"] and self.training:
            x, _ = self.base_model(x)
        else:
            x = self.base_model(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Federated learning core
# ---------------------------------------------------------------------------
def train_client(
    client_id: int,
    loader: DataLoader,
    global_weights: dict,
    class_weights: torch.Tensor,
) -> tuple[dict, float, float]:
    """
    Fine-tune the classification head on one client's local data.

    Returns
    -------
    head_state_dict : updated head weights
    avg_loss        : mean loss over all local epochs
    avg_acc         : mean accuracy over all local epochs
    """
    model = CustomECGModel(config["NUM_CLASSES"]).to(config["DEVICE"])
    model.load_state_dict(global_weights)
    model.train()

    optimizer = optim.Adam(
        model.head.parameters(),
        lr=config["INITIAL_LEARNING_RATE"],
        weight_decay=config["L2_REG"],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config["DEVICE"]))
    device_type = config["DEVICE"].split(":")[0]
    scaler = torch.amp.GradScaler(enabled=(device_type == "cuda"))

    total_loss, total_correct, total_samples = 0.0, 0, 0

    epoch_bar = tqdm(
        range(config["CLIENT_EPOCHS"]),
        desc=f"  Client {client_id + 1}",
        leave=False,
        unit="epoch",
    )
    for _ in epoch_bar:
        batch_bar = tqdm(loader, leave=False, unit="batch", desc="    batches")
        for images, labels in batch_bar:
            images = images.to(config["DEVICE"])
            labels = labels.to(config["DEVICE"])

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, preds = torch.max(outputs, 1)
            batch_loss = loss.item() * images.size(0)
            batch_correct = (preds == labels).sum().item()
            total_loss += batch_loss
            total_correct += batch_correct
            total_samples += len(labels)

            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{batch_correct / len(labels):.3f}",
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_correct / max(total_samples, 1)
    return model.head.state_dict(), avg_loss, avg_acc


def fedavg(client_weights: list[dict], client_sizes: list[int]) -> dict:
    """
    Federated Averaging (FedAvg) aggregation.

    Computes a weighted average of client model parameters, where each
    client's contribution is proportional to its local dataset size.
    """
    total = sum(client_sizes)
    aggregated = OrderedDict()

    for key in client_weights[0]:
        aggregated[key] = torch.zeros_like(
            client_weights[0][key], dtype=torch.float32
        )

    for weights, size in zip(client_weights, client_sizes):
        scale = size / total
        for key in weights:
            aggregated[key] += weights[key].to(torch.float32) * scale

    return aggregated


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """Return (loss, accuracy) on *loader*."""
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    device_type = config["DEVICE"].split(":")[0]

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(config["DEVICE"])
            labels = labels.to(config["DEVICE"])
            with torch.amp.autocast(
                device_type=device_type, enabled=(device_type == "cuda")
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
    )


# ---------------------------------------------------------------------------
# Main federated learning loop (one fold)
# ---------------------------------------------------------------------------
def run_federated_learning(
    train_dataset: Dataset,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    fold_output_dir: str,
) -> tuple[dict, str]:
    """Run the full FL process for a single cross-validation fold."""

    print("[INFO] Initialising global model …")
    global_model = CustomECGModel(config["NUM_CLASSES"]).to(config["DEVICE"])
    global_weights = global_model.state_dict()

    client_loaders, client_sizes = create_client_dataloaders(
        train_dataset, config["NUM_CLIENTS"]
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_acc = 0.0
    best_weights_path = os.path.join(
        fold_output_dir,
        f"best_{config['BASE_MODEL_NAME']}_federated.pth",
    )

    round_bar = tqdm(
        range(config["NUM_ROUNDS"]),
        desc="Federated Rounds",
        unit="round",
    )

    for round_num in round_bar:
        round_head_weights, round_losses, round_accs = [], [], []

        for client_id, loader in enumerate(client_loaders):
            head_w, loss, acc = train_client(
                client_id, loader, global_weights, class_weights
            )
            round_head_weights.append(head_w)
            round_losses.append(loss)
            round_accs.append(acc)

        if not round_head_weights:
            tqdm.write("[WARN] No clients trained this round — skipping.")
            continue

        # Aggregate and update global model
        aggregated_head = fedavg(round_head_weights, client_sizes)
        global_model.head.load_state_dict(aggregated_head)
        global_weights = global_model.state_dict()

        # Evaluate
        val_loss, val_acc = evaluate(
            global_model, val_loader, nn.CrossEntropyLoss()
        )

        avg_train_loss = float(np.mean(round_losses))
        avg_train_acc = float(np.mean(round_accs))

        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(avg_train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        round_bar.set_postfix(
            tr_loss=f"{avg_train_loss:.4f}",
            tr_acc=f"{avg_train_acc:.4f}",
            val_loss=f"{val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), best_weights_path)
            tqdm.write(
                f"  ✔ Round {round_num + 1:3d}: val_acc improved → {val_acc:.4f}  "
                f"(weights saved)"
            )

    return history, best_weights_path


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list[str],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha="right")
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    tqdm.write(f"  [PLOT] Confusion matrix → {output_path}")


def plot_training_history(history: dict, output_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot(history["train_loss"], "b-o", label="Avg Client Train Loss")
    axes[0].plot(history["val_loss"], "r-s", label="Global Val Loss")
    axes[0].set_title("Loss over Federated Rounds")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_accuracy"], "g-o", label="Avg Client Train Acc")
    axes[1].plot(history["val_accuracy"], "m-s", label="Global Val Acc")
    axes[1].set_title("Accuracy over Federated Rounds")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    tqdm.write(f"  [PLOT] Training history  → {output_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    classes: list[str],
    output_path: str,
) -> None:
    lb = LabelBinarizer()
    lb.fit(range(len(classes)))
    y_bin = lb.transform(y_true)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(classes):
        if i < y_bin.shape[1]:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    tqdm.write(f"  [PLOT] ROC curves        → {output_path}")


def predict(model: nn.Module, loader: DataLoader):
    """Return (y_true, y_pred, y_probs) arrays."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Predicting", leave=False):
            outputs = model(images.to(config["DEVICE"]))
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(np.argmax(probs, axis=1))
            all_probs.extend(probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    setup_environment()

    try:
        all_paths, all_labels, CLASS_NAMES = get_image_paths_and_labels(
            config["DATASET_PATH"]
        )
    except Exception as exc:
        print(f"[ERROR] Could not load dataset: {exc}")
        return

    os.makedirs(config["BASE_OUTPUT_DIR"], exist_ok=True)

    skf = StratifiedKFold(
        n_splits=config["K_FOLDS"],
        shuffle=True,
        random_state=config["RANDOM_STATE"],
    )
    data_transform = config["WEIGHTS_FN"].transforms()
    fold_results = []

    fold_bar = tqdm(
        enumerate(skf.split(all_paths, all_labels), start=1),
        total=config["K_FOLDS"],
        desc="K-Fold",
        unit="fold",
    )

    for fold, (train_idx, val_idx) in fold_bar:
        fold_bar.set_description(f"K-Fold  [Fold {fold}/{config['K_FOLDS']}]")
        print(f"\n{'=' * 60}")
        print(f"  FOLD {fold} / {config['K_FOLDS']}")
        print(f"{'=' * 60}")

        fold_dir = os.path.join(config["BASE_OUTPUT_DIR"], f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ds = ECGDataset(
            all_paths[train_idx], all_labels[train_idx], transform=data_transform
        )
        val_ds = ECGDataset(
            all_paths[val_idx], all_labels[val_idx], transform=data_transform
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"],
            pin_memory=True,
        )

        # Compute class weights to handle imbalance
        counts = np.bincount(all_labels[train_idx])
        class_weights = torch.tensor(
            [
                len(train_idx) / (config["NUM_CLASSES"] * c) if c > 0 else 0.0
                for c in counts
            ],
            dtype=torch.float32,
        )

        history, best_weights_path = run_federated_learning(
            train_ds, val_loader, class_weights, fold_dir
        )

        # ── Final evaluation with best checkpoint ──────────────────────────
        print(f"\n[INFO] Final evaluation — Fold {fold} (best weights) …")
        if os.path.exists(best_weights_path):
            try:
                best_model = CustomECGModel(config["NUM_CLASSES"]).to(config["DEVICE"])
                best_model.load_state_dict(
                    torch.load(best_weights_path, map_location=config["DEVICE"])
                )

                val_loss, val_acc = evaluate(
                    best_model, val_loader, nn.CrossEntropyLoss()
                )
                fold_results.append({"acc": val_acc, "loss": val_loss})
                print(f"  Val Acc : {val_acc:.4f}   Val Loss : {val_loss:.4f}")

                y_true, y_pred, y_probs = predict(best_model, val_loader)

                plot_confusion_matrix(
                    confusion_matrix(y_true, y_pred),
                    CLASS_NAMES,
                    os.path.join(fold_dir, "confusion_matrix.png"),
                )

                report = classification_report(
                    y_true,
                    y_pred,
                    target_names=CLASS_NAMES,
                    digits=4,
                    zero_division=0,
                )
                print(f"\nClassification Report — Fold {fold}:\n{report}")
                with open(
                    os.path.join(fold_dir, "classification_report.txt"), "w"
                ) as fh:
                    fh.write(report)

                plot_roc_curves(
                    y_true,
                    y_probs,
                    CLASS_NAMES,
                    os.path.join(fold_dir, "roc_curves.png"),
                )

            except Exception as exc:
                print(f"[ERROR] Evaluation failed for fold {fold}: {exc}")
                traceback.print_exc()

        plot_training_history(
            history, os.path.join(fold_dir, "training_history.png")
        )

        # Free GPU memory before next fold
        try:
            del best_model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # ── Cross-validation summary ───────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  CROSS-VALIDATION SUMMARY")
    print(f"{'=' * 60}")

    if fold_results:
        accs = [r["acc"] for r in fold_results]
        losses = [r["loss"] for r in fold_results]
        print(f"  Avg Val Accuracy : {np.mean(accs):.4f}  ± {np.std(accs):.4f}")
        print(f"  Avg Val Loss     : {np.mean(losses):.4f}  ± {np.std(losses):.4f}")
        for i, r in enumerate(fold_results, start=1):
            print(f"    Fold {i}: acc={r['acc']:.4f}  loss={r['loss']:.4f}")

        summary_path = os.path.join(config["BASE_OUTPUT_DIR"], "kfold_summary.txt")
        with open(summary_path, "w") as fh:
            fh.write(
                f"K-Fold Cross-Validation Summary\n"
                f"Model   : {config['BASE_MODEL_NAME']}\n"
                f"K-Folds : {config['K_FOLDS']}\n"
                f"Rounds  : {config['NUM_ROUNDS']}\n"
                f"Clients : {config['NUM_CLIENTS']}\n"
                + "-" * 50
                + "\n"
                f"Avg Val Accuracy : {np.mean(accs):.4f} ± {np.std(accs):.4f}\n"
                f"Avg Val Loss     : {np.mean(losses):.4f} ± {np.std(losses):.4f}\n\n"
                "Per-fold results:\n"
            )
            for i, r in enumerate(fold_results, start=1):
                fh.write(f"  Fold {i}: acc={r['acc']:.4f}  loss={r['loss']:.4f}\n")

        print(f"\n[INFO] Summary saved → {summary_path}")
    else:
        print("[WARN] No fold results to aggregate.")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
