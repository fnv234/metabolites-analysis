"""
Metabolite condition classifier: train/test with z-score or raw features,
optional feature selection, and multiple ML models (including XGBoost with label encoding).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Raw metabolite columns (non–Z-score) in data.csv order
RAW_METAB_COLS = [
    "LACTIC", "@2OHBUT", "@3OHBUT", "PYRUVIC", "CISACONITIC", "CITRIC",
    "@3OHPROPIONIC", "@3OH2MEBUT", "@3OHIVAl", "SUCCINIC", "FUMARIC",
    "@3MEGLUTACONIC", "MALIC", "@2KETOISOVALERIC", "@2KETOBUTl", "ACETOACETICl",
    "@3ME2KETOVALERIC", "@2KETOISOCAPROIC", "@2MECITRIC", "@2KETOGLUTARIC",
]


class MetaboliteConditionClassifier:
    """
    Classify Condition from metabolite features. Supports:
    - feature_set: "zscore" | "raw" | "both" or list of column names
    - Optional feature selection (SelectKBest or RF importance) then rerun
    - Multiple models including XGBoost (uses LabelEncoder for string labels)
    """

    def __init__(self, df_ml, feature_set="zscore", test_size=0.25, random_state=42):
        """
        df_ml: DataFrame with Condition_mapped and feature columns.
        feature_set: "zscore" (age + Z*), "raw" (age + raw metabolites), "both", or list of column names.
        """
        self.df_ml = df_ml
        self.test_size = test_size
        self.random_state = random_state
        self.feature_set = feature_set
        self.feature_cols_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.selector_ = None
        self.X_train_s_ = None
        self.X_test_s_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.X_filled_ = None
        self.y_ = None
        self._set_feature_columns()

    def _set_feature_columns(self):
        """Set feature_cols_ from feature_set."""
        if isinstance(self.feature_set, list):
            self.feature_cols_ = [c for c in self.feature_set if c in self.df_ml.columns]
            return
        z_cols = [c for c in self.df_ml.columns if c.startswith("Z") and len(c) > 1]
        raw_cols = [c for c in RAW_METAB_COLS if c in self.df_ml.columns]
        age = ["ageatcollection"] if "ageatcollection" in self.df_ml.columns else []
        if self.feature_set == "zscore":
            self.feature_cols_ = age + z_cols
        elif self.feature_set == "raw":
            self.feature_cols_ = age + raw_cols
        elif self.feature_set == "both":
            self.feature_cols_ = age + raw_cols + z_cols
        else:
            self.feature_cols_ = age + z_cols
        self.feature_cols_ = [c for c in self.feature_cols_ if c in self.df_ml.columns]

    def get_feature_columns(self):
        """Return current feature column list (after optional feature selection)."""
        return self.feature_cols_

    def prepare_data(self):
        """
        Build X, y; drop rows with any NaN in features; split and scale.
        Fits LabelEncoder on y_train for XGBoost. Returns self for chaining.
        """
        X_raw = self.df_ml[self.feature_cols_].copy()
        for c in self.feature_cols_:
            X_raw[c] = pd.to_numeric(X_raw[c], errors="coerce")
        X = X_raw.dropna(how="all")
        valid_idx = X.dropna().index
        X = X.loc[valid_idx].copy()
        y = self.df_ml.loc[valid_idx, "Condition_mapped"]
        X_filled = X.fillna(X.median())
        self.X_filled_ = X_filled
        self.y_ = y

        min_class = y.value_counts().min()
        stratify_arg = y if min_class >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X_filled, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify_arg
        )
        self.scaler_ = StandardScaler()
        self.X_train_s_ = self.scaler_.fit_transform(X_train)
        self.X_test_s_ = self.scaler_.transform(X_test)
        self.y_train_ = y_train
        self.y_test_ = y_test

        self.label_encoder_ = LabelEncoder()
        self.label_encoder_.fit(y_train)
        return self

    def select_features(self, method="kbest", k=10):
        """
        Select top k features from current X_train_s_ / X_test_s_.
        method: "kbest" (f_classif) or "rf" (RandomForest feature_importances_).
        Updates feature_cols_ and X_train_s_, X_test_s_ to selected features only.
        """
        if self.X_train_s_ is None or self.y_train_ is None:
            raise RuntimeError("Call prepare_data() before select_features()")
        n_features = min(k, self.X_train_s_.shape[1])
        if method == "kbest":
            self.selector_ = SelectKBest(score_func=f_classif, k=n_features)
            self.selector_.fit(self.X_train_s_, self.y_train_)
            selected_mask = self.selector_.get_support()
            self.feature_cols_ = [c for c, m in zip(self.feature_cols_, selected_mask) if m]
            self.X_train_s_ = self.selector_.transform(self.X_train_s_)
            self.X_test_s_ = self.selector_.transform(self.X_test_s_)
        elif method == "rf":
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(self.X_train_s_, self.y_train_)
            selected_idx = np.argsort(rf.feature_importances_)[-n_features:]
            self.feature_cols_ = [self.feature_cols_[i] for i in selected_idx]
            self.X_train_s_ = self.X_train_s_[:, selected_idx]
            self.X_test_s_ = self.X_test_s_[:, selected_idx]
            self.selector_ = None
        else:
            raise ValueError("method must be 'kbest' or 'rf'")
        return self

    def _get_cv(self):
        if self.y_train_ is None:
            return KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        if self.y_train_.value_counts().min() >= 2:
            return StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=5, shuffle=True, random_state=self.random_state)

    def run_models(self, include_xgb=True, verbose=True):
        """
        Fit multiple classifiers and optionally XGBoost (with integer labels).
        Returns list of dicts with Model, CV Accuracy, Test Accuracy; prints classification reports.
        """
        if self.X_train_s_ is None:
            raise RuntimeError("Call prepare_data() before run_models()")
        cv = self._get_cv()
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "SVM (RBF)": SVC(kernel="rbf", random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
        }
        results = []
        for name, model in models.items():
            scores = cross_val_score(model, self.X_train_s_, self.y_train_, cv=cv, scoring="accuracy")
            model.fit(self.X_train_s_, self.y_train_)
            y_pred = model.predict(self.X_test_s_)
            acc_test = accuracy_score(self.y_test_, y_pred)
            results.append({
                "Model": name,
                "CV Accuracy (mean)": scores.mean(),
                "CV std": scores.std(),
                "Test Accuracy": acc_test,
            })
            if verbose:
                print(f"\n--- {name} ---")
                print(f"CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
                print(f"Test accuracy: {acc_test:.3f}")
                print(classification_report(self.y_test_, y_pred, zero_division=0))

        if include_xgb:
            try:
                from xgboost import XGBClassifier
                # XGBoost requires integer labels
                y_train_int = self.label_encoder_.transform(self.y_train_)
                xgb = XGBClassifier(random_state=self.random_state, eval_metric="mlogloss")
                scores_xgb = cross_val_score(xgb, self.X_train_s_, y_train_int, cv=cv, scoring="accuracy")
                xgb.fit(self.X_train_s_, y_train_int)
                y_pred_int = xgb.predict(self.X_test_s_)
                y_pred_xgb = self.label_encoder_.inverse_transform(y_pred_int)
                acc_xgb = accuracy_score(self.y_test_, y_pred_xgb)
                results.append({
                    "Model": "XGBoost",
                    "CV Accuracy (mean)": scores_xgb.mean(),
                    "CV std": scores_xgb.std(),
                    "Test Accuracy": acc_xgb,
                })
                if verbose:
                    print("\n--- XGBoost ---")
                    print(f"CV accuracy: {scores_xgb.mean():.3f} (+/- {scores_xgb.std():.3f})")
                    print(f"Test accuracy: {acc_xgb:.3f}")
                    print(classification_report(self.y_test_, y_pred_xgb, zero_division=0))
            except ImportError:
                if verbose:
                    print("XGBoost not installed. pip install xgboost to include it.")

        return results

    def confusion_matrix_plot(self, model_name="Random Forest", ax=None):
        """Plot confusion matrix for a previously fitted model (by name). Re-fit that model for the plot."""
        cv = self._get_cv()
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=self.random_state),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            "SVM (RBF)": SVC(kernel="rbf", random_state=self.random_state),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
            "Decision Tree": DecisionTreeClassifier(random_state=self.random_state),
        }
        if model_name == "XGBoost":
            try:
                from xgboost import XGBClassifier
                model = XGBClassifier(random_state=self.random_state, eval_metric="mlogloss")
                y_train_int = self.label_encoder_.transform(self.y_train_)
                model.fit(self.X_train_s_, y_train_int)
                y_pred_int = model.predict(self.X_test_s_)
                y_pred = self.label_encoder_.inverse_transform(y_pred_int)
            except ImportError:
                raise ImportError("XGBoost not installed")
        elif model_name in models:
            model = models[model_name]
            model.fit(self.X_train_s_, self.y_train_)
            y_pred = model.predict(self.X_test_s_)
        else:
            raise ValueError("Unknown model_name")
        import matplotlib.pyplot as plt
        cm = confusion_matrix(self.y_test_, y_pred)
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_title(f"Confusion matrix: {model_name}")
        classes = sorted(self.y_test_.unique())
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        return ax
