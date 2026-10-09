"""Train, tune and evaluate AQI models. Saves the best classical model for deployment."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import BASE_FEATURES, add_features, aqi_bucket

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- 1. Data preparation ----------
df = pd.read_csv("data/city_day.csv")
print("Raw shape:", df.shape)
print("Missing values per column:\n", df[BASE_FEATURES + ["AQI"]].isna().sum())

df = df.dropna(subset=["AQI"])                      # target must exist
X = add_features(df[BASE_FEATURES])                 # feature engineering
y = df["AQI"].values
print("Rows after dropping missing AQI:", len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

# Imputer + scaler fitted on the training set only (no data leakage)
prep = Pipeline([("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())])
Xtr = prep.fit_transform(X_train)
Xte = prep.transform(X_test)


def evaluate(name, y_pred):
    true_b, pred_b = aqi_bucket(y_test), aqi_bucket(y_pred)
    return {
        "Model": name,
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2),
        "R2": round(r2_score(y_test, y_pred), 3),
        "Accuracy": round(accuracy_score(true_b, pred_b), 3),
        "F1": round(f1_score(true_b, pred_b, average="weighted"), 3),
    }


results, fitted = [], {}

# ---------- 2. Baseline models ----------
for name, model in [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=SEED)),
]:
    model.fit(Xtr, y_train)
    fitted[name] = model
    results.append(evaluate(name, model.predict(Xte)))
    print(results[-1])

# ---------- 3. Tuned Gradient Boosting ----------
search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=SEED),
    param_distributions={
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    },
    n_iter=10, cv=3, scoring="neg_root_mean_squared_error",
    random_state=SEED, n_jobs=-1)
search.fit(Xtr, y_train)
print("Best Gradient Boosting params:", search.best_params_)
fitted["Gradient Boosting (tuned)"] = search.best_estimator_
results.append(evaluate("Gradient Boosting (tuned)", search.best_estimator_.predict(Xte)))
print(results[-1])

# ---------- 4. Deep learning: MLP in PyTorch ----------
Xt = torch.tensor(Xtr, dtype=torch.float32)
yt = torch.tensor(y_train / 100.0, dtype=torch.float32).unsqueeze(1)   # scale target
Xv = torch.tensor(Xte, dtype=torch.float32)

mlp = nn.Sequential(nn.Linear(Xt.shape[1], 64), nn.ReLU(),
                    nn.Linear(64, 32), nn.ReLU(),
                    nn.Linear(32, 1))
opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(60):
    perm = torch.randperm(len(Xt))
    for i in range(0, len(Xt), 256):
        idx = perm[i:i + 256]
        opt.zero_grad()
        loss = loss_fn(mlp(Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/60  loss={loss.item():.4f}")

mlp.eval()
with torch.no_grad():
    mlp_pred = mlp(Xv).squeeze().numpy() * 100.0
results.append(evaluate("MLP (PyTorch)", mlp_pred))
print(results[-1])
torch.save(mlp.state_dict(), "model/mlp.pt")

# ---------- 5. Results and saving ----------
table = pd.DataFrame(results)
print("\n", table.to_string(index=False))
table.to_csv("model/results.csv", index=False)

# Deploy the best classical model (keeps the Docker image small, no PyTorch needed)
classical = table[table["Model"] != "MLP (PyTorch)"].sort_values("RMSE")
best_name = classical.iloc[0]["Model"]
print("\nDeploying:", best_name)
joblib.dump({"prep": prep, "model": fitted[best_name], "name": best_name},
            "model/aqi_model.joblib")
json.dump({"deployed_model": best_name}, open("model/info.json", "w"))
