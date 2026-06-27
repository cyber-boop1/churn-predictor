import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

def train_and_save_model():
    """Generate synthetic churn data and train a Gradient Boosting model."""

    np.random.seed(42)
    n = 2000

    tenure          = np.random.randint(0, 73, n)
    monthly_charges = np.random.randint(500, 10001, n)
    total_charges   = tenure * monthly_charges + np.random.randint(-500, 500, n)
    contract        = np.random.choice([0, 1, 2], n)      # 0=month, 1=1yr, 2=2yr
    internet        = np.random.choice([0, 1, 2], n)      # 0=DSL, 1=fiber, 2=no
    payment         = np.random.choice([0, 1, 2, 3], n)
    tech_support    = np.random.choice([0, 1], n)
    senior          = np.random.choice([0, 1], n, p=[0.84, 0.16])

    # Churn logic: short tenure + high charges + month-to-month = higher churn
    churn_score = (
        (tenure < 12).astype(int) * 0.4
        + (monthly_charges > 6000).astype(int) * 0.3
        + (contract == 0).astype(int) * 0.2
        + (tech_support == 0).astype(int) * 0.1
        + np.random.uniform(0, 0.3, n)
    )
    churn = (churn_score > 0.5).astype(int)

    X = np.column_stack([
        tenure, monthly_charges, total_charges,
        contract, internet, payment, tech_support, senior
    ])
    y = churn

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    with open("churn_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("churn_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Churn model trained and saved!")

if __name__ == "__main__":
    train_and_save_model()
