# fit_model.py
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle

REPO_ROOT = Path(os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parents[1])))
BASE_DIR = REPO_ROOT / "calo_configs/par04/SimpleBox/SiW_iter1"
OUTPUT_DIR = BASE_DIR / "plots"

# Load data
with open(BASE_DIR / "measured_sf.json") as f:
    data = json.load(f)

# Load parameter 'a' from configs
with open(BASE_DIR / "configs_metadata.json") as f:
    configs_meta = json.load(f)

print(f"\n{'='*60}")
print("FITTING INVERSE MODEL: a = f(SF)")
print(f"{'='*60}")

# Merge data by ID (robust to missing IDs)
id_to_sf = {d['id']: d['sf_mean'] for d in data}
id_to_a = {c['id']: c['a_parameter'] for c in configs_meta}
common_ids = sorted(set(id_to_sf.keys()) & set(id_to_a.keys()))

sf_measured = np.array([id_to_sf[i] for i in common_ids])
a_parameter = np.array([id_to_a[i] for i in common_ids])

# Check for linear relationship
ratio = sf_measured / a_parameter
print(f"Data points: {len(common_ids)}")
print(f"SF range:  [{sf_measured.min():.4f}, {sf_measured.max():.4f}]")
print(f"'a' range: [{a_parameter.min():.5f}, {a_parameter.max():.5f}]")
print(f"SF/a ratio: {ratio.mean():.2f} ± {ratio.std():.2f}")
if ratio.std() / ratio.mean() < 0.1:
    print(f"✓ Nearly linear relationship detected!\n")
else:
    print(f"⚠️ Non-linear relationship - ratio varies >10%\n")

# INVERSE FIT with proper scaling
X = sf_measured.reshape(-1, 1)  # Input: SF
y = a_parameter                  # Output: a

# ============================================================================
# DEFINE ALL MODELS
# ============================================================================
models = []

# Polynomial models
for deg in [1, 3, 4, 5]:
    models.append((
        f"Polynomial_deg{deg}",
        Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=deg)),
            ('ridge', Ridge(alpha=0.1))
        ])
    ))

# ML models
models.extend([
    ("NeuralNetwork", Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(10),
                            activation='tanh',
                            solver='lbfgs',
                            alpha=0.01,
                            max_iter=5000,
                            random_state=42))
    ])),
    ("SVM", Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.0001))
    ])),
    ("RandomForest", RandomForestRegressor(n_estimators=200, max_depth=10, 
                                           min_samples_split=5, random_state=42)),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                    learning_rate=0.05, random_state=42)),
])

# ============================================================================
# TRAIN AND EVALUATE ALL MODELS
# ============================================================================
print(f"{'Model':<20} {'CV RMSE':<12} {'Rel Error (%)':<15} {'Status'}")
print(f"{'-'*65}")

results = []
for name, model in models:
    # CV evaluation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -cv_scores.mean()
    cv_rmse = np.sqrt(cv_mse)
    cv_error_pct = (cv_rmse / y.mean()) * 100
    
    # Train and get CV predictions
    model.fit(X, y)
    cv_pred = cross_val_predict(model, X, y, cv=5)
    
    # Calculate relative errors per point
    rel_errors_pct = 100 * np.abs(cv_pred - y) / y
    
    results.append({
        'name': name,
        'model': model,
        'cv_mse': cv_mse,
        'cv_rmse': cv_rmse,
        'cv_error_pct': cv_error_pct,
        'cv_pred': cv_pred,
        'rel_errors': rel_errors_pct
    })
    
    status = "✓✓✓" if rel_errors_pct.mean() < 3 else "✓✓" if rel_errors_pct.mean() < 5 else "✓"
    print(f"{name:<20} {cv_rmse:.8f}     {rel_errors_pct.mean():.3f}%            {status}")

# Select best (lowest mean relative error, not CV MSE)
best = min(results, key=lambda x: x['rel_errors'].mean())
print(f"\n✓ Best Model: {best['name']} "
      f"(Mean Error = {best['rel_errors'].mean():.3f}%)")

# Save model
with open(BASE_DIR / "model.pkl", 'wb') as f:
    pickle.dump(best['model'], f)
print(f"✓ Saved: {BASE_DIR / 'model.pkl'}")

# Detailed analysis
cv_pred = best['cv_pred']
cv_errors_pct = best['rel_errors']

print(f"\n{'='*60}")
print(f"BEST MODEL ANALYSIS: {best['name']}")
print(f"{'='*60}")
print(f"CV Mean error:   {cv_errors_pct.mean():.2f}%")
print(f"CV Median error: {np.median(cv_errors_pct):.2f}%")
print(f"CV Max error:    {cv_errors_pct.max():.2f}%")
print(f"Configs > 10% error: {np.sum(cv_errors_pct > 10)}/{len(cv_errors_pct)}")
print(f"Configs > 20% error: {np.sum(cv_errors_pct > 20)}/{len(cv_errors_pct)}")

# ============================================================================
# COMPREHENSIVE PLOTS - MODEL COMPARISON (3x3)
# ============================================================================
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'lines.linewidth': 2,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.5,
    'ytick.major.width': 1.5,
})

fig = plt.figure(figsize=(20, 20))

# 1-8. Individual Model Predictions (sorted by performance)
sorted_results = sorted(results, key=lambda x: x['rel_errors'].mean())
lim = [y.min()*0.95, y.max()*1.05]

for i, result in enumerate(sorted_results[:8]):  # Top 8 models
    ax = plt.subplot(3, 3, 1 + i)
    pred = result['cv_pred']
    ax.scatter(y, pred, s=120, alpha=0.7, edgecolors='black', linewidths=2)
    ax.plot(lim, lim, 'r--', lw=2.5)
    ax.set_xlabel('a parameter', fontweight='bold', fontsize=26)
    if i % 3 == 0:
        ax.set_ylabel('Predicted a', fontweight='bold', fontsize=26)
    ax.set_title(f"{result['name']}\nError: {result['rel_errors'].mean():.3f}%",
                fontweight='bold', fontsize=24)
    ax.set_xlim(lim)
    ax.set_ylim(lim)

# 9. Best Model: SF vs a relationship
ax = plt.subplot(3, 3, 9)
ax.scatter(sf_measured, a_parameter, s=150, alpha=0.7, edgecolors='black', 
           linewidths=2.5, label='Measured', zorder=3)

# Plot learned function
sf_grid = np.linspace(sf_measured.min()*0.9, sf_measured.max()*1.1, 200).reshape(-1, 1)
a_pred_grid = best['model'].predict(sf_grid)
ax.plot(sf_grid, a_pred_grid, 'r-', lw=3, label=f'{best["name"]}', zorder=2)

ax.set_xlabel('Sampling Fraction', fontweight='bold', fontsize=26)
ax.set_ylabel('Parameter a', fontweight='bold', fontsize=26)
ax.set_title(f'{best["name"]}: a = f(SF)', fontweight='bold', fontsize=24)
ax.legend(fontsize=20)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Plot: {OUTPUT_DIR / 'model_comparison.png'}")

# ============================================================================
# ERROR ANALYSIS PLOTS (2x2)
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Measured vs Predicted 'a'
ax = axes[0,0]
colors = plt.cm.RdYlGn_r(cv_errors_pct / max(cv_errors_pct.max(), 10))
ax.scatter(y, cv_pred, c=colors, s=150, alpha=0.7, edgecolors='black', linewidths=2)
ax.plot(lim, lim, 'r--', lw=2, label='Perfect')
ax.set_xlabel('Measured a', fontsize=14, fontweight='bold')
ax.set_ylabel('CV Predicted a', fontsize=14, fontweight='bold')
ax.set_title(f'{best["name"]}: a Prediction', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)

# 2. Residuals
ax = axes[0,1]
residuals = y - cv_pred
ax.scatter(y, residuals, s=150, alpha=0.7, edgecolors='black', linewidths=2, c=colors)
ax.axhline(0, color='r', ls='--', lw=2)
mean_res = residuals.mean()
ax.axhline(mean_res, color='orange', ls='--', lw=1, 
           label=f'Mean bias={mean_res:.6f}')
ax.set_xlabel('Measured a', fontsize=14, fontweight='bold')
ax.set_ylabel('Residual', fontsize=14, fontweight='bold')
ax.set_title('CV Residuals', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)

# 3. SF vs a (learned relationship)
ax = axes[1,0]
ax.scatter(sf_measured, a_parameter, s=150, alpha=0.7, edgecolors='black', 
           linewidths=2, label='Measured', zorder=3)

# Plot learned function
sf_grid = np.linspace(sf_measured.min()*0.9, sf_measured.max()*1.1, 200).reshape(-1, 1)
a_pred_grid = best['model'].predict(sf_grid)
ax.plot(sf_grid, a_pred_grid, 'r-', lw=3, label=f'Model', zorder=2)

ax.set_xlabel('Sampling Fraction', fontsize=14, fontweight='bold')
ax.set_ylabel('Parameter a', fontsize=14, fontweight='bold')
ax.set_title('Learned Relationship: a = f(SF)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)

# 4. Error distribution
ax = axes[1,1]
ax.hist(cv_errors_pct, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(cv_errors_pct.mean(), color='r', ls='--', lw=2, 
          label=f'Mean={cv_errors_pct.mean():.2f}%')

ax.set_xlabel('Prediction Error (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Count', fontsize=14, fontweight='bold')
ax.set_title('Error Distribution', fontsize=16, fontweight='bold')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_analysis.png", dpi=150)
print(f"✓ Plot: {OUTPUT_DIR / 'error_analysis.png'}")

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"Model: a = {best['name']}(SF)")
print(f"Expected error: {cv_errors_pct.mean():.2f}%")
print(f"✓ Ready to generate final configs!")
print(f"{'='*60}\n")