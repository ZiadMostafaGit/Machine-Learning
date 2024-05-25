import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, n_redundant=5, n_classes=3, random_state=42)

# We need 3 sets: Split the data into train, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Calibrate the classifier
calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv='prefit')
calibrated_clf.fit(X_val, y_val)

y_pred = calibrated_clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Set up the figure and axes
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
fig.suptitle('Reliability Diagrams for Multiclass Classifier')

# Predict probabilities on the test set using both classifiers
prob_pos_base = clf.predict_proba(X_test)
prob_pos_calibrated = calibrated_clf.predict_proba(X_test)

for i, (ax, class_name) in enumerate(zip(axes, ["A", "B", "C"])):
    # Compare reliability diagrams
    fraction_of_positives_base, mean_predicted_value_base = calibration_curve(y_test == i, prob_pos_base[:, i],
                                                                              n_bins=10)
    fraction_of_positives_calibrated, mean_predicted_value_calibrated = calibration_curve(y_test == i,
                                                                                          prob_pos_calibrated[:, i],
                                                                                          n_bins=10)

    ax.plot([0, 1], [0, 1], 'k:', label='Perfect Classifier')
    ax.plot(mean_predicted_value_base, fraction_of_positives_base, 's-', label='Base Classifier')
    ax.plot(mean_predicted_value_calibrated, fraction_of_positives_calibrated, 's-', label='Calibrated Classifier')
    ax.set_ylabel('Fraction of positives')
    ax.set_xlabel('Mean predicted probability')
    ax.set_title('%s' % class_name)
    ax.legend()


plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()


