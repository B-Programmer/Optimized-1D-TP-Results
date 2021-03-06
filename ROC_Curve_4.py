import numpy as np

def normalize(array2d: np.ndarray) -> np.ndarray:
    return array2d / array2d.sum(axis=1, keepdims=True)

class_names = ["Cat", "Dog", "Pig"]
num_classes = len(class_names)
num_samples = 500

# Mock ground truth
ground_truth = np.random.choice(range(num_classes), size=num_samples, p=[0.5, 0.4, 0.1])

# Mock model predictions
perfect_model = np.eye(num_classes)[ground_truth]
noisy_model = normalize(
    perfect_model + 2 * np.random.random((num_samples, num_classes))
)
random_model = normalize(np.random.random((num_samples, num_classes)))


import metriculous

metriculous.compare_classifiers(
    ground_truth=ground_truth,
    model_predictions=[perfect_model, noisy_model, random_model],
    model_names=["Perfect Model", "Noisy Model", "Random Model"],
    class_names=class_names,
    one_vs_all_figures=True, # This line is important to include ROC curves in the output
).save_html("model_comparison.html").display()