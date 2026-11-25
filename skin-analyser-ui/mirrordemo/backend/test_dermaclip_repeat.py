import cv2
import numpy as np
import time
from dermaclip_analyzer import DermaCLIPAnalyzer

image = cv2.imread("test_skin.png")
if image is None:
    raise RuntimeError("Could not load test_skin.png")

print("Initializing DermaCLIP...")
analyzer = DermaCLIPAnalyzer.get_instance()

if not analyzer.is_loaded:
    raise RuntimeError("DermaCLIP did not load. Aborting test.")

results = []

for run in range(1, 4):
    start = time.perf_counter()
    result = analyzer.analyze(image.copy())
    elapsed = time.perf_counter() - start

    if "probabilities" not in result:
        raise RuntimeError(f"Run {run} returned fallback or invalid output")

    probs = result["probabilities"]
    values = np.array(list(probs.values()), dtype=float)

    results.append(values)

    print(f"\n========== RUN {run} ==========")
    print(f"Inference time: {elapsed * 1000:.2f} ms")
    print("Predicted class:", result["predicted_class"])
    print("Confidence:", result["confidence"])
    print("Probability sum:", values.sum())

    for name, probability in probs.items():
        print(f"{name:15s}: {probability:.8f}")

print("\n========== REPRODUCIBILITY ==========")

for i in range(1, 3):
    difference = np.max(np.abs(results[0] - results[i]))
    print(f"Run 1 vs Run {i+1}: {difference:.12f}")

print("\nTEST COMPLETED")
