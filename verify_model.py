import torch
import time
from chronos import Chronos2Pipeline

print("Loading Chronos-2...")
pipeline = Chronos2Pipeline.from_pretrained(
    "amazon/chronos-2",
    device_map="cpu",
    dtype=torch.float32,
    revision="29ec3766d36d6f73f0696f85560a422f50e8498c"
)

# Chronos-2 expects a 3D tensor: shape (n_series=1, n_variates=1, history_length=5)
dummy_prices = torch.tensor([[[2500.5, 2510.0, 2495.5, 2520.0, 2535.0]]])

print("Running inference...")
start_time = time.time()
forecast = pipeline.predict_quantiles(
    inputs=dummy_prices,
    prediction_length=1,
    quantile_levels=[0.5]
)
elapsed = time.time() - start_time

print(f"Success! Forecast: {forecast[0][0][0].item():.2f}")
print(f"Elapsed time: {elapsed:.2f} seconds")