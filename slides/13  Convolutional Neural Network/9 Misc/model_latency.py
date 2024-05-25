import torch
import time


def measure_model_latency(model, input_data, iterations=100, device=torch.device('cpu')):
    """
    Measures the latency of a PyTorch model's forward pass.

    Parameters:
    - model: The PyTorch model to evaluate.
    - input_data: A single batch of input data for the model.
    - iterations: The number of iterations to run the forward pass.
    - device: The device to run the model on ('cpu' or 'cuda').

    Returns:
    - The average latency in milliseconds for the forward pass.
    """

    # Ensure model is on the correct device
    model.to(device)

    # Ensure input data is on the correct device and in evaluation mode
    input_data = input_data.to(device)
    model.eval()

    # Warm-up run to cache memory allocations, etc.
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)

    # Synchronize GPU before starting timing (if using GPU)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

    start_time = time.time()

    # Run the forward pass multiple times and measure the time taken
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_data)
            # Synchronize for each iteration if measuring for GPU to ensure accurate timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

    # Calculate the total duration in seconds
    duration = time.time() - start_time

    # Calculate the average latency per forward pass in milliseconds
    avg_latency = (duration / iterations) * 1000

    return avg_latency

