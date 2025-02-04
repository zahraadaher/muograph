import torch

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU is available. Using {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("GPU is not available. Using CPU")

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    try:
        c = torch.ones((3, 3, 3))
        # The following two lines are needed for MPS to renounce to use linalg
        # However, running with PYTORCH_ENABLE_MPS_FALLBACK=1; permits using mps for everything but the linalg results stuff
        # d = torch.ones((3,3,3))
        # torch.linalg.svd(c)  # vh shape: (chunk_size, 3, 3)
        torch.linalg.svd(c)  # vh shape: (chunk_size, 3, 3)
        print("MPS is available. Using ", DEVICE)
    except RuntimeError as e:
        DEVICE = torch.device("cpu")
        print(f"Runtime error: {e}. MPS doesn't support stuff. Using CPU")
