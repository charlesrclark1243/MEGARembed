import torch


def get_info() -> dict[str, str]:
    """
    Collect GPU information and PyTorch information and return it.

    Returns:
        dict[str, str]: A dictionary containing the GPU and PyTorch information.
    """

    # GPU info
    cuda_available: bool = torch.cuda.is_available()
    if cuda_available:
        cuda_count: int = torch.cuda.device_count()
        cuda_version: str = torch.version.cuda
    else:
        cuda_count: int = 0
        cuda_version: str = "N/A"

    # PyTorch info
    torch_version: str = torch.__version__

    return {
        "cuda_available": "CUDA is available"
        if cuda_available
        else "CUDA is not available",
        "cuda_count": f"{cuda_count} GPU(s) detected"
        if cuda_available
        else "No GPUs detected",
        "cuda_version": f"CUDA {cuda_version}"
        if cuda_available
        else "CUDA version not available",
        "torch_version": f"PyTorch {torch_version}"
        if torch_version
        else "PyTorch version not available",
    }


def verify(cuda_available: bool) -> bool:
    """
    Verify if CUDA processing works.

    Returns:
        bool: True if CUDA processing works, False otherwise.
    """

    if not cuda_available:
        return False
    else:
        try:
            # Create a random tensor, move it to GPU, and perform a simple operation, then move it back to CPU
            tensor: torch.Tensor = torch.rand(1000, 1000).to("cuda")
            tensor = tensor * 2
            tensor = tensor.to("cpu")

            return True
        except Exception as e:  # noqa: BLE001
            print(f"CUDA processing failed: {e}")
            return False


def main() -> None:
    """
    Main function to check GPU and PyTorch information and verify CUDA processing.
    """

    info: dict[str, str] = get_info()
    for key, value in info.items():
        print(f"{key}: {value}")

    cuda_available: bool = torch.cuda.is_available()
    if cuda_available:
        if verify(cuda_available):
            print("CUDA processing works correctly.")
        else:
            print("CUDA processing failed.")
    else:
        print("CUDA is not available, skipping verification.")


if __name__ == "__main__":
    main()
