import mindspore as ms
import numpy as np
from mindspore import Tensor

def check_mindspore_gpu():
    """
    Checks if MindSpore can successfully initialize and use the GPU.
    """
    print(f"MindSpore version: {ms.__version__}")
    try:
        # 1. Attempt to set the backend device to 'GPU'.
        #    If this fails, it will raise an exception.
        ms.context.set_context(device_target="GPU")
        print("✅ Successfully set device_target to 'GPU'.")

        # 2. Perform a simple operation to verify the backend is functional.
        x = Tensor(np.ones([2, 2]), ms.float32)
        y = Tensor(np.ones([2, 2]), ms.float32)
        result = x + y

        print("✅ A simple tensor addition was executed successfully.")
        print("\n🎉 MindSpore is configured correctly and can use your GPU!")

    except RuntimeError as e:
        print("\n❌ MindSpore could not initialize the GPU backend.")
        print(f"   Error details: {e}")
        print("\n   Please check the following:")
        print("   1. Your NVIDIA drivers are installed correctly.")
        print("   2. The installed MindSpore version matches your CUDA version.")
        print("   3. You installed the 'mindspore-gpu' package.")

    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    check_mindspore_gpu()