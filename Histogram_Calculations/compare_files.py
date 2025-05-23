import numpy as np
from scipy.io import loadmat

def compare_npy_to_mat(npy_path, mat_path, mat_key, atol=1e-6):
    """
    Compare a .npy file to a variable in a .mat file.

    Args:
        npy_path (str): Path to the .npy file.
        mat_path (str): Path to the .mat file.
        mat_key (str): Key (variable name) inside the .mat file.
        atol (float): Absolute tolerance for comparison.

    Returns:
        dict: Dictionary of comparison metrics and result.
    """
    try:
        npy_data = np.load(npy_path)
    except Exception as e:
        return {"error": f"Failed to load .npy file: {e}"}

    try:
        mat_data = loadmat(mat_path)
    except Exception as e:
        return {"error": f"Failed to load .mat file: {e}"}

    if mat_key not in mat_data:
        return {"error": f"Key '{mat_key}' not found in .mat file. Found keys: {list(mat_data.keys())}"}

    mat_array = mat_data[mat_key].squeeze()  # remove singleton dimensions

    if npy_data.shape != mat_array.shape:
        return {
            "error": "Shape mismatch",
            "npy_shape": npy_data.shape,
            "mat_shape": mat_array.shape
        }

    result = {
        "npy_path": npy_path,
        "mat_path": mat_path,
        "npy_mean": np.mean(npy_data),
        "mat_mean": np.mean(mat_array),
        "npy_sum": np.sum(npy_data),
        "mat_sum": np.sum(mat_array),
        "max_diff": float(np.max(np.abs(npy_data - mat_array))),
        "allclose": bool(np.allclose(npy_data, mat_array, atol=atol)),
    }

    return result

if __name__ == "__main__":
    result = compare_npy_to_mat(
        npy_path="./ASVspoof5_train/train_data_ASVSpoof5_pmf_probs_bonafide.npy",
        mat_path="C:/Users/avish/OneDrive/Desktop/thesis_research/ASVSpoof5_Time_Embeddings/calculate_hist/ASVSpoof05/2_16/ASVspoof5_train_data.mat",
        mat_key="pmf_probs_bonafide"  # <-- Replace with your real key
    )

    if "error" in result:
        print("❌", result["error"])
    else:
        print("✅ Comparison Result:")
        for k, v in result.items():
            print(f"{k}: {v}")


    result = compare_npy_to_mat(
        npy_path='./ASVspoof5_train/train_data_ASVSpoof5_pmf_probs_spoofed.npy',
        mat_path="C:/Users/avish/OneDrive/Desktop/thesis_research/ASVSpoof5_Time_Embeddings/calculate_hist/ASVSpoof05/2_16/ASVspoof5_train_data.mat",
        mat_key="pmf_probs_spoofed"  # <-- Replace with your real key
    )

    if "error" in result:
        print("❌", result["error"])
    else:
        print("✅ Comparison Result:")
        for k, v in result.items():
            print(f"{k}: {v}")