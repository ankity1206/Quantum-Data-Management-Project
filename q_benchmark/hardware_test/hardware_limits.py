# hardware_limits.py
"""
Calculate theoretical simulation limits for your hardware
"""

import psutil
import GPUtil

def analyze_hardware_limits():
    """Calculate maximum qubits for each simulation method"""
    
    # System memory
    total_ram_gb = psutil.virtual_memory().total / (1024**3)
    available_ram_gb = psutil.virtual_memory().available / (1024**3)
    
    # GPU memory
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_mem_gb = gpus[0].memoryTotal / 1024
        else:
            gpu_mem_gb = 0
    except:
        gpu_mem_gb = 0
    
    print("=" * 80)
    print("HARDWARE CAPABILITY ANALYSIS")
    print("=" * 80)
    print(f"\nSystem RAM: {total_ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available")
    print(f"GPU VRAM: {gpu_mem_gb:.1f} GB")
    
    # Dense state vector (CPU)
    print("\n📊 DENSE STATE VECTOR (CPU) LIMITS:")
    for n in range(20, 35):
        memory_mb = (2**n * 16) / (1024 * 1024)
        if memory_mb < available_ram_gb * 1024:
            print(f"  {n} qubits: {memory_mb:.1f} MB")
        else:
            print(f"  {n} qubits: {memory_mb:.1f} MB ❌ EXCEEDS RAM")
            break
    
    # Dense state vector (GPU)
    if gpu_mem_gb > 0:
        print("\n📊 DENSE STATE VECTOR (GPU) LIMITS:")
        for n in range(20, 35):
            memory_mb = (2**n * 16) / (1024 * 1024)
            if memory_mb < gpu_mem_gb * 1024:
                print(f"  {n} qubits: {memory_mb:.1f} MB")
            else:
                print(f"  {n} qubits: {memory_mb:.1f} MB ❌ EXCEEDS GPU VRAM")
                break
    
    # Out-of-core RDBMS (theoretical - limited by disk)
    print("\n📊 OUT-OF-CORE RDBMS LIMITS:")
    print("  Limited by disk space, not RAM")
    print("  Theoretical max: ~40-50 qubits with compression")
    print("  Practical limit for sparse circuits: 30-35 qubits")

if __name__ == "__main__":
    analyze_hardware_limits()
