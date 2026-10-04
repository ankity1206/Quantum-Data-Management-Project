# comprehensive_stress_test.py
"""
Comprehensive stress test to find maximum qubits on HP Victus 16
- CPU: i5-13420H, 16GB RAM
- GPU: RTX 3060 4GB, CUDA enabled
"""

import time
import psutil
import tracemalloc
import gc
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# Import your out-of-core simulator
from out_of_core_simulator import OutOfCoreSimulator

try:
    import GPUtil
    HAS_GPU_UTIL = True
except ImportError:
    HAS_GPU_UTIL = False


class ComprehensiveStressTest:
    def __init__(self):
        self.results = {
            'cpu': [],
            'gpu': [],
            'out_of_core': []
        }
        self.start_time = time.time()
        
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage"""
        if HAS_GPU_UTIL:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed, gpus[0].memoryTotal
        return 0, 0
    
    def create_ghz_circuit(self, n_qubits):
        """Create GHZ circuit (extremely sparse)"""
        qc = QuantumCircuit(n_qubits)
        qc.h(0)
        for i in range(1, n_qubits):
            qc.cx(0, i)
        return qc
    
    def create_bv_circuit(self, n_qubits):
        """Bernstein-Vazirani circuit (sparse)"""
        import random
        qc = QuantumCircuit(n_qubits + 1, n_qubits)
        secret = random.getrandbits(n_qubits)
        qc.x(n_qubits)
        qc.h(range(n_qubits + 1))
        for i in range(n_qubits):
            if (secret >> i) & 1:
                qc.cx(i, n_qubits)
        qc.h(range(n_qubits))
        qc.measure(range(n_qubits), range(n_qubits))
        return qc
    
    def test_cpu(self, circuit, n_qubits):
        """Test CPU statevector simulation"""
        print(f"\n  🔴 CPU: {n_qubits} qubits...", end=" ", flush=True)
        
        try:
            tracemalloc.start()
            mem_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            start = time.time()
            state = Statevector.from_instruction(circuit)
            elapsed = time.time() - start
            
            mem_peak = tracemalloc.get_traced_memory()[1] / 1024 / 1024
            tracemalloc.stop()
            
            result = {
                'qubits': n_qubits,
                'time': elapsed,
                'memory_mb': mem_peak,
                'success': True,
                'method': 'CPU'
            }
            print(f"✓ {elapsed:.3f}s, {mem_peak:.1f}MB")
            return result
            
        except MemoryError as e:
            print(f"✗ OUT OF MEMORY")
            return {'qubits': n_qubits, 'success': False, 'error': 'OOM', 'method': 'CPU'}
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:40]}")
            return {'qubits': n_qubits, 'success': False, 'error': str(e), 'method': 'CPU'}
    
    def test_gpu(self, circuit, n_qubits):
        """Test GPU simulation using Aer"""
        print(f"\n  🟢 GPU: {n_qubits} qubits...", end=" ", flush=True)
        
        try:
            # Get GPU memory before
            gpu_mem_before, gpu_total = self.get_gpu_memory_usage()
            
            backend = AerSimulator(method='statevector', device='GPU')
            circuit_copy = circuit.copy()
            circuit_copy.save_statevector()
            
            # Warmup
            _ = backend.run(circuit_copy).result()
            
            start = time.time()
            result = backend.run(circuit_copy).result()
            elapsed = time.time() - start
            
            state = result.get_statevector()
            
            # Get GPU memory after
            gpu_mem_after, _ = self.get_gpu_memory_usage()
            
            result = {
                'qubits': n_qubits,
                'time': elapsed,
                'gpu_memory_mb': gpu_mem_after - gpu_mem_before,
                'success': True,
                'method': 'GPU'
            }
            print(f"✓ {elapsed:.3f}s, GPU: {result['gpu_memory_mb']:.0f}MB")
            return result
            
        except MemoryError as e:
            print(f"✗ GPU OUT OF MEMORY")
            return {'qubits': n_qubits, 'success': False, 'error': 'GPU OOM', 'method': 'GPU'}
        except Exception as e:
            error_msg = str(e)
            if "memory" in error_msg.lower():
                print(f"✗ GPU OOM")
                return {'qubits': n_qubits, 'success': False, 'error': 'GPU OOM', 'method': 'GPU'}
            print(f"✗ FAILED: {error_msg[:40]}")
            return {'qubits': n_qubits, 'success': False, 'error': error_msg, 'method': 'GPU'}
    
    def test_out_of_core(self, circuit, n_qubits, circuit_type='ghz'):
        """Test out-of-core RDBMS simulation"""
        print(f"\n  🔵 Out-of-Core: {n_qubits} qubits...", end=" ", flush=True)
        
        try:
            simulator = OutOfCoreSimulator(memory_limit_mb=500)
            simulator.connect()
            
            start = time.time()
            result = simulator.simulate_circuit(circuit, verbose=False)
            elapsed = time.time() - start
            
            # Get final stats
            simulator.cursor.execute("SELECT COUNT(*) FROM state")
            n_states = simulator.cursor.fetchone()[0]
            
            compression = (2**n_qubits) / n_states if n_states > 0 else 0
            
            result = {
                'qubits': n_qubits,
                'time': elapsed,
                'n_states': n_states,
                'compression': compression,
                'success': True,
                'method': 'Out-of-Core'
            }
            simulator.close()
            
            print(f"✓ {elapsed:.3f}s, {n_states:,} states, {compression:.0f}x compression")
            return result
            
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:40]}")
            return {'qubits': n_qubits, 'success': False, 'error': str(e), 'method': 'Out-of-Core'}
    
    def run_progressive_test(self, circuit_type='ghz', start_qubits=20, max_qubits=35, step=2):
        """
        Run progressive stress test increasing qubits until failure
        """
        print("=" * 80)
        print(f"COMPREHENSIVE STRESS TEST: {circuit_type.upper()} CIRCUIT")
        print(f"Hardware: i5-13420H, 16GB RAM, RTX 3060 4GB")
        print("=" * 80)
        
        for n in range(start_qubits, max_qubits + 1, step):
            print(f"\n{'='*50}")
            print(f"TESTING {n} QUBITS")
            print(f"{'='*50}")
            
            # Create circuit
            if circuit_type == 'ghz':
                circuit = self.create_ghz_circuit(n)
            else:
                circuit = self.create_bv_circuit(n)
            
            # Test CPU
            cpu_result = self.test_cpu(circuit, n)
            self.results['cpu'].append(cpu_result)
            
            # Test GPU (skip if CPU already OOM at this size)
            if cpu_result['success'] or n <= 26:  # GPU might still work
                gpu_result = self.test_gpu(circuit, n)
                self.results['gpu'].append(gpu_result)
            
            # Test Out-of-Core (always try)
            ooc_result = self.test_out_of_core(circuit, n, circuit_type)
            self.results['out_of_core'].append(ooc_result)
            
            # Memory cleanup
            gc.collect()
            time.sleep(1)
            
            # Stop if all three failed
            if not cpu_result['success'] and not ooc_result['success']:
                print(f"\n⚠️ All methods failed at {n} qubits. Stopping.")
                break
    
    def find_max_qubits(self):
        """Find maximum qubits for each method"""
        cpu_max = 0
        for r in self.results['cpu']:
            if r.get('success', False) and r['qubits'] > cpu_max:
                cpu_max = r['qubits']
        
        gpu_max = 0
        for r in self.results['gpu']:
            if r.get('success', False) and r['qubits'] > gpu_max:
                gpu_max = r['qubits']
        
        ooc_max = 0
        for r in self.results['out_of_core']:
            if r.get('success', False) and r['qubits'] > ooc_max:
                ooc_max = r['qubits']
        
        return cpu_max, gpu_max, ooc_max
    
    def print_summary(self):
        """Print final summary with recommendations"""
        elapsed = time.time() - self.start_time
        
        print("\n" + "=" * 80)
        print("STRESS TEST COMPLETE!")
        print(f"Total time: {elapsed/60:.1f} minutes")
        print("=" * 80)
        
        cpu_max, gpu_max, ooc_max = self.find_max_qubits()
        
        print("\n📊 MAXIMUM QUBITS ACHIEVED:")
        print("-" * 50)
        print(f"  💻 CPU (Statevector):     {cpu_max} qubits")
        print(f"  🎮 GPU (Aer):             {gpu_max} qubits")
        print(f"  💾 Out-of-Core (RDBMS):   {ooc_max}+ qubits")
        
        # Find where out-of-core beats GPU
        print("\n📊 WHERE OUT-OF-CORE EXCELS:")
        for r in self.results['out_of_core']:
            if r.get('success', False) and r.get('compression', 0) > 1000:
                print(f"  • {r['qubits']} qubits: {r['compression']:.0f}x compression ({r['time']:.3f}s)")
        
        # Performance comparison at common qubits
        print("\n📊 PERFORMANCE COMPARISON:")
        print("-" * 70)
        print(f"{'Qubits':<8} {'CPU (s)':<12} {'GPU (s)':<12} {'Out-of-Core (s)':<18} {'Winner':<10}")
        print("-" * 70)
        
        for n in range(20, min(cpu_max, gpu_max, 28) + 1, 2):
            cpu_time = None
            gpu_time = None
            ooc_time = None
            
            for r in self.results['cpu']:
                if r.get('success') and r['qubits'] == n:
                    cpu_time = r['time']
            for r in self.results['gpu']:
                if r.get('success') and r['qubits'] == n:
                    gpu_time = r['time']
            for r in self.results['out_of_core']:
                if r.get('success') and r['qubits'] == n:
                    ooc_time = r['time']
            
            times = []
            if cpu_time:
                times.append(('CPU', cpu_time))
            if gpu_time:
                times.append(('GPU', gpu_time))
            if ooc_time:
                times.append(('Out-of-Core', ooc_time))
            
            if times:
                winner = min(times, key=lambda x: x[1])
                winner_name = winner[0]
                
                cpu_str = f"{cpu_time:.3f}" if cpu_time else "N/A"
                gpu_str = f"{gpu_time:.3f}" if gpu_time else "N/A"
                ooc_str = f"{ooc_time:.3f}" if ooc_time else "N/A"
                
                print(f"{n:<8} {cpu_str:<12} {gpu_str:<12} {ooc_str:<18} {winner_name:<10}")
        
        # Final recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS FOR YOUR HP VICTUS 16")
        print("=" * 80)
        print(f"""
    ┌─────────────────────────────────────────────────────────────────┐
    │                    OPTIMAL USAGE GUIDELINES                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  📌 CIRCUITS < {cpu_max} qubits:                                  │
    │     → Use CPU (simplest, no GPU overhead)                        │
    │                                                                  │
    │  📌 CIRCUITS {cpu_max}-{gpu_max} qubits:                                 │
    │     → Use GPU (faster than CPU)                                  │
    │                                                                  │
    │  📌 CIRCUITS > {gpu_max} qubits OR SPARSE CIRCUITS:                       │
    │     → Use Out-of-Core RDBMS (only option!)                       │
    │     → Achieves {self.results['out_of_core'][-1].get('compression', 0):.0f}x compression          │
    │                                                                  │
    │  🏆 KEY TAKEAWAY:                                                │
    │     RDBMS out-of-core simulation enables quantum circuit         │
    │     analysis BEYOND your hardware's physical memory limits.      │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘
        """)
    
    def save_results(self):
        """Save results to CSV"""
        import pandas as pd
        
        cpu_df = pd.DataFrame([r for r in self.results['cpu'] if r.get('success', False)])
        gpu_df = pd.DataFrame([r for r in self.results['gpu'] if r.get('success', False)])
        ooc_df = pd.DataFrame([r for r in self.results['out_of_core'] if r.get('success', False)])
        
        filename = f'stress_test_results_{time.strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        with pd.ExcelWriter(filename) as writer:
            if not cpu_df.empty:
                cpu_df.to_excel(writer, sheet_name='CPU', index=False)
            if not gpu_df.empty:
                gpu_df.to_excel(writer, sheet_name='GPU', index=False)
            if not ooc_df.empty:
                ooc_df.to_excel(writer, sheet_name='OutOfCore', index=False)
        
        print(f"\n📁 Results saved to '{filename}'")


def run_aggressive_test():
    """
    Aggressive test pushing to 30+ qubits
    """
    print("=" * 80)
    print("AGGRESSIVE LIMIT TEST: PUSHING TO 30+ QUBITS")
    print("=" * 80)
    
    # Test single high-qubit circuit
    n = 30
    print(f"\n🔥 Testing GHZ with {n} qubits...")
    print(f"   Dense state would require {2**n * 16 / (1024**3):.1f} GB of RAM")
    print(f"   Your system has 16GB RAM - this would CRASH dense simulation!")
    print(f"   But GHZ is sparse - only 2 non-zero amplitudes...\n")
    
    simulator = OutOfCoreSimulator(memory_limit_mb=500)
    simulator.connect()
    
    # Create GHZ circuit
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(1, n):
        qc.cx(0, i)
    
    start = time.time()
    result = simulator.simulate_circuit(qc, verbose=False)
    elapsed = time.time() - start
    
    print(f"📊 RESULT:")
    print(f"   Status: SUCCESS!")
    print(f"   Time: {elapsed:.3f}s")
    print(f"   States stored: {result.get('n_states_stored', 'N/A')}")
    print(f"   DB size: {result.get('db_size_bytes', 0)} bytes")
    print(f"   Compression: {result.get('compression_ratio', 0):.0f}x")
    
    simulator.close()
    
    print("\n" + "=" * 80)
    print("✅ VERIFICATION: Out-of-core RDBMS simulated")
    print(f"   {n} qubits that would CRASH CPU/GPU simulators!")
    print("=" * 80)


if __name__ == "__main__":
    print("=" * 80)
    print("HP VICTUS 16 STRESS TEST SUITE")
    print("Hardware: i5-13420H, 16GB RAM, RTX 3060 4GB")
    print("=" * 80)
    
    print("\nSelect test mode:")
    print("  1. Quick aggressive test (30 qubits GHZ - should work)")
    print("  2. Progressive test (20-32 qubits, finds exact limits)")
    print("  3. Full progressive test with BV circuit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        run_aggressive_test()
    elif choice == '2':
        tester = ComprehensiveStressTest()
        tester.run_progressive_test(circuit_type='ghz', start_qubits=20, max_qubits=32, step=2)
        tester.print_summary()
        tester.save_results()
    elif choice == '3':
        tester = ComprehensiveStressTest()
        tester.run_progressive_test(circuit_type='bv', start_qubits=20, max_qubits=30, step=2)
        tester.print_summary()
        tester.save_results()
    else:
        print("Invalid choice")
