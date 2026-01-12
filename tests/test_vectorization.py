import subprocess
import re
import sys
import os
import pytest
import glob

def find_extension_path():
    # Try to find the extension in the same place we found it earlier
    paths = glob.glob("env/lib/python*/site-packages/tvbk/tvbk_ext.abi3.so")
    if paths:
        return paths[0]
    # Fallback to current directory or build directory if needed
    paths = glob.glob("build/**/tvbk_ext.abi3.so", recursive=True)
    if paths:
        return paths[0]
    return None

def get_disassembly(lib_path):
    try:
        # -d: disassemble, -C: demangle C++ symbols
        result = subprocess.run(["objdump", "-d", "-C", lib_path], capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        pytest.fail(f"objdump failed: {e}")
    except FileNotFoundError:
        pytest.skip("objdump not found")

def parse_disassembly(disasm):
    # Simpler regexes
    dfun_re = re.compile(r"tvbk::(\w+)::dfun")
    heun_re = re.compile(r"heun_step<tvbk::(\w+),")

    # Regex for vector instructions (AVX/AVX2 use xmm/ymm/zmm)
    vector_reg_re = re.compile(r"%(ymm|zmm)\d+")
    
    scalar_math_re = re.compile(r"call.*<(exp|log|sin|cos|pow|tan|atan|acos|asin)(f|@plt|@GLIBC).*>", re.IGNORECASE)

    current_model = None
    model_stats = {}

    for line in disasm.splitlines():
        line = line.strip()
        
        # Check start of function
        # We look for < ... >: at end of line to be sure it's a symbol definition
        if line.endswith(">:") and ("tvbk::" in line):
            found_model = None
            m_dfun = dfun_re.search(line)
            if m_dfun:
                found_model = m_dfun.group(1)
            else:
                m_heun = heun_re.search(line)
                if m_heun:
                    found_model = m_heun.group(1)
            
            if found_model:
                current_model = found_model
                if current_model not in model_stats:
                    # Initialize stats for this model
                    model_stats[current_model] = {
                        "vector_ops": 0, 
                        "scalar_math_calls": [],
                        "vector_math_calls": []
                    }
                continue
            
            # If we enter another function that is NOT a model kernel, we stop tracking stats for previous model
            current_model = None
            
        # If we are inside a function
        if current_model:
            # Check for end of function handled by start of next function or just flow
            # objdump usually leaves empty lines. But simplest is just relying entirely on the function header check above.
            # We assume code belongs to current_model until new model starts or we run out of lines.
            
            # Check for vector registers
            if vector_reg_re.search(line):
                model_stats[current_model]["vector_ops"] += 1
            
            # Check for scalar math calls
            m_scalar = scalar_math_re.search(line)
            if m_scalar:
                # Store the full line for context
                model_stats[current_model]["scalar_math_calls"].append(line)

            # Check for vector math calls (GLIBC libmvec uses _ZGV, Intel SVML uses __svml)
            if "_ZGV" in line or "__svml" in line:
                model_stats[current_model]["vector_math_calls"].append(line)

    return model_stats

@pytest.mark.skipif(sys.platform != "linux", reason="Verification linux-only")
def test_vectorization():
    lib_path = find_extension_path()
    if not lib_path:
        pytest.skip("Could not find compiled extension tvbk_ext.abi3.so")
    
    print(f"Analyzing {lib_path}...")
    disasm = get_disassembly(lib_path)
    stats = parse_disassembly(disasm)
    
    if not stats:
        pytest.fail("No 'dfun' kernels found in disassembly. Check symbol demangling or compilation.")

    failed_models = []
    
    for model, stat in stats.items():
        n_vec_ops = stat['vector_ops']
        n_scalar_calls = len(stat['scalar_math_calls'])
        n_vec_calls = len(stat["vector_math_calls"])
        
        print(f"Model {model}: {n_vec_ops} vector insts, {n_scalar_calls} scalar math calls, {n_vec_calls} vector math calls")
        
        if n_vec_ops == 0:
            failed_models.append(f"{model} has NO vector instructions (YMM/ZMM).")
        
        # If we have mixed scalar and vector, it might be partial vectorization.
        # But if we have LOTS of scalar calls and NO vector math calls, that is suspicious for models with exp/log.
        if n_scalar_calls > 0 and n_vec_calls == 0:
             # Heuristic: if it has exp/log but no vector equivalent, it failed to vectorize math.
             calls = "\n\t".join(stat['scalar_math_calls'][:5])
             failed_models.append(f"{model} uses scalar math without vector math fallback:\n\t{calls}")

    if failed_models:
        pytest.fail("Vectorization verification failed:\n" + "\n".join(failed_models))

if __name__ == "__main__":
    test_vectorization()
