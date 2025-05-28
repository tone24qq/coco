import os
import importlib
import inspect
import logging
import numpy as np
import sys
import hashlib # For duplicate file content detection
from typing import Any, Callable, Dict, List, Optional, Union, Type

# --- Logger Setup (保持詳細) ---
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Conventions for Detection (同極致版) ---
class AutoRegisteredScoringModuleBase:
    # ... (與極致版相同)
    def __init__(self, grid: np.ndarray, request_id: Optional[str] = None):
        self.grid = grid
        self.request_id = request_id
        logger.debug(f"{self.__class__.__name__} initialized with grid shape {grid.shape if isinstance(grid, np.ndarray) else 'N/A'}")

    def get_output(self) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} must implement the 'get_output' method.")

def is_ultimate_scoring_class(obj: Any, ) -> bool:
    # ... (與極致版相同，可能微調日誌)
    if not inspect.isclass(obj):
        return False
    # Strategy 1: Base class
    base_class_to_check: Optional[Type] = AutoRegisteredScoringModuleBase # Example
    marker_attribute_name: str = "_is_autoregister_module_via_decorator"
    fallback_name_keywords: List[str] = ['TensorOps', 'Module']
    fallback_name_prefixes: List[str] = ['Puzzle']
    # ... (rest of the logic from ultimate version) ...
    # (Add more logging if needed for clarity on why a class is chosen or not)
    class_name = obj.__name__ # Added for clarity
    try:
        if base_class_to_check is not None and issubclass(obj, base_class_to_check) and obj is not base_class_to_check:
            logger.debug(f"Class '{class_name}' identified by base class '{base_class_to_check.__name__}'.")
            return True
    except TypeError:
        pass
    if hasattr(obj, marker_attribute_name) and getattr(obj, marker_attribute_name) is True:
        logger.debug(f"Class '{class_name}' identified by marker attribute '{marker_attribute_name}'.")
        return True
    if any(keyword in class_name for keyword in fallback_name_keywords):
        logger.debug(f"Class '{class_name}' identified by fallback name keyword: {next(k for k in fallback_name_keywords if k in class_name)}.")
        return True
    if any(class_name.startswith(prefix) for prefix in fallback_name_prefixes):
        logger.debug(f"Class '{class_name}' identified by fallback name prefix: {next(p for p in fallback_name_prefixes if class_name.startswith(p))}.")
        return True
    return False


def create_ultimate_module_wrapper(cls_obj: type, ) -> Callable:
    # ... (與極致版相同，確保輸出類型轉換 np.array(processed_array) 包含在內)
    # ...
    #            if not isinstance(processed_array, np.ndarray):
    #                 logger.warning(f"Module {cls_obj.__name__} (request_id: {request_id}) "
    #                                f"output type is {type(processed_array)}, expected np.ndarray. Attempting conversion.")
    #                 try:
    #                     processed_array = np.array(processed_array) # Ensure this conversion attempt is robust
    #                     if not isinstance(processed_array, np.ndarray): # Check after conversion
    #                         raise ValueError("Conversion to np.ndarray failed or resulted in non-array.")
    #                     logger.info(f"Successfully converted output of {cls_obj.__name__} to np.ndarray.")
    #                 except Exception as conv_e:
    #                     logger.error(f"Failed to convert output of {cls_obj.__name__} to np.ndarray: {conv_e}. Returning zeros.", exc_info=True)
    #                     return np.zeros_like(grid)
    # ... (rest of the wrapper from ultimate version)
    def wrapper(grid: np.ndarray, request_id: Optional[str] = None) -> np.ndarray:
        logger.debug(f"Wrapper invoked for class {cls_obj.__name__} with request_id: {request_id}, grid_shape: {grid.shape if isinstance(grid, np.ndarray) else 'N/A'}")
        try:
            # ... (Instantiation logic as in ultimate version) ...
            module_instance = cls_obj(grid) # Simplified for brevity, use ultimate version's logic

            processed_array = None
            output_method_candidates: List[str] = [
                'get_output', 'get_result', 'get_processed_data', 
                'get_copy', 'process', 'run', 'transform', 'score'
            ]
            output_method_found = False
            # ... (Method finding logic as in ultimate version) ...
            for method_name in output_method_candidates:
                if hasattr(module_instance, method_name) and callable(getattr(module_instance, method_name)):
                    logger.debug(f"Attempting to call '{method_name}' on instance of {cls_obj.__name__}")
                    processed_array = getattr(module_instance, method_name)() # Simplified call
                    if processed_array is not None:
                        output_method_found = True
                        logger.info(f"Successfully retrieved output from '{method_name}' for {cls_obj.__name__}.")
                        break
            
            if not output_method_found: # Fallbacks for attributes
                if hasattr(module_instance, 'tensor') and isinstance(module_instance.tensor, np.ndarray):
                    processed_array = module_instance.tensor
                # ... other fallbacks ...
                else:
                    logger.warning(f"Could not find a suitable output method/attribute for {cls_obj.__name__}. Returning zeros.")
                    return np.zeros_like(grid)

            if not isinstance(processed_array, np.ndarray):
                logger.warning(f"Output of {cls_obj.__name__} is {type(processed_array)}, expected np.ndarray. Attempting conversion.")
                try:
                    # This is a key "auto-correction" for output type
                    candidate_array = np.array(processed_array)
                    # Check if conversion yielded a 0-dimensional array from a scalar, which might be unintended
                    if candidate_array.ndim == 0 and not isinstance(processed_array, (list, tuple, np.ndarray)): # e.g. converted a single number
                         if grid.ndim > 0 : # If grid is an array, scalar output is unlikely desired
                            logger.warning(f"Conversion of scalar output from {cls_obj.__name__} resulted in 0-dim array. This might be unintended. Returning zeros.")
                            return np.zeros_like(grid)
                    processed_array = candidate_array
                    if not isinstance(processed_array, np.ndarray): # Final check
                        raise ValueError("Conversion did not result in a NumPy ndarray.")
                    logger.info(f"Successfully converted output of {cls_obj.__name__} to np.ndarray shape {processed_array.shape}.")
                except Exception as e_conv:
                    logger.error(f"Failed to convert output of {cls_obj.__name__} to np.ndarray: {e_conv}. Returning zeros.")
                    return np.zeros_like(grid)
            
            # ... (Normalization and return logic as in ultimate version) ...
            apply_normalization = True # Example
            normalization_epsilon = 1e-8 # Example
            if apply_normalization and isinstance(processed_array, np.ndarray) and processed_array.size > 0:
                min_val = np.min(processed_array)
                peak_to_peak = np.ptp(processed_array)
                if peak_to_peak < normalization_epsilon:
                    processed_array = np.zeros_like(processed_array) 
                else:
                    processed_array = (processed_array - min_val) / peak_to_peak
            return processed_array

        except Exception as e:
            logger.error(f"Critical error executing {cls_obj.__name__}: {e}", exc_info=True)
            return np.zeros_like(grid)
    
    wrapper.__name__ = f"{cls_obj.__name__}_current_limit_wrapper"
    wrapper.__doc__ = f"Current Limit wrapper for {cls_obj.__name__}.\nOriginal doc: {cls_obj.__doc__}"
    return wrapper


def _get_file_hash(filepath: str) -> str:
    """Computes MD5 hash of a file's content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except IOError as e:
        logger.warning(f"Could not read file {filepath} for hashing: {e}")
        return "" # Return empty hash on error

def _ensure_package_path(module_path_segments: List[str], base_scan_path: str, auto_create_init_py: bool):
    """
    Ensures __init__.py exists in package directories.
    module_path_segments: e.g., ['package', 'subpackage'] for 'package.subpackage.module'
    base_scan_path: The root directory from scan_paths where this structure is found.
    """
    if not auto_create_init_py:
        return

    current_path = base_scan_path
    for segment in module_path_segments:
        current_path = os.path.join(current_path, segment)
        if os.path.isdir(current_path):
            init_py_path = os.path.join(current_path, "__init__.py")
            if not os.path.exists(init_py_path):
                try:
                    with open(init_py_path, "w", encoding="utf-8") as f:
                        f.write("# Automatically created by auto-registration script\n")
                    logger.info(f"Successfully created missing __init__.py at: {init_py_path} (auto_create_init_py=True)")
                except IOError as e:
                    logger.error(f"Failed to create __init__.py at {init_py_path}: {e}")
        else: # Should not happen if module path is valid up to the module file
            logger.warning(f"Expected directory {current_path} not found while ensuring package path.")
            break


# --- "Current Achievable Limit" Auto Module Registration ---
def current_limit_auto_register(
    target_register_dict: Dict[str, Callable],
    scan_paths: List[str] = ['.'],
    auto_create_init_py: bool = False, # New parameter
    # ... (other parameters from ultimate_auto_register_modules)
    recursive_scan: bool = True,
    module_skip_patterns: Optional[List[str]] = None,
    dir_skip_patterns: Optional[List[str]] = None,
    file_skip_patterns: Optional[List[str]] = None,
    class_identifier_func: Callable[[Any], bool] = is_ultimate_scoring_class,
    default_test_grid_factory: Callable[[], np.ndarray] = lambda: np.ones((5, 5), dtype=float),
    naming_convention_func: Callable[[str, str, str], str] = 
        lambda path, mod_name, cls_name: f"LIMIT_{mod_name.upper()}_{cls_name.upper()}_FN",
    force_override: bool = False,
    enable_testing: bool = True,
    apply_normalization_to_wrappers: bool = True

) -> List[str]:
    # ... (Default skip patterns setup as in ultimate version) ...
    _dir_skip_patterns = dir_skip_patterns or ['.git', '.venv', '__pycache__', 'tests', 'docs', 'examples']
    _file_skip_patterns = file_skip_patterns or ['setup.py', 'conftest.py', '__init__.py'] # __init__.py handled separately by auto_create
    _module_skip_patterns = module_skip_patterns or ['auto_module_register', 'brain', 'main', 'analyzer']

    processed_logs = []
    modules_with_classes_registered = 0
    
    python_files_to_process = set()
    scanned_file_hashes: Dict[str, List[str]] = {} # For duplicate content detection

    for start_path_orig in scan_paths:
        # ... (File scanning logic from ultimate version, with hashing) ...
        # Ensure start_path is absolute for consistent path ops
        abs_start_path = os.path.abspath(start_path_orig)
        if not os.path.exists(abs_start_path):
            logger.warning(f"Scan path '{abs_start_path}' does not exist. Skipping.")
            continue
        
        # Simplified scanning logic here, refer to ultimate for full detail
        if os.path.isdir(abs_start_path):
            for root, dirs, files in os.walk(abs_start_path, topdown=True):
                dirs[:] = [d for d in dirs if d not in _dir_skip_patterns and not d.startswith('.')]
                for fname in files:
                    if fname.endswith('.py'): # Basic filter
                        # More advanced filtering based on skip patterns
                        module_basename = os.path.splitext(fname)[0]
                        if fname in _file_skip_patterns or module_basename in _module_skip_patterns or fname.startswith('.'):
                            logger.debug(f"Skipping file by pattern: {os.path.join(root, fname)}")
                            continue
                        
                        filepath = os.path.join(root, fname)
                        python_files_to_process.add(filepath)
                        
                        # Duplicate content check
                        file_hash = _get_file_hash(filepath)
                        if file_hash: # If hashing was successful
                            if file_hash in scanned_file_hashes:
                                msg = (f"DUPLICATE CONTENT: File '{filepath}' has identical content to "
                                       f"'{scanned_file_hashes[file_hash][0]}'. Hash: {file_hash[:8]}")
                                logger.warning(msg)
                                processed_logs.append(msg)
                                scanned_file_hashes[file_hash].append(filepath)
                            else:
                                scanned_file_hashes[file_hash] = [filepath]
        elif os.path.isfile(abs_start_path) and abs_start_path.endswith('.py'): # Single file scan
            python_files_to_process.add(abs_start_path)
            # Also hash single files if needed for broader duplicate detection
            file_hash = _get_file_hash(abs_start_path)
            if file_hash:
                if file_hash in scanned_file_hashes:
                    scanned_file_hashes[file_hash].append(abs_start_path)
                else:
                    scanned_file_hashes[file_hash] = [abs_start_path]


    logger.info(f"Identified {len(python_files_to_process)} Python files for potential registration.")
    # ... (sys.path management as in ultimate version) ...
    original_sys_path = list(sys.path) # Make a copy

    for filepath in python_files_to_process:
        module_dir = os.path.dirname(filepath)
        module_name_from_file = os.path.splitext(os.path.basename(filepath))[0]

        # Determine the "importable" module name (e.g., package.subpackage.module)
        # This requires finding which scan_path is the root for this filepath
        importable_module_name = module_name_from_file
        relative_path_from_a_scan_root = ""
        current_scan_root_for_module = ""

        for sp_orig in scan_paths:
            sp = os.path.abspath(sp_orig)
            if filepath.startswith(sp) and os.path.isdir(sp): # File is under this scan path dir
                relative_path = os.path.relpath(filepath, sp)
                relative_path_from_a_scan_root = os.path.splitext(relative_path)[0]
                path_segments = relative_path_from_a_scan_root.split(os.sep)
                importable_module_name = ".".join(path_segments)
                current_scan_root_for_module = sp
                logger.debug(f"Determined importable name '{importable_module_name}' for file '{filepath}' relative to scan root '{sp}'")
                break
        
        path_added_to_sys = False
        # Ensure the *root* of the package structure (the scan_path containing it) is in sys.path
        # Example: if scan_paths=['projects/my_project'], and module is 'projects/my_project/pkg/mod.py'
        # then 'projects/my_project' should be in sys.path, and we import 'pkg.mod'
        # The ultimate version's logic adds module_dir, which is fine for simple cases, but for packages,
        # the Python path should point to the directory *containing* the top-level package.
        
        # Simplified sys.path logic for now: Add the scan_root if it's not there.
        # A more robust solution considers Python's existing sys.path and import rules.
        if current_scan_root_for_module and current_scan_root_for_module not in sys.path:
            sys.path.insert(0, current_scan_root_for_module)
            path_added_to_sys = True
            logger.debug(f"Temporarily added scan root '{current_scan_root_for_module}' to sys.path for importing '{importable_module_name}'")
        elif not current_scan_root_for_module and module_dir not in sys.path : # Fallback for non-package like structure or direct file scan
             sys.path.insert(0, module_dir)
             path_added_to_sys = True
             logger.debug(f"Temporarily added module directory '{module_dir}' to sys.path for importing '{importable_module_name}'")


        # Auto-create __init__.py if needed for package imports
        if auto_create_init_py and "." in importable_module_name and current_scan_root_for_module:
            package_segments = importable_module_name.split('.')[:-1] # Get package parts, e.g., ['pkg', 'sub']
            if package_segments:
                _ensure_package_path(package_segments, current_scan_root_for_module, auto_create_init_py)
        
        try:
            logger.info(f"Attempting to import module: '{importable_module_name}' (from file: {filepath})")
            module = importlib.import_module(importable_module_name)
            module = importlib.reload(module) # Reload for development
            # ... (Rest of the module processing, class identification, wrapping, testing from ultimate version) ...
            msg = f"Successfully imported module: '{importable_module_name}' from '{filepath}'"
            logger.info(msg)
            processed_logs.append(msg)
            
            found_classes_in_this_module = 0
            for member_name, member_obj in inspect.getmembers(module):
                if inspect.isclass(member_obj) and member_obj.__module__ != module.__name__:
                    continue # Skip imported classes

                if class_identifier_func(member_obj):
                    found_classes_in_this_module +=1
                    registration_key = naming_convention_func(filepath, importable_module_name, member_name)
                    
                    if registration_key in target_register_dict and not force_override:
                        # ... (log skip) ...
                        continue

                    wrapper_func = create_ultimate_module_wrapper( # Using the ultimate wrapper
                        member_obj, 
                    
                    )
                    target_register_dict[registration_key] = wrapper_func
                    msg = f"Registered: '{registration_key}' (from {importable_module_name}.{member_name})."
                    logger.info(msg)
                    processed_logs.append(msg)

                    if enable_testing:
                        # ... (testing logic) ...
                        try:
                            test_grid = default_test_grid_factory()
                            output = wrapper_func(np.copy(test_grid), request_id=f"AUTOTEST_{registration_key}")
                            if isinstance(output, np.ndarray):
                                logger.info(f"Test of '{registration_key}' OK. Output shape: {output.shape}")
                            else:
                                logger.warning(f"Test of '{registration_key}' output type: {type(output)}.")
                        except Exception as te:
                            logger.error(f"Test of '{registration_key}' FAILED: {te}")


            if found_classes_in_this_module > 0:
                modules_with_classes_registered +=1
            else:
                logger.debug(f"No scorable classes found in '{importable_module_name}'.")


        except ModuleNotFoundError as mnfe:
            # This is where __init__.py creation might have helped or other path issues.
            msg = (f"Failed to import module '{importable_module_name}' from '{filepath}': {mnfe}. "
                   "Check for missing __init__.py in package directories (if applicable and auto_create_init_py=False), "
                   "or ensure the correct scan_paths are provided. Python sys.path includes: " + ";".join(sys.path[:5]) + "...")
            logger.error(msg, exc_info=True) # exc_info=True provides more details on ModuleNotFoundError
            processed_logs.append(msg)

        except SyntaxError as se:
            msg = f"Syntax error in module '{importable_module_name}' ({filepath}): {se}. This module cannot be loaded or fixed automatically."
            logger.error(msg, exc_info=True)
            processed_logs.append(msg)
            # Suggestion for user
            processed_logs.append(f"MANUAL ACTION REQUIRED: Please check and fix the syntax in '{filepath}'.")


        except Exception as e:
            msg = f"Unexpected error processing module '{importable_module_name}' ({filepath}): {e}"
            logger.error(msg, exc_info=True)
            processed_logs.append(msg)
        finally:
            if path_added_to_sys:
                # Restore sys.path carefully
                if current_scan_root_for_module and sys.path[0] == current_scan_root_for_module:
                    sys.path.pop(0)
                    logger.debug(f"Removed scan root '{current_scan_root_for_module}' from sys.path.")
                elif not current_scan_root_for_module and module_dir == sys.path[0]:
                    sys.path.pop(0)
                    logger.debug(f"Removed module directory '{module_dir}' from sys.path.")
                # Else: path manipulation might have been complex, log a warning or reconsider strategy

    # ... (Fallback for non-callable entries - "Dummy補件" - as in ultimate version) ...
    logger.info(f"Current Limit auto-registration completed. Registered classes from {modules_with_classes_registered} modules.")
    return processed_logs

# --- Example Usage (Similar to ultimate, adjust parameters like auto_create_init_py) ---
if __name__ == '__main__':
    print(">>>>已進入簡化版 <<<<<")
    from typing import Dict, Callable
    import numpy as np

    class DummyBrain:
        REGISTERED_MODULES_BRAIN: Dict[str, Callable] = {}

    brain = DummyBrain()

    registration_logs = current_limit_auto_register(
        target_register_dict=brain.REGISTERED_MODULES_BRAIN,
        scan_paths=['.'],  # <<<<<< 這一行要用 '.'，千萬不能再寫 project_root
        auto_create_init_py=True,
        recursive_scan=True,
        force_override=True,
        enable_testing=True
    )

    print("\n=== 註冊結果 ===")
    for log in registration_logs:
        print(log)
    print("\n=== 註冊到的 modules ===")
    for name in brain.REGISTERED_MODULES_BRAIN:
        print(name)


    logger.info("--- Starting 'Current Achievable Limit' Auto Module Registration Example ---")
    
    registration_logs = current_limit_auto_register(
        target_register_dict=brain.REGISTERED_MODULES_BRAIN,
        scan_paths=['.'], # Scan the project root
        auto_create_init_py=True,  # <<<<<< Enable this new feature
        recursive_scan=True,
        force_override=True,
        enable_testing=True
        # class_identifier_func can be further customized if needed
    )
    
    logger.info("\n--- Registered Modules ---")
    if brain.REGISTERED_MODULES_BRAIN:
        for name, func in brain.REGISTERED_MODULES_BRAIN.items():
            logger.info(f"Registered: {name}")
            if name.endswith("LISTRETURNERMODULE_FN"): # Test the list returner specifically
                test_grid = np.array([[1,2],[3,4]])
                output = func(test_grid)
                logger.info(f"Output of {name} (should be ndarray due to coercion): type={type(output)}, shape={output.shape if isinstance(output,np.ndarray) else 'N/A'}")
                assert isinstance(output, np.ndarray)
    else:
        logger.info("No modules registered.")
    
    # Check if __init__.py was created
    expected_init_path = os.path.join(modules_pkg_dir, "__init__.py")
    if os.path.exists(expected_init_path):
        logger.info(f"SUCCESS: __init__.py was automatically created at '{expected_init_path}'")
    else:
        logger.error(f"FAILURE: __init__.py was NOT created at '{expected_init_path}' (check logs for errors if auto_create_init_py was True)")

    # import shutil
    # shutil.rmtree(project_root, ignore_errors=True)