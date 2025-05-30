#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ##############################################################################
# # Python 全自動代碼維護修復工具 (auto_fix.py)                             #
# # 適配 GitHub Actions 環境                                                 #
# ##############################################################################
#
# 🎯 功能：
# 1.  遞迴掃描整個 repo 中的所有 .py 檔案 (從腳本執行位置開始)
# 2.  使用 autopep8 自動修正縮排與格式問題
# 3.  使用 autoflake 自動刪除未使用的 import 與變數
# 4.  使用 flake8 檢查語法風格錯誤
# 5.  使用 mypy 檢查靜態型別錯誤
# 6.  嘗試自動加入缺失的標準庫 import (使用 LibCST)
# 7.  所有操作與結果詳細記錄到 scan_report.txt
#
# 🛠️ 必要模組安裝指令 (通常在 GitHub Actions workflow 中處理)：
# # pip install autopep8 flake8 mypy autoflake libcst
#
# ℹ️ 注意：
# - 此腳本設計為在 GitHub Actions 環境中執行，預期相關工具已安裝。
# - 自動加入 import 功能目前僅限於 COMMON_STD_LIBS 中定義的標準庫。
# - 建議在版本控制系統下運行此腳本，以便追蹤和審核變更。
#
# ##############################################################################

import os
import sys
import subprocess
import pathlib
import shutil
import re
from datetime import datetime
from typing import List, Tuple, Dict, Set, Optional, Any

# 第三方模組，確保已安裝 (在 GHA 環境中應已處理)
try:
    import libcst as cst
except ImportError:
    print("錯誤：LibCST 模組未安裝。請確保 'pip install libcst' 已執行。")
    sys.exit(1)

# --- Configuration ---
# TARGET_FOLDER is now implicitly the current working directory where the script is run.
REPORT_FILE = "scan_report.txt" # Output report file name
# 常見標準庫列表，用於嘗試自動 import
COMMON_STD_LIBS = {
    "os", "sys", "json", "re", "collections", "datetime", "math", "itertools",
    "functools", "pathlib", "logging", "subprocess", "shutil", "tempfile",
    "time", "argparse", "csv", "glob", "hashlib", "heapq", "pickle", "random",
    "socket", "sqlite3", "statistics", "string", "tarfile", "threading",
    "uuid", "urllib", "xml", "zipfile", "typing", "decimal", "asyncio",
    "concurrent", "contextlib", "dataclasses", "enum"
}
# --- End Configuration ---

class ReportLogger:
    """統一記錄日誌到檔案和控制台"""
    def __init__(self, report_filepath: str):
        self.report_filepath = report_filepath
        self._initialize_report_file()

    def _initialize_report_file(self):
        # In GitHub Actions, the CWD is the root of the repository
        repo_root = os.getcwd()
        with open(self.report_filepath, "w", encoding="utf-8") as f:
            f.write(f"自動代碼掃描維護報告\n")
            f.write(f"執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"掃描根目錄：{repo_root}\n")
            f.write("=" * 80 + "\n\n")

    def log(self, message: str, console: bool = True):
        if console:
            print(message) # Output to GitHub Actions log
        with open(self.report_filepath, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def log_section(self, title: str):
        self.log("\n" + "-" * 20 + f" {title} " + "-" * (57 - len(title)), console=False)

    def log_subsection(self, title: str):
        self.log(f"\n--- {title} ---", console=False)

    def log_error(self, message: str):
        self.log(f"[錯誤] {message}")

    def log_warning(self, message: str):
        self.log(f"[警告] {message}")

    def log_info(self, message: str):
        self.log(f"[資訊] {message}")

    def log_fix(self, file_path: str, fix_description: str):
        self.log(f"  [自動修復] {file_path}: {fix_description}")

    def log_suggestion(self, file_path: str, line: Optional[int], code: Optional[str], issue: str, suggestion: str):
        location = f"{file_path}"
        if line:
            location += f":{line}"
        if code:
            location += f" ({code})"
        self.log(f"  [問題] {location}: {issue}")
        self.log(f"    💡 [建議] {suggestion}")

    def log_skipped(self, file_path: str, reason: str, operation: str):
        self.log(f"  [SKIPPED: {reason}] {file_path}: 未執行 {operation}")

# --- Global Report Logger ---
logger: Optional[ReportLogger] = None

# --- Helper Functions ---
def find_py_files(folder: str) -> List[pathlib.Path]:
    """遞迴尋找指定資料夾下的所有 .py 檔案"""
    # Exclude .venv, venv, .git, __pycache__ directories common in projects
    excluded_dirs = {".venv", "venv", ".git", "__pycache__", ".mypy_cache", ".pytest_cache", "build", "dist", "docs"}
    all_files = []
    root_path = pathlib.Path(folder)
    for item in root_path.rglob("*.py"):
        # Check if any part of the path is in excluded_dirs
        if not any(part in excluded_dirs for part in item.relative_to(root_path).parts):
            all_files.append(item)
    return sorted(all_files)


def run_command(command: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
    """執行外部指令並回傳回傳碼、標準輸出和標準錯誤"""
    global logger
    try:
        process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=cwd)
        return process.returncode, process.stdout, process.stderr
    except FileNotFoundError:
        logger.log_error(f"指令 '{command[0]}' 未找到。請確保它已安裝並在 PATH 中。")
        return -1, "", f"Command not found: {command[0]}"
    except Exception as e:
        logger.log_error(f"執行指令 '{' '.join(command)}' 時發生錯誤: {e}")
        return -1, "", str(e)

def get_file_content(file_path: pathlib.Path) -> str:
    global logger
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.log_error(f"讀取檔案 {file_path} 失敗: {e}")
        return ""
# --- Tooling Functions ---

def format_with_autopep8(file_path: pathlib.Path) -> bool:
    """使用 autopep8 修復格式錯誤"""
    global logger
    logger.log_subsection(f"autopep8 格式化: {file_path.name}")
    if not shutil.which("autopep8"):
        logger.log_warning(f"autopep8 未安裝，跳過格式化 {file_path}")
        return False

    original_content = get_file_content(file_path)
    if original_content == "" and not file_path.exists(): # Handle read error or empty file correctly
        return False

    command = ["autopep8", "--in-place", "--aggressive", "--aggressive", str(file_path)]
    returncode, stdout, stderr = run_command(command)

    if returncode != 0:
        logger.log_error(f"autopep8 執行失敗於 {file_path}。錯誤碼: {returncode}")
        if stderr: logger.log(f"  Stderr: {stderr.strip()}", console=False)
        return False
    
    modified_content = get_file_content(file_path)
    if modified_content == original_content:
        logger.log(f"  {file_path.name}: 無格式變更。", console=False)
        return False
    else:
        logger.log_fix(str(file_path), "使用 autopep8 自動格式化代碼。")
        return True

def remove_unused_with_autoflake(file_path: pathlib.Path) -> bool:
    """使用 autoflake 自動刪除未使用的 import 和變數"""
    global logger
    logger.log_subsection(f"autoflake 清理: {file_path.name}")
    if not shutil.which("autoflake"):
        logger.log_warning(f"autoflake 未安裝，跳過清理 {file_path}")
        return False

    original_content = get_file_content(file_path)
    if original_content == "" and not file_path.exists():
        return False
        
    command = [
        "autoflake",
        "--in-place",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        "--ignore-init-module-imports",
        str(file_path)
    ]
    returncode, stdout, stderr = run_command(command)

    if returncode != 0:
        logger.log_error(f"autoflake 執行失敗於 {file_path}。錯誤碼: {returncode}")
        # autoflake often prints to stderr for changes, so only log if it's an actual error context.
        # Here, a non-zero return code implies an error.
        if stderr: logger.log(f"  Stderr: {stderr.strip()}", console=False)
        return False

    modified_content = get_file_content(file_path)
    if modified_content == original_content:
        logger.log(f"  {file_path.name}: autoflake 未做任何變更。", console=False)
        return False
    else:
        fix_message = "使用 autoflake 清理未使用的 import/變數。"
        # Try to parse stderr for more specific messages if autoflake provides them for successful changes
        if stderr: # autoflake successful changes might also go to stderr
            changes = [line for line in stderr.strip().split('\n') if "removing" in line.lower()]
            if changes:
                for change in changes:
                    logger.log_fix(str(file_path), f"autoflake: {change.strip()}")
                # If specific changes were logged, we might not need the generic message,
                # but keeping it for overall action summary.
            else: # If no specific "removing" lines, use generic.
                 logger.log_fix(str(file_path), fix_message)
        else: # No stderr output, assume general success.
            logger.log_fix(str(file_path), fix_message)
        return True


def analyze_with_flake8(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    """使用 flake8 掃描語法錯誤和風格問題"""
    global logger
    logger.log_subsection(f"flake8 分析: {file_path.name}")
    if not shutil.which("flake8"):
        logger.log_warning(f"flake8 未安裝，跳過分析 {file_path}")
        return []

    command = ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s", str(file_path)]
    returncode, stdout, stderr = run_command(command)

    issues = []
    if stdout:
        for line in stdout.strip().split('\n'):
            if not line: continue
            # Flake8 output can sometimes be prefixed by path if run on multiple files,
            # but here we run per file, so path is somewhat redundant but good for consistency.
            # The format is: path:row:col:code:text
            match = re.match(r"(.+?):(\d+):(\d+):\s*([A-Z]\d{3,})\s*(.*)", line)
            if match:
                _path_str, line_num_str, col_num_str, code_str, msg_str = match.groups()
                try:
                    issue = {
                        "path": _path_str, # or str(file_path) for consistency if needed
                        "line": int(line_num_str),
                        "col": int(col_num_str),
                        "code": code_str.strip(),
                        "message": msg_str.strip()
                    }
                    issues.append(issue)
                    suggestion = f"請檢查代碼風格/語法問題：{issue['message']}"
                    if issue['code'] == 'F821':
                        suggestion = f"變數或模組 '{issue['message'].split(\"'")[1]}' 未定義。請檢查拼寫、是否已 import 或定義。"
                    elif issue['code'].startswith('F401'):
                        suggestion = f"模組 '{issue['message'].split(\"'")[1]}' 已 import 但未使用。autoflake 通常會處理此類問題。"
                    elif issue['code'] == 'E501':
                        suggestion = "行過長。請遵守專案的行長度限制 (autopep8 可能已處理部分)。"

                    logger.log_suggestion(str(file_path.relative_to(os.getcwd())), issue['line'], issue['code'], issue['message'], suggestion)
                except ValueError:
                    logger.log_warning(f"無法解析 flake8 輸出行: {line}")
            elif line.strip():
                 logger.log_info(f"  {str(file_path.relative_to(os.getcwd()))}: Flake8 原始輸出: {line.strip()}")

    if not issues and returncode == 0 :
        logger.log(f"  {file_path.name}: flake8 未發現問題。", console=False)
    
    if stderr:
        logger.log_warning(f"flake8 執行於 {file_path} 時產生 stderr: {stderr.strip()}")
    return issues


def analyze_with_mypy(file_path: pathlib.Path) -> List[Dict[str, Any]]:
    """使用 mypy 進行靜態型別檢查"""
    global logger
    logger.log_subsection(f"mypy 型別檢查: {file_path.name}")
    if not shutil.which("mypy"):
        logger.log_warning(f"mypy 未安裝，跳過型別檢查 {file_path}")
        return []

    command = ["mypy", str(file_path), "--show-column-numbers", "--no-error-summary", "--pretty", "--allow-redefinition"]
    returncode, stdout, stderr = run_command(command) # Mypy often has non-zero exit code for type errors

    issues = []
    # path:line:col: level: message [error-code]
    pattern = re.compile(r"^(.*?):(\d+):(?:(\d+):)?\s*(error|note|warning):\s*(.*?)(?:\s*\[(.*?)\])?$")

    if stdout:
        for line in stdout.strip().split('\n'):
            if not line or "Success: no issues found" in line: continue
            match = pattern.match(line)
            if match:
                p, l, c, level, msg, code = match.groups()
                try:
                    issue = {
                        "path": p, "line": int(l), "col": int(c) if c else None,
                        "level": level, "message": msg.strip(), "code": code.strip() if code else None
                    }
                    issues.append(issue)
                    suggestion = f"請檢查型別提示或相關邏輯：{issue['message']}"
                    if issue['code'] == 'import-untyped':
                        suggestion = f"模組 '{issue['message'].split(\"'")[1]}' 沒有型別提示。考慮為其添加 stubs 或使用 --ignore-missing-imports。"
                    elif issue['code'] == 'name-defined':
                         suggestion = f"名稱 '{issue['message'].split(\"'")[1]}' 未定義。請檢查拼寫、import 或定義。"
                    
                    logger.log_suggestion(str(file_path.relative_to(os.getcwd())), issue['line'], issue['code'], f"({issue['level']}) {issue['message']}", suggestion)
                except ValueError:
                    logger.log_warning(f"無法解析 mypy 輸出行: {line}")
            elif line.strip() and "Found " not in line and " errors in " not in line and " note: " not in line :
                 logger.log_info(f"  {str(file_path.relative_to(os.getcwd()))}: mypy 原始輸出: {line.strip()}")

    if not issues and ("Success: no issues found" in stdout or returncode == 0): # Mypy can return 0 if no issues
        logger.log(f"  {file_path.name}: mypy 未發現型別問題。", console=False)
    
    # Mypy error summary often goes to stderr, or other notes.
    if stderr:
        # Filter out common non-error messages from mypy stderr if any
        # (e.g. "Found X errors in Y files (checked Z source files)")
        filtered_stderr = "\n".join(l for l in stderr.strip().split('\n') 
                                    if not ("Found " in l and " errors in " in l) and 
                                       not ("checked " in l and " source files" in l) and
                                       not ("note:" in l.lower() and "suggestion:" in l.lower()) # some notes are useful
                                   )
        if filtered_stderr.strip(): # If anything substantial remains
            logger.log_warning(f"mypy 執行於 {file_path} 時產生 stderr: {filtered_stderr.strip()}")
    return issues
# --- LibCST Based Import Adder ---

class AddImportTransformer(cst.CSTTransformer):
    """
    A LibCST Transformer to add specified imports to a module if they don't already exist.
    It tries to place new imports after existing ones and adds an [AUTO-FIX] comment.
    """
    def __init__(self, modules_to_add: Set[str], file_path_for_logging: str):
        super().__init__()
        self.modules_to_add = modules_to_add
        self.existing_imports: Set[str] = set() # Tracks existing imported module names
        self._file_path_for_logging = file_path_for_logging # For logging from within transformer
        self.imports_actually_added_names: Set[str] = set() # Track what was really added

    def visit_Import(self, node: cst.Import) -> Optional[bool]:
        for alias in node.names:
            self.existing_imports.add(alias.name.value)
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> Optional[bool]:
        # This logic primarily helps avoid adding 'import X' if 'from X import Y' already exists.
        # It's a simplification; a truly robust check would be more complex.
        module_name_node = node.module
        if isinstance(module_name_node, cst.Name):
            self.existing_imports.add(module_name_node.value)
        elif isinstance(module_name_node, cst.Attribute): 
            # e.g., from collections.abc import ... -> 'collections'
            current = module_name_node
            while isinstance(current, cst.Attribute):
                current = current.value
            if isinstance(current, cst.Name):
                 self.existing_imports.add(current.value)
        return True

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        global logger
        imports_to_insert_cst_nodes = []
        
        for module_name in self.modules_to_add:
            if module_name not in self.existing_imports:
                import_statement_node = cst.Import(names=[cst.ImportAlias(name=cst.Name(module_name))])
                auto_fix_comment = cst.Comment(f" # [AUTO-FIX] Added by auto_fix.py")
                
                statement_line = cst.SimpleStatementLine(
                    body=[import_statement_node],
                    trailing_whitespace=cst.TrailingWhitespace(
                        whitespace=cst.SimpleWhitespace(" "), # Space before comment
                        comment=auto_fix_comment,
                        newline=cst.Newline() 
                    )
                )
                imports_to_insert_cst_nodes.append(statement_line)
                logger.log_fix(self._file_path_for_logging, f"規劃加入 import {module_name} # [AUTO-FIX]")
                self.imports_actually_added_names.add(module_name)

        if not imports_to_insert_cst_nodes:
            return updated_node

        new_body_list = list(updated_node.body)
        insert_idx = 0

        # 1. Skip potential docstring
        if new_body_list and isinstance(new_body_list[0], cst.SimpleStatementLine):
            first_expr = new_body_list[0].body[0]
            if isinstance(first_expr, cst.Expr) and isinstance(first_expr.value, (cst.SimpleString, cst.ConcatenatedString)):
                insert_idx = 1
        
        # 2. Skip __future__ imports
        for i in range(insert_idx, len(new_body_list)):
            stmt_line = new_body_list[i]
            if isinstance(stmt_line, cst.SimpleStatementLine) and stmt_line.body:
                first_stmt_in_line = stmt_line.body[0]
                if isinstance(first_stmt_in_line, cst.ImportFrom) and \
                   first_stmt_in_line.module and \
                   isinstance(first_stmt_in_line.module, cst.Name) and \
                   first_stmt_in_line.module.value == "__future__":
                    insert_idx = i + 1
                    continue
            break 
        
        # 3. Find the line after the last existing import to group new imports
        last_import_line_idx = -1
        for i in range(insert_idx, len(new_body_list)):
            stmt_line = new_body_list[i]
            if isinstance(stmt_line, cst.SimpleStatementLine) and stmt_line.body:
                first_stmt_in_line = stmt_line.body[0]
                if isinstance(first_stmt_in_line, (cst.Import, cst.ImportFrom)):
                    last_import_line_idx = i
        
        if last_import_line_idx != -1:
            insert_idx = last_import_line_idx + 1
            # Add a blank line before new imports if previous line wasn't an import and not empty.
            # And if the insertion point isn't the very start (after docstring/future).
            if insert_idx > 0 and insert_idx <= len(new_body_list): # Check bounds
                prev_line_node = new_body_list[insert_idx-1]
                # If the previous line (last import) doesn't have a blank line after it via its own newlines.
                # LibCST often handles this, but we can ensure.
                # If the last import doesn't have multiple newlines in its trailing_whitespace
                # and the new imports are not the first statements.
                if not (isinstance(prev_line_node, cst.EmptyLine) or \
                        (isinstance(prev_line_node, cst.SimpleStatementLine) and len(prev_line_node.leading_lines) > 0)):
                     # Check if the last import line has enough newlines after it.
                    if isinstance(prev_line_node, cst.SimpleStatementLine) and not prev_line_node.trailing_whitespace.newline.value.count('\n') > 1:
                        # Add an empty line if we are inserting after other code (not directly after another import block)
                        # This is complex; LibCST often does a good job by default.
                        # For simplicity, let's assume imports are somewhat grouped.
                        pass


        # Insert new import CST nodes
        final_body_list = new_body_list[:insert_idx] + imports_to_insert_cst_nodes + new_body_list[insert_idx:]
        
        # Ensure a blank line after the new import block if there's subsequent code (PEP 8)
        if imports_to_insert_cst_nodes and (insert_idx + len(imports_to_insert_cst_nodes)) < len(final_body_list):
            node_after_imports = final_body_list[insert_idx + len(imports_to_insert_cst_nodes)]
            if not isinstance(node_after_imports, cst.EmptyLine) and \
               not (isinstance(node_after_imports, cst.SimpleStatementLine) and len(node_after_imports.leading_lines) > 0):
                # Add an empty line if not already present
                # Check the last added import's trailing whitespace
                last_added_import_node = imports_to_insert_cst_nodes[-1]
                if isinstance(last_added_import_node, cst.SimpleStatementLine): # Should be
                    # If it doesn't already force two newlines (e.g. by ending with `\n\n`)
                    if last_added_import_node.trailing_whitespace.newline.value.count('\n') < 2:
                         final_body_list.insert(insert_idx + len(imports_to_insert_cst_nodes), cst.EmptyLine())
        
        return updated_node.with_changes(body=tuple(final_body_list))


def add_missing_standard_library_imports(file_path: pathlib.Path, flake8_issues: List[Dict[str, Any]]) -> bool:
    """
    Tries to add missing standard library imports based on flake8 F821 errors.
    Uses LibCST to modify the file, preserving formatting. Returns True if changes were made.
    """
    global logger
    logger.log_subsection(f"自動加入標準庫 import: {file_path.name}")

    undefined_names = set()
    for issue in flake8_issues:
        if issue["code"] == "F821": 
            match = re.search(r"undefined name '([^']*)'", issue["message"])
            if match:
                undefined_names.add(match.group(1))

    if not undefined_names:
        logger.log(f"  {file_path.name}: 未偵測到 F821 (undefined name) 錯誤，無需加入 import。", console=False)
        return False

    modules_to_attempt_add = undefined_names.intersection(COMMON_STD_LIBS)
    if not modules_to_attempt_add:
        logger.log(f"  {file_path.name}: 偵測到的 undefined names 不在通用標準庫列表中。", console=False)
        for name in undefined_names:
            logger.log(f"    - Undefined: '{name}' (非標準庫或未列出)", console=False)
        return False

    try:
        relative_file_path_str = str(file_path.relative_to(os.getcwd()))
        original_code = get_file_content(file_path)
        if original_code == "" and not file_path.exists(): return False

        module_cst = cst.parse_module(original_code)
        transformer = AddImportTransformer(modules_to_attempt_add, relative_file_path_str)
        
        modified_module_cst = module_cst.visit(transformer) # visit will call leave_Module
        
        if not transformer.imports_actually_added_names: # Check if transformer planned any additions
             logger.log(f"  {file_path.name}: LibCST 分析後未進行 import 添加 (可能已存在或無有效目標)。", console=False)
             return False

        modified_code = modified_module_cst.code
        
        if original_code == modified_code: # Should not happen if imports_actually_added_names is populated
            logger.log(f"  {file_path.name}: LibCST 未修改檔案 (程式碼無變化)。", console=False)
            return False # No actual change in code content
        else:
            file_path.write_text(modified_code, encoding="utf-8")
            # Logging of specific added imports is handled within AddImportTransformer
            logger.log(f"  {file_path.name}: 成功使用 LibCST 更新 imports。", console=False)
            return True # Code was changed

    except cst.ParserSyntaxError as e:
        logger.log_error(f"LibCST 解析檔案 {file_path} 失敗: {e}")
        logger.log_skipped(str(file_path.relative_to(os.getcwd())), "UNSAFE (LibCST ParserSyntaxError)", "自動加入 import")
        return False
    except Exception as e:
        logger.log_error(f"使用 LibCST 為 {file_path} 添加 import 時發生未知錯誤: {e}")
        logger.log_skipped(str(file_path.relative_to(os.getcwd())), f"UNSAFE (Exception: {type(e).__name__})", "自動加入 import")
        return False
# --- Main Orchestration ---
def main():
    global logger
    logger = ReportLogger(REPORT_FILE) # Report file in CWD
    logger.log("開始全自動代碼掃描與維護流程 (GitHub Actions)...", console=True)

    # In GitHub Actions, CWD is the root of the repository.
    scan_directory = "." 
    py_files = find_py_files(scan_directory)

    if not py_files:
        logger.log_warning(f"在 '{os.path.abspath(scan_directory)}' 中未找到任何 .py 檔案 (已排除常見虛擬環境目錄)。")
        logger.log("流程結束。")
        return

    logger.log(f"找到 {len(py_files)} 個 .py 檔案進行處理。\n")

    # Check for tool availability once at the start (should be handled by GHA setup)
    tools_to_check = {"autopep8": shutil.which("autopep8"),
                      "autoflake": shutil.which("autoflake"),
                      "flake8": shutil.which("flake8"),
                      "mypy": shutil.which("mypy")}
    
    all_tools_available = True
    for tool_name, path in tools_to_check.items():
        if not path:
            logger.log_warning(f"工具 '{tool_name}' 未安裝或不在 PATH 中。依賴此工具的功能將被跳過。")
            all_tools_available = False
    
    # LibCST is imported, so it's checked at startup.

    files_changed_by_script = False

    for file_path in py_files:
        relative_path_str = str(file_path.relative_to(os.getcwd()))
        logger.log_section(f"處理檔案: {relative_path_str}")
        
        changed_by_autopep8 = format_with_autopep8(file_path)
        if changed_by_autopep8: files_changed_by_script = True
        
        changed_by_autoflake = remove_unused_with_autoflake(file_path)
        if changed_by_autoflake: files_changed_by_script = True

        # First pass of flake8 to find undefined names for import addition
        pre_flake8_issues = []
        if tools_to_check["flake8"]:
            # Run flake8 quietly just to get data for import addition
            command = ["flake8", "--format=%(path)s:%(row)d:%(col)d:%(code)s:%(text)s", str(file_path)]
            _, stdout, _ = run_command(command)
            if stdout:
                for line in stdout.strip().split('\n'):
                    if not line: continue
                    match = re.match(r"(.+?):(\d+):(\d+):\s*([A-Z]\d{3,})\s*(.*)", line)
                    if match:
                        _p, l, _c, cde, msg = match.groups()
                        pre_flake8_issues.append({
                            "path": _p, "line": int(l), "code": cde.strip(), "message": msg.strip()
                        })

        # Attempt to add missing standard library imports using LibCST
        if tools_to_check["flake8"]: # Relies on flake8 output
            try:
                changed_by_libcst = add_missing_standard_library_imports(file_path, pre_flake8_issues)
                if changed_by_libcst: files_changed_by_script = True
            except Exception as e: 
                logger.log_error(f"嘗試為 {relative_path_str} 自動加入 import 時發生嚴重錯誤: {e}")
                logger.log_skipped(relative_path_str, f"CRITICAL ERROR ({type(e).__name__})", "自動加入 import")

        # Full Flake8 scan (on potentially modified file)
        if tools_to_check["flake8"]:
            analyze_with_flake8(file_path) # Logging happens inside

        # Mypy static type checking (on potentially modified file)
        if tools_to_check["mypy"]:
            analyze_with_mypy(file_path) # Logging happens inside

        logger.log(f"檔案 {relative_path_str} 處理完成。\n", console=False)

    logger.log_section("總結")
    logger.log(f"所有檔案處理完畢。詳細報告已儲存至: {os.path.abspath(REPORT_FILE)}")
    
    if files_changed_by_script:
        logger.log("腳本已對部分檔案進行了自動修復。在 GitHub Actions 中，這些變更可能會被自動提交。")
    else:
        logger.log("腳本未對任何檔案進行自動修復性變更 (僅格式化或清理可能發生，但程式碼實質內容無新增)。")
    
    print(f"\n✅ 掃描與維護完成。詳細報告已儲存至: {os.path.abspath(REPORT_FILE)}")
    # In GitHub Actions, a non-zero exit code might fail the step.
    # We are reporting, so exit 0 unless a critical error in the script itself occurs.

if __name__ == "__main__":
    main()
