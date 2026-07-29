# -*- Python -*-

import os

import lit.formats
from lit.llvm import llvm_config

config.name = "NPUIR-Interpreter"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.excludes = ["Inputs", "CMakeLists.txt", "README.md",
                   "lit.cfg.py", "lit.site.cfg.py", "Precision"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.npuir_interpreter_obj_root, "test")

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.npuir_interpreter_tools_dir,
                             append_path=True)
llvm_config.add_tool_substitutions(
    ["npuir-interp", "FileCheck", "count", "not"],
    [config.npuir_interpreter_tools_dir, config.llvm_tools_dir])

config.environment["FILECHECK_OPTS"] = (
    "-enable-var-scope --allow-unused-prefixes=false")
