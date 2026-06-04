import os

import ast

class RepoScanner:
    def __init__(self, root_dir="."):
        self.root_dir = root_dir
        self.index = {
            "structure": {},
            "signatures": [],
            "critical_logic": {}
        }

    def scan(self):
        for root, dirs, files in os.walk(self.root_dir):
            if ".git" in dirs:
                dirs.remove(".git")
            if "__pycache__" in dirs:
                dirs.remove("__pycache__")

            rel_path = os.path.relpath(root, self.root_dir)
            self.index["structure"][rel_path] = files

            for file in files:
                if file.endswith(".py"):
                    filepath = os.path.join(root, file)
                    self._parse_python_file(filepath)
        return self.index

    def _parse_python_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

                rel_filepath = os.path.relpath(filepath, self.root_dir)
                file_sigs = []

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [arg.arg for arg in node.args.args]
                        file_sigs.append({
                            "name": node.name,
                            "args": args,
                            "line": node.lineno
                        })
                        # Heuristic for important logic
                        keywords = ["secure", "auth", "crypto", "gate", "verify", "patch", "engine", "model", "block", "attention", "moe", "rec", "norm", "layer", "orchestra", "analyze", "process", "forward"]
                        if any(k in node.name.lower() for k in keywords):
                            lines = content.splitlines()
                            end_line = getattr(node, "end_lineno", node.lineno + 10)
                            logic_snippet = lines[node.lineno-1 : end_line]
                            self.index["critical_logic"][f"{rel_filepath}:{node.name}"] = "\n".join(logic_snippet)

                    elif isinstance(node, ast.ClassDef):
                        file_sigs.append({
                            "type": "class",
                            "name": node.name,
                            "line": node.lineno
                        })

                self.index["signatures"].append({
                    "file": rel_filepath,
                    "definitions": file_sigs
                })
        except Exception as e:
            import logging
            logging.error(f"Error parsing {filepath}: {e}")

if __name__ == "__main__":
    scanner = RepoScanner()
    index = scanner.scan()
    print(f"Scanned {len(index['signatures'])} files.")
    print(f"Identified {len(index['critical_logic'])} critical logic blocks.")
