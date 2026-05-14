
import sys
import re
from unittest.mock import MagicMock

# Define a function to fix end document placement by reading it from the source file directly.
# This avoids all dependency issues while ensuring we are testing the actual production code.
def get_fix_end_document_placement():
    with open("nanoresearch/agents/writing/latex_figure_placement.py", "r") as f:
        content = f.read()

    # Extract the _fix_end_document_placement method
    # It starts with '    @staticmethod' followed by '    def _fix_end_document_placement'
    start_pattern = r'    @staticmethod\n    def _fix_end_document_placement\(text: str\) -> str:'
    start_match = re.search(start_pattern, content)
    if not start_match:
        raise Exception("Could not find start of _fix_end_document_placement")

    start_pos = start_match.end()

    # The method ends at the next @classmethod or @staticmethod at the same indentation level
    end_match = re.search(r'\n    @(classmethod|staticmethod)', content[start_pos:])
    if not end_match:
        # If no next method, it might be the end of the class/file
        # Look for the next class or end of string
        end_match = re.search(r'\nclass ', content[start_pos:])
        if not end_match:
             end_pos = len(content)
        else:
             end_pos = start_pos + end_match.start()
    else:
        end_pos = start_pos + end_match.start()

    method_body = content[start_pos:end_pos]

    # Dedent the body (remove 8 spaces from each line, and add 4 spaces for the function body)
    indented_body = ""
    for line in method_body.split("\n"):
        if line.startswith("        "):
            indented_body += "    " + line[8:] + "\n"
        elif line.strip() == "":
            indented_body += "\n"
        else:
            indented_body += "    " + line + "\n"

    # Create a function from the body
    local_vars = {}
    import re as re_module
    exec_globals = {"re": re_module}
    exec_code = f"def _fix_end_document_placement(text: str):\n{indented_body}"
    try:
        exec(exec_code, exec_globals, local_vars)
    except Exception:
        print("Failed to exec code:")
        for i, line in enumerate(exec_code.split("\n")):
            print(f"{i+1:3}: {line}")
        raise

    return local_vars["_fix_end_document_placement"]

def test_fix_end_document_placement():
    _fix_end_document_placement = get_fix_end_document_placement()

    test_cases = [
        (
            "Standard (no change)",
            r"\begin{document} Hello \bibliographystyle{unsrt} \bibliography{refs} \end{document}",
            [r"\begin{document} Hello \bibliographystyle{unsrt} \bibliography{refs} \end{document}"]
        ),
        (
            "Bib after end",
            r"\begin{document} Hello \end{document} \bibliographystyle{unsrt} \bibliography{refs}",
            [r"Hello", r"\bibliographystyle{unsrt}", r"\bibliography{refs}", r"\end{document}"]
        ),
        (
            "Multiple end",
            r"\begin{document} Hello \end{document} World \end{document}",
            [r"Hello World", r"\end{document}"]
        ),
        (
            "Space in bib command",
            r"\begin{document} Hello \bibliography {refs} \end{document}",
            [r"Hello", r"bibliography {refs}", r"\end{document}"]
        ),
        (
            "Text after single end",
            r"\begin{document} Hello \bibliographystyle{unsrt} \bibliography{refs} \end{document} Garbage",
            [r"Hello Garbage", r"\bibliographystyle{unsrt}", r"\bibliography{refs}", r"\end{document}"]
        ),
        (
            "Inline bibliography",
            r"\begin{document} Hello \begin{thebibliography} item \end{thebibliography} \end{document} Extra",
            [r"Hello", r"Extra", r"\begin{thebibliography} item \end{thebibliography}", r"\end{document}"]
        ),
        (
            "No end document",
            r"\begin{document} Hello",
            [r"\bibliographystyle{plainnat}", r"\bibliography{references}", r"\end{document}"]
        )
    ]

    for name, input_text, expected_in_output in test_cases:
        print(f"Running test: {name}")
        result = _fix_end_document_placement(input_text)

        # Check that all expected strings are in the output
        for expected in expected_in_output:
            assert expected in result, f"Failed {name}: {expected} not in result.\nResult was: {result}"

        # Check that it ends with \end{document}
        assert result.strip().endswith(r"\end{document}"), f"Failed {name}: does not end with \end{{document}}"

        # Check that there is only one \end{document}
        assert len(re.findall(r'\\end\{document\}', result)) == 1, f"Failed {name}: more than one \end{{document}}"

def test_fix_end_document_placement_no_begin():
    _fix_end_document_placement = get_fix_end_document_placement()
    input_text = r"No begin document here \end{document}"
    result = _fix_end_document_placement(input_text)
    assert result == input_text

if __name__ == "__main__":
    try:
        test_fix_end_document_placement()
        test_fix_end_document_placement_no_begin()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
