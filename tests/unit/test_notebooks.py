#
# Tests jupyter notebooks
#
import os
import sys
import unittest

import liionpack as lp


class TestNotebooks(unittest.TestCase):
    def test_notebooks(self):
        examples_folder = os.path.join(lp.utils.ROOT_DIR, "docs", "examples")
        for filename in os.listdir(examples_folder):
            if os.path.splitext(filename)[1] == ".ipynb":
                path = os.path.join(examples_folder, filename)
                print("Testing notebook:", filename)

            # Make sure the notebook has a pip install command, for using Google Colab
            with open(path, "r") as f:
                if (
                    "!pip install -q git+https://github.com/pybamm-team/liionpack.git@main"  # noqa: E501
                    not in f.read()
                ):
                    # Print error and exit
                    print("\n" + "-" * 115)
                    print("ERROR")
                    print(
                        "Installation command '!pip install -q git+https://github.com/pybamm-team/liionpack.git@main' not found in notebook"  # noqa: E501
                    )
                    print("-" * 115)
                    return sys.exit(1)


if __name__ == "__main__":
    unittest.main()
