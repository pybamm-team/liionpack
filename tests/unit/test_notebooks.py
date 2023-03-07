#
# Tests jupyter notebooks
#
import os
import subprocess
import unittest

import nbconvert

import liionpack as lp


class TestNotebooks(unittest.TestCase):
    def test_notebooks(self):
        examples_folder = os.path.join(lp.ROOT_DIR, "docs", "examples")
        for filename in os.listdir(examples_folder):
            if os.path.splitext(filename)[1] == ".ipynb":
                print("-" * 80)
                print("Testing notebook:", filename)
                print("-" * 80)

                # Load notebook, convert to python
                path = os.path.join(examples_folder, filename)
                e = nbconvert.exporters.PythonExporter()
                code, __ = e.from_filename(path)

                # Make sure the notebook has pip install command, for using Google Colab
                self.assertIn(
                    "pip install -q git+https://github.com/pybamm-team/liionpack.git@main",  # noqa: E501
                    code,
                    "Installation command '!pip install -q git+https://github.com/pybamm-team/liionpack.git@main' not found in notebook",  # noqa: E501
                )

                # Comment out the pip install command to avoid reinstalling
                code = code.replace("get_ipython().system('pip", "#")

                # Run in subprocess
                cmd = ["python", "-c", code]
                p = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                self.assertEqual(p.returncode, 0)


if __name__ == "__main__":
    unittest.main()
