import liionpack as lp
import numpy as np
import pandas as pd
import pathlib
import pybamm
import unittest


class utilsTest(unittest.TestCase):

    def setUp(self):
        currents = [4, 5.2, 8, 10]
        volts = np.array([[3.5, 3.6], [3.54, 3.58]])
        output = {'currents': currents, 'volts': volts}

        lp.save_to_csv(output, path='.')
        lp.save_to_npy(output, path='.')
        lp.save_to_npzcomp(output, path='.')

        self.currents = currents
        self.volts = volts

    def test_interp_current(self):
        d = {"Time": [0, 10], "Cells Total Current": [2.0, 4.0]}
        df = pd.DataFrame(data=d)
        f = lp.interp_current(df)
        assert f(5) == 3.0

    def test_add_events_to_model(self):
        model = pybamm.lithium_ion.SPMe()
        model = lp.add_events_to_model(model)
        events_in = False
        for key in sorted(model.variables.keys()):
            if "Event:" in key:
                events_in = True
                break
        assert events_in

    def test_currents_csv(self):
        with open('currents.csv', 'r') as f:
            current = float(f.readline())
        self.assertEqual(self.currents[0], current)

    def test_volts_csv(self):
        with open('volts.csv', 'r') as f:
            volts = list(map(float, f.readline().split(', ')))
        self.assertEqual(self.volts[0][0], volts[0])

    def test_currents_npy(self):
        currents = np.load('currents.npy')
        self.assertEqual(self.currents[0], currents[0])

    def test_volts_npy(self):
        volts = np.load('volts.npy')
        self.assertEqual(self.volts[0, 0], volts[0, 0])

    def test_currents_npz(self):
        output = np.load('output.npz')
        currents = output['currents']
        self.assertEqual(self.currents[0], currents[0])

    def test_volts_npz(self):
        output = np.load('output.npz')
        volts = output['volts']
        self.assertEqual(self.volts[0, 0], volts[0, 0])

    def tearDown(self):
        path = pathlib.Path('currents.csv')
        path.unlink(missing_ok=True)

        path = pathlib.Path('volts.csv')
        path.unlink(missing_ok=True)

        path = pathlib.Path('currents.npy')
        path.unlink(missing_ok=True)

        path = pathlib.Path('volts.npy')
        path.unlink(missing_ok=True)

        path = pathlib.Path('output.npz')
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
