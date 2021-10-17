import pybamm

def test_experiment():
    r'''
    retun a standard experiment

    Returns
    -------
    experiment : pybamm.Experiment
        A pybamm experiment class describing a simple CC charge/discharge

    '''
    experiment = pybamm.Experiment(
        ["Charge at 50 A for 300 seconds",
        "Rest for 150 seconds",
        "Discharge at 50 A for 300 seconds",
        "Rest for 300 seconds"],
        period="10 seconds",
    )
    return experiment