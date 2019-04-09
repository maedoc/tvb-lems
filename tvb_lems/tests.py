import os
import numpy as np
from pyneuroml import pynml
from tvb.simulator.models import Generic2dOscillator

_here = os.path.abspath(os.path.dirname(__file__))

def do_lems_sim_g2do():
    """
    Run G2DO simulation from LEMS file.
    :return:  time, V time series
    """
    example_lems_file = os.path.join(_here, 'LEMS_TestG2DO.xml')
    results1 = pynml.run_lems_with_jneuroml(example_lems_file, nogui=True, load_saved_data=True)
    t, v = results1['t'], results1['Pop1[0]/V']
    return t, v


def do_tvb_sim_g2do():
    """
    Run G2DO simulation using TVB (single node Euler step)
    :return: time, V time series
    """
    # TODO read parameters from LEMS test file?
    osc = Generic2dOscillator(a=0.0, d=0.009)
    t, v = osc.stationary_trajectory(np.zeros((2, 1)), -2.0 * np.ones((2, 1)), n_step=5000, n_skip=1, dt=1.0)
    v = v[:, 0, 0, 0]
    return t * 1e-3, v


def test_g2do():
    t = np.r_[:1.0:20j]
    lv = np.interp(t, *do_lems_sim_g2do())
    tv = np.interp(t, *do_tvb_sim_g2do())
    np.testing.assert_allclose(lv, tv)
