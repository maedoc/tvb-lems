"""Generate LEMS from a source.Model instance. Disparities include

- model may specify diffusion coefficients for SDE. LEMS?
- LEMS expects units, TVB doesn't.

"""

import lems.api as lems
from . import source


def model_instances():
    for key in dir(source):
        member = getattr(source, key)
        if isinstance(member, source.Model):
            yield member


def build_lems_for_model(src: source.Model):
    model = lems.Model()

    model.add(lems.Dimension('time', t=1))
    model.add(lems.Dimension('au'))

    # primary element of the model is a mass model component
    mass = lems.ComponentType(src.name)
    model.add(mass)

    for key, val in src.const.items():
        mass.add(lems.Constant(key, 'au'))  # TODO units

    for key in src.param:
        mass.add(lems.Parameter(key, 'au'))  # TODO units

    for key, val in src.auxex:
        val = val.replace('**', '^')
        mass.dynamics.add(lems.DerivedVariable(key, value=val))

    for key in src.obsrv:
        mass.add(lems.Exposure(key, 'au'))

    for src_svar in src.state_space:
        name = src_svar.name
        ddt = src_svar.drift.replace('**', '^')
        mass.dynamics.add(lems.StateVariable(name, 'au', name))
        mass.dynamics.add(lems.TimeDerivative(name, ddt))

    return model



if __name__ == '__main__':
    import os
    from lems.base.util import validate_lems
    here = os.path.dirname(os.path.abspath(__file__))
    for model in model_instances():
        fname = os.path.join(here, model.name + '.lems.xml')
        try:
            build_lems_for_model(model).export_to_file(fname)
            validate_lems(fname)
        except Exception as exc:
            print(exc)
