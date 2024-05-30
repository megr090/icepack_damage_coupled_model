# Copyright (C) 2018-2023 by Daniel Shapero <shapero@uw.edu> and Andrew Hoffman
# <hoffmaao@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

r"""Description of the continuum damage mechanics model

This module contains a solver for the conservative advection equation that
describes the evolution of ice damage (Albrecht and Levermann 2014).
"""

from operator import itemgetter
import firedrake
from firedrake import inner, sqrt, det, min_value, conditional
from .transport import TransportEquation
from icepack.constants import year
from icepack.utilities import eigenvalues


class DamageTransportNoSrc(TransportEquation):
    def __init__(
        self,
        damage_stress=0.07,
        damage_rate=0.3,
        healing_strain_rate=2e-10 * year,
        healing_rate=0.1,
    ):
        super(DamageTransportNoSrc, self).__init__(
            field_name="damage", source_name=None, conservative=False
        )
        self.damage_stress = damage_stress
        self.damage_rate = damage_rate
        self.healing_strain_rate = healing_strain_rate
        self.healing_rate = healing_rate

    def sources(self, **kwargs):
        #keys = ("damage", "velocity", "strain_rate", "membrane_stress")
        #D, u, ε, M = itemgetter(*keys)(kwargs)

        src = firedrake.Constant(0)

        return src
