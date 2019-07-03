# ----------------------------------------------------------------------
# Hierachrical Prediction Memory
# Copyright (C) 2019, HDILab.  Unless you have an agreement
# with HDILab, for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
# ----------------------------------------------------------------------

"""
Feeder class
Feeder feeds input to the Temporal Memory

There can be child classes including txtfeeder and temporalfeeder

Usage concept
txtfeeder = TXTFeeder("input.txt")
L1 = temporalmemory(feeder=txtfeeder)
L2feeder = TemporalFeeder(input=L1)
L2 = temporalmemory(feeder=L2feeder)
def feedforward()
    input=feeder.feed()
    self.predict(input)
"""


class Feeder(object):
    """
      Class implementing the Feeder.

      :param input: (obj) input for feeder

      :param numBits: (int) Number of bits for SDR. Default value ``512``

      :param numOnBits: (int) Number of Active bits for SDR. Default value ``10``.
            It is 2% sparcity for 512 bit

      :param seed: (int) Seed for the random number generator. Default value ``42``.
    """

    def __init__(self,
                 numBits=512,
                 numOnBits=10,
                 seed=42):
        self.numBits = numBits
        self.numOnBits = numOnBits

