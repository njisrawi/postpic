#
# This file is part of postpic.
#
# postpic is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# postpic is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with postpic. If not, see <http://www.gnu.org/licenses/>.
#
# Copyright 2015 Stephan Kuschel, Stefan Tietze
'''
.. _PIConGPU: https://github.com/ComputationalRadiationPhysics/picongpu

Reader for the hdf5 files written by the PIConGPU_ Code.

Dependecies:
  - h5py: python package for reading hdf5 files

Written by:
Stephan Kuschel 2015
Stefan Tietze 2015
'''

from . import Dumpreader_ifc
from . import Simulationreader_ifc
import numpy as np
import re
from .. import helper

__all__ = ['Picongpuh5reader', 'PicongpuSimreader']


class Picongpuh5reader(Dumpreader_ifc):
    '''
    The Reader implementation for Data written by the PIConGPU_ Code
    in .h5 format.

    Args:
      sdffile : String
        A String containing the relative Path to the .h5 file.
    '''

    def __init__(self, h5file, **kwargs):
        super(self.__class__, self).__init__(h5file, **kwargs)
        import os.path
        import h5py
        if not os.path.isfile(h5file):
            raise IOError('File "' + str(h5file) + '" doesnt exist.')
        self._h5 = h5py.File(h5file, 'r')
        self._iteration = int(self._h5['data'].keys()[0])
        self._data = self._h5['/data/{:d}/'.format(self._iteration)]
        self.attrs = self._data.attrs

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]

    def getdata(self, key):
        dataset = self[key]
        try:
            ret = dataset.value * dataset.attrs['sim_unit']
        except(KeyError):
            ret = dataset.value
        if len(ret.shape) > 1:
            # PIConGPU axis order: y, x, z (WTF?)
            ret = np.swapaxes(ret, 0, 1)
        return np.float64(ret)

    def timestep(self):
        return self._iteration

    def time(self):
        return np.float64(self.attrs['delta_t'] * self.attrs['unit_time'] * self.timestep())

    def dataE(self, axis):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
        ret = np.float64(self.getdata('fields/FieldE/' + axsuffix))
        return ret

    def dataB(self, axis):
        axsuffix = {0: 'x', 1: 'y', 2: 'z'}[helper.axesidentify[axis]]
        ret = np.float64(self.getdata('fields/FieldB/' + axsuffix))
        return ret

    def grid(self, axis):  # todo
        # just use the grid of Ex for now, needs generalization
        axid = helper.axesidentify[axis]
        picongpuweiredaxisorder = {0: 1, 1: 0, 2: 2}
        try:
            n = self['fields/FieldE/x'].shape[picongpuweiredaxisorder[axid]]
            dx = self.dx(axis)
        except (IndexError):
            raise KeyError
        import numpy as np
        return np.linspace(0, n * dx, n)

    def dx(self, axis):
        axid = helper.axesidentify[axis]
        converter = {1: 'height', 0: 'width', 2: 'depth'}
        ret = self.attrs['cell_' + converter[axid]] * self.attrs['unit_length']
        return ret

    def listSpecies(self):
        return self['particles'].keys()

    def getSpecies(self, species, attrib):
        """
        Returns one of the attributes out of (x,y,z,px,py,pz,weight,ID, mass, charge) of
        this particle species.
        returning None means that this particle property wasnt dumped.
        Note that this is different from returning an empty list!
        """
        attribid = helper.attribidentify[attrib]
        options = {9: lambda s: self.getdata(s + 'weighting'),
                   0: lambda s: self.getdata(s + 'position/x') +
                   self.getdata(s + 'globalCellIdx/x') * self.dx('x'),
                   1: lambda s: self.getdata(s + 'position/y') +
                   self.getdata(s + 'globalCellIdx/y') * self.dx('y'),
                   2: lambda s: self.getdata(s + 'position/z') +
                   self.getdata(s + 'globalCellIdx/y') * self.dx('z'),
                   3: lambda s: self.getdata(s + 'momentum/x'),
                   4: lambda s: self.getdata(s + 'momentum/y'),
                   5: lambda s: self.getdata(s + 'momentum/z'),
                   10: lambda s: self['ID'],  # ids not yet implemented: will raise KeyError
                   11: lambda s: self[s].attrs['mass'] * self.attrs['unit_mass'],
                   12: lambda s: self[s].attrs['charge'] * self.attrs['unit_charge']}
        try:
            ret = np.float64(options[attribid]('particles/{}/'.format(species)))
        except(IndexError):
            raise KeyError
        return ret

    def getderived(self):
        '''
        Returns all derived fields
        '''
        ret = self['fields'].keys()
        for key in ['FieldB', 'FieldE']:
            try:
                ret.remove(key)
            except(ValueError):
                pass
        return ret

    def __str__(self):
        return '<PIConGPUh5Reader at "' + str(self.dumpidentifier) + '">'






















