#!/usr/bin/env python

# Copyright (c) 2023 Boston University 
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Wrapper for non-ego agents required for tracking and checking of just the lidar sensor
"""

from __future__ import print_function
import math
import os

import carla
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from leaderboard.autoagents.autonomous_agent import Track

from leaderboard.envs.sensor_interface import CallBack, SensorInterface


DATAGEN = int(os.environ.get('DATAGEN'))


class VehicleWrapper(object):

    """
    Wrapper for autonomous agents required for tracking and checking of used sensors
    """
    
    _agent = None
    _sensors_list = []

    def __init__(self, vehicle, sensor_spec):
        """
        Set the autonomous agent
        """
        self._vehicle = vehicle
        self._sensor_spec = sensor_spec
        self._sensor_interface = SensorInterface()

    def setup_sensor(self, debug_mode=False):
        """
        Create the sensors defined by the user and attach them to the ego-vehicle
        :param vehicle: ego vehicle
        :return:
        """
        bp_library = CarlaDataProvider.get_world().get_blueprint_library()
        
        bp = bp_library.find(str(self._sensor_spec['type']))
        
        if self._sensor_spec['type'].startswith('sensor.lidar'):
            bp.set_attribute('range', str(85))
            if DATAGEN==1:
                bp.set_attribute('rotation_frequency', str(self._sensor_spec['rotation_frequency']))
                bp.set_attribute('points_per_second', str(self._sensor_spec['points_per_second']))
            else:
                bp.set_attribute('rotation_frequency', str(10))
                bp.set_attribute('points_per_second', str(600000))
            bp.set_attribute('channels', str(64))
            bp.set_attribute('upper_fov', str(10))
            bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
            bp.set_attribute('dropoff_general_rate', str(0.45))
            bp.set_attribute('dropoff_intensity_limit', str(0.8))
            bp.set_attribute('dropoff_zero_intensity', str(0.4))
            sensor_location = carla.Location(x=self._sensor_spec['x'], y=self._sensor_spec['y'],
                                                z=self._sensor_spec['z'])
            sensor_rotation = carla.Rotation(pitch=self._sensor_spec['pitch'],
                                                roll=self._sensor_spec['roll'],
                                                yaw=self._sensor_spec['yaw'])
        else:
            raise ValueError("Sensor not implemented as a part of NPC Agent")
        
        # create sensor
        sensor_transform = carla.Transform(sensor_location, sensor_rotation)
        sensor = CarlaDataProvider.get_world().spawn_actor(bp, sensor_transform, self._vehicle)
        # setup callback
        
        sensor.listen(CallBack(self._sensor_spec['id'], self._sensor_spec['type'], sensor, self._sensor_interface))
        self._sensors_list.append(sensor)

        # Tick once to spawn the sensors
        # CarlaDataProvider.get_world().tick() # - Tick is automatically called in the scenario manager once scenario starts
        # We essentially do not collect data on this first tick, we just set up the agent's sensor, and collect data in the next tick


    def cleanup(self):
        """
        Remove and destroy all sensors
        """
        for i, _ in enumerate(self._sensors_list):
            if self._sensors_list[i] is not None:
                self._sensors_list[i].stop()
                self._sensors_list[i].destroy()
                self._sensors_list[i] = None
        self._sensors_list = []