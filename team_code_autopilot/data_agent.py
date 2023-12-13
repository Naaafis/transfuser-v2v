from copy import deepcopy
import cv2
import carla

import random
import torch
import numpy as np
import pygame
import json

from utils import lts_rendering
from utils.map_utils import MapImage, encode_npy_to_pil, PIXELS_PER_METER
from autopilot import AutoPilot
from vehicle_wrapper import VehicleWrapper


def get_entry_point():
    return 'DataAgent'


class DataAgent(AutoPilot):
    def setup(self, path_to_conf_file, route_index=None):
        super().setup(path_to_conf_file, route_index)

        self.cam_config = {
            'width': 320,
            'height': 160,
            'fov': 60
        }

        self.weathers = {
            'Clear': carla.WeatherParameters.ClearNoon,
            'Cloudy': carla.WeatherParameters.CloudySunset,
            'Wet': carla.WeatherParameters.WetSunset,
            'MidRain': carla.WeatherParameters.MidRainSunset,
            'WetCloudy': carla.WeatherParameters.WetCloudySunset,
            'HardRain': carla.WeatherParameters.HardRainNoon,
            'SoftRain': carla.WeatherParameters.SoftRainSunset,
        }

        self.azimuths = [45.0 * i for i in range(8)]

        self.daytimes = {
            'Night': -80.0,
            'Twilight': 0.0,
            'Dawn': 5.0,
            'Sunset': 15.0,
            'Morning': 35.0,
            'Noon': 75.0,
        }

        self.weathers_ids = list(self.weathers)

        if self.save_path is not None:
            (self.save_path / 'topdown').mkdir()
            (self.save_path / 'lidar').mkdir()
            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'label_raw').mkdir()
            (self.save_path / 'semantics').mkdir()
            (self.save_path / 'depth').mkdir()
            (self.save_path / 'lidar_2').mkdir()
            (self.save_path / 'vehicle_2_pos').mkdir()  

        self._active_traffic_light = None

    def _init(self, hd_map):
        super()._init(hd_map)
        self._sensors = self.sensor_interface._sensors_objects

        self.vehicle_template = torch.ones(1, 1, 22, 9, device='cuda')
        self.walker_template = torch.ones(1, 1, 10, 7, device='cuda')
        self.traffic_light_template = torch.ones(1, 1, 4, 4, device='cuda')

        # create map for renderer
        map_image = MapImage(self._world, self.world_map, PIXELS_PER_METER)
        make_image = lambda x: np.swapaxes(pygame.surfarray.array3d(x), 0, 1).mean(axis=-1)
        road = make_image(map_image.map_surface)
        lane = make_image(map_image.lane_surface)
        
        self.global_map = np.zeros((1, 15,) + road.shape)
        self.global_map[:, 0, ...] = road / 255.
        self.global_map[:, 1, ...] = lane / 255.

        self.global_map = torch.tensor(self.global_map, device='cuda', dtype=torch.float32)
        world_offset = torch.tensor(map_image._world_offset, device='cuda', dtype=torch.float32)
        self.map_dims = self.global_map.shape[2:4]

        self.renderer = lts_rendering.Renderer(world_offset, self.map_dims, data_generation=True)
        
        
    def sensors(self):
        result = super().sensors()
        if self.save_path is not None:
            result += [
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_front'
                    },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_left'
                    },
                    {
                        'type': 'sensor.camera.rgb',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'rgb_right'
                    },
                    {
                        'type': 'sensor.lidar.ray_cast',
                        'x': 1.3, 'y': 0.0, 'z': 2.5,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                        'rotation_frequency': 20,
                        'points_per_second': 1200000,
                        'id': 'lidar'
                    },
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_front'
                    },
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_left'
                    },
                    {
                        'type': 'sensor.camera.semantic_segmentation',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'semantics_right'
                    },
                    {
                        'type': 'sensor.camera.depth',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'depth_front'
                    },
                    {
                        'type': 'sensor.camera.depth',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'depth_left'
                    },
                    {
                        'type': 'sensor.camera.depth',
                        'x': 1.3, 'y': 0.0, 'z':2.3,
                        'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
                        'width': self.cam_config['width'], 'height': self.cam_config['height'], 'fov': self.cam_config['fov'],
                        'id': 'depth_right'
                    },
                    ]

        return result

    def tick(self, input_data, vehicle_2_data=None):
        result = super().tick(input_data)

        if self.save_path is not None:
            rgb = []
            semantics = []
            depth = []
            for pos in ['left', 'front', 'right']:
                rgb_cam = 'rgb_' + pos
                semantics_cam = 'semantics_' + pos
                depth_cam = 'depth_' + pos
                semantics_img = input_data[semantics_cam][1][:, :, 2]
                depth_img = input_data[depth_cam][1][:, :, :3] 
                _semantics = np.copy(semantics_img)
                _depth = self._get_depth(depth_img)
                self._change_seg_tl(_semantics, _depth, self._active_traffic_light)

                rgb.append(cv2.cvtColor(input_data[rgb_cam][1][:, :, :3], cv2.COLOR_BGR2RGB))
                semantics.append(_semantics)
                depth.append(depth_img)

            rgb = np.concatenate(rgb, axis=1)
            semantics = np.concatenate(semantics, axis=1)
            depth =  np.concatenate(depth, axis=1)

            result['topdown'] = self.render_BEV()
            
            lidar = input_data['lidar']
            
            cars = self.get_bev_cars(lidar=lidar)

            if vehicle_2_data is not None:
                lidar_2 = vehicle_2_data['data']['lidar']
                vehicle_2_loc = vehicle_2_data['transform'].location
                vehicle_2_yaw = vehicle_2_data['transform'].rotation.yaw
                vehicle_2_pitch = vehicle_2_data['transform'].rotation.pitch
                vehicle_2_roll = vehicle_2_data['transform'].rotation.roll
                vehicle_2_pos = [vehicle_2_loc.x, vehicle_2_loc.y, vehicle_2_loc.z, vehicle_2_yaw, vehicle_2_pitch, vehicle_2_roll]
                result.update({'lidar': lidar,
                                'lidar_2': lidar_2,
                                'vehicle_2_pos': vehicle_2_pos, 
                                'rgb': rgb,
                                'cars': cars,
                                'semantics': semantics,
                                'depth': depth})
                
            else:
                result.update({'lidar': lidar,
                               'lidar_2': None, 
                                'vehicle_2_pos': None,
                                'rgb': rgb,
                                'cars': cars,
                                'semantics': semantics,
                                'depth': depth})

        return result

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not ('hd_map' in input_data.keys()) and not self.initialized:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        control = super().run_step(input_data, timestamp)
        
        # Collect data from the closest vehicle
        vehicle_2_data = self.find_closest_vehicle()

        if self.step % self.save_freq == 0:
            if self.save_path is not None:
                tick_data = self.tick(input_data, vehicle_2_data)
                self.save_sensors(tick_data)
                self.shuffle_weather()
            
        return control
    
    def find_closest_vehicle(self):
        closest_vehicle = None
        min_distance = 30  # 30 meters threshold
        max_distance = 50  # 50 meters threshold
        ego_location = self._vehicle.get_location()
        vehicles = self._world.get_actors().filter('*vehicle*')

        for vehicle in vehicles:
            if vehicle.id != self._vehicle.id:
                distance = ego_location.distance(vehicle.get_location())
                if distance < min_distance:
                    min_distance = distance
                    closest_vehicle = vehicle
                if distance > max_distance:
                    # clean up the vechiel_wrapper if the vehicle is now too far away
                    if vehicle.id in self._wrapped_vehicles:
                        # run clean up first 
                        self._wrapped_vehicles[vehicle.id].cleanup()
                        self._wrapped_vehicles.pop(vehicle.id)

        if closest_vehicle and closest_vehicle.id in self._wrapped_vehicles:
            # sensors have already been set up for this vehicle
            closets_vehicle_transform = closest_vehicle.get_transform() 
            # return both the vehicle and the sensor data
            
            #return self.collect_vehicle_data(self._wrapped_vehicles[closest_vehicle.id])
            sensor_data = self.collect_vehicle_data(self._wrapped_vehicles[closest_vehicle.id])
            
            vehicle_2_info = {
                'id': closest_vehicle.id,
                'transform': closets_vehicle_transform,
                'data': sensor_data
            }
            
            return vehicle_2_info
            
        elif closest_vehicle and closest_vehicle.id not in self._wrapped_vehicles:
            # Create sensor_spec based on the vehicle's current transform
            vehicle_transform = closest_vehicle.get_transform()
            sensor_spec = {
                'type': 'sensor.lidar.ray_cast',
                'x': vehicle_transform.location.x,
                'y': vehicle_transform.location.y,
                'z': vehicle_transform.location.z,
                'roll': vehicle_transform.rotation.roll,
                'pitch': vehicle_transform.rotation.pitch,
                'yaw': vehicle_transform.rotation.yaw - 90,  # Adjusting yaw
                'rotation_frequency': 20,
                'points_per_second': 1200000,
                'id': 'lidar'
            }

            # Wrap the vehicle and add it to the dictionary
            vehicle_wrapper = VehicleWrapper(closest_vehicle, sensor_spec)
            self._wrapped_vehicles[closest_vehicle.id] = vehicle_wrapper
            # Setup sensor for the vehicle
            vehicle_wrapper.setup_sensor()
            # we will not collect data on this tick, we just set up the sensor, and collect data in the next tick
            
        return None
    
    def collect_vehicle_data(self, vehicle_wrapper):
        # Logic to collect data from the vehicle wrapper
        return vehicle_wrapper._sensor_interface.get_data()

    def shuffle_weather(self):
        # change weather for visual diversity
        index = random.choice(range(len(self.weathers)))
        dtime, altitude = random.choice(list(self.daytimes.items()))
        altitude = np.random.normal(altitude, 10)
        self.weather_id = self.weathers_ids[index] + dtime

        weather = self.weathers[self.weathers_ids[index]]
        weather.sun_altitude_angle = altitude
        weather.sun_azimuth_angle = np.random.choice(self.azimuths)
        self._world.set_weather(weather)

        # night mode
        vehicles = self._world.get_actors().filter('*vehicle*')
        if weather.sun_altitude_angle < 0.0:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))
        else:
            for vehicle in vehicles:
                vehicle.set_light_state(carla.VehicleLightState.NONE)

    def save_sensors(self, tick_data):
        frame = self.step // self.save_freq

        # CV2 uses BGR internally so we need to swap the image channels before saving.
        img = cv2.cvtColor(tick_data['rgb'],cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.save_path / 'rgb' / ('%04d.png' % frame)), img)

        img = encode_npy_to_pil(np.asarray(tick_data['topdown'].squeeze().cpu()))
        img_save=np.moveaxis(img,0,2)
        cv2.imwrite(str(self.save_path / 'topdown' / ('encoded_%04d.png' % frame)), img_save)

        semantics = tick_data['semantics']
        cv2.imwrite(str(self.save_path / 'semantics' / ('%04d.png' % frame)), semantics)

        depth = cv2.cvtColor(tick_data['depth'], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(self.save_path / 'depth' / ('%04d.png' % frame)), depth)

        np.save(self.save_path / 'lidar' / ('%04d.npy' % frame), tick_data['lidar'], allow_pickle=True)
        self.save_labels(self.save_path / 'label_raw' / ('%04d.json' % frame), tick_data['cars'])
        
        if tick_data['lidar_2'] is not None and tick_data['vehicle_2_pos'] is not None:
            np.save(self.save_path / 'lidar_2' / ('%04d.npy' % frame), tick_data['lidar_2'], allow_pickle=True)
            self.save_vehicle_pos(self.save_path / 'vehicle_2_pos' / ('%04d.json' % frame), tick_data['vehicle_2_pos'])
        
    def save_labels(self, filename, result):
        with open(filename, 'w') as f:
            json.dump(result, f, indent=4)
        return
    
    def save_vehicle_pos(self, filename, points):
        with open(filename, 'w') as f:
            json.dump(points, f, indent=4)

    def save_points(self, filename, points):
        points_to_save = deepcopy(points[1])
        points_to_save[:, 1] = -points_to_save[:, 1]
        np.save(filename, points_to_save)
        return
    
    def destroy(self):
        # print for debugging
        print('DataAgent destroyed')
        
        # clean up all wrapped vehicles before destroying the agent
        for vehicle_id in self._wrapped_vehicles:
            self._wrapped_vehicles[vehicle_id].cleanup()
            # self._wrapped_vehicles.pop(vehicle_id)
            
        # Only delete global_map if it exists
        if hasattr(self, 'global_map'):
            del self.global_map
        if hasattr(self, 'vehicle_template'):
            del self.vehicle_template
        if hasattr(self, 'walker_template'):
            del self.walker_template
        if hasattr(self, 'traffic_light_template'):
            del self.traffic_light_template
        if hasattr(self, 'map_dims'):
            del self.map_dims
        
        torch.cuda.empty_cache()

    def get_bev_cars(self, lidar=None):
        results = []
        ego_rotation = self._vehicle.get_transform().rotation
        ego_matrix = np.array(self._vehicle.get_transform().get_matrix())

        ego_extent = self._vehicle.bounding_box.extent
        ego_dx = np.array([ego_extent.x, ego_extent.y, ego_extent.z]) * 2.
        ego_yaw =  ego_rotation.yaw/180*np.pi
        
        # also add ego box for visulization
        relative_yaw = 0
        relative_pos = self.get_relative_transform(ego_matrix, ego_matrix)

        # add vehicle velocity and brake flag
        ego_transform = self._vehicle.get_transform()
        ego_control   = self._vehicle.get_control()
        ego_velocity  = self._vehicle.get_velocity()
        ego_speed = self._get_forward_speed(transform=ego_transform, velocity=ego_velocity) # In m/s
        ego_brake = ego_control.brake
        
        # the position is in lidar coordinates
        result = {"class": "Car",
                  "extent": [ego_dx[2], ego_dx[0], ego_dx[1]], # NOTE: height stored in first dimension
                  "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                  "yaw": relative_yaw,
                  "num_points": -1, 
                  "distance": -1, 
                  "speed": ego_speed, 
                  "brake": ego_brake,
                  "id": int(self._vehicle.id),
                  'ego_matrix': self._vehicle.get_transform().get_matrix()
                }
        results.append(result)
        
        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < 50):
                if (vehicle.id != self._vehicle.id):
                    vehicle_rotation = vehicle.get_transform().rotation
                    vehicle_matrix = np.array(vehicle.get_transform().get_matrix())
                    vehicle_id = vehicle.id

                    vehicle_extent = vehicle.bounding_box.extent
                    dx = np.array([vehicle_extent.x, vehicle_extent.y, vehicle_extent.z]) * 2.
                    yaw =  vehicle_rotation.yaw/180*np.pi

                    relative_yaw = yaw - ego_yaw
                    relative_pos = self.get_relative_transform(ego_matrix, vehicle_matrix)

                    vehicle_transform = vehicle.get_transform()
                    vehicle_control   = vehicle.get_control()
                    vehicle_velocity  = vehicle.get_velocity()
                    vehicle_speed = self._get_forward_speed(transform=vehicle_transform, velocity=vehicle_velocity) # In m/s
                    vehicle_brake = vehicle_control.brake

                    # filter bbox that didn't contains points of contains less points
                    if not lidar is None:
                        num_in_bbox_points = self.get_points_in_bbox(ego_matrix, vehicle_matrix, dx, lidar)
                    else:
                        num_in_bbox_points = -1

                    distance = np.linalg.norm(relative_pos)

                    result = {
                        "class": "Car",
                        "extent": [dx[2], dx[0], dx[1]], # NOTE: height stored in first dimension
                        "position": [relative_pos[0], relative_pos[1], relative_pos[2]],
                        "yaw": relative_yaw,
                        "num_points": int(num_in_bbox_points), 
                        "distance": distance, 
                        "speed": vehicle_speed, 
                        "brake": vehicle_brake,
                        "id": int(vehicle_id),
                        "ego_matrix": vehicle.get_transform().get_matrix()
                    }
                    results.append(result)
                    
        return results

    def get_points_in_bbox(self, ego_matrix, vehicle_matrix, dx, lidar):
        # inverse transform
        Tr_lidar_2_ego = self.get_lidar_to_vehicle_transform()
        
        # construct transform from lidar to vehicle
        Tr_lidar_2_vehicle = np.linalg.inv(vehicle_matrix) @ ego_matrix @ Tr_lidar_2_ego

        # transform lidar to vehicle coordinate
        lidar_vehicle = Tr_lidar_2_vehicle[:3, :3] @ lidar[1][:, :3].T + Tr_lidar_2_vehicle[:3, 3:]

        # check points in bbox
        x, y, z = dx / 2.
        # why should we use swap?
        x, y = y, x
        num_points = ((lidar_vehicle[0] < x) & (lidar_vehicle[0] > -x) & 
                      (lidar_vehicle[1] < y) & (lidar_vehicle[1] > -y) & 
                      (lidar_vehicle[2] < z) & (lidar_vehicle[2] > -z)).sum()
        return num_points

    def get_relative_transform(self, ego_matrix, vehicle_matrix):
        """
        return the relative transform from ego_pose to vehicle pose
        """
        relative_pos = vehicle_matrix[:3, 3] - ego_matrix[:3, 3]
        rot = ego_matrix[:3, :3].T
        relative_pos = rot @ relative_pos
        
        # transform to right handed system
        relative_pos[1] = - relative_pos[1]

        # transform relative pos to virtual lidar system
        rot = np.eye(3)
        trans = - np.array([1.3, 0.0, 2.5])
        relative_pos = rot @ relative_pos + trans

        return relative_pos

    def get_lidar_to_vehicle_transform(self):
        # yaw = -90
        rot = np.array([[0, 1, 0],
                        [-1, 0, 0],
                        [0, 0, 1]], dtype=np.float32)
        T = np.eye(4)

        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.5
        T[:3, :3] = rot
        return T

    def get_vehicle_to_lidar_transform(self):
        return np.linalg.inv(self.get_lidar_to_vehicle_transform())

    def get_image_to_vehicle_transform(self):
        # yaw = 0.0 as rot is Identity
        T = np.eye(4)
        T[0, 3] = 1.3
        T[1, 3] = 0.0
        T[2, 3] = 2.3

        # rot is from vehicle to image
        rot = np.array([[0, -1, 0],
                        [0, 0, -1],
                        [1, 0, 0]], dtype=np.float32)
        
        # so we need a transpose here
        T[:3, :3] = rot.T
        return T

    def get_vehicle_to_image_transform(self):
        return np.linalg.inv(self.get_image_to_vehicle_transform())

    def get_lidar_to_image_transform(self):
        Tr_lidar_to_vehicle = self.get_lidar_to_vehicle_transform()
        Tr_image_to_vehicle = self.get_image_to_vehicle_transform()
        T_lidar_to_image = np.linalg.inv(Tr_image_to_vehicle) @ Tr_lidar_to_vehicle
        return T_lidar_to_image

    def render_BEV(self):
        semantic_grid = self.global_map
        
        vehicle_position = self._vehicle.get_location()
        ego_pos_list =  [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
        ego_yaw_list =  [self._vehicle.get_transform().rotation.yaw/180*np.pi]

        # fetch local birdview per agent
        ego_pos =  torch.tensor([self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y], device='cuda', dtype=torch.float32)
        ego_yaw =  torch.tensor([self._vehicle.get_transform().rotation.yaw/180*np.pi], device='cuda', dtype=torch.float32)
        birdview = self.renderer.get_local_birdview(
            semantic_grid,
            ego_pos,
            ego_yaw
        )

        self._actors = self._world.get_actors()
        vehicles = self._actors.filter('*vehicle*')
        for vehicle in vehicles:
            if (vehicle.get_location().distance(self._vehicle.get_location()) < self.detection_radius):
                if (vehicle.id != self._vehicle.id):
                    pos =  torch.tensor([vehicle.get_transform().location.x, vehicle.get_transform().location.y], device='cuda', dtype=torch.float32)
                    yaw =  torch.tensor([vehicle.get_transform().rotation.yaw/180*np.pi], device='cuda', dtype=torch.float32)
                    veh_x_extent = int(max(vehicle.bounding_box.extent.x*2, 1) * PIXELS_PER_METER)
                    veh_y_extent = int(max(vehicle.bounding_box.extent.y*2, 1) * PIXELS_PER_METER)

                    self.vehicle_template = torch.ones(1, 1, veh_x_extent, veh_y_extent, device='cuda')
                    self.renderer.render_agent_bv(
                        birdview,
                        ego_pos,
                        ego_yaw,
                        self.vehicle_template,
                        pos,
                        yaw,
                        channel=5
                    )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Pedestrian rendering
        # -----------------------------------------------------------
        walkers = self._actors.filter('*walker*')
        for walker in walkers:
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([walker.get_transform().location.x, walker.get_transform().location.y])
            yaw_batched.append([walker.get_transform().rotation.yaw/180*np.pi])
            channel_batched.append(6)
            template_batched.append(np.ones([20, 7]))

        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.float32)

            self.renderer.render_agent_bv_batched(
                birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        ego_pos_batched = []
        ego_yaw_batched = []
        pos_batched = []
        yaw_batched = []
        template_batched = []
        channel_batched = []

        # -----------------------------------------------------------
        # Traffic light rendering
        # -----------------------------------------------------------
        traffic_lights = self._actors.filter('*traffic_light*')
        for traffic_light in traffic_lights:
            trigger_box_global_pos = traffic_light.get_transform().transform(traffic_light.trigger_volume.location)
            trigger_box_global_pos = carla.Location(x=trigger_box_global_pos.x, y=trigger_box_global_pos.y, z=trigger_box_global_pos.z)
            if (trigger_box_global_pos.distance(vehicle_position) > self.light_radius):
                continue
            ego_pos_batched.append(ego_pos_list)
            ego_yaw_batched.append(ego_yaw_list)
            pos_batched.append([traffic_light.get_transform().location.x, traffic_light.get_transform().location.y])
            yaw_batched.append([traffic_light.get_transform().rotation.yaw/180*np.pi])
            template_batched.append(np.ones([4, 4]))
            if str(traffic_light.state) == 'Green':
                channel_batched.append(4)
            elif str(traffic_light.state) == 'Yellow':
                channel_batched.append(3)
            elif str(traffic_light.state) == 'Red':
                channel_batched.append(2)

        if len(ego_pos_batched)>0:
            ego_pos_batched_torch = torch.tensor(ego_pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            ego_yaw_batched_torch = torch.tensor(ego_yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            pos_batched_torch = torch.tensor(pos_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            yaw_batched_torch = torch.tensor(yaw_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            template_batched_torch = torch.tensor(template_batched, device='cuda', dtype=torch.float32).unsqueeze(1)
            channel_batched_torch = torch.tensor(channel_batched, device='cuda', dtype=torch.int)

            self.renderer.render_agent_bv_batched(
                birdview,
                ego_pos_batched_torch,
                ego_yaw_batched_torch,
                template_batched_torch,
                pos_batched_torch,
                yaw_batched_torch,
                channel=channel_batched_torch,
            )

        return birdview

    def _change_seg_tl(self, seg_img, depth_img, tl, _region_size=4):
        """Adds 3 traffic light classes (green, yellow, red) to the segmentation image
        Args:
            seg_img ([type]): [description]
            depth_img ([type]): [description]
            traffic_lights ([type]): [description]
            _region_size (int, optional): [description]. Defaults to 4.
        """
        if tl is not None:
            _dist = self._get_distance_from_camera(tl.get_transform().location)
            _region = np.abs(depth_img - _dist)

            if tl.get_state() == carla.TrafficLightState.Red:
                state = 23
            elif tl.get_state() == carla.TrafficLightState.Yellow:
                state = 24
            else: # do not change class
                state = 18

            seg_img[(_region < _region_size) & (seg_img == 18)] = state

    def _get_distance_from_camera(self, target):
        """Returns the distance from the (rgb) camera to the target
        Args:
            target ([type]): [description]
        Returns:
            [type]: [description]
        """        
        sensor_transform = self._sensors['rgb_front'].get_transform()

        distance = np.sqrt(
                (sensor_transform.location.x - target.x) ** 2 +
                (sensor_transform.location.y - target.y) ** 2 +
                (sensor_transform.location.z - target.z) ** 2)

        return distance

    def _get_depth(self, data):
        """Transforms the depth image into meters
        Args:
            data ([type]): [description]
        Returns:
            [type]: [description]
        """        

        data = data.astype(np.float32)

        normalized = np.dot(data, [65536.0, 256.0, 1.0]) 
        normalized /=  (256 * 256 * 256 - 1)
        in_meters = 1000 * normalized

        return in_meters
