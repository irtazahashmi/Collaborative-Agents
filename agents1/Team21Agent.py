import itertools
import sys
import random  # type: ignore
from typing import Dict
import numpy as np  # type: ignore
from matrx.actions import MoveNorth, OpenDoorAction, Action, DropObject, GrabObject  # type: ignore
from matrx.actions.move_actions import MoveEast, MoveSouth, MoveWest  # type: ignore
from matrx.agents import StateTracker, Navigator
from matrx.agents.agent_utils.state import State  # type: ignore
from matrx.messages import Message
from matrx.objects import AreaTile

from bw4t.BW4TBrain import BW4TBrain


class RandomAgent(BW4TBrain):
    """
    This agent makes random walks and opens any doors it hits upon
    """

    def __init__(self, settings: Dict[str, object]):
        super().__init__(settings)
        self._moves = [MoveNorth.__name__, MoveEast.__name__, MoveSouth.__name__, MoveWest.__name__]
        self.required_blocks = []  # The blocks we want to drop at the drop off location
        self.useful_objects = []  # To keep the track of possible deliverable blocks
        self.is_selected_list = []  # The possible rooms that are selected for traversing
        self.roomAssigned = None  # The current room that is assigned and needs to be traversed
        self.state_tracker = None
        self.navigator = None
        self.currently_on_the_move_to_a_room = False
        self.phase = 'mapping'  # Different Phases -> 'mapping', 'assigning', 'fetching', 'delivering'
        self.currently_traversing_room = False
        self.previous_action = None
        self.message_queue_size = 0
        self.assigned_rooms_tile_list = None  # Coordinates of the assigned rooms
        self.going_somewhere = False
        self.just_opened_door = False
        self.agent_id_and_location = None
        self.assigned_block = None  # The next block that needs to be collected
        self.assignable_blocks = []  # Blocks that can be assigned
        self.collectable_info = []  # Block id and information about collectable blocks
        self.assignable_block_type_list = []  # List of assignable block types
        self.assignable_block_type = None  # Shape, color, drop off location about the required block
        self.required_blocks_list_flag = False  # checking condition
        self.no_conflict_flag = False  # checking condition
        self.drop_location = None  # the location to drop off the assigned block
        self.stop_initializing_blocks = False  # checking condition
        self.tie_breaker_value = np.random.randint(10000000)  # Used for conceding in a conflict
        self.tie_breaker = False  # Flag for conceding in a conflict
        self.tie_breaker_counter = 0  # checking condition
        self.update_useful_list_flag = True  # checking condition
        self.handicapped = (settings.get("colorblind") is not None
                            and settings.get("colorblind") is not False) \
                           or (settings.get("shapeblind") is not None
                               and settings.get("shapeblind") is not False)

    def initialize(self):
        super().initialize()
        self._door_range = 1
        # Init state tracker
        self.state_tracker = StateTracker(agent_id=self.agent_id)
        # init navigator
        self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                   algorithm=Navigator.A_STAR_ALGORITHM)
        # send message for the tie breaker
        self.send_message_tie_breaker()

    def filter_bw4t_observations(self, state) -> State:
        self.update_agent_location(state)
        self.state_tracker.update(state)

        # Before choosing the next action , filter the objects around the agent
        if self.phase == 'mapping':
            self.get_objects_around_me(state)

        # As soon as you filter the objects, parse the messages that the agent received
        if len(self.received_messages) > self.message_queue_size:  # if list is not empty
            # Assumptions:
            # 1. All agents have the same message order
            # 2. Received message list keeps past messages
            for index in range(self.message_queue_size, len(self.received_messages)):
                current_message = self.received_messages[
                    index]  # TODO Make sure that this is the contents and not a message
                self.on_message_received(current_message)

            self.message_queue_size = len(self.received_messages)

        # On the first round, init useful and assignable block list
        if self.check_required_blocks_features() and not self.required_blocks_list_flag:
            self.initialize_useful_list()
            self.initialize_assignable_block_types_list()
            self.required_blocks_list_flag = True

        return state

    def is_empty_list(self, l) -> bool:
        if not l:
            return True
        else:
            return False

    def decide_on_bw4t_action(self, state: State):
        params = {}
        act = None

        # Required_blocks represent desirable blocks and is_selected_list represent rooms
        if not self.required_blocks and not self.is_selected_list:
            self.initialize_rooms(state)
            self.initialize_objects(state)

        # Making sure there are no conflicts in tie breaker
        if self.tie_breaker and self.tie_breaker_counter <= 15:
            self.send_message_tie_breaker()
            self.tie_breaker_counter += 1

        # Updating color and shape in useful objects list from required blocks
        if self.check_required_blocks_features() and self.update_useful_list_flag:
            for obj in self.useful_objects:
                for block in self.required_blocks:
                    colour = block.colour
                    blockshape = block.shape
                    if obj.colour is None and obj.shape is not None:
                        obj.colour = colour
                    if obj.shape is None and obj.colour is not None:
                        obj.shape = blockshape

            self.update_useful_list_flag = False

        # Phase for finding possible objects for the drop off zone
        if self.phase == 'mapping':
            if self.roomAssigned is None:  # Will enter this loop only until it picks its first move
                available = self.pick_a_room()  # Case that we dont have a room yet

                # Switching from mapping phase to assigning next required block if all rooms traversed
                if not available:
                    self.phase = 'assigning'
                return act, {}
            else:  # If agent has a room assigned to him
                if self.currently_on_the_move_to_a_room:
                    if self.currently_traversing_room:
                        act = self.continue_traversing_room(state)
                        if act is None:
                            # If act is None means that agent is done traversing
                            # So exit room -> Find a new room and if there is no
                            # Then it will go to fetching state
                            self.exit_room()

                    else:
                        doors_nearby, door_location, _id = self.open_doors(state)

                        try:
                            points = self.navigator.get_all_waypoints()[-1][1]
                        except:
                            points = (-1, -1)

                        # Open the door when near of the door
                        if doors_nearby and not self.just_opened_door and door_location == points:
                            act = OpenDoorAction.__name__
                            params['object_id'] = _id
                            self.currently_on_the_move_to_a_room = False
                            self.just_opened_door = True
                            self.navigator.reset_full()

                        elif self.handicapped and door_location == points and not self.just_opened_door:
                            self.currently_on_the_move_to_a_room = False
                            self.just_opened_door = True
                            self.navigator.reset_full()
                        elif door_location == points and not self.just_opened_door:
                            self.currently_on_the_move_to_a_room = False
                            self.just_opened_door = True
                            self.navigator.reset_full()
                        else:
                            # Case that we decided on a room on already
                            act = self.continue_on_planned_path(state)

                            if act is None and self.just_opened_door:
                                self.just_opened_door = False
                                act = self.start_exploring_room(state)
                else:
                    # Case that we are still trying to get to the room
                    act = self.go_to_room(state)

                self.previous_action = act
                return act, params

        # Assigning a required block to the agent if there is no tie
        elif self.phase == 'assigning' and self.tie_breaker:

            # Initializing assignable block list
            if not self.stop_initializing_blocks:
                self.initialize_assignable_blocks_list()
                self.stop_initializing_blocks = True

            # Finding a block to assign
            self.pick_a_block_type()
            # Finding the nearest block to the agent that was assigned
            self.find_closest_matching_block()

            # Adding the location of the assigned block
            self.navigator.reset_full()
            self.navigator.add_waypoints([self.assigned_block.location])

            # Switching phase to fetching -> going to fetch the block and going to drop off location
            self.phase = 'fetching'
            return None, {}

        elif self.phase == 'fetching' and self.tie_breaker:

            # If agent is at the block -> Grab it
            if self.agent_id_and_location[1] == self.assigned_block.location \
                    and not self.assigned_block.has_been_collected:
                act = GrabObject.__name__, {'object_id': self.assigned_block.block_id}
                self.assigned_block.has_been_collected = True
                return act

            # If the block has not been collected yet -> Go to the assigned block location
            if not self.assigned_block.has_been_collected:
                act = self.go_to_assigned_block(state)
                return act, {}

            # If the block is grabbed, go to drop off location
            if self.agent_id_and_location[1] == self.assigned_block.location \
                    and self.assigned_block.has_been_collected:
                act = self.go_to_drop_off_location()
                self.phase = 'delivering'
                return act, {}

            return None, {}

        # Delivering the block
        elif self.phase == 'delivering' and self.tie_breaker:

            # If location of agent is at the drop off location -> drop the object
            if self.agent_id_and_location[1] == self.drop_location:
                act = DropObject.__name__, {'object_id': self.assigned_block.block_id}

                # Updating that the block has been delivered so it is not assigned again
                for block in self.required_blocks:
                    if block.location == self.drop_location:
                        block.has_been_delivered = True

                self.assigned_block.location = self.drop_location
                self.assigned_block.has_been_delivered = True

                self.phase = 'assigning'
                return act
            else:
                # If agent is on the way to destination, keep going
                act = self.navigator.get_move_action(self.state_tracker)
                return act, {}
        else:
            # If agent is not a tie breaker, do nothing
            return None, {}

    # Finding the drop off location of a block with a certain colour and shape
    def find_drop_off_location(self, shape, colour):
        for block in self.required_blocks:

            if shape == block.shape and colour == block.colour and \
                    not block.has_been_delivered:
                return block.location

            if shape == block.shape and colour == block.colour and block.has_been_delivered:
                return block.location

    def on_message_received(self, message: dict) -> None:
        """
        This method should be called every time a message is received

        Currently handles messages of type: 

        <*> pick_rooms
        <*> object_found
        <*> tie_breaker
        <*> update_block_features
        <*> assign_block_type
        """
        action = message.get('action')
        agent_id = message.get('agent_id')
        handicaped = message.get('handicaped')

        if action == 'pick_rooms':
            room_id = message.get('room_id')
            if str(agent_id) == str(self.agent_id):
                specific_room = None
                for room in self.is_selected_list:
                    if room[0] == room_id:
                        specific_room = room
                        break
                if specific_room is not None and specific_room[1]:
                    # If that specific room is already selected then pick a new one
                    self.pick_a_room()
                else:  # ELse assign it to urself
                    self.roomAssigned = room_id
                    self.update_is_selected_list_room_to_true(str(room_id))
            else:
                if not handicaped and not self.handicapped:
                    self.update_is_selected_list_room_to_true(str(room_id))

        elif action == 'object_found':
            # For sharing block info with all agents in order to find matching blocks
            _object = message.get('object')
            location = (_object.get('x_coord'), _object.get('y_coord'))
            shape = _object.get('shape')
            colour = _object.get('colour')
            block_id = _object.get('block_id')
            self.add_object_to_useful_objects(block_id, location, shape, colour)

        elif action == "tie_breaker":
            # For avoiding assigning and delivering conflicts
            sender = message.get('agent_id')
            value = message.get('tie_breaker_value')

            number_of_agents = len([s for s in self.state.keys() if 'agent' in s])
            if number_of_agents == 1:
                self.tie_breaker = True

            if sender != self.agent_id:
                if value > self.tie_breaker_value:
                    self.tie_breaker = False
                if value < self.tie_breaker_value:
                    self.tie_breaker = True

        elif action == "update_block_features":
            # Update disabled agents about block features
            block_location = message.get('block_location')
            shape = message.get('shape')
            colour = message.get('colour')
            sender = message.get('agent_id')

            if sender != self.agent_id:
                for block in self.required_blocks:
                    if block.location == block_location:
                        if block.shape is None and shape is not None:
                            block.shape = shape
                        if block.colour is None and colour is not None:
                            block.colour = colour

        elif action == 'assign_block_type':
            # Avoid conflicts when assigning block types
            sender = message.get('agent_id')
            _object = message.get('object')
            block_type_id = _object.get('block_type_id')
            location = _object.get('location')
            has_been_picked = _object.get('has_been_picked')

            self.phase = 'fetching'

            number_of_agents = len([s for s in self.state.keys() if 'agent' in s])
            if number_of_agents == 1:
                self.no_conflict_flag = True
                self.phase = 'fetching'

            if self.assignable_block_type is None and sender != self.agent_id:
                for block_type in self.assignable_block_type_list:
                    if block_type.location == location:
                        block_type.has_been_picked = True
                self.no_conflict_flag = True
                self.phase = 'assigning'

            if sender != self.agent_id and self.assignable_block_type is not None:
                # Handles conflict
                if has_been_picked:
                    if self.assignable_block_type.block_type_id == block_type_id:
                        for block_type in self.assignable_block_type_list:
                            if block_type.block_type_id == self.assignable_block_type.block_type_id:
                                block_type.has_been_picked = False
                                self.pick_a_block_type()
                    for block_type in self.assignable_block_type_list:
                        if block_type.block_type_id == block_type_id:
                            block_type.has_been_picked = True
                else:
                    # No conflict
                    self.no_conflict_flag = True
        else:
            pass

    # Check if color and shape are known
    def check_required_blocks_features(self) -> bool:
        if self.is_empty_list(self.required_blocks):
            return False

        for block in self.required_blocks:
            if block.shape is None or block.colour is None:
                return False

        return True

    def add_object_to_useful_objects(self, block_id, location, shape, colour) -> None:
        """
        Main logic of useful objects 

        If an object is exactly what we want, then it is added in the locations of that object

        If it is partially what we want then :

             Tries to find if it complements a previous finding and updates it
                or
             Creates a new partial finding for someone else to complement

        If the object is not useful it is discarded

        """
        # if agent is not handicapped
        if colour is not None and shape is not None:
            for i, block in enumerate(self.useful_objects):
                if block.colour == colour and block.shape == shape:
                    block.location.add((block_id, location))
                    self.required_blocks[i].has_been_observed = True

        elif shape is None and colour is None:
            # Probably some error occurred. Shouldn't happen.
            return

        # shape blind
        elif shape is None:
            # 3 options
            # 1. Color is useless
            # 2. color is probably useful but never observed.
            # 3. This object has been observed but we only know the shape

            useful = False
            for block in self.required_blocks:
                if block.colour == colour:
                    useful = True
            if not useful:
                return
            # Check if there is any object in a similar location missing its colour
            same_colour_index = -1
            for i in range(len(self.required_blocks), len(self.useful_objects)):
                block = self.useful_objects[i]
                if block.colour is None:
                    for loc_id, loc in block.location:
                        if loc == location:
                            block.location.remove((loc_id, loc))
                            self.add_object_to_useful_objects(block_id, location, block.shape,
                                                              colour)
                            break

                if block.colour == colour:
                    same_colour_index = i

            if same_colour_index != -1:
                self.useful_objects[same_colour_index].location.add((block_id, location))
            else:
                self.useful_objects.append(ObjectFound(colour, None, (block_id, location)))

        elif colour is None:
            # 3 options
            # 1. shape is useless
            # 2. shape is probably usefull but never observed.
            # 3. This object has been observed but we only know the colour
            # 
            useful = False
            for block in self.required_blocks:
                if block.shape == shape:
                    useful = True
            if not useful:
                return
            # Check if there is any object in a similar location missing its shape
            same_shape_index = -1
            for i in range(len(self.required_blocks), len(self.useful_objects)):
                block = self.useful_objects[i]
                if block.shape is None:
                    for loc_id, loc in block.location:
                        if loc == location:
                            block.location.remove((loc_id, loc))
                            self.add_object_to_useful_objects(block_id, location, shape,
                                                              block.colour)
                            break

                if block.shape == shape:
                    same_shape_index = i

            if same_shape_index != -1:
                self.useful_objects[same_shape_index].location.add((block_id, location))
            else:
                self.useful_objects.append(ObjectFound(None, shape, (block_id, location)))
        else:
            return

        # Check if we have at least one of each required object in order
        # to move to the fetching state
        for block in self.required_blocks:
            if not block.has_been_observed:
                return

    def initialize_useful_list(self):
        """
        Copies the required blocks into the useful list
        """
        for block in self.required_blocks:
            object_found = ObjectFound(block.colour, block.shape)
            self.useful_objects.append(object_found)

    def initialize_assignable_blocks_list(self) -> None:
        """
        Initializing assignable blocks
        """
        for obj in self.useful_objects:
            col = obj.colour
            shape = obj.shape

            for loc_id, loc in obj.location:
                self.assignable_blocks.append(
                    RequiredBlock(loc, col, shape, True, False, False, loc_id, False))

    def initialize_assignable_block_types_list(self) -> None:
        """
        Converting required blocks to assignable block type object
        """
        counter = 0
        has_been_picked = False
        for block in self.required_blocks:
            self.assignable_block_type_list.append(
                AssignableBlockType(counter, block.shape, block.colour, block.location,
                                    has_been_picked))
            counter += 1

    def pick_a_block_type(self) -> bool:
        """
        Assign a block shape and colour to the agent
        """
        counter = 0
        size = len(self.assignable_block_type_list)

        for block in self.assignable_block_type_list:
            if block.has_been_picked:
                counter += 1

        if counter == size:
            return False

        max_loc_y = -1
        max_block_type = None
        for block_type in self.assignable_block_type_list:
            if not block_type.has_been_picked:
                curr_loc_y = block_type.location[1]
                if curr_loc_y > max_loc_y:
                    max_block_type = block_type
                    max_loc_y = curr_loc_y

        self.assignable_block_type = max_block_type
        self.assignable_block_type.has_been_picked = True
        self.send_message_to_assign_block_type(self.assignable_block_type)
        return True

    def find_closest_matching_block(self) -> None:
        """
        Finding the closest matching block from the agent
        """
        colour = self.assignable_block_type.colour
        shape = self.assignable_block_type.shape

        min_dist = sys.maxsize
        closest_block = None

        for block in self.assignable_blocks:
            if not block.has_been_collected:
                if shape == block.shape and colour == block.colour:
                    dist = self.euclidean(self.agent_id_and_location[1], block.location)
                    if dist < min_dist:
                        min_dist = dist
                        closest_block = block

        self.assigned_block = closest_block
        self.assigned_block.has_been_assigned = True

        # Updating the assignable block list
        for block in self.assignable_blocks:
            if self.assigned_block.shape == block.shape and self.assigned_block.colour == block.colour:
                block.has_been_assigned = True

    def pick_a_room(self) -> bool:
        """
        Find a random available room that is available and broadcast that you want it.
        """
        choices = []
        for room in self.is_selected_list:
            if not room[1]:
                choices.append(room[0])
        if not choices:
            return False
        self.send_message_to_get_room(random.choice(choices))
        return True

    def go_to_room(self, state: State) -> Action:
        """
        This method should be called once for each room that the agent wants
        to visit

        Will initialize the Tracker and Navigator in order to move
        """

        room_id = self.roomAssigned
        self.currently_on_the_move_to_a_room = True

        # Get room doors returns a list
        # of dicts (in case there are multiple doors) so get 0 (we assume only one door) and
        # get location then
        location = state.get_room_doors(room_id)[0].get('location')
        self.navigator.add_waypoint((location[0], location[1] + 1))  # Add one to y to go just below
        # the door, We assume that the door is always at the bottom of the room

        action = self.navigator.get_move_action(self.state_tracker)
        return action

    def exit_room(self):
        """
        Method that initiates exit from a room .

        Which basically is:

        Find a new room or enter the fetching state

        Purpose of this method is to clean the parameters basically
        """
        self.roomAssigned = None
        self.currently_on_the_move_to_a_room = False
        self.currently_traversing_room = False
        self.assigned_rooms_tile_list = None
        self.going_somewhere = False
        ans = self.pick_a_room()
        if not ans:
            self.phase = 'assigning'

    def continue_on_planned_path(self, state: State):
        """
        Returns the action to do in order 
        to continue down the road of going to the target that was already set
        """
        action = self.navigator.get_move_action(self.state_tracker)
        return action

    def start_exploring_room(self, state: State):
        """
        Method that initiates a room traversal 

        Create the queue of tiles that the agent will follow
        Basically all the room tiles (for now)

        Which the agent will consume one by one 

        The tiles are ordered in a 
        -------|
        |-------      
        -------|

        fashion but flattened 

        :return: The first action 
        """
        room_id = self.roomAssigned
        self.currently_traversing_room = True

        area_tiles = self.get_room_area_tiles(room_id, state)
        locations_of_area_tiles = [tile['location'] for tile in area_tiles]

        sorted_on_y = sorted(locations_of_area_tiles, key=lambda y: y[1])

        i = 0
        twoD_array = [[]]
        previous = sorted_on_y[0]
        twoD_array[0].append(previous)

        for j in range(1, len(sorted_on_y)):
            if sorted_on_y[j][1] == previous[1]:
                twoD_array[i].append(sorted_on_y[j])
            else:
                i += 1
                twoD_array.append([])
                twoD_array[i].append(sorted_on_y[j])

            previous = sorted_on_y[j]

        is_right = True

        for k in range(len(twoD_array)):
            if is_right:
                twoD_array[k] = sorted(twoD_array[k], key=lambda x: -x[0])
                # Assumed that minus only sorts on descending order (doesnt change value)
                is_right = False
            else:
                twoD_array[k] = sorted(twoD_array[k], key=lambda x: x[0])
                is_right = True

        final_list = list(itertools.chain.from_iterable(twoD_array))
        room_id = self.roomAssigned

        location_ = state.get_room_doors(room_id)[0]['location']
        final_list.append(location_)
        self.assigned_rooms_tile_list = final_list

        first_tile = self.assigned_rooms_tile_list.pop()

        self.navigator.reset_full()
        self.navigator.add_waypoint((first_tile[0], first_tile[1]))
        action = self.navigator.get_move_action(self.state_tracker)  # Debug this

        self.going_somewhere = True
        return action

    def get_room_area_tiles(self, room_name, state: State) -> list or None:
        """
        Method that will return all the tiles in a room
        """

        # Locate method to identify doors of the right room

        def is_content(obj):
            if 'class_inheritance' in obj.keys():
                chain = obj['class_inheritance']
                if AreaTile.__name__ in chain and obj['room_name'] == room_name:
                    return obj
            else:  # the object is not a Door
                return None

        room_objs = state.get_room(room_name)
        if room_objs is None:  # No room was found with the given room name
            return None

        # Filter out all doors
        area_tiles = map(is_content, room_objs)
        area_tiles = [c for c in area_tiles if c is not None]

        return area_tiles

    def continue_traversing_room(self, state: State):
        """
        Continue traversing the room that you are,

        if you find an object broadcast it

        :return: actions to do 
        """
        if self.going_somewhere:
            # Will return the first action points

            action = self.navigator.get_move_action(self.state_tracker)
            if action is None:
                self.going_somewhere = False
            else:
                self.going_somewhere = False
                return action

        if self.assigned_rooms_tile_list:
            location = self.assigned_rooms_tile_list.pop()
            # self.navigator.reset_full()
            self.navigator.reset_full()
            self.navigator.add_waypoint((location[0], location[1]))
            action = self.navigator.get_move_action(self.state_tracker)
            return action

    def broadcast_object(self, block_id, x_coord: int, y_coord: int, shape: int = None,
                         colour: str = None) -> None:
        """
        object_found = (x coord, y coord, shape_id, colour in hex)

        uses channel  `object_found`
        """

        dic = {
            "action": "object_found",
            "agent_id": self.agent_id,
            "object": {
                "block_id": block_id,
                "x_coord": x_coord,
                "y_coord": y_coord,
                "shape": shape,
                "colour": colour
            }
        }

        msg = Message(content=dic, from_id='me')
        self.send_message(msg)

    def send_message_to_get_room(self, room_id: str) -> None:
        """
        Broadcast that you want the room with room_id

        uses channel `pick_rooms`
        """

        dic = {
            "action": "pick_rooms",
            "room_id": room_id,
            "agent_id": self.agent_id,
            "handicaped": self.handicapped
        }

        msg = Message(content=dic, from_id='me')
        self.send_message(msg)

    def send_message_to_assign_block_type(self, assignable_type):
        """
        Broadcast that you want this type of block and its dropoff location
        uses channel `pick_rooms`
        """
        dic = {
            "action": "assign_block_type",
            "agent_id": self.agent_id,
            "object": {
                "type_id": assignable_type.block_type_id,
                "location": assignable_type.location,
                "shape": assignable_type.shape,
                "colour": assignable_type.colour,
                "has_been_picked": assignable_type.has_been_picked
            }
        }

        msg = Message(content=dic, from_id='me')
        self.send_message(msg)

    def initialize_rooms(self, state: State) -> None:
        """
        Initializing rooms
        """
        rooms = [s for s in state.get_all_room_names() if 'room' in s]
        selected = False

        for room in rooms:
            is_selected = [room, selected]
            self.is_selected_list.append(is_selected)

    def update_is_selected_list_room_to_true(self, room_id: str) -> None:
        """
        Method to update the is_selected_list attribute of a room to True because tuples are immutable
        """
        for index, room in enumerate(self.is_selected_list):
            if room[0] == room_id:
                self.is_selected_list[index] = (room_id, True)
                break

    def initialize_objects(self, state: State):
        """
        Initializing required blocks
        """
        drop_off_zones = [s for s in state.keys() if 'Drop_off' in s]
        collectables = [s for s in state.keys() if 'Collect_Block' in s]

        # Adding information relating to collectables. [Collectable, has_been_observed, has_been_collected]
        has_been_observed = False
        has_been_assigned = False
        has_been_collected = False

        # Creating list of desirable blocks, ordered
        for i in drop_off_zones:
            drop_off_location = state.get(i)['location']
            for j in collectables:
                collectable_location = state.get(j)['location']
                colour = None
                shape = None

                try:
                    colour = state.get(j)['visualization']['colour']
                except KeyError:
                    "colour blind agent!"

                try:
                    shape = state.get(j)['visualization']['shape']
                except KeyError:
                    "Shape blind agent!"

                if drop_off_location == collectable_location:
                    required_block = RequiredBlock(drop_off_location, colour, shape, has_been_observed,
                                                   has_been_assigned, has_been_collected, j, False)
                    self.required_blocks.append(required_block)

        for collectable in collectables:
            this_collectable = [collectable, has_been_observed, has_been_collected]
            self.collectable_info.append(this_collectable)

        # Broadcast current known block features
        for block in self.required_blocks:
            self.send_message_required_block_features(block.location, block.shape, block.colour)

        agent_id = self.agent_id
        agent_location = state.get(agent_id)['location']
        x = agent_location[0]
        y = agent_location[1]
        self.agent_id_and_location = [agent_id, (x, y)]

    def update_agent_location(self, state: State) -> None:
        """
        Updating the agent's location
        """
        agent_id = self.agent_id
        agent_location = state.get(agent_id)['location']
        x = agent_location[0]
        y = agent_location[1]
        self.agent_id_and_location = [agent_id, (x, y)]


    def go_to_assigned_block(self, state: State) -> Action:
        """
        Navigates to closest desirable block and picks it up
        """
        action = self.navigator.get_move_action(self.state_tracker)
        return action


    def go_to_drop_off_location(self) -> Action:
        """
        Navigates to drop zone and delivers the block
        """
        self.navigator.reset_full()
        self.drop_location = self.find_drop_off_location(self.assigned_block.shape,
                                                         self.assigned_block.colour)
        self.navigator.add_waypoints([self.drop_location])
        action = self.navigator.get_move_action(self.state_tracker)
        return action

    def _nearbyDoors(self, state: State):
        # copy from human agent
        # Get all doors from the perceived objects
        objects = list(state.keys())
        doors = [obj for obj in objects if 'is_open' in state[obj]]
        doors_in_range = []
        for object_id in doors:
            # Select range as just enough to grab that object
            dist = int(np.ceil(np.linalg.norm(
                np.array(state[object_id]['location']) - np.array(
                    state[self.agent_id]['location']))))
            if dist <= self._door_range:
                doors_in_range.append(object_id)
        return doors_in_range

    def send_message_tie_breaker(self):
        """
        Broadcasting tie_breaker_value to find a random agent, to determine who will do
        tiebreaker stuff
        """
        dic = {
            "action": "tie_breaker",
            "tie_breaker_value": self.tie_breaker_value,
            "agent_id": self.agent_id
        }
        msg = Message(content=dic, from_id='me')
        self.send_message(msg)

    def send_message_required_block_features(self, loc, shape, colour) -> None:
        """
        Broadcast the features of required blocks the agent has observed, to help other colorblind
        or shape-blind agents.
        """
        dic = {
            "action": "update_block_features",
            "block_location": loc,
            "shape": shape,
            "colour": colour,
            "agent_id": self.agent_id
        }
        msg = Message(content=dic, from_id='me')
        self.send_message(msg)

    def euclidean(self, p1, p2):
        """
        Finding the distance between two locations
        """
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_objects_around_me(self, state: State):
        """
        Method that looks for any collectable object near the agent
        If such an object exists, then it is broadcasted
        """
        closest_objects = state.get_closest_objects()

        for obj in closest_objects:
            class_inheritance = obj.get('class_inheritance')
            is_movable = obj.get('is_movable')
            if 'CollectableBlock' in class_inheritance:
                location = obj['location']
                shape = None
                colour = None
                try:
                    shape = obj['visualization']['shape']
                except KeyError:
                    pass
                try:
                    colour = obj['visualization']['colour']
                except KeyError:
                    pass
                self.broadcast_object(obj['obj_id'], location[0], location[1], shape, colour)

    def open_doors(self, state: State):
        """
        Method that checks if there is an available door close, and if yes it returns True,  its location and id
        Else it returns False, None, None
        """
        doors_nearby = False
        _id = None
        door_location = None
        for doorId in self._nearbyDoors(state):
            if not state[doorId]['is_open']:
                doors_nearby = True
                _id = doorId
                door_location = state[doorId]['location']
                door_location = (door_location[0], door_location[1] + 1)
            if state[doorId]['is_open']:
                door_location = state[doorId]['location']
                door_location = (door_location[0], door_location[1] + 1)
        return doors_nearby, door_location, _id


class ObjectFound:
    """
    A class to represent the objects that were found
    """
    def __init__(self, colour, shape, location=None):
        self.location = set()
        self.colour = colour
        self.shape = shape
        if location is not None:
            self.location.add(location)


class RequiredBlock:
    """
    A class to represent the objects that are required
    """
    def __init__(self, location, colour, shape, has_been_observed, has_been_assigned, has_been_collected, block_id,
                 has_been_delivered):
        self.location = location
        self.colour = colour
        self.shape = shape
        self.has_been_observed = has_been_observed
        self.has_been_assigned = has_been_assigned
        self.has_been_collected = has_been_collected
        self.block_id = block_id
        self.has_been_delivered = has_been_delivered


class AssignableBlockType:
    """
    A class to represent the objects that can be assigned to agents
    """
    def __init__(self, block_type_id, shape, colour, location, has_been_picked):
        self.block_type_id = block_type_id
        self.location = location
        self.colour = colour
        self.shape = shape
        self.has_been_picked = has_been_picked