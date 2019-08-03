# coding=utf-8
"""Class name to id mappings."""

# for using a COCO model to finetuning with DIVA data.
targetClass1to1 = {
    "Vehicle":"car",
    "Person":"person",
    "Parking_Meter":"parking meter",
    "Tree":"potted plant",
    "Other":None,
    "Trees":"potted plant",
    "Construction_Barrier":None,
    "Door":None,
    "Dumpster":None,
    "Push_Pulled_Object":"suitcase", # should be a dolly
    "Construction_Vehicle":"truck",
    "Prop":"handbag",
    "Bike":'bicycle',
    "Animal":'dog',
    "Articulated_Infrastructure":None,
}
""" # the number of bbox in DIVA training set
Vehicle:3618465
Person:809684
Tree:686551
Other:303851
Trees:227180
Construction_Barrier:146712
Parking_Meter:103627
Door:81362
Dumpster:68441
Push_Pulled_Object:44788
Construction_Vehicle:16981
Prop:14441
Bike:13031
Animal:4055
"""

targetClass2id = {
    "BG":0,
    "Vehicle":1,
    "Person":2,
    "Parking_Meter":3,
    "Tree":4,
    "Other":5,
    "Trees":6,
    "Construction_Barrier":7,
    "Door":8,
    "Dumpster":9,
    "Push_Pulled_Object":10,
    "Construction_Vehicle":11,
    "Prop":12,
    "Bike":13,
    "Animal":14,
    "Articulated_Infrastructure":15,
}

targetClass2id_new = {
    "BG":0,
    "Vehicle":1,
    "Person":2,
    "Parking_Meter":3,
    "Tree":4,
    "Skateboard":5,
    "Prop_Overshoulder":6,
    "Construction_Barrier":7,
    "Door":8,
    "Dumpster":9,
    "Push_Pulled_Object":10,
    "Construction_Vehicle":11,
    "Prop":12,
    "Bike":13,
    "Animal":14,
    # person-object classes
    "Bike_Person":15,
    "Prop_Person": 16,
    "Skateboard_Person": 17,
    'Prop_Overshoulder_Person': 18
}

targetClass2id_new_nopo = {
    "BG":0,
    "Vehicle":1,
    "Person":2,
    "Parking_Meter":3,
    "Tree":4,
    "Skateboard":5,
    "Prop_Overshoulder":6,
    "Construction_Barrier":7,
    "Door":8,
    "Dumpster":9,
    "Push_Pulled_Object":10,
    "Construction_Vehicle":11,
    "Prop":12,
    "Bike":13,
    "Animal":14,
}

targetClass2id_mergeProp = {
    "BG":0,
    "Vehicle":1,
    "Person":2,
    "Parking_Meter":3,
    "Tree":4,
    "Other":5,
    "Trees":6,
    "Construction_Barrier":7,
    "Door":8,
    "Dumpster":9,
    "Push_Pulled_Object":10,
    "Construction_Vehicle":11,
    "Prop":12,
    "Bike":13,
    "Animal":14,
    "Articulated_Infrastructure":15,
    "Prop_plus_Push_Pulled_Object":16,
}
# original ratio [h/w] (0.5,1.0,2.0)
# tall: (1.0,2.0,3.0,4.0)
targetClass2id_tall = {
    "BG":0,
    "Person":1,
    "Parking_Meter":2,
    "Tree":3,
    "Other":4,
    "Trees":5,
    "Door":6,
}
# wide: (0.3,0.6,1.0,1.5)
targetClass2id_wide = {
    "BG":0,
    "Vehicle":1,
    "Prop":2,
    "Push_Pulled_Object":3,
    "Bike":4,
    "Construction_Barrier":5,
    "Dumpster":6,
    "Construction_Vehicle":7,
    "Animal":8,
}
# wide: (0.3,0.6,1.0,1.5)
targetClass2id_wide_v2 = {
    "BG":0,
    "Prop":1,
    "Push_Pulled_Object":2,
    "Bike":3,
    "Construction_Barrier":4,
    "Dumpster":5,
    "Construction_Vehicle":6,
    "Animal":7,
}

targetAct2id = {
    "BG":0,
    "activity_walking": 1,
    "vehicle_moving": 2,
    "activity_standing": 3,
    "vehicle_stopping": 4,
    "activity_carrying": 5,
    "vehicle_starting": 6,
    "vehicle_turning_right": 7,
    "vehicle_turning_left": 8,
    "activity_gesturing": 9,
    "Closing": 10,
    "Opening": 11,
    "Interacts": 12,
    "Exiting": 13,
    "Entering": 14, # 3, 0.014
    "Talking": 15, # (4, '0.045'), (3, '0.224')
    "Transport_HeavyCarry": 16, # (3, '0.156')
    "Unloading": 17, # (4, '0.250'), (2, '0.273'), (3, '0.477')
    "Pull": 18,
    "Loading": 19, # (4, '0.132'), (2, '0.342'), (3, '0.526')
    "Open_Trunk": 20, # (3, '0.114')
    "Closing_Trunk": 21, # (3, '0.194')
    "Riding": 22,
    "specialized_texting_phone": 23,
    "Person_Person_Interaction": 24,
    "specialized_talking_phone": 25,
    "activity_running": 26,
    #"specialized_miscellaneous": 0,
    "vehicle_u_turn": 27,
    "PickUp": 28, # [(3, '0.364'), (2, '0.636')]
    "specialized_using_tool": 29,
    #"SetDown": 2, # [(4, '0.100'), (3, '0.400'), (2, '0.500')]
    "activity_crouching": 30,
    "activity_sitting": 31,
    "Object_Transfer": 32, #[(2, '0.375'), (3, '0.625')]
    "Push": 33,
    "PickUp_Person_Vehicle": 34,
    #"Misc": 0,
    "DropOff_Person_Vehicle": 35,
    #"Drop": 0,
    #"specialized_umbrella":0,
}

targetAct2id_wide = {
    "BG":0,
    #"vehicle_moving": 2,
    #"vehicle_stopping": 4,
    #"vehicle_starting": 6,
    "vehicle_turning_right": 1,
    "vehicle_turning_left": 2,
    "Closing": 3,
    "Opening": 4,
    "Interacts": 5,
    "Exiting": 6,
    "Entering": 7, # 3, 0.014
    "Talking": 8, # (4, '0.045'), (3, '0.224')
    "Unloading": 9, # (4, '0.250'), (2, '0.273'), (3, '0.477')
    "Pull": 10,
    "Loading": 11, # (4, '0.132'), (2, '0.342'), (3, '0.526')
    "Open_Trunk": 12, # (3, '0.114')
    "Closing_Trunk": 13, # (3, '0.194')
    "Person_Person_Interaction": 14,
    "vehicle_u_turn": 15,
    "specialized_using_tool": 16,
    "activity_crouching": 17,
    "activity_sitting": 18,
    "Object_Transfer": 19, #[(2, '0.375'), (3, '0.625')]
    "Push": 20,
    "PickUp_Person_Vehicle": 21,
    "DropOff_Person_Vehicle": 22,
}
targetAct2id_tall = {
    "BG":0,
    "activity_walking": 1,
    #"activity_standing": 3,
    "activity_carrying": 2,
    "activity_gesturing": 3,
    "Transport_HeavyCarry": 4, # (3, '0.156')
    "Riding": 5,
    "specialized_texting_phone": 6,
    "specialized_talking_phone": 7,
    "activity_running": 8,
    "PickUp": 9, # [(3, '0.364'), (2, '0.636')]
}

# for single box act, here we use all
targetSingleAct2id = {
    "BG":0,
    "activity_walking": 1,
    "vehicle_moving": 2,
    "activity_standing": 3,
    "vehicle_stopping": 4,
    "activity_carrying": 5,
    "vehicle_starting": 6,
    "vehicle_turning_right": 7,
    "vehicle_turning_left": 8,
    "activity_gesturing": 9,
    "Closing": 10,
    "Opening": 11,
    "Interacts": 12,
    "Exiting": 13,
    "Entering": 14, # 3, 0.014
    "Talking": 15, # (4, '0.045'), (3, '0.224')
    "Transport_HeavyCarry": 16, # (3, '0.156')
    "Unloading": 17, # (4, '0.250'), (2, '0.273'), (3, '0.477')
    "Pull": 18,
    "Loading": 19, # (4, '0.132'), (2, '0.342'), (3, '0.526')
    "Open_Trunk": 20, # (3, '0.114')
    "Closing_Trunk": 21, # (3, '0.194')
    "Riding": 22,
    "specialized_texting_phone": 23,
    "Person_Person_Interaction": 24,
    "specialized_talking_phone": 25,
    "activity_running": 26,
    #"specialized_miscellaneous": 0,
    "vehicle_u_turn": 27,
    "PickUp": 28, # [(3, '0.364'), (2, '0.636')]
    "specialized_using_tool": 29,
    #"SetDown": 2, # [(4, '0.100'), (3, '0.400'), (2, '0.500')]
    "activity_crouching": 30,
    "activity_sitting": 31,
    "Object_Transfer": 32, #[(2, '0.375'), (3, '0.625')]
    "Push": 33,
    "PickUp_Person_Vehicle": 34,
    #"Misc": 0,
    "DropOff_Person_Vehicle": 35,
    #"Drop": 0,
    #"specialized_umbrella":0,
}

targetPairAct2id = {
    "BG":0,
    "Closing": 1,
    "Opening": 2,
    "Interacts": 3,
    "Exiting": 4,
    "Entering": 5, # 3, 0.014
    "Talking": 6, # (4, '0.045'), (3, '0.224')
    "Transport_HeavyCarry": 7, # (3, '0.156')
    "Unloading": 8, # (4, '0.250'), (2, '0.273'), (3, '0.477')
    "Pull": 9,
    "Loading": 10, # (4, '0.132'), (2, '0.342'), (3, '0.526')
    "Open_Trunk": 11, # (3, '0.114')
    "Closing_Trunk": 12, # (3, '0.194')
    "Riding": 13,
    "Person_Person_Interaction": 14,
    "PickUp": 15, # [(3, '0.364'), (2, '0.636')]
    "SetDown": 16, # [(4, '0.100'), (3, '0.400'), (2, '0.500')]
    "Object_Transfer": 17, #[(2, '0.375'), (3, '0.625')]
    "Push": 18,
    "PickUp_Person_Vehicle": 19,
    "DropOff_Person_Vehicle": 20,
}

targetAct2id_19 = {
    "BG":0,
    "activity_carrying": 1,
    "vehicle_turning_right": 2,
    "vehicle_turning_left": 3,
    "Closing": 4,
    "Opening": 5,
    "Interacts": 6,
    "Exiting": 7,
    "Entering": 8, # 3, 0.014
    "Talking": 9, # (4, '0.045'), (3, '0.224')
    "Transport_HeavyCarry": 10, # (3, '0.156')
    "Unloading": 11, # (4, '0.250'), (2, '0.273'), (3, '0.477')
    "Pull": 12,
    "Loading": 13, # (4, '0.132'), (2, '0.342'), (3, '0.526')
    "Open_Trunk": 14, # (3, '0.114')
    "Closing_Trunk": 15, # (3, '0.194')
    "Riding": 16,
    "specialized_texting_phone": 17,
    "specialized_talking_phone": 18,
    "vehicle_u_turn": 19,
}

# ---------------- 07/2019, like BUPT

"""

Person-Vehicle
    "Closing"
    "Opening"
    "Exiting"
    "Entering"
    "Unloading"
    "Loading"
    "Open_Trunk"
    "Closing_Trunk"

Car-Turning:
    "vehicle_turning_right"
    "vehicle_turning_left"
    "vehicle_u_turn"

person-only:
    "activity_carrying"
    "Transport_HeavyCarry"
    "Talking"
    "Pull"
    "Riding"
    "specialized_texting_phone"
    "specialized_talking_phone"

"""

targetAct2id_bupt = {
    "BG":0,
    "Person-Vehicle": 1,

    "Vehicle-Turning": 2,

    "activity_carrying": 3,
    "Transport_HeavyCarry": 4,
    "Talking": 5,
    "Pull": 6,
    "Riding": 7,
    "specialized_texting_phone": 8,
    "specialized_talking_phone": 9,
}

bupt_act_mapping = {
    "Closing": "Person-Vehicle",
    "Opening": "Person-Vehicle",
    "Exiting": "Person-Vehicle",
    "Entering": "Person-Vehicle",
    "Unloading": "Person-Vehicle",
    "Loading": "Person-Vehicle",
    "Open_Trunk": "Person-Vehicle",
    "Closing_Trunk": "Person-Vehicle",

    "vehicle_turning_right": "Vehicle-Turning",
    "vehicle_turning_left": "Vehicle-Turning",
    "vehicle_u_turn": "Vehicle-Turning"
}

targetAct2id_meva = {
    "BG":0,
    "Person-Vehicle": 1,
    "Person-Structure": 2,
    "Vehicle-Turning": 3,

    "Person_Heavy_Carry": 4,
    "People_Talking": 5,
    "Riding": 6,
    "Person_Texting_on_Phone": 7,
    "Person_Talking_on_Phone": 8,
    "Person_Sitting_Down": 9,
    "Person_Sets_Down_Object": 10,
    "Person_Standing_Up": 11,
    "Person_Picks_Up_Object": 12,
    "Person_Purchasing": 13,
    "Person_Reading_Document": 14,
    "Object_Transfer": 15,
    "Hand_Interaction": 16,
    "Person-Person_Embrace": 17,
    "Person-Laptop_Interaction": 18,

    "Vehicle_Stopping": 19,
    "Vehicle_Starting": 20,
    "Vehicle_Reversing": 21,
}

meva_act_mapping = {
    "Person_Exits_Vehicle": "Person-Vehicle",
    "Person_Enters_Vehicle": "Person-Vehicle",
    "Person_Opens_Vehicle_Door": "Person-Vehicle",
    "Person_Closes_Vehicle_Door": "Person-Vehicle",
    "Vehicle_Drops_Off_Person": "Person-Vehicle",
    "Person_Unloads_Vehicle": "Person-Vehicle",
    "Person_Loads_Vehicle": "Person-Vehicle",
    "Person_Opens_Trunk": "Person-Vehicle",
    "Person_Closes_Trunk": "Person-Vehicle",
    "Vehicle_Picks_Up_Person": "Person-Vehicle",

    "Vehicle_Turning_Right": "Vehicle-Turning",
    "Vehicle_Turning_Left": "Vehicle-Turning",
    "Vehicle_U-Turn": "Vehicle-Turning",

    "Person_Enters_Through_Structure": "Person-Structure",
    "Person_Exits_Through_Structure": "Person-Structure",
    "Person_Opens_Facility_Door": "Person-Structure",
    "Person_Closes_Facility_Door": "Person-Structure",
}

# 81 classes
coco_obj_classes = [
    "BG",
    "person",  # upper case to be compatable with actev class
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
coco_obj_class_to_id = {
    coco_obj_classes[i]: i for i in xrange(len(coco_obj_classes))}
coco_obj_id_to_class = {
    coco_obj_class_to_id[o]: o for o in coco_obj_class_to_id}

coco_obj_to_actev_obj = {
    "person": "Person",
    "car": "Vehicle",
    "bus": "Vehicle",
    "truck": "Vehicle",
    # "motorcycle": "Vehicle",
    "bicycle": "Bike",
}
