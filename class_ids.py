# coding=utf-8
# 

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