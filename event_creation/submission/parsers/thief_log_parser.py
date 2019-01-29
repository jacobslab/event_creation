from .base_log_parser import BaseSessionLogParser#, UnknownExperimentError
from ..viewers.recarray import strip_accents, to_json, to_dict
import numpy as np
import os
from .. import fileutil
import argparse

class ThiefSessionLogParser(BaseSessionLogParser):

	@staticmethod
	def _thief_fields():
		"""
		Returns the template for a new th field
		:return:
		"""
		return (
			('condition','none','S32'),
			('env','nan','S32'),
			('stage','nan','S32'),
			('action','nan','S32'),
			('reward', 0, 'int16'),
			('whichroom','nan','S32'),
			('whichtraj','nan','S32'),
			('posX',-1,'float'),
			('posZ',-1,'float'),
			('temporal_event','nan','S32'),
			('music','nan','S32')
		)


	TH_RADIUS_SIZE = 13.0
	_MSTIME_INDEX = 0
	_TYPE_INDEX = 1

	def __init__(self, protocol, subject, montage, experiment, session, files):

		# create parsed log file treasure.par from original log file before BaseSessionLogParser
		# is initialized
		files['thief_par'] = os.path.join(os.path.dirname(files['session_log']), 'thief.par')
		# print(files['thief_par'])
		
		self.parse_raw_log(files['session_log'],files['thief_par'])
		super(ThiefSessionLogParser, self).__init__(protocol, subject, montage, experiment, session, files,
												 primary_log='thief_par',
												 include_stim_params=True, allow_unparsed_events=True)

		# remove msoffset field because it does not exist in the TH log
		self._fields = tuple([x for x in self._fields if x[0] != 'msoffset'])
		self._log_header = ''
		self._condition="none"
		self._env=None
		self._stage=None
		self._action=None
		self._reward = 0
		self._whichroom=None
		self._whichtraj=None
		self._posX=-1
		self._posZ=-1
		self._temporal_event=None
		self._music=None

		# self._version = ''
		# kind of hacky, 'type' is the second entry in the header
		self._add_fields(*self._thief_fields())
		self._add_type_to_new_event(
			condition=self.event_header,
			none=self.event_line, #removed because it causes the JSON file to be exponentially large
            reward_reval=self.event_line,
            trans_reval=self.event_line
		)
		# self._add_type_to_modify_events(
		# 	condition=self.modify_header
		# )

	# Any reason to not have all the fields persist in TH?
	@staticmethod
	def persist_fields_during_stim(event):
		return [field[0] for field in ThiefSessionLogParser._thief_fields()]


	@property
	def _empty_event(self):
		"""
		Overiding BaseSessionLogParser._empty_event because we don't have msoffset field
		"""
		# print("empty event")
		event = self.event_from_template(self._fields)
		event.protocol = self._protocol
		event.subject = self._subject
		event.montage = self._montage
		event.experiment = self._experiment
		event.session = self._session
		return event

	def event_default(self, split_line):
		"""
		Override base class's default event to TH specific events.
		"""
		# print("default event")
		event = self._empty_event
		event.mstime = int(split_line[self._MSTIME_INDEX])
		event.condition=self._condition
		event.env=self._env
		event.stage=self._stage
		event.action = self._action
		event.reward = self._reward
		event.whichroom=self._whichroom
		event.whichtraj = self._whichtraj
		event.posX = self._posX
		event.posZ= self._posZ
		event.temporal_event = self._temporal_event
		event.music = self._music

		
		return event

	def event_header(self, split_line):
		"""I don't really want an event for this line, I'm just doing it to get the header. Getting the header because
		some old log files don't have the stimList column, and we need to know if that exists. Could do it based on the
		column number, but I feel like this is safer in case the log file changes."""
		# print("event line " + str(split_line))
		self._log_header = split_line
		split_line[0] = -999
		# split_line[1] = 'dummy'
		return self.event_default(split_line)

	def modify_header(self, events):
		"""Remove dummy event"""
		events = events[events['condition'] != 'dummy']
		return events

	def set_event_properties(self, split_line):

		# print("log head " + str(self._log_header))
		ind = self._log_header.index('condition')
		self._condition = split_line[ind]

		ind = self._log_header.index('env')
		self._env = split_line[ind]

		ind = self._log_header.index('stage')
		self._stage = split_line[ind]

		ind = self._log_header.index('action')
		self._action = split_line[ind]

		ind = self._log_header.index('whichroom')
		self._whichroom = split_line[ind]

		ind = self._log_header.index('whichtraj')
		self._whichtraj = split_line[ind]

		ind = self._log_header.index('temporal_event')
		self._temporal_event = split_line[ind]

		ind = self._log_header.index('music')
		self._music = split_line[ind]

		ind = self._log_header.index('reward')
		self._reward = int(split_line[ind])


		ind = self._log_header.index('posX')
		self._posX = float(split_line[ind])
		ind = self._log_header.index('posZ')
		self._posZ = float(split_line[ind])


	def event_line(self, split_line):

		# set all the values in the line
		self.set_event_properties(split_line)

		event = self.event_default(split_line)
		return event

	def modify_rec(self, events):
		"""This adds list length field to current rec event and all prior CHEST events of the current trial.
		No current way to know this ahead of time."""
		pres_events = (events['condition'] == self._condition)
		pres_events_inc_empty = (events['condition'] == self._condition)
		listLength = np.sum(pres_events)
		for ind in np.where(pres_events_inc_empty)[0]:
			events[ind].listLength = listLength
		events[-1].listLength = listLength
		return events


	@staticmethod
	def parse_raw_log(raw_log_file, out_file_path):
		def writeToFile(f,data,subject):
			columnOrder = ['mstime','condition','env','stage','action','reward','whichroom','whichtraj','posX','posZ','temporal_event','music']
			strToWrite = ''
			# print(data)
			for col in columnOrder:
				line = data[col]
				# print(line)
				if col != columnOrder[-1]:
					strToWrite += '%s\t'%(line)
				else:
					strToWrite += '%s\t%s\n'%(line,subject)
			f.write(strToWrite)

		def makeEmptyDict(mstime=None,condition=None,env=None,stage=None,action=None,reward=None,whichroom=None,whichtraj=None,posX=None,posZ=None,
			temporal_event=None,music=None):
			fields = ['mstime','condition','env','stage','action','reward','whichroom','whichtraj','posX','posZ','temporal_event','music']
			vals = [mstime,condition,env,stage,action,reward,whichroom,whichtraj,posX,posZ,temporal_event,music]
			emptyDict = dict(zip(fields,vals))
			return emptyDict


		#deprecated
		def getPresDictKey(data,recItem,trialNum):
			for key in data:
				if data[key]['condition'] == recItem and data[key]['type'] == 'CHEST' and data[key]['trial'] == trialNum:
					return key

		# open raw log file to read and treasure.par to write new abridged log

		sess_dir, log_file = os.path.split(raw_log_file)
		in_file = open(raw_log_file, 'r')
		out_file = open(out_file_path, 'w')

		# file to keep track of player pather
		playerPathsFile = open(os.path.join(sess_dir,"playerPaths.par"), 'w')

		# write log header
		columnOrder = ['mstime','condition','env','stage','action','reward','whichroom','whichtraj','posX','posZ','temporal_event','music']
		subject = log_file[:-4]
		out_file.write('\t'.join(columnOrder) + '\tsubject\n')

		# initial values
		treasureInfo = {}
		data = {}
		chest = None
		env=None
		condition="none"
		stage=None
		action=None
		posX=-1
		posZ=-1
		reward=0
		temporal_event=None
		music=None
		activeCorridor=-1 # to keep track of which corridor the player is in; 0 = left ; 1= right
		whichroom=None
		whichtraj=None
		camPos=-1000 
		action_event=None
		ogLeft=np.zeros([2,1],dtype=float)
		ogRight=np.zeros([2,1],dtype=float)
		camDistance=100
		room_img_dict= { 'RestaurantRoom': 'RoomFive_Space', 'CryoRoom' : 'RoomSix_Space',
					  'Restroom' : 'RoomFive_Office', 'ConferenceRoom' : 'RoomSix_Office',
					   'RoomOne' : 'RoomSix_WesternTown' , 'RoomTwo' : 'RoomFive_WesternTown',
				   'EmptyHouse_RoomTwo': 'RoomFive_Apartment', 'LivingRoom_RoomOne' : 'RoomSix_Apartment'}

		music_track_dict={'Grebe': 1, 'Picnic':2,'Tours':3,'Harph':4,'Chords':5,'Soularflair':6,'Mezcal':7,'SONNIK':8,'Xia':9,'Warriors':10,
		'Morning':11,'Dubwegians':12,'tabla':13,'Elgon':14,'Demonization':15,'Trace':16,'Horizon':17,'Air':18,'Glouglou':19,'Confidence':20,
		'Metalmania':21,'Ombroso':22,'Fragmental':23,'cybercity':24}

		# loop over all lines in raw log
		for s in in_file.readlines():

			s = s.replace('\r','')
			tokens = s[:-1].split('\t')
			if len(tokens)>1:

				mstime=tokens[0]
				temporal_event=None
				# action_event=None

				#check which stage
				if tokens[2]=="TRAINING":
					stage="pretraining"
				if tokens[2]=="LEARNING":
					stage="learning"
				if tokens[2]=="RE-EVALUATION":
					stage="relearning"
				if tokens[2]=="TESTING":
					stage="test"
				if tokens[2]=="POST-TEST":
					stage="postnav_noreward"
					measure=False
				if tokens[2]=="MUSIC_BASELINE":
					stage="music_baseline"
				if tokens[2]=="IMAGE_BASELINE":
					stage="image_baseline"
				if tokens[2]=="SILENT_TRAVERSAL":
					stage="silent_nav"
				if tokens[2]=="END_SESSION":
					stage="session_complete"

				if tokens[2]=="LEFT_CORRIDOR":
					activeCorridor = 0
					whichtraj="first"
				if tokens[2]=="RIGHT_CORRIDOR":
					activeCorridor=1
					whichtraj="second"

				#condition
				if tokens[2]=="TRANSITION_REEVAL":
					condition="trans_reval"
				if tokens[2]=="REWARD_REEVAL":
					condition="reward_reval"
				if tokens[2]=="END_ENVIRONMENT_STAGE":
					if tokens[3]=="ON":
						condition="none"
						music=None



				#position of player; we parse this first to create an event line with the player position and are OK if it gets overwritten with additional info like temporal_event and action further down the script 
				if tokens[2]=="DecisionBody" and stage!=None:
					posX = float(tokens[4])
					posZ= float(tokens[6])
					# mstime = tokens[0]
					# data[mstime] = makeEmptyDict(mstime,condition,env,stage,None,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)


				#actions

					
				#for comparative sliders
				if tokens[2]=="COMPARATIVE_PREF_SLIDER":
					#check if the rooms being queried are starting rooms -- RoomOne and RoomTwo
					if "One" in tokens[4] or "Two" in tokens[4]: 
						action_event="slide_start_rooms"
						#else they are middle rooms
					if "Three" in tokens[4] or "Four":
						action_event="slide_middle_rooms"

				#for solo sliders
				if tokens[2]=="SOLO_PREF_SLIDER_IMAGE":
					action_event="solo_slide_start_room"


				#for multiple choice questions
				if tokens[2]=="MULTIPLE_CHOICE_FOCUS_IMAGE":
					action_event="map_multichoice_rooms"

				if tokens[2]=="NegativeFeedback" and tokens[4]=="1":
					action_event="cam_press"

				if tokens[2]=="CAM_SNEAKING":
					action_event="cam_press"

				#check which environment
				if tokens[2] == "ENVIRONMENT_CHOSEN":
					measure=True
					first_dev_index=0
					firstTime=True
					second_dev_index=0    
					solo_first_index=0
					solo_second_index=0
					solo_index=0
#                     print("NEW ENVIRONMENT")
					if tokens[3] == "Office":
						env="office"
						secondEnvName= "Office"
						envIndex=1
					elif tokens[3]=="SpaceStation":
						env="space"
						firstEnvName = "SpaceStation"
						envIndex=0
					elif tokens[3] == "WesternTown":
						env="west"
						envIndex=0
						firstEnvName = "WesternTown"
					elif tokens[3] == "Apartment":
						env="apt"
						envIndex = 1
						secondEnvName = "Apartment"


				#calc distance from cam
				if env=="space":
					camDistance = abs(posX-camPos)
				else:
					camDistance = abs(posZ-camPos)

				#check to see if the current mstime key has already been made before we parse through temporal events; this is to prevent overwriting of events
				keyExists=False
				for key in data:
					if key==mstime:
						keyExists=True



				#below are time-sensitive events that are ordered based on their priority; chest rewards are parsed through first and less time-sensitive events like "nav" are parsed at the end

				#check if we are at the chest reward event
				if tokens[2] == "REGISTER_REWARD":
					if tokens[4]=="LEFT":
						leftReward=int(tokens[3])
						mstime = tokens[0]
						reward=leftReward
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,leftReward,whichroom,whichtraj,posX,posZ,temporal_event,music)
						if ogLeft[envIndex,0]==0:
							ogLeft[envIndex,0]=leftReward
					else:
						rightReward = int(tokens[3])
						mstime = tokens[0]
						reward=rightReward
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,rightReward,whichroom,whichtraj,posX,posZ,temporal_event,music)
						if ogRight[envIndex,0]==0:
							ogRight[envIndex,0]=rightReward


				#second we parse through button presses which are of high-importance
				elif tokens[2]=="ACTION_BUTTON_PRESSED":
					mstime = tokens[0]
					#this is the closest we have to a logged line to chest opening/ when the button press to open chest is logged happens right after chest opening animation begins 
					if temporal_event=="chest_pos":
						temporal_event="chest_opens"
					data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					action_event=None	
				#for reward opening 
				elif tokens[2]=="WAITING_FOR_REGISTER_PRESS":
					if tokens[3]=="STARTED":
						camDistance=1000 #set this as a flag to stop looking for camera_pos events
						action_event="reward_chest_press"
						temporal_event="chest_pos"
						mstime = tokens[0]
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,None,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					
				#for door opening
				elif tokens[2]=="WAITING_FOR_DOOR_PRESS":
					if tokens[3]=="STARTED":
						if action_event !="reward_chest_press": #log this only if we aren't also waiting for a reward_chest_press event
							action_event="door_press"
							temporal_event="door_pos"
							mstime = tokens[0]
							data[mstime] = makeEmptyDict(mstime,condition,env,stage,None,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
			

				#third we parse through room-entry

				#check whichroom player is in currently
				elif tokens[2]=="ROOM_1_MOVE":
					if tokens[3]=="STARTED":
						if activeCorridor==0:
							whichroom="room1"
						else:
							whichroom="room2"

				elif tokens[2]=="ROOM_2_MOVE":
					if tokens[3]=="STARTED":
						temporal_event="door_opens"
						if activeCorridor==0:
							whichroom="room3"
						else:
							whichroom="room4"
						mstime = tokens[0]
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					
							
				elif tokens[2]=="ROOM_3_MOVE":
					if tokens[3]=="STARTED":
						temporal_event="door_opens"
						if activeCorridor==0:
							whichroom="room5"
						else:
							whichroom="room6"
						mstime = tokens[0]
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)


				#check which temporal event
				elif tokens[2]=="INSTRUCTION_VIDEO":
					if tokens[3]=="STARTED":
						temporal_event="instruction_video"
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					if tokens[3]=="ENDED":
						temporal_event=None

				elif tokens[2]=="IntertrialScreen":
					if tokens[4]=="1":
						temporal_event="ITI_msg"
						music=None #make sure to turn off the music marker in case "AUDIO_STOPPED" message doesn't get logged
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					if tokens[4]=="0":
						temporal_event=None

				elif tokens[2]=="PositiveFeedback":
					if tokens[4]=="1":
						temporal_event="cam_correct_msg"
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					if tokens[4]=="0":
						temporal_event=None

				elif tokens[2]=="NegativeFeedback":
					if tokens[4]=="1":
						temporal_event="cam_incorrect_msg"
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					if tokens[4]=="0":
						temporal_event=None

				elif tokens[2]=="REWARD_TEXT":
					if tokens[3]=="ON":
						temporal_event="reward_shows"
						if not keyExists:
							data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
						else:	
							data[mstime]['temporal_event']=temporal_event
						# data[mstime] = makeEmptyDict(mstime,condition,env,phase,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					if tokens[3]=="OFF":
						reward=0
						temporal_event=None
				elif camDistance < 10 and stage!=None:
					temporal_event="camera_pos"
					if not keyExists:
						data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
					else:	
						data[mstime]['temporal_event']=temporal_event
				elif stage!=None:
					if temporal_event==None:
						temporal_event="nav"
						if not keyExists:
							data[mstime] = makeEmptyDict(mstime,condition,env,stage,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music)
						else:	
							data[mstime]['temporal_event']=temporal_event

				#music tracks
				if "_Audio" in tokens[2]:
					if tokens[3]=="AUDIO_PLAYING":
						for tracks in music_track_dict:
							if tracks in tokens[5]:
								music=music_track_dict[tracks]
					# if tokens[3]=="AUDIO_STOPPED":
					# 	music=None

								# data[mstime]= makeEmptyDict(mstime,condition,env,phase,action_event,reward,whichroom,whichtraj,posX,posZ,temporal_event,music_track_dict[tracks])

				if "CamZone" in tokens[2] and tokens[3]=="POSITION":
					if env=="space":
						camPos = float(tokens[4]) #take the x-position as the movement happens along that axis
					else:
						camPos= float(tokens[6]) #else take the z-axis position



				
		# make sure all the events are in order, and write to new file
		# print(data)
		sortedKeys = sorted(data)
		for key in sortedKeys:
			writeToFile(out_file,data[key],subject)

		# close files
		in_file.close()
		out_file.close()
		os.chmod(out_file_path, 0o644)
		playerPathsFile.close()

		# save out total score
		# scoreFile = open(os.path.join(sess_dir,"totalScore.txt"), 'w')
		# scoreFile.write('%d'%(totalScore))
		# scoreFile.close()

def thief_test(protocol, subject, montage, experiment, session, base_dir='/data/eeg/'):
	exp_path = os.path.join(base_dir, subject, 'behavioral', experiment)
	# files = {'session_log': os.path.join(exp_path, 'session_%d' % session, 'treasure.par'),
			 # 'annotations': ''}
	files = {'thief_par': os.path.join(exp_path,'thief.par'),
			 'session_log': os.path.join(exp_path, subject+'Log.txt')}



	parser = ThiefSessionLogParser(protocol, subject, montage, experiment, session, files)
	return parser

# if __name__ == '__main__':
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument("subject", help="subject name")
# 	parser.add_argument("session", help="session number", type=int)
# 	parser.add_argument("log_path", help="path to log file")
# 	parser.add_argument("--protocol", help="protocol string, for compatability with other events", default='jacobs_beh')
# 	parser.add_argument("--montage", help="montage number, for compatability with ECoG events", default=0, type=int)
# 	args = parser.parse_args()

# 	# get the base directory. Will save events in same place as log file
# 	log_dir = os.path.split(args.log_path)[0]
# 	files = {'thief_par': os.path.join(log_dir,'thief.par'),
# 			 'session_log': args.log_path}

# 	# create the parser
# 	parser = ThiefSessionLogParser(args.protocol, args.subject, args.montage,
# 			 'TH_BEH', args.session, files)

# 	# parse the data to create the events
# 	events = parser.parse()

# 	# save the events
# 	save_fname = os.path.join(log_dir, args.subject+'_session_%d.json' % args.session)
# 	with fileutil.open_with_perms(save_fname, 'w') as f:
# 		f.write(to_json(events))

















