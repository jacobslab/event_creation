"""
Datatype specifications for use by parser classes,
to make sure that different parsers for the same experiment return arrays with the same fields.
Fields are specified as (name, default_value,dtype_string)
"""



# Defaults

base_fields = (
    ('protocol', '', 'S64'),
    ('subject', '', 'S64'),
    ('montage', '', 'S64'),
    ('experiment', '', 'S64'),
    ('session', -1, 'int16'),
    ('type', '', 'S64'),
    ('mstime', -1, 'int64'),
    ('msoffset', -1, 'int16'),
    ('eegoffset', -1, 'int64'),
    ('eegfile', '', 'S256'),
    ('lfpoffset', -1, 'int64'),
    ('lfpfile', '', 'S256'),
    ('phase','','S16')
)

stim_fields = (
        ('anode_number', -1, 'int16'),
        ('cathode_number', -1, 'int16'),
        ('anode_label', '', 'S64'),
        ('cathode_label', '', 'S64'),
        ('amplitude', -1, 'float16'),
        ('pulse_freq', -1, 'int16'),
        ('n_pulses', -1, 'int16'),
        ('burst_freq', -1, 'int16'),
        ('n_bursts', -1, 'int16'),
        ('pulse_width', -1, 'int16'),
        ('stim_on', False, bool),
        ('stim_duration', -1, 'int16'),
        ('biomarker_value', -1, 'float64'),
        ('host_time',-999,'float64'),
        ('id', '', 'S64'),
        ('position', '', 'S64'),
        ('_remove', True, 'b'),  # This field is removed before saving, and it used to mark whether it should be output
                                 # to JSON
    )

# FR

fr_fields = (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('item_name', 'X', 'U64'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intrusion', -999, 'int16'),
            ('exp_version', '', 'S64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
            ('rectime',-999,'int16'),
            # Recognition stuff goes here
            ('recognized', -999, 'int16'),
            ('rejected', -999, 'int16'),
            ('recog_resp', -999, 'int16'),
            ('recog_rt', -999, 'int16'),
)

# catFR
category_fields = (
        ('category','X','S64'),
        ('category_num',-999,'int16'),
)

#PAL

pal_fields = (
            ('list', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('probepos', -999, 'int16'),
            ('study_1', '', 'S16'),
            ('study_2', '', 'S16'),
            ('cue_direction', -999, 'int16'),
            ('probe_word', '', 'S16'),
            ('expecting_word', '', 'S16'),
            ('resp_word', '', 'S16'),
            ('correct', -999, 'int16'),
            ('intrusion', -999, 'int16'),
            ('resp_pass', 0 , 'int16'),
            ('vocalization', -999, 'int16'),
            ('RT', -999, 'int16'),
            ('exp_version', '', 'S16'),
            ('stim_type', '', 'S16'),
            ('stim_list', 0, 'b1'),
            ('is_stim', False, 'b1'),
)

#Math

math_fields =  (
            ('list', -999, 'int16'),
            ('test', -999, 'int16', 3),
            ('answer', -999, 'int16'),
            ('iscorrect', -999, 'int16'),
            ('rectime', -999, 'int32'),
        )

# LTP

ltp_fr2_fields = (
            ('trial', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('begin_distractor', -999, 'int16'),
            ('final_distractor', -999, 'int16'),
            ('begin_math_correct', -999, 'int16'),
            ('final_math_correct', -999, 'int16'),
            ('item_name', '', 'S16'),
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intruded', 0, 'int16'),
            ('rectime', -999, 'int32'),
            ('intrusion', -999, 'int16'),
)

ltp_fields = (
            ('artifactMS', -1, 'int32'),
            ('artifactNum', -1, 'int32'),
            ('artifactFrac', -1, 'float16'),
            ('artifactMeanMS', -1, 'float16'),
            ('badEvent', False, 'b1'),
            ('badEventChannel', '', 'S8', 132)  # Because recarrays require fields of type array to be a fixed length,
                                                # all badEventChannel entries must be length 132
)

ltp_fr_fields = ( ('trial', -999, 'int16'),
            ('studytrial', -999, 'int16'),
            ('listtype', -999, 'int16'),
            ('serialpos', -999, 'int16'),
            ('distractor', -999, 'int16'),
            ('final_distractor', -999, 'int16'),
            ('math_correct', -999, 'int16'),
            ('final_math_correct', -999, 'int16'),
            ('task', -999, 'int16'),
            ('resp', -999, 'int16'),
            ('rt', -999, 'int16'),
            ('recog_resp', -999, 'int16'),
            ('recog_conf', -999, 'int16'),
            ('recog_rt', -999, 'int32'),
            ('item_name', '', 'S16'),  # Calling this 'item' will break things, due to the built-in recarray.item method
            ('item_num', -999, 'int16'),
            ('recalled', False, 'b1'),
            ('intruded', 0, 'int16'),
            ('finalrecalled', False, 'b1'),
            ('recognized', False, 'b1'),
            ('rectime', -999, 'int32'),
            ('intrusion', -999, 'int16'),
            ('color_r', -999, 'float16'),
            ('color_g', -999, 'float16'),
            ('color_b', -999, 'float16'),
            ('font', '', 'S32'),
            ('case', '', 'S8'),
            ('rejected', False, 'b1'),
            ('rej_time', -999, 'int32'),
)
# PS2-3
ps_fields = (
            ('exp_version', '', 'S16'),
            ('ad_observed', 0, 'b1'),
            ('is_stim', 0, 'b1')
        )


system2_ps_fields = (
        ('hosttime', -1, 'int64'),
        ('file_index', -1, 'int16')
    )


# PS4

location_subfields = (
        ('loc_name','','S16'),
        ('amplitude',-999,'float64'),
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('snr',-999,'float64')
    )

sham_subfields = (
        ('delta_classifier',-999,'float64'),
        ('sem',-999,'float64'),
        ('p_val',-999,'float64',),
        ('t_stat',-999,'float64',),
    )

decision_subfields = (
        ('p_val',-999.0, 'float64'),
        ('t_stat',-999.0,'float64'),
        ('best_location','','S16'),
        ('tie',-1,'int16'),

    )


# TH
th_fields = (
            ('trial', -999, 'int16'),
            ('item_name', '', 'S64'),
            ('chestNum', -999, 'int16'),
            ('block', -999, 'int16'),
            ('locationX', -999, 'float64'),
            ('locationY', -999, 'float64'),
            ('chosenLocationX', -999, 'float64'),
            ('chosenLocationY', -999, 'float64'),
            ('navStartLocationX', -999, 'float64'),
            ('navStartLocationY', -999, 'float64'),
            ('recStartLocationX', -999, 'float64'),
            ('recStartLocationY', -999, 'float64'),
            ('isRecFromNearSide', False, 'b1'),
            ('isRecFromStartSide', False, 'b1'),
            ('reactionTime', -999, 'float64'),
            ('confidence', -999, 'int16'),
            ('radius_size', -999, 'float64'),
            ('listLength', -999, 'int16'),
            ('distErr', -999, 'float64'),
            ('normErr', -999, 'float64'),
            ('recalled', False, 'b1'),
            ('exp_version', '', 'S64'),
            ('stim_list', False, 'b1'),
            ('is_stim', False, 'b1'),
        )


thr_fields =(
    ('trial', -999, 'int16'),
    ('item_name', '', 'S64'),
    ('resp_word', '', 'S64'),
    ('serialpos', -999, 'int16'),
    ('probepos', -999, 'int16'),
    ('block', -999, 'int16'),
    ('locationX', -999, 'float64'),
    ('locationY', -999, 'float64'),
    ('navStartLocationX', -999, 'float64'),
    ('navStartLocationY', -999, 'float64'),
    ('recStartLocationX', -999, 'float64'),
    ('recStartLocationY', -999, 'float64'),
    ('reactionTime', -999, 'float64'),
    ('list_length', -999, 'int16'),
    ('recalled', False, 'b1'),
    ('exp_version', '', 'S64'),
    ('stim_list', False, 'b1'),
    ('is_stim', False, 'b1'),
)