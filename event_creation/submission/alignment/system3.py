import json
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from ..log import logger
from ..parsers.system3_log_parser import System3LogParser
from ..exc import AlignmentError
import itertools


class System3Aligner(object):

    TASK_TIME_FIELD = 'mstime'
    ENS_TIME_FIELD = 'eegoffset'
    EEG_FILE_FIELD = 'eegfile'

    MAXIMUM_ALLOWED_RESIDUAL = 1000
    MAXIMUM_NUMBER_EXCESSIVE_RESIDUALS = 2

    FROM_LABELS = (('orig_timestamp', 1000,('STIM','FEATURES',
                                            'BIOMARKER',)),
                   )
    TO_LABELS = ('t_event', 't0')

    def __init__(self, events, files, plot_save_dir=None):

        self.files = files

        self.events_logs = files['event_log']

        self.electrode_config = files['electrode_config']

        self.plot_save_dir = plot_save_dir

        self.events = events
        self.session_attrs={prop:events[0][prop] for prop in ['protocol','session','experiment','subject','montage']}
        self.session_attrs['files'] = files

        self.residuals =  []
        # vocalization_events = VocalizationParser(**session_attrs).parse()
        # if vocalization_events.shape:
        #     self.merged_events = np.concatenate([self.events,vocalization_events]).view(np.recarray).sort('mstime')
        # else:
        self.merged_events = self.events
        for ((from_label, from_rate, exclude), to_label) in itertools.product(self.FROM_LABELS, self.TO_LABELS):
            try:
                self.task_to_ens_coefs, self.task_ends = \
                    self.get_coefficients_from_event_log(from_label, 'offset', from_rate,exclude)
                self.host_to_ens_coefs, self.host_ends = \
                    self.get_coefficients_from_event_log(to_label, 'offset', 1,exclude)
                logger.debug("Found coefficient with label {}".format(from_label))
            except KeyError as key_error:
                if key_error.message != from_label:
                    raise
                logger.debug("Couldn't find coefficient with label {}".format(from_label))
                continue
            except AlignmentError as ae:
                logger.debug("Couldn't align coefficient with label {}".format(from_label))
                continue
            self.label = from_label
            self.from_multiplier = from_rate
            break
        else:
            raise AlignmentError("Could not find alignable label in events")

        self.eeg_info = json.load(open(files['eeg_sources']))

    def stim_event_to_mstime(self, event):

        if event['host_time'].shape:
            host_time = event['host_time'][0]
        else:
            host_time = event['host_time']

        coef_inds = np.where(host_time <= self.host_ends)[0]

        if len(coef_inds) > 0:
            coef_ind = coef_inds[-1]
        else:
            if host_time - self.host_ends[-1] > 2000:
                logger.error("Time extends beyond end of log file!")
            coef_ind = len(self.host_ends)-1

        ens_time = self.apply_coefficients(host_time, self.host_to_ens_coefs[coef_ind])
        task_time = self.apply_coefficients_backwards(ens_time, self.task_to_ens_coefs[coef_ind])

        return task_time

    def add_stim_events(self, event_template, persistent_fields=lambda *_: tuple()):
        # Merge in the stim events

        logger.debug("Generating system 3 log parser")
        s3lp = System3LogParser(self.events_logs, self.electrode_config)
        logger.debug("Merging events")
        self.merged_events = s3lp.merge_events(self.events, event_template, self.stim_event_to_mstime, persistent_fields)

        # Have to manually correct subject and session due to events appearing before start of session
        self.merged_events['subject'] = self.events[-1].subject
        self.merged_events['session'] = self.events[-1].session

        return self.merged_events

    def get_coefficients_from_event_log(self, from_label, to_label, rate,exclude=(None,)):

        ends = []
        coefs = []

        for i, event_log in enumerate(self.events_logs):

            event_dict = json.load(open(event_log))['events']

            froms = [float(event[from_label]) * 1000. / rate for event in event_dict \
                    if from_label in event and to_label in event and event['event_label'] not in exclude]
            tos = [float(event[to_label]) for event in event_dict\
                   if from_label in event and to_label in event and event['event_label'] not in exclude]

            froms = np.array(froms)
            tos = np.array(tos)

            froms = froms[tos > 0]
            tos = tos[tos > 0]

            if len(froms) <= 1:
                continue

            coefs.append(scipy.stats.theilslopes(x=froms, y=tos)[:2])
            ends.append(froms[-1])

            self.plot_fit(froms, tos, coefs[-1], '.', 'fit_{}_{}_{}'.format(from_label,to_label,i))
            residuals = self.check_fit(froms, tos, coefs[-1])

            if from_label == 'orig_timestamp':
                for time,residue in zip(froms,residuals):
                    at_time = self.events['mstime']==time
                    if at_time.any():
                        new_event = self.events[at_time]
                        new_event['msoffset'] = int(residue)
                        self.events[at_time] = new_event


        if len(coefs) == 0:
            raise AlignmentError("Could not find enough events to determine coefficients!")

        return np.array(coefs), np.array(ends)

    def align(self, start_type=None):

        new_events = deepcopy(self.merged_events)
        unaligned_events = new_events[new_events['eegoffset'] == -1]

        if start_type:
            starting_entries= np.where(unaligned_events['type'] == start_type)[0]
            # Don't have to align until one after the starting type (which is normally SESS_START)
            starts_at = starting_entries[0] + 1 if len(starting_entries) > 0 else 1
        else:
            starts_at = 0

        ens_times = self.align_source_to_dest(unaligned_events[self.TASK_TIME_FIELD],
                                              self.task_to_ens_coefs,
                                              self.task_ends, starts_at)
        ens_times[ens_times < 0] = -1
        unaligned_events[self.ENS_TIME_FIELD] = ens_times
        new_events[new_events['eegoffset'] == -1] = unaligned_events

        untimed_events = new_events[new_events['mstime'] == -1]

        task_times = self.align_source_to_dest(untimed_events[self.ENS_TIME_FIELD],
                                               self.task_to_ens_coefs,
                                               self.task_ends,backwards=True)

        untimed_events[self.TASK_TIME_FIELD] = task_times
        new_events[new_events['mstime'] == -1] = untimed_events

        return new_events

    def apply_eeg_file(self, events):

        eeg_info = sorted(self.eeg_info.items(), key= lambda info:info[1]['start_time_ms'])

        
        for eegfile, info in eeg_info:
            start_time_host = info['start_time_ms']
            inds = np.where(self.host_ends > start_time_host)[0]
            if len(inds) == 0:
                ind = len(self.host_ends)-1
            else:
                ind = inds[0]

            task_time = self.apply_coefficients_backwards(0, self.task_to_ens_coefs[ind])

            mask = events[self.TASK_TIME_FIELD] >= task_time
            events[self.EEG_FILE_FIELD][mask] = eegfile

    @classmethod
    def align_source_to_dest(cls, source, coefs, ends, align_start_index=0,backwards=False):

        dest = np.full(len(source), np.nan)
        dest[source == -1] = -1

        sorted_ends, sorted_coefs = zip(*sorted(zip(ends, coefs)))

        for (start, coef) in zip((0,)+sorted_ends, coefs):
            time_mask = (source >= start)
            if backwards:
                dest[time_mask] = cls.apply_coefficients_backwards(
                    source[time_mask], coef)
            else:
                dest[time_mask] = cls.apply_coefficients(source[time_mask], coef)

        still_nans = np.where(np.isnan(dest))[0]
        if len(still_nans) > 0:
            if (np.array(still_nans) <= align_start_index).all():
                logger.warn('Warning: Could not align events %s' % still_nans)
                dest[np.isnan(dest)] = -1
            else:
                logger.error("Events {} could not be aligned! Session starts at event {}".format(still_nans, align_start_index))
                raise AlignmentError('Could not convert {} times past start of session'.format(len(still_nans)))
        return dest



    @staticmethod
    def apply_coefficients_backwards(dest, coefficients):
        """
        Applies a set of coefficients in reverse, going from destination to source
        :param dest: "destination" time to be converted to source times
        :param coefficients: coefficients to be applied to destination times
        :return: "source" times
        """
        return (dest - coefficients[1]) / coefficients[0]

    @staticmethod
    def apply_coefficients(source, coefficients):
        """
        Applies a set of coefficients (slope, intercept) to an array or list of source values
        :param source: times to be converted
        :param coefficients: (slope, intercept)
        :return: converted times
        """
        return coefficients[0] * np.array(source) + coefficients[1]

    @classmethod
    def check_fit(cls, x, y, coefficients):
        fit = coefficients[0] * np.array(x) + coefficients[1]
        residuals = np.array(y) - fit
        if abs(1 - coefficients[0]) > .05:
            raise AlignmentError(
                "Maximum deviation from slope is .1, current slope is {}".format(coefficients[0])
            )

        return residuals
    @classmethod
    def plot_fit(cls, x, y, coefficients, plot_save_dir, plot_save_label):
        """
        Plots a fit between two values (both plotting fit itself and residuals
        :param x:
        :param y:
        :param coefficients:
        :param plot_save_dir: Where to save the plot
        :param plot_save_label: What to name the saved plot
        :return: None
        """
        fit = coefficients[0] * np.array(x) + coefficients[1]
        plt.figure(figsize=(20,10))
        plt.subplot(121)
        plt.plot(x, y, 'g.', x, fit, 'b-')
        plt.title("EEG Samples vs Tim   estamps")
        plt.xlabel("Timestamp (ms)")
        plt.ylabel("EEG Samples")
        plt.xlim(min(x), max(x))

        plt.subplot(122)
        plt.plot(x, y - fit, 'g.-', [min(x), max(x)], [0, 0], 'k-')
        plt.title("Fit residuals")
        plt.xlabel("Timestamp (ms)")
        plt.ylabel("Best-fit residuals")
        plt.xlim(min(x), max(x))
        try:
            plt.savefig(os.path.join(plot_save_dir, '{label}_fit{ext}'.format(label=plot_save_label,
                                                                                ext='.png')))
        except Exception:
            logger.debug("Could not save plot %s"%plot_save_label)
        plt.show()
        plt.close()


class System3FourAligner(System3Aligner):
    """
    Aligner class for System 3.4+
    Behaves just like System3Aligner, except that the task times and the
    host times are identical.
    """

    def __init__(self, events, files, plot_save_dir=None):
        super(System3FourAligner, self).__init__(events,files, plot_save_dir)
        self.task_to_ens_coefs = self.host_to_ens_coefs
        self.task_ends = self.host_ends
        
    def apply_eeg_file(self, events):
        eeg_info = self.eeg_info.items()
        if len(eeg_info) == 1:
            mask = events['eegoffset'] >= 0
            events[self.EEG_FILE_FIELD][mask] = eeg_info[0][0]
            return events
        else:
            return super(System3FourAligner, self).apply_eeg_file(events)


if __name__ == '__main__':
    files = {
        'electrode_config' : '/Volumes/PATRIOT/R1999X/behavioral/FR1/host_pc/session_0/20170210_105441/config_files/R1170J_ALLCHANNELSSTIM.csv',
        'event_log' : ['/Volumes/PATRIOT/R1999X/behavioral/FR1/host_pc/session_0/20170210_105441/data_incremental/event_log.json']
             }
    aligner = System3Aligner(events=None,files=files)
