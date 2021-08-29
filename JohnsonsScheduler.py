import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


class JohnsonsScheduler:

    def __init__(self, processing_times_A, processing_times_B, names=None, seed=None):
        # Check if same number of jobs at both work centers
        if len(processing_times_A) != len(processing_times_B):
            print(
                f'Error: Criteria of same number of jobs at work center A and B is not met. {len(processing_times_A)} number of jobs detected for work center A and {len(processing_times_B)} detected for work center B.')
        if seed != None:
            random.seed(seed)
        self.n_jobs = len(processing_times_A)
        self.processing_times_A = processing_times_A
        self.processing_times_B = processing_times_B
        if names == None:
            self.names = [i for i in range(self.n_jobs)]
        else:
            self.names = names
        self.jobs = pd.DataFrame(
            index=self.names, data=
            {'A': self.processing_times_A,
             'B': self.processing_times_B})

    def calculate_schedule(self):
        """
        Perform Johnsons schedule/rule
        """
        self.sequence = [0 for i in range(self.n_jobs)]
        jobs_copy = self.jobs.copy()
        for i in range(self.n_jobs):
            min_val_A = jobs_copy['A'].min()
            min_index_A = jobs_copy['A'].idxmin()
            min_val_B = jobs_copy['B'].min()
            min_index_B = jobs_copy['B'].idxmin()
            if min_val_A <= min_val_B:
                # Job at work center A has the shortest processing time
                self.sequence[self.sequence.index(0)] = min_index_A
                drop_index = min_index_A
            else:
                # Job at work center B has the shortest processing time
                self.sequence[self.n_jobs - 1 - self.sequence[::-1].index(0)] = min_index_B
                drop_index = min_index_B
            jobs_copy = jobs_copy.drop(drop_index)
        self.schedule = pd.DataFrame([self.jobs.loc[i] for i in self.sequence])
        self.schedule['start_A'] = np.zeros(self.n_jobs)
        self.schedule['end_A'] = np.zeros(self.n_jobs)
        self.schedule['start_B'] = np.zeros(self.n_jobs)
        self.schedule['end_B'] = np.zeros(self.n_jobs)
        for i, name in enumerate(self.sequence):
            # Calculate schedule for work center A
            if i == 0:
                self.schedule['start_A'].iloc[i] = 0
            else:
                self.schedule['start_A'].iloc[i] = self.schedule['end_A'].iloc[i - 1]
            self.schedule['end_A'].iloc[i] = self.schedule['start_A'].iloc[i] + self.jobs.at[name, 'A']
            # Calculate schedule for work center B
            if i == 0:
                self.schedule['start_B'].iloc[i] = self.schedule['end_A'].iloc[i]
            else:
                self.schedule['start_B'].iloc[i] = max(
                    [self.schedule['end_A'].iloc[i], self.schedule['end_B'].iloc[i - 1]])
            self.schedule['end_B'].iloc[i] = self.schedule['start_B'].iloc[i] + self.jobs.at[name, 'B']

    def visualize(self, color_map=plt.cm.Set3, plot_size=(15, 5)):
        """
        Visualize the schedule
        """
        fig, ax = plt.subplots(figsize=plot_size)
        colors = iter([color_map(i) for i in range(20)])
        for index, row in self.schedule.iterrows():
            xs = [(row.start_A, row.end_A), (row.start_B, row.end_B)]
            xmins = np.array([row.start_A, row.start_B])
            xmaxs = np.array([row.end_A, row.end_B])
            cs = xmins + (xmaxs - xmins) / 2
            ys = [2, 1]
            ax.hlines(
                y=ys,
                xmin=xmins,
                xmax=xmaxs,
                linestyle='solid',
                linewidth=50,
                label=index,
                colors=[next(colors)])
            for c, y in zip(cs, ys):
                ax.text(
                    x=c,
                    y=y,
                    s=index,
                    ha='center',
                    va='center')
        plt.yticks([1, 2], ['Work center B', 'Work center A'])
        plt.ylim(0, 3)
        ax.set_xlabel('Time')
        ax.set_ylabel('Resource')
        ax.set_title('Productioon schedule')
        ax.grid(axis='x')
        plt.show()

    def get_metrics(self):
        idle_A, idle_B = 0, 0
        for i in range(self.n_jobs):
            if i > 0 and i < self.n_jobs - 1:
                idle_A += self.schedule['start_A'].iloc[i] - self.schedule['end_A'].iloc[i - 1]
            elif i == self.n_jobs - 1:
                idle_A += self.schedule['end_B'].iloc[i] - self.schedule['end_A'].iloc[i]
            if i == 0:
                idle_B += self.schedule['start_B'].iloc[i]
            else:
                idle_B += self.schedule['start_B'].iloc[i] - self.schedule['end_B'].iloc[i - 1]
        metrics = ['makespan',
                   'average_flow_A',
                   'average_flow_B',
                   'idle_time_A',
                   'idle_time_B']
        values = [self.schedule['end_B'].iloc[-1],
                  np.mean(self.schedule['end_A']),
                  np.mean(self.schedule['end_B']),
                  idle_A,
                  idle_B]
        self.metrics = pd.DataFrame(data=values, index=metrics, columns=['value'])