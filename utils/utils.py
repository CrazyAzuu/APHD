import numpy as np
import torch
import time
import math
import os
import random
from utils.logger import logger

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

def soft_update(target, source, tau=0.001):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Progress:

	def __init__(self, total, name='Progress', ncol=3, max_length=20, indent=0, line_width=100, speed_update_freq=100):
		self.total = total
		self.name = name
		self.ncol = ncol
		self.max_length = max_length
		self.indent = indent
		self.line_width = line_width
		self._speed_update_freq = speed_update_freq

		self._step = 0
		self._prev_line = '\033[F'
		self._clear_line = ' ' * self.line_width

		self._pbar_size = self.ncol * self.max_length
		self._complete_pbar = '#' * self._pbar_size
		self._incomplete_pbar = ' ' * self._pbar_size

		self.lines = ['']
		self.fraction = '{} / {}'.format(0, self.total)

		self.resume()

	def update(self, description, n=1):
		self._step += n
		if self._step % self._speed_update_freq == 0:
			self._time0 = time.time()
			self._step0 = self._step
		self.set_description(description)

	def resume(self):
		self._skip_lines = 1
		print('\n', end='')
		self._time0 = time.time()
		self._step0 = self._step

	def pause(self):
		self._clear()
		self._skip_lines = 1

	def set_description(self, params=[]):

		if type(params) == dict:
			params = sorted([
				(key, val)
				for key, val in params.items()
			])

		############
		# Position #
		############
		self._clear()

		###########
		# Percent #
		###########
		percent, fraction = self._format_percent(self._step, self.total)
		self.fraction = fraction

		#########
		# Speed #
		#########
		speed = self._format_speed(self._step)

		##########
		# Params #
		##########
		num_params = len(params)
		nrow = math.ceil(num_params / self.ncol)
		params_split = self._chunk(params, self.ncol)
		params_string, lines = self._format(params_split)
		self.lines = lines

		description = '{} | {}{}'.format(percent, speed, params_string)
		print(description)
		self._skip_lines = nrow + 1

	def append_description(self, descr):
		self.lines.append(descr)

	def _clear(self):
		position = self._prev_line * self._skip_lines
		empty = '\n'.join([self._clear_line for _ in range(self._skip_lines)])
		print(position, end='')
		print(empty)
		print(position, end='')

	def _format_percent(self, n, total):
		if total:
			percent = n / float(total)

			complete_entries = int(percent * self._pbar_size)
			incomplete_entries = self._pbar_size - complete_entries

			pbar = self._complete_pbar[:complete_entries] + self._incomplete_pbar[:incomplete_entries]
			fraction = '{} / {}'.format(n, total)
			string = '{} [{}] {:3d}%'.format(fraction, pbar, int(percent * 100))
		else:
			fraction = '{}'.format(n)
			string = '{} iterations'.format(n)
		return string, fraction

	def _format_speed(self, n):
		num_steps = n - self._step0
		t = time.time() - self._time0
		speed = num_steps / t
		string = '{:.1f} Hz'.format(speed)
		if num_steps > 0:
			self._speed = string
		return string

	def _chunk(self, l, n):
		return [l[i:i + n] for i in range(0, len(l), n)]

	def _format(self, chunks):
		lines = [self._format_chunk(chunk) for chunk in chunks]
		lines.insert(0, '')
		padding = '\n' + ' ' * self.indent
		string = padding.join(lines)
		return string, lines

	def _format_chunk(self, chunk):
		line = ' | '.join([self._format_param(param) for param in chunk])
		return line

	def _format_param(self, param):
		k, v = param
		return '{} : {}'.format(k, v)[:self.max_length]

	def stamp(self):
		if self.lines != ['']:
			params = ' | '.join(self.lines)
			string = '[ {} ] {}{} | {}'.format(self.name, self.fraction, params, self._speed)
			self._clear()
			print(string, end='\n')
			self._skip_lines = 1
		else:
			self._clear()
			self._skip_lines = 0

	def close(self):
		self.pause()

class Silent:

	def __init__(self, *args, **kwargs):
		pass

	def __getattr__(self, attr):
		return lambda *args: None

class EarlyStopping(object):
	def __init__(self, tolerance=5, min_delta=0):
		self.tolerance = tolerance
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False

	def __call__(self, train_loss, validation_loss):
		if (validation_loss - train_loss) > self.min_delta:
			self.counter += 1
			if self.counter >= self.tolerance:
				return True
		else:
			self.counter = 0
		return False

def generate_horizon_sequence(horizon_up):
    sequence = []
    current = 1
    while current <= horizon_up:
        sequence.append(current)
        current *= 2
    return tuple(sequence)

# Set a random number seed for each subprocess
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
	
def set_seed(seed):
	# os.environ['PYTHONHASHSEED'] = str(seed)

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.enabled = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	# torch.use_deterministic_algorithms(True)  

def formate_print(top_k, reward:dict, sps:dict):
	print("========================================")
	print(f"top_k: {top_k}")
	print("========================================")
	for key in reward:
		print(f"{key}: {reward[key]}")
	print("========================================")
	for key in sps:
		print(f"{key}: {sps[key]}")
	print("========================================")

def record(epoch, loss, reward_td, sps_td):
	logger.record_tabular('Trained Epochs', epoch)
	logger.record_tabular('Low Planner Loss', loss['loss/planner_low'])
	logger.record_tabular('Up Planner Loss', loss['loss/planner_up'])
	logger.record_tabular('Model Loss', loss['loss/model'])
	logger.record_tabular('Actor Loss', loss['loss/actor'])
	logger.record_tabular('Critic Loss', loss['loss/critic'])
	logger.record_tabular('Value Loss', loss['loss/value'])
	logger.dump_tabular()

	logger.record_tabular('Average Reward', round(reward_td["reward/avg"], 3))
	logger.record_tabular('Average N-Reward', round(reward_td["reward/avg_normalized"], 3))
	logger.record_tabular('Standard Deviation Reward', round(reward_td["reward/std"], 3))
	logger.record_tabular('Standard Deviation N-Reward', round(reward_td["reward/std_normalized"], 3))
	logger.record_tabular('Max Reward', round(reward_td["reward/max"], 3))
	logger.record_tabular('Max N-Reward', round(reward_td["reward/max_normalized"], 3))
	logger.dump_tabular()

	logger.record_tabular('Average Seconds-Per-Step', round(sps_td["sps/avg"], 2))
	logger.record_tabular('Min Seconds-Per-Step', round(sps_td["sps/min"], 2))
	logger.dump_tabular()