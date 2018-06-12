from __future__ import print_function
import csv
import random
import numpy as np


# import tensorflow as tf

# FLAGS = tf.flags.FLAGS

num_words = 1000 # 400000
c_rsize = 137
c_small_rsize = 99
# c_b_learn_hd = True

glove_fn = '../../data/glove/glove.6B.50d.txt'
# tf.flags.DEFINE_float('nn_lrn_rate', 0.001,
# 					  'base learning rate for nn ')
c_key_dim = 50
c_bitvec_size = 50
c_train_fraction = 0.8
c_num_centroids = 7 # should be 200
c_num_ham_winners = 10
c_num_closest_for_change = 10 # The number of closest in train db used to change the bits
c_vote_spreader = 5
c_keep_away_from_avg_factor = 1.
c_eval_every = 1
c_num_percentile_stops = 10
c_val_thresh_step_size = 100 # the val thresh is the threshold for individual input values not the aggrtegate of these threshes
c_num_input_samp = 1 # how many input values to sample and apply thresh to
c_improve_bad = 0.001 # chance that we will ignore the pre-score and select anyway
c_default_thresh_val = 0.

def find_cd_single_closest(train_arr, test_arr):
	l_i_test_closest = []
	for test_vec in test_arr:
		cd = np.dot(train_arr, test_vec)
		l_i_test_closest.append(np.argmax(cd))
	return l_i_test_closest
		# cd_winners = np.delete(cd_winners, np.where(cd_winners == tester))

c_rnd_of_k = 32

def find_cd_train_closest(train_arr):
	k = c_num_closest_for_change
	numrecs = train_arr.shape[0]
	l_closest_l_i , l_closest_l_hd = [], []
	for itest, test_vec in enumerate(train_arr):
		cd = np.dot(train_arr, test_vec)
		cd_winners = np.argpartition(cd, -((c_rnd_of_k*k) + 1))[-((c_rnd_of_k*k) + 1):]
		cd_winners = np.delete(cd_winners, np.where(cd_winners == itest))
		cd_of_winners = cd[cd_winners]
		iwinners = np.argsort(-cd_of_winners)
		cd_idx_sorted = cd_winners[iwinners]
		cd_of_winners = cd[cd_idx_sorted]
		hd_of_winners = np.round_((c_bitvec_size / 2) - (cd_of_winners * (c_bitvec_size / 2)))
		idx_best = cd_idx_sorted[:(k/2)]
		hd_best = hd_of_winners[:(k/2)]
		rnd_next = np.random.randint(k/2, high=k*c_rnd_of_k, size=k/2)
		idx_next = cd_idx_sorted[rnd_next]
		hd_next = hd_of_winners[rnd_next]

		cd_middle = np.argpartition(cd, numrecs/2)[:(k*c_rnd_of_k)]
		rnd_middle = np.random.randint(0, high=k*c_rnd_of_k, size=k/2)
		cd_of_middle = cd[cd_middle[rnd_middle]]
		hd_of_middle = np.round_((c_bitvec_size / 2) - (cd_of_middle * (c_bitvec_size / 2)))

		cd_losers = np.argpartition(cd, c_rnd_of_k*k)[:(c_rnd_of_k*k)]
		rnd_losers = np.random.randint(0, high=k*c_rnd_of_k, size=k/2)
		cd_of_losers = cd[cd_losers[rnd_losers]]
		hd_of_losers = np.round_((c_bitvec_size / 2) - (cd_of_losers * (c_bitvec_size / 2)))

		l_closest_l_i.append(np.concatenate((idx_best, idx_next, cd_middle[rnd_middle], cd_losers[rnd_losers])))
		l_closest_l_hd.append(np.concatenate((hd_best, hd_next, hd_of_middle, hd_of_losers)))
		# winner_outputs = nd_phrase_bits_db[hd_idx_sorted]
		# l_i_test_closest.append(np.argmax(cd))
	return l_closest_l_i, l_closest_l_hd

def get_score(bin_db, l_closest_l_i, l_closest_l_hd):
	score = 0.
	for itest, test_vec in enumerate(bin_db):
		e = bin_db[l_closest_l_i[itest]].astype(float)
		g = np.sum(test_vec != e, axis=1)
		l_hd = l_closest_l_hd[itest]
		divisor = np.where(np.arange(c_num_closest_for_change * 2) < c_num_closest_for_change, np.ones_like(l_hd), np.ones_like(l_hd)*3.0)
		score += np.average(np.abs((g - l_hd)/divisor))

	return score / bin_db.shape[0]

def find_target_bits(word_arr, sel_mat, bin_db, l_closest_l_i, bit_counts, l_closest_l_hd, bprint):
	# bprint = True
	numrecs = bin_db.shape[0]
	bitcomps, thresh, nd_steps = sel_mat
	a = np.tile(np.expand_dims(word_arr, axis=-1), reps=[1, c_bitvec_size])
	b = np.tile(np.expand_dims(bitcomps, axis=0), reps=[numrecs, 1, 1])
	c = np.where(a>b, np.ones_like(a), np.zeros_like(a))
	nd_sel_mat_change = np.zeros((c_key_dim, c_bitvec_size))
	score = 0.
	for itest, test_vec in enumerate(bin_db):
		d = c[itest]
		e = bin_db[l_closest_l_i[itest]].astype(float)
		g = np.sum(test_vec != e, axis=1)
		ichange = np.random.randint(c_bitvec_size)
		changed_vec = np.copy(test_vec)
		changed_vec[ichange] = 1 - test_vec[ichange]
		h = np.sum(changed_vec != e, axis=1)
		l_hd = l_closest_l_hd[itest]
		m = np.square(h - l_hd) < np.square(g - l_hd)
		if bprint:
			if np.any(l_hd == 0):
				score += 1.
			else:
				score += np.average(np.abs((g - l_hd)/l_hd))
		if np.sum(m) < (l_hd.shape[0] / 2):
			continue
		# f = (np.sum((e.transpose() * find_target_bits.factor).transpose(), axis=0)
		# 	+ ((1. - bit_counts) * c_keep_away_from_avg_factor)) \
		# 	/ (find_target_bits.factor_sum + c_keep_away_from_avg_factor)
		f = np.full(c_bitvec_size, 0.5)
		f[ichange] = changed_vec[ichange]
		# if bprint:
		# 	score += np.average(np.square(f - test_vec.astype(float)))
		# for ibit, target_val in enumerate(f):
		# 	h = d[:, ibit]
		# 	g = h * (0.5 - target_val) if target_val < 0.5 else (1.0 - h) * (0.5 - target_val)
		# 	# g = d[:, ibit] * (target_val - 0.5)
		# 	nd_sel_mat_change[:, ibit] += g
		# m_plus = np.where(f>0.5)
		ga = np.where(f<0.5, d * (0.5 - f), (1.0 - d) * (.5 - f))
		nd_sel_mat_change += ga
	if bprint:
		print('Score from diffs:', score / numrecs)
	return nd_sel_mat_change / numrecs
		# c = np.tile(np.expand_dims(b, axis=0), reps=[c_key_dim, 1])
		# d = np.tile(np.expand_dims(test_vec, axis=0), reps=[c_key_dim, 1])

		# d = np.where()
		# nd_sel_mat_change[itest] =

find_target_bits.factor = 1.0 / np.array(range(c_vote_spreader , c_num_closest_for_change+c_vote_spreader))
find_target_bits.factor_sum = np.sum(find_target_bits.factor)

def create_baseline(nd_full_db):
	# arr_median = np.tile(np.median(nd_full_db, axis=1), (nd_full_db.shape[1], 1)).transpose()
	# return np.greater(nd_full_db, arr_median).astype(np.float32)
	arr_median = np.median(nd_full_db, axis=0)
	return np.where(nd_full_db > arr_median, np.ones_like(nd_full_db), np.zeros_like(nd_full_db)).astype(np.int)


def test(train_bin_db, test_bin_arr, l_i_test_best, rat):
	num_hits, num_poss = 0.0, 0.0
	for itest, test_vec in enumerate(test_bin_arr):
		hd = np.sum(np.where(np.not_equal(test_vec, train_bin_db), np.ones_like(train_bin_db), np.zeros_like(train_bin_db)), axis=1)
		hd_winners = np.argpartition(hd, (rat + 1))[:(rat + 1)]
		num_hits += 1.0 if np.any(hd_winners == l_i_test_best[itest]) else 0.0

	return num_hits / float(test_bin_arr.shape[0])



def load_word_dict():
	global g_word_vec_len
	glove_fh = open(glove_fn, 'rb')
	glove_csvr = csv.reader(glove_fh, delimiter=' ', quoting=csv.QUOTE_NONE)

	word_dict = {}
	word_arr = []
	for irow, row in enumerate(glove_csvr):
		word = row[0]
		vec = [float(val) for val in row[1:]]
		vec = np.array(vec, dtype=np.float32)
		en = np.linalg.norm(vec, axis=0)
		vec = vec / en
		word_dict[word] = vec
		word_arr.append(vec)
		if irow > num_words:
			break
	# print(row)

	glove_fh.close()
	g_word_vec_len = len(word_dict['the'])
	random.shuffle(word_arr)
	return word_dict, np.array(word_arr)


def create_sel_mat(word_arr):
	# nd_percentile_stops = np.zeros((c_num_percentile_stops, c_key_dim))
	nd_min, nd_max = np.min(word_arr, axis=0), np.max(word_arr, axis=0)
	l_decile_stops = [float(decile) * (100.0 / float(c_num_percentile_stops)) for decile in range(c_num_percentile_stops+1)]
	nd_percentile = np.percentile(word_arr, l_decile_stops, axis=0)
	nd_steps = (nd_percentile[-2] - nd_percentile[1]) / c_val_thresh_step_size
	nd_mins, nd_maxs = nd_percentile[0], nd_percentile[-1]
	# nd_val_thresh = np.random.choice(a=np.arange(2,7), size=(c_key_dim, c_bitvec_size))
	# nd_thresh = np.zeros((c_key_dim, c_bitvec_size))
	# for ival in range(c_key_dim):
	# 	nd_thresh[ival, :] = np.take(nd_percentile[:, ival], nd_val_thresh[ival, :])
	# nd_thresh_ivals = np.asarray([np.take(nd_percentile[:, ival], nd_val_thresh[ival, :]) for ival in range(c_key_dim)])
	num_samps = c_num_input_samp * c_bitvec_size
	l_samp_src = []
	num_placed = 0
	while num_placed < num_samps:
		num_to_place = min(num_samps-num_placed, c_key_dim)
		l_samp_src += range(num_to_place)
		num_placed += num_to_place
	random.shuffle(l_samp_src)
	# l_samp_thresh = [nd_percentile[random.randint(0,c_num_percentile_stops), samp] for samp in l_samp_src]
	l_samp_thresh = [nd_percentile[c_num_percentile_stops/2, samp] for samp in l_samp_src]
	l_samp_steps = [nd_steps[samp] for samp in l_samp_src]
	l_mins, l_maxs = [nd_mins[samp] for samp in l_samp_src], [nd_maxs[samp] for samp in l_samp_src]
	nd_samp_src = np.asarray(l_samp_src).reshape((c_num_input_samp, c_bitvec_size))
	nd_samp_thresh = np.asarray(l_samp_thresh).reshape((c_num_input_samp, c_bitvec_size))
	nd_samp_steps = np.asarray(l_samp_steps).reshape((c_num_input_samp, c_bitvec_size))
	nd_samp_mins = np.asarray(l_mins).reshape((c_num_input_samp, c_bitvec_size))
	nd_samp_maxs = np.asarray(l_maxs).reshape((c_num_input_samp, c_bitvec_size))
	# nd_thresh_vals = np.zeros((c_num_input_samp, c_bitvec_size))
	# for ival in range(c_key_dim):
	# 	r = range(ival, min(ival+c_num_input_samp, c_key_dim)) + range(ival - c_key_dim + c_num_input_samp)
	# 	nd_thresh_vals[:,ival] = nd_thresh_ivals[r,ival]
	# nd_thresh_sum_thresh = np.sum(1.0 - (nd_val_thresh.astype(float) / float(c_num_percentile_stops)), axis=0)
	nd_thresh_sum_thresh = np.full((c_num_input_samp, c_bitvec_size), c_default_thresh_val)
	# return (np.random.rand(c_key_dim, c_bitvec_size), np.full(c_key_dim, c_bitvec_size/2))
	return (nd_samp_src, nd_samp_thresh, nd_thresh_sum_thresh, nd_samp_steps, nd_samp_mins, nd_samp_maxs)

def create_bit_db(word_arr, sel_mat):
	numrecs = word_arr.shape[0]
	bitcomps, thresh, nd_steps = [],[],[]
	nd_samp_src, nd_samp_thresh, nd_thresh_sum_thresh, _, _, _ = sel_mat
	bitcomps = np.full((c_key_dim, c_bitvec_size), 1.1)
	for j in range(c_bitvec_size):
		for i in range(c_num_input_samp):
			bitcomps[nd_samp_src[i,j], j] = nd_samp_thresh[i,j]
	a = np.tile(np.expand_dims(word_arr, axis=-1), reps=[1, c_bitvec_size])
	b = np.tile(np.expand_dims(bitcomps, axis=0), reps=[numrecs, 1, 1])
	c = np.where(a>b, np.ones_like(a), np.zeros_like(a))
	# g = np.sum(c, axis=1)
	d = np.where(np.sum(c, axis=1) > nd_thresh_sum_thresh, np.ones((numrecs, c_bitvec_size), dtype=np.uint8),
				 np.zeros((numrecs, c_bitvec_size), dtype=np.uint8))
	# e = np.sum(d, axis=0).astype(float) / float(numrecs)
	# for iiter in xrange(300):
	# 	thresh += (e - 0.5) * 0.01
	# 	d = np.where(np.sum(c, axis=1) > thresh, np.ones((numrecs, c_bitvec_size), dtype=np.uint8),
	# 				 np.zeros((numrecs, c_bitvec_size), dtype=np.uint8))
	# 	e = np.sum(d, axis=0).astype(float) / float(numrecs)
	# 	f = np.sum(d, axis=0)
	# 	g = np.sum(c, axis=1)
	return d

c_improve_pair = 0.5
def improve_sel_mat(word_arr, test_arr, sel_mat, l_i_test_closest):
	old_bin_db = nd_bin_db = create_bit_db(word_arr, sel_mat)

	numrecs = word_arr.shape[0]
	for iiter in xrange(100000):
		# bitcomps, thresh, nd_steps = sel_mat
		nd_samp_src, nd_samp_thresh, nd_thresh_sum_thresh, nd_samp_steps, nd_samp_mins, nd_samp_maxs = sel_mat
		bit_counts = np.sum(nd_bin_db, axis=0).astype(float) / float(numrecs)
		l_closest_l_i, l_closest_l_hd = find_cd_train_closest(word_arr)
		pre_score = get_score(nd_bin_db, l_closest_l_i, l_closest_l_hd)
		print('score before change', pre_score)
		for ibititer in xrange(100):
			changes = []
			ibit, isamp = random.randint(0, c_bitvec_size-1), random.randint(0, c_num_input_samp-1)
			val = nd_samp_thresh[isamp, ibit] - nd_samp_steps[isamp, ibit]
			if val > nd_samp_mins[isamp, ibit]:
				changes.append([isamp, ibit, val])
			val = nd_samp_thresh[isamp, ibit] + nd_samp_steps[isamp, ibit]
			if val < nd_samp_maxs[isamp, ibit]:
				changes.append([isamp, ibit, val])
			num_changes = len(changes)
			if random.random() < c_improve_pair:
				ibit, isamp = random.randint(0, c_bitvec_size - 1), random.randint(0, c_num_input_samp - 1)
				val = nd_samp_thresh[isamp, ibit] - nd_samp_steps[isamp, ibit]
				if val > nd_samp_mins[isamp, ibit]:
					changes[0] += [isamp, ibit, val]
					if num_changes > 1:
						changes[1] += [isamp, ibit, val]
				val = nd_samp_thresh[isamp, ibit] + nd_samp_steps[isamp, ibit]
				if val < nd_samp_maxs[isamp, ibit]:
					changes.append(changes[0][:3] + [isamp, ibit, val])
					if num_changes > 1:
						changes.append(changes[1][:3] + [isamp, ibit, val])
			# for ibit in range(c_bitvec_size):
			# for isamp in range(c_num_input_samp):
			scores = []
			for change in changes:
				if len(change) > 3:
					bpair = True
					isamp, ibit, val, isamp2, ibit2, val2 = change
				else:
					bpair = False
					isamp, ibit, val = change
				new_samp_thresh = np.copy(nd_samp_thresh)
				new_samp_thresh[isamp, ibit] = val
				if bpair:
					new_samp_thresh[isamp2, ibit2] = val2
				sel_mat = (nd_samp_src, new_samp_thresh, nd_thresh_sum_thresh, nd_samp_steps, nd_samp_mins, nd_samp_maxs)
				nd_bin_db = create_bit_db(word_arr, sel_mat)
				scores.append(get_score(nd_bin_db, l_closest_l_i, l_closest_l_hd))
			bbad = random.random() < c_improve_bad
			print('iter:', iiter, 'bit iter:', ibititer, 'bad:', bbad, 'scores:', scores, 'vs pre:', pre_score)
			best_iscore = np.argmin(scores)
			if scores[best_iscore] < pre_score or bbad:
				pre_score = scores[best_iscore]
				best_change = changes[best_iscore]
				nd_samp_thresh[best_change[0], best_change[1]] = best_change[2]
				if len(best_change) > 3:
					nd_samp_thresh[best_change[3], best_change[4]] = best_change[5]
				# if scores[0] < pre_score and scores[1] > pre_score:
				# 	pre_score = scores[0]
				# 	nd_samp_thresh[isamp, ibit] -= nd_samp_steps[isamp, ibit]
				# elif scores[0] > pre_score and scores[1] < pre_score:
				# 	pre_score = scores[1]
				# 	nd_samp_thresh[isamp, ibit] += nd_samp_steps[isamp, ibit]

		sel_mat = (nd_samp_src, nd_samp_thresh, nd_thresh_sum_thresh, nd_samp_steps, nd_samp_mins, nd_samp_maxs)

		"""
		sel_mat_change =  find_target_bits(word_arr, sel_mat, nd_bin_db, l_closest_l_i, bit_counts, l_closest_l_hd,
										   bprint=(iiter % c_eval_every == 0))
		new_bitcomps = bitcomps + (sel_mat_change.transpose() * nd_steps).transpose()

		sel_mat = new_bitcomps, thresh, nd_steps
		nd_bin_db = create_bit_db(word_arr, sel_mat)
		bit_counts = np.sum(nd_bin_db, axis=0).astype(float) / float(numrecs)
		thresh += (bit_counts - 0.5) * .0
		sel_mat = new_bitcomps, thresh, nd_steps
		"""
		#note. In general, I don't bother recalculating the bin db after th thresh has changed
		if iiter % c_eval_every == 0:
			nd_bin_db, nd_bin_q = create_bit_db(word_arr, sel_mat), create_bit_db(test_arr, sel_mat)
			rat10 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 10)
			print('iiter', iiter, 'r@: 10:', rat10)
			print('Sum of bits in bin array:', np.sum(nd_bin_db))
			print('bit counts:', bit_counts)
			print('num_bit_chages:', np.sum(old_bin_db != nd_bin_db))
			old_bin_db = nd_bin_db
	pass

def main():
	word_dict, word_arr = load_word_dict()
	num_recs_total = word_arr.shape[0]
	train_limit = int(num_recs_total * c_train_fraction)
	nd_train_recs = word_arr[:train_limit, :]
	nd_q_recs = word_arr[train_limit:, :]
	l_i_test_closest = find_cd_single_closest(nd_train_recs, nd_q_recs)
	# l_closest_l_i, l_closest_l_hd = find_cd_train_closest(nd_train_recs)
	nd_bin_db, nd_bin_q = create_baseline(nd_train_recs), create_baseline(nd_q_recs)
	rat10 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 10)
	# rat100 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 100)
	# rat1000 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 1000)
	# print('r@: 10, 100, 1000:', rat10, rat100, rat1000)
	print('baseline r@: 10:', rat10)
	sel_mat = create_sel_mat(nd_train_recs)
	nd_bin_db, nd_bin_q = create_bit_db(nd_train_recs, sel_mat), create_bit_db(nd_q_recs, sel_mat)
	rat10 = test(nd_bin_db, nd_bin_q, l_i_test_closest, 10)
	print('starting r@: 10:', rat10)
	improve_sel_mat(nd_train_recs, nd_q_recs, sel_mat, l_i_test_closest)
	return


main()
print('done')


