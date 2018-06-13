from __future__ import print_function
# import struct
import numpy as np
import os

c_b_small = False
if c_b_small:
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	import tensorflow as tf
	fn_db = '../../data/sift/siftsmall_base.fvecs'
	fn_q = '../../data/sift/siftsmall_query.fvecs'
	fn_gt = '../../data/sift/siftsmall_groundtruth.ivecs'
	c_num_k_eval = 2
	c_num_clusters_q = 100
	c_num_clusters = 800
	c_kmeans_num_db_segs = 2
	c_rat = 100
else:
	# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
	import tensorflow as tf
	fn_db = '../../data/sift/sift_base.fvecs'
	fn_q = '../../data/sift/sift_query.fvecs'
	fn_gt = '../../data/sift/sift_groundtruth.ivecs'
	c_num_k_eval = 10
	c_num_clusters_q = 256
	c_num_clusters = 8192
	c_kmeans_num_db_segs = 40
	c_rat = 1000

c_key_dim = 128
c_kmeans_num_batches = 1
c_kmeans_iters = 6
c_global_rat = c_num_k_eval * c_num_clusters_q
c_b_test_baseline = True # Useful information but slow down terribly on large db's

"""
with open(fn, "rb") as binary_file:
	# Read the whole file at once
	data = binary_file.read()
	i = struct.unpack('<H', data[:2])
	print(i)
"""

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(filename, c_contiguous=True):
    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()

	en = np.linalg.norm(fv, axis=1)
	fv = (fv.transpose() / en).transpose()

	return fv

def make_norm(fv):
	en = np.linalg.norm(fv, axis=1)
	fv = (fv.transpose() / en).transpose()
	return fv

def load_vecs():
	nd_db = fvecs_read(fn_db)
	nd_q = fvecs_read(fn_q)
	nd_gt = ivecs_read(fn_gt)
	return nd_db, nd_q, nd_gt

def create_rot():
	nd_rnd = (np.random.rand(c_key_dim, c_key_dim) -0.5) * 2.
	q,_ = np.linalg.qr(nd_rnd)
	return q

def create_baseline(nd_full_db, arr_median):
	return np.where(nd_full_db > arr_median, np.ones_like(nd_full_db), np.zeros_like(nd_full_db)).astype(np.int)

def test(train_bin_db, test_bin_arr, l_i_test_best, rat):
	num_hits, num_poss = 0.0, 0.0
	for itest, test_vec in enumerate(test_bin_arr):
		if itest % 10 == 0:
			print('Baseline: tested', itest, 'records. Score:', num_hits / float(itest) if itest > 0 else 'n/a')
		hd = np.sum(np.where(np.not_equal(test_vec, train_bin_db), np.ones_like(train_bin_db), np.zeros_like(train_bin_db)), axis=1)
		hd_winners = np.argpartition(hd, (rat + 1))[:(rat + 1)]
		num_hits += 1.0 if np.any(hd_winners == l_i_test_best[itest]) else 0.0

	return num_hits / float(test_bin_arr.shape[0])

def test_cd_single_closest(train_arr, test_arr, l_i_test_closest):
	# l_i_test_closest = []
	score = 0.0
	for itest, test_vec in enumerate(test_arr):
		if (itest % 1000) == 0:
			print('Found single closest cd for', itest, 'records')
		cd = np.dot(train_arr, test_vec)
		if l_i_test_closest[itest] == np.argmax(cd):
			score += 1.0
	score /= float(len(test_arr))
	print('Test of cd and l2 norm:', score)
	return score


def make_db_cg(numrecs):
	v_db_norm = tf.Variable(tf.zeros([numrecs, c_key_dim], dtype=tf.float32), trainable=False)
	ph_db_norm = tf.placeholder(dtype=tf.float32, shape=[numrecs, c_key_dim], name='ph_db_norm')
	op_db_norm_assign = tf.assign(v_db_norm, ph_db_norm, name='op_db_norm_assign')
	return v_db_norm, ph_db_norm, op_db_norm_assign

def get_closest_clusters(l_centroids, l_cluster_idxs, q, k, rat):
	l_i_cd_closest, l_l_k = [], []
	for test_vec in q:
		cd = np.dot(l_centroids, test_vec)
		cd_winners = np.argpartition(cd, -k)[-k:]
		cd_of_winners = cd[cd_winners]
		iwinners = np.argsort(-cd_of_winners)
		cd_idx_sorted = cd_winners[iwinners]
		cd_of_winners = cd[cd_idx_sorted]
		# imax, imin = np.argmax(cd_of_winners), np.argmin(cd_of_winners)
		cdmax, cdmin = cd_of_winners[0], cd_of_winners[-1]
		w = (cd_of_winners - cdmin) / (cdmax - cdmin)
		w /= sum(w)
		l_k, rat_left, aw_used = [], rat, 0.0
		for iw, aw in enumerate(w):
			# keep calculating the fractions as the top is removed
			wlen = l_cluster_idxs[cd_idx_sorted[iw]].shape[0]
			if aw_used > .999:
				break
			wfactor = aw / (1. - aw_used)
			aw_used += aw
			if (rat_left * wfactor) > wlen:
				l_k.append(wlen+1)
				rat_left -= wlen
			else:
				nhere = int(round(rat_left * wfactor))
				rat_left -= nhere
				l_k.append(nhere)

		l_l_k.append(l_k)
		# iwinners = np.argsort(-cd_of_winners)
		# cd_idx_sorted = cd_winners[iwinners]
		l_i_cd_closest.append(cd_idx_sorted)
	return l_i_cd_closest, l_l_k


def make_per_batch_init_cg(numrecs, v_db_norm, num_centroids):
	# The goal is to cluster the convolution vectors so that we can perform dimension reduction
	# KMeans implementation
	# Intitialize the centroids indicies. Shape=[num_centroids]
	t_centroids_idxs_init = tf.random_uniform([num_centroids], 0, numrecs - 1, dtype=tf.int32,
											  name='t_centroids_idxs_init')
	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_centroids = tf.Variable(tf.zeros([num_centroids, c_key_dim], dtype=tf.float32), name='v_centroids')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))

	return v_centroids, op_centroids_init

def make_closest_idxs_cg(v_db_norm, v_all_centroids, num_db_seg_entries, num_tot_db_entries):
	ph_i_db_seg = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_db_seg')
	# Create actual centroids as seeds. Shape=[num_centroids, c_kernel_size]
	# op_centroids_init = tf.assign(v_centroids, tf.gather(v_db_norm, t_centroids_idxs_init, name='op_centroids_init'))
	# Do cosine distances for all centroids on all elements of the db. Shape [num_centroids, num_db_seg_entries]
	t_all_CDs = tf.matmul(v_all_centroids, v_db_norm[ph_i_db_seg*num_db_seg_entries:(ph_i_db_seg+1)*num_db_seg_entries, :], transpose_b=True, name='t_all_CDs')
	# For each entry in the chunk database, find the centroid that's closest.
	# Basically, we are finding which centroid had the highest cosine distance for each entry of the chunk db
	# This holds the index to the centroid which we can then use to create an average among the entries that voted for it
	# Shape=[num_db_seg_entries]
	t_closest_idxs_seg = tf.argmax(t_all_CDs, axis=0, name='t_closest_idxs_seg')
	# unconnected piece of cg building. Create a way of assigning np complete array back into the tensor Variable
	# code remains here because it would  be nice to replace with an in-graph assignment like TensorArray
	ph_closest_idxs = tf.placeholder(	shape=[num_tot_db_entries], dtype=tf.int32,
										name='ph_closest_idxs')
	v_closest_idxs = tf.Variable(tf.zeros(shape=[num_tot_db_entries], dtype=tf.int32),
								 name='v_closest_idxs')
	op_closest_idxs_set = tf.assign(v_closest_idxs, ph_closest_idxs, name='op_closest_idxs_set')

	return ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs

def vote_for_centroid_cg(v_db_norm, v_closest_idxs, num_tot_db_entries):
	# create placehoder to tell the call graph which iteration, i.e. which centroid we are working on
	ph_i_centroid = tf.placeholder(dtype=tf.int32, shape=(), name='ph_i_centroid')
	# Create an array of True if the closest index was this centroid
	# Shape=[num_centroids]
	t_vote_for_this = tf.equal(v_closest_idxs, ph_i_centroid, name='t_vote_for_this')
	# Count the number of trues in the vote_for_tis tensor
	# Shape=()
	t_vote_count = tf.reduce_sum(tf.cast(t_vote_for_this, tf.float32), name='t_vote_count')
	# Create the cluster. Use the True positions to put in the values from the v_db_norm and put zeros elsewhere.
	# This means that instead of a short list of the vectors in this cluster we use the full size with zeros for non-members
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_this_cluster = tf.where(t_vote_for_this, v_db_norm,
							  tf.zeros([num_tot_db_entries, c_key_dim]), name='t_this_cluster')
	# Sum the values for each property to get the aveage property
	# Shape=[c_kernel_size]
	t_cluster_sum = tf.reduce_sum(t_this_cluster, axis=0, name='t_cluster_sum')
	# Shape=[c_kernel_size]
	t_avg = tf.cond(t_vote_count > 0.0,
					lambda: tf.divide(t_cluster_sum, t_vote_count),
					lambda: tf.zeros([c_key_dim]),
					name='t_avg')

	return ph_i_centroid, t_avg, t_vote_count, t_vote_for_this

def update_centroids_cg(v_db_norm, v_all_centroids, v_closest_idxs, num_tot_db_entries, num_centroids):
	ph_new_centroids = tf.placeholder(dtype=tf.float32, shape=[num_centroids, c_key_dim], name='ph_new_centroids')
	ph_votes_count = tf.placeholder(dtype=tf.float32, shape=[num_centroids], name='ph_votes_count')
	# Do random centroids again. This time for filling in
	t_centroids_idxs = tf.random_uniform([num_centroids], 0, num_tot_db_entries - 1, dtype=tf.int32, name='t_centroids_idxs')
	# Shape = [num_centroids, c_kernel_size]
	# First time around I forgot that I must normalize the centroids as required for shperical k-means. Avg, as above, will not produce a normalized result
	t_new_centroids_norm = tf.nn.l2_normalize(ph_new_centroids, dim=1, name='t_new_centroids_norm')
	# Shape=[num_centroids]
	t_votes_count = ph_votes_count
	# take the new random idxs and gather new centroids from the db. Only used in case count == 0. Shape=[num_centroids, c_kernel_size]
	t_centroids_from_idxs = tf.gather(v_db_norm, t_centroids_idxs, name='t_centroids_from_idxs')
	# Assign back to the original v_centroids so that we can go for another round
	op_centroids_update = tf.assign(v_all_centroids, tf.where(tf.greater(ph_votes_count, 0.0), t_new_centroids_norm,
														  t_centroids_from_idxs, name='centroids_where'),
									name='op_centroids_update')

	# The following section of code is designed to evaluate the cluster quality, specifically the average distance of a conv fragment from
	# its centroid.
	# t_closest_idxs is an index for each element in the database, specifying which cluster it belongs to. So we use that to
	# replicate the centroid of that cluster to the locations alligned with each member of the database
	# Shape=[num_tot_db_entries, c_kernel_size]
	t_centroid_broadcast = tf.gather(v_all_centroids, v_closest_idxs, name='t_centroid_broadcast')
	# element-wise multiplication of each property and the sum down the properties. It is reallt just a CD but we aren't using matmul
	# Shape=[num_tot_db_entries]
	t_cent_dist = tf.reduce_sum(tf.multiply(v_db_norm, t_centroid_broadcast), axis=1, name='t_cent_dist')
	# Extract a single number representing the kmeans error. This is the mean of the distances from closest centers. Shape=()
	t_kmeans_err = tf.reduce_mean(t_cent_dist, name='t_kmeans_err')
	return t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count

def eval_clusters(l_clusters, l_cluster_idxs, q, k_src, l_i_cluster_closest, l_i_test_closest, l_l_k):
	num_hit = 0.
	for itest, test_vec in enumerate(q):
		for iicluster, icluster in enumerate(l_i_cluster_closest[itest]):
			# k = k_src

			if len(l_l_k[itest]) <= iicluster:
				break
			k = l_l_k[itest][iicluster]

			# k = 100
			if k == 0:
				continue
			elif k >= l_clusters[icluster].shape[0]:
				cd_winners = range(l_clusters[icluster].shape[0])
			else:
				cd = np.dot(l_clusters[icluster], test_vec)
				cd_winners = np.argpartition(cd, -k)[-k:]
			if l_i_test_closest[itest] in l_cluster_idxs[icluster][cd_winners]:
				num_hit += 1.
				break
	score = num_hit / q.shape[0]
	return score

def build_bin_clusters(l_clusters, nd_median):
	l_bin_clusters = []
	for nd_cluster in l_clusters:
		l_bin_clusters.append(np.where(nd_cluster > nd_median,
									   np.ones_like(nd_cluster), np.zeros_like(nd_cluster)).astype(np.int))

	return l_bin_clusters

def eval_clusters_on_bin(l_cluster_idxs, l_bin_clusters, l_i_cluster_closest, l_l_k, test_bin_arr, l_i_test_best):
	# def test(train_bin_db, test_bin_arr, l_i_test_best, rat):
	num_hits, num_poss = 0.0, 0.0
	for itest, test_vec in enumerate(test_bin_arr):
		for iicluster, icluster in enumerate(l_i_cluster_closest[itest]):
			cluster = l_bin_clusters[icluster]
			if len(l_l_k[itest]) <= iicluster:
				break
			k = l_l_k[itest][iicluster]
			# k = c_num_k_eval
			if k == 0:
				continue
			elif k >= cluster.shape[0]:
				hd_winners = range(cluster.shape[0])
			else:
				hd = np.sum(np.where(np.not_equal(test_vec, cluster),
									 np.ones_like(cluster), np.zeros_like(cluster)), axis=1)
				hd_winners = np.argpartition(hd, k)[:k]
				# idx_winners = [l_cluster_idxs[icluster][idx] for ]
			if np.any(l_cluster_idxs[icluster][hd_winners] == l_i_test_best[itest]):
				num_hits += 1.0
				break

	return num_hits / float(test_bin_arr.shape[0])


def prep_learn():
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	return sess

def learn(sess, nd_train_recs):
	numrecs = nd_train_recs.shape[0]
	# v_full_db = tf.Variable(tf.zeros([numrecs, c_key_dim], dtype=tf.float32), name='v_full_db')
	v_db_norm, ph_db_norm, op_db_norm_assign = make_db_cg(numrecs)
	v_centroids, op_centroids_init = make_per_batch_init_cg(numrecs, v_db_norm, c_num_clusters)

	# Get the centroids variable ready. Must persist between loops. Shape=[c_num_convs*c_num_files_per_batch, c_kernel_size]
	v_all_centroids = tf.Variable(tf.zeros([c_num_clusters * c_kmeans_num_batches, c_key_dim],
										   dtype=tf.float32), name='v_centroids')
	ph_all_centroids = tf.placeholder(dtype=tf.float32,
									  shape=[c_num_clusters * c_kmeans_num_batches, c_key_dim],
									  name='ph_all_centroids')
	op_all_centroids_set = tf.assign(v_all_centroids, ph_all_centroids, name='op_all_centroids_set')

	ph_random_centroids = tf.placeholder(dtype=tf.float32,
										 shape=[c_num_clusters * c_kmeans_num_batches, c_key_dim],
										 name='ph_random_centroids')
	ph_centroid_sums  = tf.placeholder(dtype=tf.float32,
									   shape=[c_num_clusters * c_kmeans_num_batches, c_key_dim],
									   name='ph_centroid_sums')
	ph_count_sums  = tf.placeholder(dtype=tf.float32,
									shape=[c_num_clusters * c_kmeans_num_batches, c_key_dim],
									name='ph_count_sums')
	t_new_centroids = tf.where(ph_count_sums > 0.0, ph_centroid_sums/ph_count_sums, ph_random_centroids)
	op_all_centroids_norm_set = tf.assign(v_all_centroids, tf.nn.l2_normalize(t_new_centroids, dim=1),
										  name='op_all_centroids_norm_set')

	# cg to create the closest_idxs for one segment of one batch of the v_db
	ph_i_db_seg, t_closest_idxs_seg, ph_closest_idxs, op_closest_idxs_set, v_closest_idxs \
		= make_closest_idxs_cg(	v_db_norm, v_all_centroids,
								num_db_seg_entries = numrecs / c_kmeans_num_db_segs,
								num_tot_db_entries = numrecs)

	# Create cg that calculates the votes for just one centroid, must be fed the index of the centroid to calculate for
	ph_i_centroid, t_avg, t_vote_count, t_vote_for_this \
		= vote_for_centroid_cg(	v_db_norm, v_closest_idxs,
								num_tot_db_entries = numrecs)

	t_kmeans_err, op_centroids_update, ph_new_centroids, ph_votes_count, t_votes_count \
		= update_centroids_cg(	v_db_norm, v_all_centroids, v_closest_idxs,
								num_tot_db_entries = numrecs * c_kmeans_num_batches,
								num_centroids = c_num_clusters * c_kmeans_num_batches)


	nd_all_controids = np.zeros([c_num_clusters * c_kmeans_num_batches, c_key_dim], dtype=np.float32)

	nd_db_norm = sess.run(op_db_norm_assign, feed_dict={ph_db_norm: nd_train_recs})
	# nd_db_norm = sess.run(v_db_norm)
	for ibatch in range(c_kmeans_num_batches):
		nd_all_controids[ibatch * c_num_clusters:(ibatch + 1) * c_num_clusters] = sess.run(op_centroids_init)
		print('building initial db. ibatch=', ibatch)

	sess.run(op_all_centroids_set, feed_dict={ph_all_centroids:nd_all_controids})

	for iter_kmeans in range(c_kmeans_iters):
		l_centroid_avgs = []
		l_centroid_counts = []
		l_kmeans_err = []
		l_cluster_idxs = [[] for _ in range(c_num_clusters * c_kmeans_num_batches)]
		for ibatch in range(c_kmeans_num_batches):
			# nd_y_db = sess.run(t_y_db)
			# sess.run(op_db_norm_assign, feed_dict={ph_db_norm: nd_train_recs})
			for iseg in range(c_kmeans_num_db_segs):
				n1 = sess.run(t_closest_idxs_seg, feed_dict={ph_i_db_seg:iseg})
				if iseg == 0:
					nd_closest_idxs = n1
				else:
					nd_closest_idxs = np.concatenate([nd_closest_idxs, n1], axis=0)
			sess.run(op_closest_idxs_set, feed_dict={ph_closest_idxs:nd_closest_idxs})
			nd_new_centroids = np.ndarray(dtype = np.float32, shape = [c_num_clusters * c_kmeans_num_batches, c_key_dim])
			nd_votes_count = np.ndarray(dtype = np.float32, shape = [c_num_clusters * c_kmeans_num_batches])
			for icent in range(c_num_clusters * c_kmeans_num_batches):
				if icent != 0 and icent % 100 == 0:
					print('building kmeans db. centroid # =', icent)
				r_cent_avg, r_cent_vote_count, r_vote_for_this \
					= sess.run([t_avg, t_vote_count, t_vote_for_this], feed_dict={ph_i_centroid:icent})
				nd_new_centroids[icent, : ]  = r_cent_avg
				nd_votes_count[icent] = r_cent_vote_count
				nd_cluster = np.extract(r_vote_for_this, range(numrecs))
				if l_cluster_idxs[icent] == []:
					l_cluster_idxs[icent] = nd_cluster
				else:
					np.concatenate([l_cluster_idxs[icent], nd_cluster], axis=0)
			r_votes_count, r_centroids, r_kmeans_err \
				= sess.run(	[t_votes_count, op_centroids_update, t_kmeans_err],
							feed_dict={ph_new_centroids:nd_new_centroids, ph_votes_count:nd_votes_count})
			l_centroid_avgs.append(r_centroids)
			l_centroid_counts.append(r_votes_count)
			l_kmeans_err.append(r_kmeans_err)
			print('building kmeans db. ibatch=', ibatch)
		np_centroid_avgs = np.stack(l_centroid_avgs)
		np_centroid_counts = np.stack(l_centroid_counts)
		np_count_sums = np.tile(np.expand_dims(np.sum(np_centroid_counts, axis=0), axis=-1), reps=[1, c_key_dim])
		np_br_centroid_counts = np.tile(np.expand_dims(np_centroid_counts, axis=-1), reps=[1, c_key_dim])
		np_centroid_facs = np.multiply(np_centroid_avgs, np_br_centroid_counts)
		np_centroid_sums = np.sum(np_centroid_facs, axis=0)
		np.random.shuffle(nd_all_controids)
		r_centroids = sess.run(	op_all_centroids_norm_set,
								feed_dict={	ph_random_centroids: nd_all_controids,
											ph_centroid_sums: np_centroid_sums,
											ph_count_sums: np_count_sums})
		print('kmeans iter:', iter_kmeans, 'kmeans err:', np.mean(np.stack(l_kmeans_err)))

	return r_centroids, l_cluster_idxs


def main():
	nd_db, nd_q, nd_gt = load_vecs()
	q_of_qr = create_rot()
	nd_db, nd_q = np.matmul(nd_db, q_of_qr), np.matmul(nd_q, q_of_qr)
	# nd_db, nd_q = make_norm(nd_db), make_norm(nd_q)
	nd_median = np.median(nd_db, axis=0)
	nd_bin_db, nd_bin_q = create_baseline(nd_db, nd_median), create_baseline(nd_q, nd_median)
	l_i_test_best = [nd_gt[iq][0] for iq in xrange(nd_q.shape[0])]
	if c_b_small:
		test_cd_single_closest(nd_db, nd_q, l_i_test_best)
		rat = test(nd_bin_db, nd_bin_q, l_i_test_best, c_rat)
		print('baseline r@:', c_rat, ':', rat)
	sess = prep_learn()
	l_centroids, l_cluster_idxs = learn(sess, nd_db)
	l_clusters = []
	for cluster_idxs in l_cluster_idxs:
		l_clusters.append(nd_db[cluster_idxs])
	l_i_cluster_closest, l_k = get_closest_clusters(l_centroids, l_cluster_idxs, nd_q, c_num_clusters_q, c_global_rat)
	score = eval_clusters(l_clusters, l_cluster_idxs, nd_q, c_num_k_eval, l_i_cluster_closest, l_i_test_best, l_k)
	print(c_num_clusters_q, 'out of', c_num_clusters, 'clusters, k =', c_num_k_eval, 'per cluster,', nd_db.shape[0], 'training records', nd_q.shape[0], 'queries. score:', score)

	l_bin_clusters = build_bin_clusters(l_clusters, nd_median)
	final_score = eval_clusters_on_bin(l_cluster_idxs, l_bin_clusters, l_i_cluster_closest, l_k, nd_bin_q, l_i_test_best)
	print('r@:',c_global_rat, ':', final_score)
	return

main()
print('done')
