#!/usr/bin/env python

"""
Compute distance and nearest neighbours to find twins in the HCP database,
based on Champollion V1 embeddings.
"""

import pandas as pd
import numpy as np
import os
import os.path as osp
import sklearn
import csv
import cycler
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import glob
import json

# paths for data and embeddings (valid at Neurospin on the intra network)

participants_file = '/neurospin/dico/data/bv_databases/human/not_labeled/hcp/participants.csv'
restricted_file = '/neurospin/dico/jchavas/RESTRICTED_jchavas_1_18_2022_3_17_51.csv'
embeddings_dir = '/neurospin/dico/data/deep_folding/current/models/Champollion_V1_after_ablation'
sub_embeddings = '*/hcp_random_embeddings/full_embeddings.csv'
#embeddings_dir = '/neurospin/dico/jlaval/Runs_jl277509/yAwareContrastiveLearning/dataset/hcp_embeddings_nibabel.csv'
out_dist_file = '/neurospin/dico/data/bv_databases/human/not_labeled/hcp/tables/BL/twin_distances_champollion_%s.csv'
#out_dist_file = '/neurospin/dico/data/bv_databases/human/not_labeled/hcp/tables/BL/twin_distances_benoit_%s.csv'
out_dir = '/neurospin/dico/driviere/hcp_twin_stats/champollion'
#out_dir = '/neurospin/dico/driviere/hcp_twin_stats/benoit'
regions_list_f = '/neurospin/dico/data/deep_folding/current/sulci_regions_champollion_V1.json'


def read_embeddings(embeddings_dir=embeddings_dir, regions_list=None):
    """ Read either a single embeddings file (.csv) or a series (a set of
    regions) from a directory, and possibly a regions list.

    If embeddings_dir is a directory and regions_list is None, try to consider
    every sub-directory of embeddings_dir as a region.
    If regions_list is provided, it should be a hierarchical dict of regions,
    whose root node (single key) is "brain". The 2nd level is the regions we
    are looking for. Regions directories in embeddings_dir are obtained by
    removing dots (".") from the regions_list regions names.

    A region sub-directory should contain a file
    `hcp_random_embeddings/full_embeddings.csv`

    returns a Pandas DataFrame. When several regions are read, embeddings are
    concatenated (columns).
    """
    if embeddings_dir.endswith('.csv'):
        # single file, whole brain model
        embeddings = pd.read_csv(embeddings_dir, index_col='ID',
                                 dtype={'ID': str})
        return embeddings

    embeddings_l = []
    embeddings = None
    nr = 0
    if regions_list is not None:
        regions = [r.replace('.', '') for r in regions_list['brain']]
    else:
        regions = os.listdir(embeddings_dir)

    for region in regions:
        region_dir = osp.join(embeddings_dir, region)
        if not osp.isdir(region_dir):
            continue
        emb_files = []
        for d in os.listdir(region_dir):
            md = osp.join(region_dir, d)
            emb_f = osp.join(md, 'hcp_random_embeddings/full_embeddings.csv')
            if not osp.exists(emb_f):
                continue
            emb_files.append(emb_f)
        if len(emb_files) == 0:
            continue  # not a model dir
        emb_files = sorted(emb_files)
        if len(emb_files) > 1:
            print('warning: several model/embeddings for region', region, ':')
            print(emb_files)
            print('taking lastest one')
        emb_f = emb_files[-1]

        embedding = pd.read_csv(emb_f, index_col='ID', dtype={'ID': str})
        cols = list(embedding.columns)
        cols[:] = [(region, c) for c in cols]
        embedding.columns = cols
        embeddings_l.append(embedding)
        del embedding
        nr += 1

    print('nb of regions:', nr)
    # embeddings = pd.concat(embeddings_l, axis=1, keys=('ID', ))
    embeddings = embeddings_l[0]
    embeddings = embeddings.join(embeddings_l[1:])
    # for embedding in embeddings_l[1:]:
    #     embeddings.join(embedding, 'ID')
    del embeddings_l
    return embeddings


def get_twins(participants, twin_types, df, monoz):
    """ Get twin pairs and metadata dicts from a participants dataframe and
    subjects information.

    Parameters
    ----------
    participants: pandas.DataFrame
        participants table
    twin_types: sequence
        each element is a string identifying the twins type (monozygote,
        dizygote). Not used, actually.
    df: sequence
        each element is a DataFrame for a given kind of twins (monozygopte,
        dizygote). We find twins as having the same mother and father ID in the
        table. So we don't handle cases when a family has several twin pairs.
        It doesn't seem to happen in HCP.
    monoz: sequence
        each element is a bool, True for monozygote.

    Returns
    -------
    twins: dict
        {pair_name: [subject1, subject2]}
    tmeta: dict
        {pair_name: {"monozygote": bool, "genre": "M"/"F"/"B"}}
    """
    done = set()
    num = 0
    for tt, tp, mono in zip(twin_types, df, monoz):
        tdic = {}
        twins = {}
        tmeta = {}

        for row in range(tp.shape[0]):
            subject = tp.index[row]
            if subject in done:
                continue
            mother = tp.Mother_ID.iloc[row]
            father = tp.Father_ID.iloc[row]
            other = tp[(tp.Mother_ID == mother) & (tp.Father_ID == father)]
            #            & (tp.Age_in_Yrs == tp.Age_in_Yrs.iloc[row])]
            if len(other) != 2:
                print('unmatching twins:', other.index)
            else:
                tname = 'twin_%04d' % num
                num += 1
                twins[tname] = [str(x) for x in sorted(other.index)]
                tmeta[tname] = {'monozygote': mono}
                p = participants[participants.index.isin(other.index)]
                if np.all(p.Gender == 'F'):
                    tmeta[tname]['genre'] = 'F'
                elif np.all(p.Gender == 'M'):
                    tmeta[tname]['genre'] = 'M'
                else:
                    tmeta[tname]['genre'] = 'B'
                    if mono:
                        print('WARNING: monozygotes with differing '
                              'gender !', other)
            done.update(other.index)
    return twins, tmeta


def euclidean_dist(emb1, emb2):
    return np.sqrt(np.sum((emb2 - emb1) ** 2, axis=1))


def cosine_dist(emb1, emb2):
    return 1. - np.sum(emb1 * emb2, axis=1) \
        / (np.sqrt(np.sum(emb1 ** 2, axis=1))
           * np.sqrt(np.sum(emb2 ** 2, axis=1)))


def twin_dist(twins, embeddings, dist_func):
    dist = {}
    for tname, twin_pair in twins.items():
        if twin_pair[0] in embeddings.index \
                    and twin_pair[1] in embeddings.index:
            emb1 = embeddings.loc[twin_pair[0]]
            emb2 = embeddings.loc[twin_pair[1]]
            d = dist_func(emb1.to_numpy().reshape(1, -1),
                          emb2.to_numpy().reshape(1, -1))[0]
            dist[tname] = d
    return dist


def random_non_twins_pairs(embeddings, twins, n=10000):
    done = set()
    twinp = set(tuple(x) for x in twins.values())
    while len(done) != n:
        m = n - len(done)
        s1 = np.random.choice(embeddings.index, size=m)
        s2 = np.random.choice(embeddings.index, size=m)
        pairs = np.sort(np.array((s1, s2)).T, axis=1)
        self_p = np.where(pairs[:, 0] == pairs[:, 1])
        pairs = [tuple(x) for i, x in enumerate(pairs) if i not in self_p[0]]
        done.update([x for x in pairs if x not in twinp])
    return done


def rank_dispersion(ranks):
    r = []
    for tname in ranks[:, 0]:
        r.append(np.where(ranks == tname)[0])
    r = np.array(r)
    avg = np.average(r, axis=1)
    # med = np.median(r, axis=1)
    std = np.std(r, axis=1)
    return (r, avg, std)


def find_nneighbour(embeddings, twins, dist_func, nneigh=1):
    tnames = set(np.array(list(twins.values())).ravel())
    inv_twins = {t[0]: k for k, t in twins.items()}
    inv_twins.update({t[1]: k for k, t in twins.items()})
    nneighbours = {}
    for s in tnames:
        try:
            emb1 = embeddings.loc[s]
        except KeyError:
            continue
        # dist = np.apply_along_axis(partial(dist_func, emb1), 1, embeddings)
        dist = dist_func(embeddings.to_numpy(), emb1.to_numpy().reshape(1, -1))
        adist = np.argsort(dist)
        neigh = []
        nneighbours[s] = neigh
        for ind in adist[1:nneigh+1]:
            neigh.append({'subject': embeddings.index[ind],
                          'distance': dist[ind],
                          'is_his_twin': (
                              inv_twins[s]
                              == inv_twins.get(embeddings.index[ind]))})
    return nneighbours


def draw_dist_stats(all_dists, labels=None, distnames=None, show_plots=True,
                    out_plots_dir=None):
    if not show_plots and out_plots_dir is None:
        return

    import matplotlib.pyplot as plt

    npl = all_dists[0].shape[1]
    bdata = [[p[:, i] for p in all_dists] for i in range(npl)]
    fig, axs = plt.subplots(1, npl)
    kw = {}
    if labels:
        kw = {'tick_labels': labels}
    colors = ['peachpuff', 'orange', 'tomato']
    for i in range(npl):
        bplot = axs[i].boxplot(bdata[i], patch_artist=True, **kw)
        if distnames:
            axs[i].set_title(distnames[i])
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    fig.set_tight_layout(True)

    if out_plots_dir is not None:
        if not osp.exists(out_plots_dir):
            os.makedirs(out_plots_dir)
        fig.savefig(osp.join(out_plots_dir, 'distances.svg'))
    if show_plots:
        fig.show()
    else:
        plt.close(fig)


def draw_dispersion(avg, std, style='bo', show_plots=True, out_plots_dir=None):
    if not show_plots and out_plots_dir is None:
        return

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes()
    plt.plot(avg, std, style)
    plt.xlabel('average rank')
    plt.ylabel('rank std')
    if out_plots_dir is not None:
        fig.savefig(osp.join(out_plots_dir, 'rank_dispersion.svg'))
    if show_plots:
        fig.show()
    else:
        plt.close(fig)


def draw_neighbours(nbcum, show_plots=True, out_plots_dir=None):
    if not show_plots and out_plots_dir is None:
        return

    import matplotlib.pyplot as plt

    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0f19cf', '#a2cf95'])

    fig = plt.figure()
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    for dname, p in nbcum.items():
        plt.plot(range(1, 6), p, label=dname)
    plt.xlabel('nb neighbours')
    plt.ylabel('% twin found')
    ax.legend(nbcum.keys(), loc='upper left', bbox_to_anchor=[1.05, 1.])
    if out_plots_dir is not None:
        fig.savefig(osp.join(out_plots_dir, 'twin_found.svg'))
    if show_plots:
        fig.show()
    else:
        plt.close(fig)


def draw_nregions(nbcum, show_plots=True, out_plots_dir=None):
    if not show_plots and out_plots_dir is None:
        return

    import matplotlib.pyplot as plt

    plt.rcParams['axes.prop_cycle'] = cycler.cycler('color', [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#0f19cf', '#a2cf95'])

    fig = plt.figure()
    ax = plt.axes()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    for dname, p in nbcum.items():
        plt.plot(range(1, len(p) + 1), p, label=dname)
    plt.xlabel('nb best regions')
    plt.ylabel('% twin found')
    ax.legend(nbcum.keys(), loc='upper left', bbox_to_anchor=[1.05, 1.])
    if out_plots_dir is not None:
        fig.savefig(osp.join(out_plots_dir, 'twin_found_nregions.svg'))
    if show_plots:
        fig.show()
    else:
        plt.close(fig)


def display_stats(summary, show_plots=True, out_plots_dir=None):
    if not show_plots and out_plots_dir is None:
        return

    import matplotlib.pyplot as plt

    regions = ['all'] + sorted(k for k in summary.keys() if k != 'all')
    #sep_byreg_best = [np.average(summary[r]['separability']
                                 #[summary[r]['best_dist_index']])
                      #for r in regions]

    twmz = []
    twdz = []
    for r in regions:
        fval = [v[0] for v in summary[r]['twin_found'].values()]
        val_best = np.argmax(fval[:6])
        twmz.append(fval[val_best])
        twdz.append(fval[val_best + 6])

    fig = plt.figure()
    plt.bar(np.arange(len(regions)) - 0.2, twmz, width=0.4, tick_label=regions)
    plt.bar(np.arange(len(regions)) + 0.2, twdz, width=0.4)
    ax = fig.axes[0]
    ax.tick_params(axis='x', labelrotation=80)
    fig.legend(['MZ', 'DZ'])
    fig.tight_layout()
    if out_plots_dir is not None:
        fig.savefig(osp.join(out_plots_dir, 'twin_found_all.svg'))
    if show_plots:
        fig.show()
    else:
        plt.close(fig)

    #import anatomist.api as ana
    #from soma import aims

    #a = ana.Anatomist()
    #model_l_f = aims.carto.Paths.findResourceFile(
        #'models/models_2019/descriptive_models/segments/talairach_spam_left/'
        #'meshes/Lspam_model_meshes_1.arg')
    #model_r_f = aims.carto.Paths.findResourceFile(
        #'models/models_2019/descriptive_models/segments/talairach_spam_left/'
        #'meshes/Rspam_model_meshes_1.arg')
    #model_l = a.loadObject(model_l_f)
    #model_r = a.loadObject(model_r_f)



def build_all_twin_distances(twins, embeddings_list, distances):
    nd = len(embeddings_list) * len(distances)
    all_dist = []
    sorted_twins = []

    for distance in distances:
        for embeddings in embeddings_list:
            dist = twin_dist(twins, embeddings, distance)
            sorted_twins.append(sorted(dist, key=lambda x: dist[x]))
            all_dist.append(list(dist.values()))
            if len(dist) != len(twins):
                to_remove = [k for k in twins if k not in dist]
                for k in to_remove:
                    del twins[k]

    return np.array(all_dist).T, np.array(sorted_twins).T


def dist_separability(dist_mz, dist_dz, dist_nt, niter=300):
    """ Compute a separability ratio between MZ and others and between twins
        and non-twins.
    """
    nmz = len(dist_mz)
    ndz = len(dist_dz)
    nt = nmz + ndz
    nnt = len(dist_nt)
    X = np.array(list(dist_mz) + list(dist_dz) + list(dist_nt)).reshape(-1, 1)
    clf = make_pipeline(StandardScaler(),
                        SGDClassifier(max_iter=1000, tol=1e-3))
    mz_mz_rates = []
    mz_nt_rates = []
    dz_dz_rates = []
    dz_nt_rates = []
    for i in range(int(np.ceil(niter / nmz))):
        tok = 0
        dok = 0
        nok = 0
        # LOO on mz
        for j in range(nmz):
            sj = list(range(X.shape[0]))
            dj = np.random.choice(ndz) + nmz - 1
            nj = np.random.choice(nnt) + nmz + ndz - 2
            del sj[nj]
            del sj[dj]
            del sj[j]
            Y = np.array([1] * (nmz - 1) + [2] * (ndz - 1) + [2] * (nnt - 1))
            clf.fit(X[sj, :], Y)
            Yp = clf.predict(X[[j, dj, nj]])
            if Yp[0] == 1:
                tok += 1
            if Yp[1] == 2:
                dok += 1
            if Yp[2] == 2:
                nok += 1
        mz_mz_rates.append(tok / nmz)
        mz_nt_rates.append((dok + nok) / (nmz * 2))

    for i in range(int(np.ceil(niter / nt))):
        tok = 0
        nok = 0
        for j in range(nt):
            sj = list(range(X.shape[0]))
            nj = np.random.choice(nnt) + nmz + ndz - 2
            del sj[nj]
            del sj[j]
            Y = np.array([1] * (nt - 1) + [2] * (nnt - 1))
            clf.fit(X[sj, :], Y)
            Yp = clf.predict(X[[j, nj]])
            if Yp[0] == 1:
                tok += 1
            if Yp[1] == 2:
                nok += 1
        dz_dz_rates.append(tok / nt)
        dz_nt_rates.append(nok / nt)

    mz_mz_rate = np.average(mz_mz_rates)
    mz_nt_rate = np.average(mz_nt_rates)
    dz_dz_rate = np.average(dz_dz_rates)
    dz_nt_rate = np.average(dz_nt_rates)

    return ((mz_mz_rate, mz_nt_rate),
            (dz_dz_rate, dz_nt_rate))


def twin_found_aggregative_regions(best_regions, region_embeddings,
                                   participants, twins, dz_twins, nontwins,
                                   summary, nmin=1, nmax=None):
    if nmax is None:
        nmax = len(best_regions)
    best_n = {}
    if nmin == 1:
        for k, v in summary[best_regions[0][1]]['twin_found'].items():
            best_n[k] = [v[0]]
    for i in range(nmin, nmax):
        regions = [x[1] for x in best_regions[:i + 1]]
        best_emb = pd.concat([region_embeddings[r] for r in regions], axis=1)
        missing = np.where(np.isnan(best_emb))
        if len(missing[0]) != 0:
            missing_s = np.unique(missing[0])
            best_emb.drop(index=best_emb.index[missing_s], inplace=True)
        out_sub_dir = osp.join(out_dir, f'best_{i}')
        sub_res = do_all(participants, twins, dz_twins, nontwins,
                         best_emb, out_dist_file=None,
                         out_plots_dir=out_sub_dir, out_dir=None,
                         show_plots=False, do_separability=False)
        with open(osp.join(out_sub_dir, 'regions.json'), 'w') as f:
            json.dump(regions, f)
        for k, v in sub_res['twin_found'].items():
            best_n[k].append(v[0])
    with open(osp.join(out_dir, 'best_regions.json'), 'w') as f:
        json.dump(best_n, f)
    draw_nregions(best_n, show_plots=False, out_plots_dir=out_dir)


def do_all(participants, twins, dz_twins, nontwins, embeddings,
           out_dist_file=None, show_plots=True, out_plots_dir=None,
           out_dir=None, do_separability=True):

    embed_pca = pd.DataFrame(sklearn.decomposition.PCA(
        n_components=0.99, whiten=True,
        svd_solver='full').fit_transform(embeddings),
        index=embeddings.index)
    embed_pca20 = pd.DataFrame(sklearn.decomposition.PCA(
        n_components=20, whiten=True,
        svd_solver='full').fit_transform(embeddings),
        index=embeddings.index)
    all_embeddings = [embeddings, embed_pca, embed_pca20]
    distances = [euclidean_dist, cosine_dist]

    embed_names = ['', 'PCA99%', 'PCA20']
    dist_names = ['eucl', 'cos']
    category_names = ['MZ', 'DZ', 'NT']

    all_dist, all_sorted = build_all_twin_distances(twins, all_embeddings,
                                                    distances)

    # compare to DZ
    all_dz_dist, all_dz_sorted = build_all_twin_distances(
        dz_twins, all_embeddings, distances)

    # compare to non-twins
    all_nt_dist, all_nt_sorted = build_all_twin_distances(
        nontwins, all_embeddings, distances)

    draw_dist_stats((all_dist, all_dz_dist, all_nt_dist),
                    labels=category_names,
                    distnames=('eucl.', 'PCA99%/eucl.', 'PCA20/eucl.', 'cos',
                               'PCA99%/cos', 'PCA20/cos'),
                    show_plots=show_plots, out_plots_dir=out_plots_dir)

    if do_separability:
        print('compute separability...')
        separability = [dist_separability(all_dist[:, i], all_dz_dist[:, i],
                                          all_nt_dist[:, i]) for i in range(6)]
        avg_sep = [np.average(s) for s in separability]
        best_dist_i = np.argmax(avg_sep)

        best_dist = '-'.join([x for x in [dist_names[best_dist_i // 3],
                                          embed_names[best_dist_i % 3]]
                              if x != ''])
        sorted_twins_best = all_sorted[:, best_dist_i]
        twinsm = {k: i for i, k in enumerate(twins)}
        dist_best = all_dist[:, best_dist_i]
        sorted_dz_best = all_dz_sorted[:, best_dist_i]
        dist_dz_best = all_dz_dist[:, best_dist_i]
        dz_twinsm = {k: i for i, k in enumerate(dz_twins)}

        if out_dist_file is not None:
            with open(out_dist_file % best_dist, 'w') as f:
                w = csv.writer(f, delimiter=',')
                w.writerow(['ID1', 'ID2', 'MZ', 'dist', 'rank'])
                for i, twin in enumerate(sorted_twins_best):
                    twin_i = twinsm[twin]
                    dist = dist_best[twin_i]
                    s1, s2 = twins[twin]
                    w.writerow([s1, s2, 'true', dist, i])
                for i, twin in enumerate(sorted_dz_best):
                    twin_i = dz_twinsm[twin]
                    dist = dist_dz_best[twin_i]
                    s1, s2 = dz_twins[twin]
                    w.writerow([s1, s2, 'false', dist, i])

    r, avg, std = rank_dispersion(ranks=all_sorted)
    draw_dispersion(avg, std, 'bo', show_plots, out_plots_dir)

    dist_set = [
        ('eucl_MZ', embeddings, twins, euclidean_dist),
        ('eucl_pca_MZ', embed_pca, twins, euclidean_dist),
        ('eucl_pca20_MZ', embed_pca20, twins, euclidean_dist),
        ('cos_MZ', embeddings, twins, cosine_dist),
        ('cos_pca_MZ', embed_pca, twins, cosine_dist),
        ('cos_pca20_MZ', embed_pca20, twins, cosine_dist),
        ('eucl_DZ', embeddings, dz_twins, euclidean_dist),
        ('eucl_pca_DZ', embed_pca, dz_twins, euclidean_dist),
        ('eucl_pca20_DZ', embed_pca20, dz_twins, euclidean_dist),
        ('cos_DZ', embeddings, dz_twins, cosine_dist),
        ('cos_pca_DZ', embed_pca, dz_twins, cosine_dist),
        ('cos_pca20_DZ', embed_pca20, dz_twins, cosine_dist),
    ]

    nbest = {}
    nearest = {}
    nneigh = 5
    nbcum = {}
    for dname, emb, tw, dis in dist_set:
        print('find nearest for', dname)
        nneighbours = find_nneighbour(emb, tw, dis, nneigh=nneigh)
        nearest[dname] = nneighbours
        nbestd = [len([s for s, ns in nneighbours.items()
                       if ns[i]['is_his_twin']])
                  for i in range(nneigh)]
        nbest[dname] = nbestd
        nbcum[dname] = np.cumsum(nbestd) / len(nneighbours)

    draw_neighbours(nbcum, show_plots=show_plots, out_plots_dir=out_plots_dir)

    res = {
        'embeddings': all_embeddings,
        'distances': {
            'mz': all_dist,
            'dz': all_dz_dist,
            'nt': all_nt_dist,
        },
        'sorted_twins': {
            'mz': all_sorted,
            'dz': all_dz_sorted,
            'nt': all_nt_sorted,
        },
        'dispersion': {
            'r': r,
            'avg': avg,
            'std': std,
        },
        'twin_found': nbcum,
    }
    if do_separability:
        res['separability'] = separability
        res['best_dist'] = best_dist
        res['best_dist_index'] = best_dist_i
    return res


class NpEncoder(json.JSONEncoder):
    """ Json encoder for numpy arrays
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def main():
    """ equivalent to calling the command champo_hcp_twin_distance.py
    """
    participants = pd.read_csv(participants_file, dtype={'Subject': str},
                               index_col='Subject')
    restricted = pd.read_csv(restricted_file,
                             dtype={'Subject': str, 'Mother_ID': str,
                                    'Father_ID': str}, index_col='Subject')

    mz = restricted[restricted.ZygosityGT == 'MZ']
    dz = restricted[restricted.ZygosityGT == 'DZ']
    # nt = restricted[restricted.ZygosityGT == ' ']

    twins, tmeta = get_twins(participants, twin_types=('mononzyg', ),
                             df=(mz, ), monoz=(True, ))
    dz_twins, dz_tmeta = get_twins(participants, twin_types=('dizyg', ),
                                   df=(dz, ), monoz=(False, ))
    all_twins = dict(twins)
    all_twins.update(dz_twins)

    regions_list = None
    if regions_list_f is not None:
        with open(regions_list_f) as f:
            regions_list = json.load(f)

    embeddings = read_embeddings(embeddings_dir, regions_list)

    nontwins = {'NT_%05d' % i: x
                for i, x in enumerate(
                    random_non_twins_pairs(embeddings, all_twins, n=10000))}

    res = do_all(participants, twins, dz_twins, nontwins, embeddings,
                 out_dist_file=None, out_plots_dir=out_dir, out_dir=out_dir,
                 show_plots=False)

    all_res = {'all': res}
    region_embeddings = {}

    if sub_embeddings:
        if regions_list is None:
            sub_emb_fl = glob.glob(osp.join(embeddings_dir, sub_embeddings))
            regions_list = {
                'brain': [osp.dirname(osp.dirname(f)) for f in sub_emb_fl]}

        for i, region_s in enumerate(regions_list['brain']):
            print(f'region {(i + 1) / len(regions_list["brain"])}: {region_s}')
            region = region_s.replace('.', '')
            sub_emb_fp = osp.join(embeddings_dir, region, sub_embeddings)
            sub_emb_f = glob.glob(sub_emb_fp)
            if len(sub_emb_f) == 0:
                continue
            sub_emb_f = sub_emb_f[0]
            out_sub_dir = osp.join(out_dir, region)
            if not osp.exists(out_sub_dir):
                os.makedirs(out_sub_dir)
            embeddings_r = read_embeddings(sub_emb_f)
            region_embeddings[region] = embeddings_r
            sub_res = do_all(participants, twins, dz_twins, nontwins,
                             embeddings_r,
                             out_dist_file=None, out_plots_dir=out_sub_dir,
                             out_dir=out_sub_dir, show_plots=False)
            all_res[region] = sub_res

        best_sep = 0
        best_sep_reg = None
        best_sep_dist = None
        best_find = 0
        best_find_reg = None
        best_find_dist = None
        reg_sep = {}
        reg_find = {}
        for region, res in all_res.items():
            if region == 'all':  # exclude all, it will win every time
                continue
            separability = res['separability']
            avg_sep = [np.average(s) for s in separability]
            best_dist_i = np.argmax(avg_sep)
            reg_sep[region] = avg_sep[best_dist_i]
            if avg_sep[best_dist_i] > best_sep:
                best_sep = avg_sep[best_dist_i]
                best_sep_reg = region
                best_sep_dist = best_dist_i
            twin_found = res['twin_found']
            t0 = [tf[0] for tf in twin_found.values()]
            tbest_i = np.argmax(t0)
            tbest = t0[tbest_i]
            reg_find[region] = tbest
            if best_find < tbest:
                best_find = tbest
                best_find_reg = region
                best_find_dist = tbest_i

        print('Best separability:', best_sep_reg, ', dist:', best_sep_dist,
              ', sep:', best_sep)
        print('Best twin find:', best_find_reg, ', dist:', best_find_dist,
              ', sep:', best_find)

    summary = {}
    for region, rdesc in all_res.items():
        rs = {k: v for k, v in rdesc.items()
              if k in ('separability', 'best_dist', 'best_dist_index',
                       'twin_found')}
        rs['distances'] = {
            'mz': rdesc['distances']['mz'],
            'dz': rdesc['distances']['dz']
        }
        rs['distances_avg_nt'] = np.average(rdesc['distances']['nt'])
        rs['sorted_twins'] = {
            'mz': rdesc['sorted_twins']['mz'],
            'dz': rdesc['sorted_twins']['dz'],
        }
        summary[region] = rs

    out_res_f = osp.join(out_dir, 'summary.json')
    with open(out_res_f, 'w') as f:
        json.dump(summary, f, cls=NpEncoder)

    display_stats(summary, out_plots_dir=out_dir, show_plots=False)

    # concatenate best regions
    best_regions = sorted([(np.max([y[0] for y in x['twin_found'].values()]),
                            k)
                           for k, x in summary.items() if k != 'all'],
                          reverse=True)
    twin_found_aggregative_regions(best_regions, region_embeddings,
                                   participants, twins, dz_twins, nontwins,
                                   summary, nmin=1, nmax=None)


if __name__ == '__main__':
    main()
