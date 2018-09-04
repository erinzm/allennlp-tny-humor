import sys
import os
import pandas as pd

if __name__ == '__main__':
    summary_dir = sys.argv[1]
    summary_filenames = os.listdir(summary_dir)
    contests = set()
    for filename in summary_filenames:
        if 'munging' in filename:
            continue
        contests.add(int(filename.split('_')[0]))

    to_read = []
    for n in contests:
        for name in [f'{n}_summary_KLUCB.csv', f'{n}_summary_LilUCB.csv']:
            if name in summary_filenames:
                to_read.append((n, name))
                break

    dfs = []
    hierarchical_keys = []
    for contest_idx, filename in to_read:
        df = pd.read_csv(os.path.join(summary_dir, filename), index_col=0)
        # we don't care about target ids, and they're not universally added to the summary files anyway
        df = df.drop(columns='target_id', errors='ignore')
        # ranks aren't universally added either, and we'll recompute them from scores if we need them ┐(ツ)┌
        df = df.drop(columns='rank', errors='ignore')
        # some contests have a 'mean' column which is the same as the score column; just get rid of it
        df = df.drop(columns='mean', errors='ignore')
        df = df.drop(columns='contest', errors='ignore')

        # some contests have 'somewhat funny' columns instead of 'somewhat_funny'; normalize.
        df.rename(columns={'somewhat funny': 'somewhat_funny'}, inplace=True)

        # make sure dataframe is sorted by descending score
        df.sort_values(by='score', ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)

        # drop nan captions
        df = df[~df.caption.isna()]
        # and ones without letters
        df = df[df.caption.str.contains('[a-zA-Z]', regex=True)]

        dfs.append(df)
        hierarchical_keys.append(contest_idx)

    df = pd.concat(dfs, keys=hierarchical_keys, sort=False)
    df.index.set_names(['contest', 'idx'], inplace=True)

    df.to_csv('condensed.csv')
