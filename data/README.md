# Data

Nothing in this folder is committed. GitHub rejects any file over 100 MB and
`FineFake.pkl` alone is ~186 MB.

## FineFake - primary corpus (train and test)

16,909 articles, 6 topics, 8 platforms, full body text included.

- Repository: https://github.com/Accuser907/FineFake
- Download the **text data** pickle from the Google Drive link in that repo.
- Place it here as `FineFake.pkl`.

The image and knowledge-graph files are **not needed** - this project is
text-only by design.

## ISOT - transfer test only (Approach D)

44,898 articles. Never trained on.

- https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/
- Place `True.csv` and `Fake.csv` here.

## After downloading

Run `notebooks/00_profile.ipynb` once. It writes a slimmed parquet of the
four columns actually used, so later loads take a second instead of a minute.
