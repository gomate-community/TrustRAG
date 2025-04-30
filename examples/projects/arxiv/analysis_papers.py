import pandas as pd

papers_df=pd.read_parquet("papers/papers_metadata.parquet")

print(papers_df.shape)
print(papers_df)
print(papers_df.columns)
print(papers_df.dtypes)

# Index(['entry_id', 'updated', 'published', 'title', 'authors', 'summary',
#        'comment', 'journal_ref', 'doi', 'primary_category', 'categories',
#        'links', 'pdf_url', 'download_time', 'content', 'topic'],
#       dtype='object')


print(papers_df.isnull().sum())

print(papers_df["content"])
print(papers_df["topic"].value_counts())