import pandas as pd
import os

print("Loading test data...")
s1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False)
s2 = pd.read_csv('dataset/test/test_source2.tsv', sep='\t', dtype=str, keep_default_na=False)
s3 = pd.read_csv('dataset/test/test_source3.tsv', sep='\t', dtype=str, keep_default_na=False)

s23 = pd.concat([s2, s3], ignore_index=True)

print("Performing simple exact name matching...")
s1['key'] = s1['business_name'].str.lower().str.strip()
s23['key'] = s23['business_name'].str.lower().str.strip()

# Group S2/S3 by exact lowercased name to form candidate lists
s23_grouped = s23[s23['key'] != ''].groupby('key')['entity_id'].apply(lambda x: ','.join(x.unique())).reset_index()
s23_grouped.rename(columns={'entity_id': 'matched_entity_ids'}, inplace=True)

res = s1.merge(s23_grouped, on='key', how='left')
res['matched_entity_ids'] = res['matched_entity_ids'].fillna('')

out_matching = res[['entity_id', 'matched_entity_ids']].rename(columns={'entity_id': 'source1_entity_id'})
out_candidate = out_matching.rename(columns={'matched_entity_ids': 'candidate_entity_ids'})

print("Writing TSV files...")
os.makedirs('output', exist_ok=True)
out_matching.to_csv('output/matching_results.tsv', sep='\t', index=False)
out_candidate.to_csv('output/candidate_pairs.tsv', sep='\t', index=False)
print("Success! TSV files generated in output/ directory.")
