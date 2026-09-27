import pandas as pd
import sys

def validate():
    try:
        # Load test set entities
        s1 = set(pd.read_csv('dataset/test/test_source1.tsv', sep='\t')['entity_id'])
        s2 = set(pd.read_csv('dataset/test/test_source2.tsv', sep='\t')['entity_id'])
        s3 = set(pd.read_csv('dataset/test/test_source3.tsv', sep='\t')['entity_id'])
        valid_s2_s3 = s2.union(s3)
        
        # Load output files
        matching = pd.read_csv('output/matching_results.tsv', sep='\t', dtype=str)
        candidate = pd.read_csv('output/candidate_pairs.tsv', sep='\t', dtype=str)
        
        errors = []
        
        # Check columns
        if list(matching.columns) != ['source1_entity_id', 'matched_entity_ids']:
            errors.append("matching_results.tsv columns are incorrect")
        if list(candidate.columns) != ['source1_entity_id', 'candidate_entity_ids']:
            errors.append("candidate_pairs.tsv columns are incorrect")
            
        # Check if every S1 entity is present exactly once
        matching_s1 = list(matching['source1_entity_id'])
        candidate_s1 = list(candidate['source1_entity_id'])
        
        if set(matching_s1) != s1:
            errors.append("matching_results.tsv missing or extra S1 entities")
        if len(matching_s1) != len(set(matching_s1)):
            errors.append("matching_results.tsv has duplicate S1 entities")
            
        if set(candidate_s1) != s1:
            errors.append("candidate_pairs.tsv missing or extra S1 entities")
        if len(candidate_s1) != len(set(candidate_s1)):
            errors.append("candidate_pairs.tsv has duplicate S1 entities")
            
        # Check rules for each row
        matching_dict = {}
        for _, row in matching.iterrows():
            s1_id = row['source1_entity_id']
            m_ids = row['matched_entity_ids']
            m_list = str(m_ids).split(',') if pd.notna(m_ids) and str(m_ids).strip() != '' else []
            matching_dict[s1_id] = m_list
            
            if len(m_list) != len(set(m_list)):
                errors.append(f"matching_results: duplicate IDs in list for {s1_id}")
            for mid in m_list:
                if mid not in valid_s2_s3:
                    errors.append(f"matching_results: invalid ID {mid} for {s1_id}")
                    
        candidate_dict = {}
        for _, row in candidate.iterrows():
            s1_id = row['source1_entity_id']
            c_ids = row['candidate_entity_ids']
            c_list = str(c_ids).split(',') if pd.notna(c_ids) and str(c_ids).strip() != '' else []
            candidate_dict[s1_id] = c_list
            
            if len(c_list) != len(set(c_list)):
                errors.append(f"candidate_pairs: duplicate IDs in list for {s1_id}")
            for cid in c_list:
                if cid not in valid_s2_s3:
                    errors.append(f"candidate_pairs: invalid ID {cid} for {s1_id}")
                    
        # Subset check
        for s1_id in s1:
            m_list = matching_dict.get(s1_id, [])
            c_list = candidate_dict.get(s1_id, [])
            for mid in m_list:
                if mid not in c_list:
                    errors.append(f"Subset error: {mid} matched for {s1_id} but not in candidates")
                    
        if errors:
            for e in errors[:20]:
                print(e)
            if len(errors) > 20:
                print(f"...and {len(errors)-20} more errors")
            sys.exit(1)
        else:
            print("PASS")
            
    except Exception as e:
        print(f"Exception: {e}")
        sys.exit(1)

if __name__ == '__main__':
    validate()
