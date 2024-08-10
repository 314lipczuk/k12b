import pandas as pd

def sample_snps(file_path, sample_fraction=0.01, output_file='sampled_snps.csv'):
    # Define data types for each column if known
    dtype_dict = {
        'rsid': 'str',
        'chromosome': 'str',
        'position': 'int',
        'genotype': 'str'
    }
    
    snp_df = pd.read_csv(file_path, dtype=dtype_dict)
    
    # Randomly sample a fraction of SNPs
    sampled_snp_df = snp_df.sample(frac=sample_fraction, random_state=42)
    
    # Save the sampled SNPs to a CSV file
    sampled_snp_df.to_csv(output_file, index=False)

file_path = 'k12b/samples/genome_Melinda_Chaperlo_v5_Full_20240730223601.csv'  # Replace with your file path
sample_snps(file_path, sample_fraction=0.01, output_file='sampled_snps.csv')  # Sample 1% of SNPs
